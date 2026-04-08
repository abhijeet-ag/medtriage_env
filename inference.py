"""
inference.py — MedTriageEnv baseline evaluation script.

Run from the project root:
    python hz/medtriage_env/inference.py

Required environment variables:
    API_BASE_URL   — Base URL of the inference endpoint (e.g. https://api-inference.huggingface.co/models/xxx)
    MODEL_NAME     — Model identifier forwarded in every chat-completion request
    HF_TOKEN       — HuggingFace / gateway bearer token

The script:
    1. Spins up one episode per difficulty tier (easy, medium, hard) against the
       MedTriageEnv server running at localhost:8000.
    2. Uses a thin LLM agent that reads observations and emits JSON actions.
    3. Prints per-episode scores and a final summary table.

Disqualification guards (enforced at import time):
    - No secret values hard-coded — all config via os.environ.
    - Total wall-clock budget: 20 minutes (enforced per episode with a timeout).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — all values from environment
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "")
MODEL_NAME:   str = os.environ.get("MODEL_NAME",   "")
HF_TOKEN:     str = os.environ.get("HF_TOKEN",     "")

ENV_SERVER_URL: str = "http://localhost:8000"

# Per-episode wall-clock budget (seconds).  3 difficulties × 6 min = 18 min < 20 min limit.
EPISODE_TIMEOUT_SECONDS: int = 360

# Maximum steps the agent will attempt before force-submitting a diagnosis.
# Must match MedTriageState.max_steps (default 20).
MAX_STEPS: int = 20

# ---------------------------------------------------------------------------
# System prompt (verbatim as specified in the hackathon brief)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are a clinical diagnostic AI agent. Your goal is to:
1. Ask the patient targeted questions to understand their symptoms (action_type: ask_question)
2. Order the most relevant diagnostic tests (action_type: order_test)
3. Submit a final diagnosis and triage level (action_type: submit_diagnosis)

Always respond with valid JSON matching exactly one of these formats:
{"action_type": "ask_question", "question": "Do you have chest pain?"}
{"action_type": "order_test", "test_name": "ECG"}
{"action_type": "submit_diagnosis", "diagnosis": "STEMI", "triage_level": "immediate"}
"""

# Default action used whenever the LLM returns unparseable output.
FALLBACK_ACTION: Dict[str, Any] = {
    "action_type": "ask_question",
    "question": "Can you describe your main symptom?",
}

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_env() -> None:
    """Abort early with a clear message if any required variable is missing."""
    missing = [name for name, val in [
        ("API_BASE_URL", API_BASE_URL),
        ("MODEL_NAME",   MODEL_NAME),
        ("HF_TOKEN",     HF_TOKEN),
    ] if not val]
    if missing:
        sys.exit(
            f"[inference.py] Missing required environment variable(s): {', '.join(missing)}\n"
            "Export them before running this script."
        )

# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def _build_llm_client() -> OpenAI:
    """Return an OpenAI-compatible client pointed at API_BASE_URL."""
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def _call_llm(client: OpenAI, messages: List[Dict[str, str]]) -> str:
    """Call the chat-completion endpoint and return the raw assistant text.

    Returns an empty string on any API error so the caller can fall back
    gracefully without crashing the episode.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=256,
            temperature=0.2,  # low temperature for deterministic clinical reasoning
        )
        return response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        print(f"    [LLM error] {exc}")
        return ""

# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

_JSON_PATTERN: re.Pattern = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_action(raw_text: str) -> Dict[str, Any]:
    """Extract the first JSON object from *raw_text* and return it as a dict.

    Parsing strategy:
      1. Find the first {...} block via regex.
      2. Attempt json.loads on that block.
      3. Validate that the result contains a recognised "action_type".
      4. Fall back to FALLBACK_ACTION on any failure.

    This function must never raise — bad LLM output should degrade gracefully.
    """
    match = _JSON_PATTERN.search(raw_text)
    if not match:
        return dict(FALLBACK_ACTION)

    try:
        action = json.loads(match.group())
    except json.JSONDecodeError:
        return dict(FALLBACK_ACTION)

    valid_types = {"ask_question", "order_test", "submit_diagnosis"}
    if action.get("action_type") not in valid_types:
        return dict(FALLBACK_ACTION)

    return action

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# MedTriageEnv WebSocket client (wraps the typed EnvClient)
# ---------------------------------------------------------------------------

class MedTriageClient:
    """Synchronous WebSocket client for MedTriageEnv using the typed EnvClient."""

    def __init__(self, base_url: str = ENV_SERVER_URL) -> None:
        from medtriage_env.client import MedTriageEnv as _MedTriageEnv
        self._sync_client = _MedTriageEnv(base_url=base_url).sync()
        self._ctx = self._sync_client.__enter__()

    def reset(self, difficulty: str) -> Dict[str, Any]:
        result = self._ctx.reset(difficulty=difficulty)
        return result.observation.model_dump()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        from medtriage_env.models import MedTriageAction
        act = MedTriageAction(**action)
        result = self._ctx.step(act)
        obs = result.observation.model_dump()
        obs["episode_done"] = result.done
        return obs

    def close(self):
        try:
            self._sync_client.__exit__(None, None, None)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Observation → human-readable string for the LLM conversation
# ---------------------------------------------------------------------------

def _obs_to_text(obs: Dict[str, Any]) -> str:
    """Format an environment observation as a user-turn message for the LLM.

    Includes the observation type, content, available actions, and any
    relevant info fields so the agent has full situational awareness.
    """
    lines: List[str] = []

    obs_type = obs.get("observation_type", "unknown")
    content  = obs.get("content", "")
    available = obs.get("available_actions", [])
    info      = obs.get("info", {})

    lines.append(f"[{obs_type.upper()}] {content}")

    if info.get("steps_remaining") is not None:
        lines.append(f"Steps remaining: {info['steps_remaining']}")

    if info.get("tests_ordered"):
        lines.append(f"Tests ordered so far: {', '.join(info['tests_ordered'])}")

    if info.get("questions_asked"):
        lines.append(f"Questions asked so far: {len(info['questions_asked'])}")

    if available:
        lines.append(f"Available actions: {', '.join(available)}")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_episode(
    llm_client: OpenAI,
    env_client: MedTriageClient,
    difficulty: str,
) -> Tuple[float, Dict[str, Any]]:
    """Run a single episode and return (final_score, result_dict).

    The agent maintains a running conversation (system + alternating
    user/assistant turns) so the LLM has full episode context.

    The episode terminates when:
      - The environment sets episode_done=True, OR
      - The agent has taken MAX_STEPS steps, OR
      - EPISODE_TIMEOUT_SECONDS wall-clock seconds have elapsed
        (the agent force-submits a generic diagnosis to close the episode).

    Args:
        llm_client: Configured OpenAI-compatible inference client.
        env_client: Configured MedTriageEnv HTTP client.
        difficulty: "easy", "medium", or "hard".

    Returns:
        (score, result_dict) where result_dict contains episode metadata and
        the full breakdown returned by the grader.
    """
    episode_start = time.monotonic()

    # ── Reset environment ────────────────────────────────────────────────────
    obs = env_client.reset(difficulty=difficulty)

    # Build the conversation history.  The system prompt is prepended once.
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": _obs_to_text(obs)},
    ]

    final_score  = 0.0
    result_dict: Dict[str, Any] = {}

    for step in range(MAX_STEPS):
        # ── Timeout guard ────────────────────────────────────────────────────
        elapsed = time.monotonic() - episode_start
        if elapsed > EPISODE_TIMEOUT_SECONDS:
            print(f"    [timeout] Episode exceeded {EPISODE_TIMEOUT_SECONDS}s — force-submitting.")
            # Force a submission so the environment can score what we have.
            action: Dict[str, Any] = {
                "action_type": "submit_diagnosis",
                "diagnosis": "unspecified — timeout",
                "triage_level": "urgent",
            }
        else:
            # ── Ask the LLM ──────────────────────────────────────────────────
            raw_response = _call_llm(llm_client, messages)
            action = _parse_action(raw_response)

            # Append assistant turn so future calls see the full conversation.
            messages.append({"role": "assistant", "content": raw_response or json.dumps(action)})

        # ── Send action to environment ────────────────────────────────────────
        obs = env_client.step(action)

        # Append environment response as the next user turn.
        obs_text = _obs_to_text(obs)
        messages.append({"role": "user", "content": obs_text})

        # ── Check terminal conditions ─────────────────────────────────────────
        if obs.get("episode_done", False):
            # The grader score lives in the observation's info dict on episode end.
            result_dict = obs.get("info", {})
            final_score = float(obs.get("cumulative_reward", 0.0))
            # Some server implementations put the final graded score here:
            if "total_score" in result_dict:
                final_score = float(result_dict["total_score"])
            break

        # If we reach MAX_STEPS without episode_done, the loop exits naturally.
        # The cumulative reward at that point is used as the score.
        if step == MAX_STEPS - 1:
            result_dict = obs.get("info", {})
            final_score = float(obs.get("cumulative_reward", 0.0))

    return final_score, result_dict

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DIFFICULTIES: List[str] = ["easy", "medium", "hard"]


def main() -> None:
    """Entry point — run all three episodes and print the score table."""
    _validate_env()

    print("\n=== MedTriageEnv Baseline Evaluation ===\n")

    llm_client = _build_llm_client()

    scores: Dict[str, float] = {}

    env_client = MedTriageClient(ENV_SERVER_URL)
    for difficulty in DIFFICULTIES:
        label = f"Running {difficulty} episode..."
        # Pad label to a fixed width so the "Score:" column lines up.
        print(f"{label:<28}", end="", flush=True)

        try:
            score, result = run_episode(llm_client, env_client, difficulty)
            scores[difficulty] = score

            print(f"Score: {score:.2f}")

            # Print the grader breakdown for debugging — indented so it doesn't
            # clutter the summary table that follows.
            breakdown = result.get("breakdown", {})
            if breakdown:
                sym  = breakdown.get("symptom",   {})
                tst  = breakdown.get("test",       {})
                diag = breakdown.get("diagnosis",  {})
                wc   = breakdown.get("weighted_contributions", {})
                print(
                    f"    symptom  {result.get('symptom_score',  0.0):.2f} "
                    f"(covered {sym.get('covered_count', '?')}/{sym.get('total', '?')})"
                    f"  ×0.30 → {wc.get('symptom', 0.0):.3f}"
                )
                print(
                    f"    test     {result.get('test_score',     0.0):.2f} "
                    f"(hit {len(tst.get('hit', []))}/{len(tst.get('gold_key_tests', []))},"
                    f" penalty {tst.get('penalty', 0.0):.2f})"
                    f"  ×0.30 → {wc.get('test', 0.0):.3f}"
                )
                print(
                    f"    diagnosis {result.get('diagnosis_score', 0.0):.2f} "
                    f"({diag.get('match_type', '?')},"
                    f" triage {'✓' if diag.get('triage_correct') else '✗'})"
                    f"  ×0.40 → {wc.get('diagnosis', 0.0):.3f}"
                )

        except Exception as exc:  # noqa: BLE001
            # A failed episode does not crash the script — score 0.0 is recorded.
            scores[difficulty] = 0.0
            print(f"Score: 0.00  [ERROR: {exc}]")

    # ── Summary table ────────────────────────────────────────────────────────
    average = sum(scores.values()) / len(scores) if scores else 0.0

    print()
    print("BASELINE SCORES:")
    for difficulty in DIFFICULTIES:
        print(f"{difficulty:<8} {scores.get(difficulty, 0.0):.2f}")
    print(f"{'average':<8} {average:.2f}")
    print()


if __name__ == "__main__":
    main()

