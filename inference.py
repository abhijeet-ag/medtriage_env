"""
inference.py — MedTriageEnv baseline evaluation script.

Run from the project root with the server already running at localhost:8000:
    python inference.py

Required environment variables:
    API_BASE_URL   — Base URL of the inference endpoint
    MODEL_NAME     — Model identifier
    HF_TOKEN       — HuggingFace / gateway bearer token

Stdout format (mandatory per hackathon spec):
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
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
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME:   str = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN:     str = os.environ.get("HF_TOKEN",     "")

ENV_SERVER_URL: str    = "http://localhost:8000"
BENCHMARK:      str    = "medtriage_env"
EPISODE_TIMEOUT_SECONDS: int = 360
MAX_STEPS: int = 20
SUCCESS_SCORE_THRESHOLD: float = 0.1

# ---------------------------------------------------------------------------
# Mandatory stdout logging (hackathon spec)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # Sanitise action string — no newlines allowed on a single [STEP] line
    action_clean = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
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

FALLBACK_ACTION: Dict[str, Any] = {
    "action_type": "ask_question",
    "question": "Can you describe your main symptom?",
}

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_env() -> None:
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
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def _call_llm(client: OpenAI, messages: List[Dict[str, str]]) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=256,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        print(f"    [LLM error] {exc}", flush=True)
        return ""

# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

_JSON_PATTERN: re.Pattern = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_action(raw_text: str) -> Dict[str, Any]:
    match = _JSON_PATTERN.search(raw_text)
    if not match:
        return dict(FALLBACK_ACTION)
    try:
        action = json.loads(match.group())
    except json.JSONDecodeError:
        return dict(FALLBACK_ACTION)
    if action.get("action_type") not in {"ask_question", "order_test", "submit_diagnosis"}:
        return dict(FALLBACK_ACTION)
    return action

# ---------------------------------------------------------------------------
# Environment client
# ---------------------------------------------------------------------------

class MedTriageClient:
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
        obs["_reward"] = result.reward or 0.0
        return obs

    def close(self):
        try:
            self._sync_client.__exit__(None, None, None)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Observation formatter
# ---------------------------------------------------------------------------

def _obs_to_text(obs: Dict[str, Any]) -> str:
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
# Agent loop — runs one episode, emits [START]/[STEP]/[END]
# ---------------------------------------------------------------------------

def run_episode(
    llm_client: OpenAI,
    env_client: MedTriageClient,
    difficulty: str,
) -> Tuple[float, Dict[str, Any]]:

    episode_start = time.monotonic()
    rewards: List[float] = []
    steps_taken = 0
    success = False
    final_score = 0.0
    result_dict: Dict[str, Any] = {}

    log_start(task=difficulty, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env_client.reset(difficulty=difficulty)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _obs_to_text(obs)},
        ]

        for step in range(1, MAX_STEPS + 1):
            elapsed = time.monotonic() - episode_start
            error_str: Optional[str] = None

            if elapsed > EPISODE_TIMEOUT_SECONDS:
                action: Dict[str, Any] = {
                    "action_type": "submit_diagnosis",
                    "diagnosis": "unspecified — timeout",
                    "triage_level": "urgent",
                }
                error_str = "episode_timeout"
            else:
                raw_response = _call_llm(llm_client, messages)
                action = _parse_action(raw_response)
                messages.append({"role": "assistant", "content": raw_response or json.dumps(action)})

            obs = env_client.step(action)
            reward = float(obs.get("_reward", 0.0))
            done   = bool(obs.get("episode_done", False))

            rewards.append(reward)
            steps_taken = step

            # Emit mandatory [STEP] line
            action_str = json.dumps(action)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_str)

            messages.append({"role": "user", "content": _obs_to_text(obs)})

            if done:
                result_dict  = obs.get("info", {})
                final_score  = float(obs.get("cumulative_reward", 0.0))
                if "total_score" in result_dict:
                    final_score = float(result_dict["total_score"])
                break

            if step == MAX_STEPS:
                result_dict = obs.get("info", {})
                final_score = float(obs.get("cumulative_reward", 0.0))

        success = final_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        error_str = str(exc)
        # Emit a final STEP line so the log is never truncated mid-episode
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=error_str)

    finally:
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

    return final_score, result_dict

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DIFFICULTIES: List[str] = ["easy", "medium", "hard"]


def main() -> None:
    _validate_env()

    print("\n=== MedTriageEnv Baseline Evaluation ===\n", flush=True)

    llm_client = _build_llm_client()
    env_client = MedTriageClient(ENV_SERVER_URL)

    scores: Dict[str, float] = {}

    for difficulty in DIFFICULTIES:
        print(f"\n--- Episode: {difficulty} ---", flush=True)
        try:
            score, result = run_episode(llm_client, env_client, difficulty)
            scores[difficulty] = score

            # Human-readable breakdown (not part of mandatory format)
            breakdown = result.get("breakdown", {})
            if breakdown:
                sym  = breakdown.get("symptom",  {})
                tst  = breakdown.get("test",      {})
                diag = breakdown.get("diagnosis", {})
                wc   = breakdown.get("weighted_contributions", {})
                print(
                    f"    symptom  {result.get('symptom_score',  0.0):.2f} "
                    f"(covered {sym.get('covered_count','?')}/{sym.get('total','?')})"
                    f"  x0.30 -> {wc.get('symptom', 0.0):.3f}",
                    flush=True,
                )
                print(
                    f"    test     {result.get('test_score', 0.0):.2f} "
                    f"(hit {len(tst.get('hit',[]))}/{len(tst.get('gold_key_tests',[]))},"
                    f" penalty {tst.get('penalty', 0.0):.2f})"
                    f"  x0.30 -> {wc.get('test', 0.0):.3f}",
                    flush=True,
                )
                print(
                    f"    diagnosis {result.get('diagnosis_score', 0.0):.2f} "
                    f"({diag.get('match_type','?')},"
                    f" triage {'Y' if diag.get('triage_correct') else 'N'})"
                    f"  x0.40 -> {wc.get('diagnosis', 0.0):.3f}",
                    flush=True,
                )
        except Exception as exc:
            scores[difficulty] = 0.0
            print(f"Episode failed: {exc}", flush=True)

    env_client.close()

    # Summary
    average = sum(scores.values()) / len(scores) if scores else 0.0
    print("\nBASELINE SCORES:", flush=True)
    for difficulty in DIFFICULTIES:
        print(f"{difficulty:<8} {scores.get(difficulty, 0.0):.2f}", flush=True)
    print(f"{'average':<8} {average:.2f}", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
