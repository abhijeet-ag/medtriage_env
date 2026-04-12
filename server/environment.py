from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any, List, Optional
from dataclasses import dataclass
from openenv.core.env_server.interfaces import Environment
from dataclasses import dataclass


from medtriage_env.models import MedTriageAction, MedTriageObservation, MedTriageState

from server.graders import MedTriageGrader, _symptom_covered


def _norm(s: Any) -> str:
    return str(s).lower().strip().replace("_", " ").replace("-", " ")

def _norm_compact(s: Any) -> str:
    """Norm with all spaces removed for fuzzy matching."""
    return _norm(s).replace(" ", "")


def _with_cumulative_reward(obs: MedTriageObservation, cumulative: float) -> MedTriageObservation:
    try:
        if hasattr(obs, "model_copy"):
            return obs.model_copy(update={"cumulative_reward": cumulative}, deep=True)
        return obs.copy(update={"cumulative_reward": cumulative})
    except Exception:
        try:
            data = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
            data["cumulative_reward"] = cumulative
            return MedTriageObservation(**data)
        except Exception:
            return obs


def _best_symptom_response_key(question: str, symptom_responses: dict) -> Optional[str]:
    """Pick the longest matching key using case-insensitive substring overlap."""
    if not str(question).strip() or not symptom_responses:
        return None
    q = _norm(question)
    matches: List[Any] = []
    for key in symptom_responses:
        k = _norm(key)
        if not k:
            continue
        if k in q or q in k:
            matches.append(key)
    if not matches:
        return None
    return max(matches, key=lambda k: len(_norm(k)))


def _ci_in_list(name: str, items: List[str]) -> Optional[str]:
    """Return the canonical list entry if *name* matches any item case-insensitively."""
    nl = _norm(name)
    # Exact match first
    for item in items:
        if _norm(item) == nl:
            return item
    # Substring fallback: item contained in name or name contained in item
    for item in items:
        ik = _norm(item)
        if ik in nl or nl in ik:
            return item
    # Compact fallback: remove all spaces for xray vs x-ray style mismatches
    nl_compact = _norm_compact(name)
    for item in items:
        if _norm_compact(item) == nl_compact:
            return item
    # Partial word overlap fallback for "CT scan of the head" vs "CT_head"
    nl_words = set(nl.split())
    for item in items:
        ik_words = set(_norm(item).split())
        if ik_words and ik_words.issubset(nl_words):
            return item
    return None


def _lookup_test_result(test_name: str, test_results: dict) -> Optional[str]:
    """Resolve test result string; keys matched case-insensitively."""
    if not test_results:
        return None
    hit = _ci_in_list(test_name, list(test_results.keys()))
    if hit is not None:
        return str(test_results[hit])
    return None


@dataclass
class StepResult:
    observation: MedTriageObservation
    reward: float
    done: bool


class MedTriageEnvironment(Environment):
    """Clinical triage simulation: questions, tests, and diagnosis submission."""

    max_steps: int = 20

    def __init__(self) -> None:
        super().__init__()
        self.cases: List[dict] = []
        self.easy_cases: List[dict] = []
        self.medium_cases: List[dict] = []
        self.hard_cases: List[dict] = []
        self.current_case: Optional[dict] = None
        self.current_state: Optional[MedTriageState] = None
        self._cumulative_reward: float = 0.0
        self._terminated: bool = False

        cases_path = Path(__file__).resolve().parent.parent / "data" / "cases.json"
        try:
            raw = cases_path.read_text(encoding="utf-8")
            self.cases = json.loads(raw)
            if not isinstance(self.cases, list):
                raise ValueError("cases.json must contain a JSON array of case objects")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"MedTriageEnvironment could not find case data at {cases_path}. "
                "Create data/cases.json with a list of case dicts."
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"MedTriageEnvironment failed to parse JSON at {cases_path}: {e}"
            ) from e

        for c in self.cases:
            tier = str(c.get("difficulty", "")).lower()
            if tier == "easy":
                self.easy_cases.append(c)
            elif tier == "medium":
                self.medium_cases.append(c)
            elif tier == "hard":
                self.hard_cases.append(c)

    def reset(self, difficulty: str = "random") -> MedTriageObservation:
        episode_id = str(uuid.uuid4())
        case = self._select_case(difficulty)
        self.current_case = case
        self._cumulative_reward = 0.0
        self._terminated = False

        case_id = str(case.get("case_id", case.get("id", "unknown")))
        diff = str(case.get("difficulty", "unknown"))

        self.current_state = MedTriageState(
            episode_id=episode_id,
            case_id=case_id,
            difficulty=diff,
            step_count=0,
            max_steps=self.max_steps,
            questions_asked=[],
            tests_ordered=[],
            diagnosis_submitted=False,
            final_score=None,
        )

        content = self._format_initial_patient_message(case)
        return MedTriageObservation(
            observation_type="patient_response",
            content=content,
            available_actions=["ask_question", "order_test", "submit_diagnosis"],
            step_reward=0.0,
            cumulative_reward=self._cumulative_reward,
            episode_done=False,
            info={
                "episode_id": episode_id,
                "case_id": case_id,
                "steps_remaining": self.max_steps,
            },
        )

    def step(self, action: MedTriageAction) -> MedTriageObservation:
        if self.current_state is None or self.current_case is None:
            raise RuntimeError("Call reset() first")
        if self._terminated:
            return MedTriageObservation(
                observation_type="error",
                content="Episode has already ended.",
                available_actions=[],
                step_reward=0.0,
                cumulative_reward=self._cumulative_reward,
                episode_done=True,
                info={"final_score": self.current_state.final_score},
            )

        if action is None:
            result = self._error_obs("step() requires a MedTriageAction; got None.")
        else:
            try:
                if action.action_type == "ask_question":
                    result = self._step_ask_question(action)
                elif action.action_type == "order_test":
                    result = self._step_order_test(action)
                elif action.action_type == "submit_diagnosis":
                    result = self._step_submit_diagnosis(action)
                else:
                    result = self._error_obs(f"Unknown action_type: {action.action_type!r}")
            except Exception as e:
                result = self._error_obs(f"Unexpected error: {e}")

        obs, step_reward, done = result
        self._cumulative_reward += step_reward
        obs = _with_cumulative_reward(obs, self._cumulative_reward)
        if done:
            self._terminated = True
        return obs

    def state(self) -> MedTriageState:
        if self.current_state is None:
            raise RuntimeError("Call reset() first")
        return self.current_state

    # --- internals ---

    def _select_case(self, difficulty: str) -> dict:
        d = difficulty.lower()
        if d == "random":
            pool: List[dict] = self.cases
        elif d == "easy":
            pool = self.easy_cases
        elif d == "medium":
            pool = self.medium_cases
        elif d == "hard":
            pool = self.hard_cases
        else:
            raise ValueError(
                f"difficulty must be 'easy', 'medium', 'hard', or 'random', got {difficulty!r}"
            )
        if not pool:
            raise RuntimeError(f"No cases available for difficulty {difficulty!r}")
        return random.choice(pool)

    def _format_initial_patient_message(self, case: dict) -> str:
        chief = case.get("chief_complaint", "")
        age = case.get("age", "?")
        gender = case.get("gender", "?")
        vit = case.get("vitals") if isinstance(case.get("vitals"), dict) else case
        bp = vit.get("bp", case.get("bp", "?"))
        hr = vit.get("hr", case.get("hr", "?"))
        temp = vit.get("temp", case.get("temp", "?"))
        rr = vit.get("rr", case.get("rr", "?"))
        spo2 = vit.get("spo2", case.get("spo2", "?"))
        return (
            f"Patient: {chief}\n"
            f"Age: {age}, {gender}\n"
            f"Vitals: BP {bp}, HR {hr}, Temp {temp}°C, RR {rr}, SpO2 {spo2}%"
        )

    def _info_dict(self) -> dict[str, Any]:
        assert self.current_state is not None
        s = self.current_state
        return {
            "episode_id": s.episode_id,
            "case_id": s.case_id,
            "questions_asked": list(s.questions_asked),
            "tests_ordered": list(s.tests_ordered),
            "steps_remaining": max(0, s.max_steps - s.step_count),
            "final_score": s.final_score,
        }

    def _error_obs(self, msg: str) -> tuple[MedTriageObservation, float, bool]:
        obs = MedTriageObservation(
            observation_type="error",
            content=msg,
            available_actions=["ask_question", "order_test", "submit_diagnosis"],
            step_reward=0.0,
            cumulative_reward=self._cumulative_reward,
            episode_done=False,
            info=self._info_dict(),
        )
        return obs, 0.0, False

    def _question_covers_critical_symptom(self, question: str) -> bool:
        hints = self.current_case.get("scoring_hints", {}) or {}
        critical = hints.get("critical_symptoms", []) or []
        if not isinstance(critical, list):
            return False
        for sym in critical:
            if _symptom_covered(str(sym), [question]):
                return True
        return False

    def _maybe_force_episode_end(
        self,
        base_obs: MedTriageObservation,
        step_reward: float,
    ) -> tuple[MedTriageObservation, float, bool]:
        assert self.current_state is not None
        st = self.current_state
        if st.step_count >= st.max_steps and not st.diagnosis_submitted:
            st.final_score = 0.0
            updates = {
                "observation_type": "episode_end",
                "episode_done": True,
                "available_actions": [],
                "info": {
                    **dict(base_obs.info),
                    **self._info_dict(),
                    "reason": "max_steps_reached",
                },
            }
            if hasattr(base_obs, "model_copy"):
                obs = base_obs.model_copy(update=updates, deep=True)
            else:
                obs = base_obs.copy(update=updates)
            return obs, step_reward, True
        return base_obs, step_reward, False

    def _step_ask_question(
        self, action: MedTriageAction
    ) -> tuple[MedTriageObservation, float, bool]:
        assert self.current_case is not None and self.current_state is not None
        if action.question is None:
            return self._error_obs("ask_question requires action.question to be set.")

        symptom_responses = self.current_case.get("symptom_responses") or {}
        if not isinstance(symptom_responses, dict):
            symptom_responses = {}

        best_key = _best_symptom_response_key(action.question, symptom_responses)
        if best_key is None:
            reply = "The patient says: I don't understand that question."
        else:
            reply = str(symptom_responses[best_key])

        self.current_state.questions_asked.append(action.question)
        reward = 0.05 if self._question_covers_critical_symptom(action.question) else 0.01
        self.current_state.step_count += 1

        obs = MedTriageObservation(
            observation_type="patient_response",
            content=reply,
            available_actions=["ask_question", "order_test", "submit_diagnosis"],
            step_reward=reward,
            cumulative_reward=self._cumulative_reward,
            episode_done=False,
            info=self._info_dict(),
        )
        return self._maybe_force_episode_end(obs, reward)

    def _step_order_test(
        self, action: MedTriageAction
    ) -> tuple[MedTriageObservation, float, bool]:
        assert self.current_case is not None and self.current_state is not None
        if action.test_name is None:
            return self._error_obs("order_test requires action.test_name to be set.")

        available = self.current_case.get("available_tests") or []
        if not isinstance(available, list):
            available = []

        canonical = _ci_in_list(action.test_name, [str(x) for x in available])
        if canonical is None:
            return self._error_obs("That test is not available for this patient.")

        ordered_lower = {_norm(t) for t in self.current_state.tests_ordered}
        if _norm(canonical) in ordered_lower:
            return self._error_obs("You already ordered that test.")

        test_results = self.current_case.get("test_results") or {}
        if not isinstance(test_results, dict):
            test_results = {}
        result_text = _lookup_test_result(canonical, test_results)
        if result_text is None:
            result_text = f"No result recorded for {canonical!r}."

        self.current_state.tests_ordered.append(canonical)
        gold_keys = self.current_case.get("gold_key_tests") or []
        gold_lower = {_norm(str(g)) for g in gold_keys}
        reward = 0.1 if _norm(canonical) in gold_lower else 0.02
        self.current_state.step_count += 1

        obs = MedTriageObservation(
            observation_type="test_result",
            content=result_text,
            available_actions=["ask_question", "order_test", "submit_diagnosis"],
            step_reward=reward,
            cumulative_reward=self._cumulative_reward,
            episode_done=False,
            info=self._info_dict(),
        )
        return self._maybe_force_episode_end(obs, reward)

    def _step_submit_diagnosis(
        self, action: MedTriageAction
    ) -> tuple[MedTriageObservation, float, bool]:
        assert self.current_case is not None and self.current_state is not None
        diagnosis = action.diagnosis
        triage_level = action.triage_level

        try:
            grader = MedTriageGrader(self.current_case)
            grade: dict[str, Any] = grader.grade_episode(
                questions_asked=list(self.current_state.questions_asked),
                tests_ordered=list(self.current_state.tests_ordered),
                submitted_diagnosis=diagnosis,
                submitted_triage=triage_level,
            )
        except Exception as e:
            return self._error_obs(f"Grading failed: {e}")

        try:
            total = float(grade["total_score"])
        except (KeyError, TypeError, ValueError) as e:
            return self._error_obs(f"Invalid grader result (missing total_score): {e}")
        self.current_state.diagnosis_submitted = True
        self.current_state.step_count += 1
        self.current_state.final_score = total

        try:
            grade_json = json.dumps(grade, indent=2, default=str)
        except (TypeError, ValueError):
            grade_json = repr(grade)

        content = (
            f"Episode graded.\nFinal score: {total:.4f}\n\n"
            f"{grade_json}"
        )
        obs = MedTriageObservation(
            observation_type="episode_end",
            content=content,
            available_actions=[],
            step_reward=total,
            cumulative_reward=self._cumulative_reward,
            episode_done=True,
            info={**self._info_dict(), "grading": grade},
        )
        return obs, total, True
