"""Basic smoke tests for MedTriageEnvironment."""
import pytest
from medtriage_env.models import MedTriageAction, MedTriageObservation, MedTriageState
from server.environment import MedTriageEnvironment


def test_models_instantiate():
    a = MedTriageAction(action_type="ask_question", question="Do you have chest pain?")
    assert a.action_type == "ask_question"

    o = MedTriageObservation(
        observation_type="patient_response",
        content="Yes",
        available_actions=["ask_question", "order_test", "submit_diagnosis"],
    )
    assert o.episode_done is False

    s = MedTriageState(episode_id="test-1", case_id="case_001", difficulty="easy")
    assert s.step_count == 0


def test_environment_reset():
    env = MedTriageEnvironment()
    obs = env.reset()
    assert obs is not None
    assert obs.episode_done is False
    assert obs.observation_type == "patient_response"


def test_environment_ask_question():
    env = MedTriageEnvironment()
    env.reset()
    result = env.step(MedTriageAction(action_type="ask_question", question="do you have chest pain"))
    assert result is not None


def test_environment_submit_diagnosis():
    env = MedTriageEnvironment()
    env.reset()
    case = env.current_case
    result = env.step(MedTriageAction(
        action_type="submit_diagnosis",
        diagnosis=case["gold_diagnosis"],
        triage_level=case["gold_triage_level"],
    ))
    assert result.episode_done is True
    assert result.info.get("total_score") is not None
