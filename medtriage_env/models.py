from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from openenv.core.env_server.interfaces import Action, Observation, State


class MedTriageAction(Action):
    """An action taken by the triage agent during a clinical episode.

    The agent may ask the patient a clarifying question, order a diagnostic
    test, or submit a final diagnosis with an associated triage level.
    Exactly one of ``question``, ``test_name``, or ``diagnosis`` should be
    populated depending on ``action_type``.
    """

    action_type: Literal["ask_question", "order_test", "submit_diagnosis"] = Field(
        ...,
        description=(
            "The category of action the agent is taking. "
            "'ask_question' gathers anamnesis from the patient, "
            "'order_test' requests a diagnostic investigation, and "
            "'submit_diagnosis' concludes the episode with a clinical assessment."
        ),
    )
    question: Optional[str] = Field(
        None,
        description=(
            "The natural-language question posed to the patient. "
            "Required when action_type is 'ask_question'; ignored otherwise."
        ),
    )
    test_name: Optional[str] = Field(
        None,
        description=(
            "The name of the diagnostic test to order (e.g. 'CBC', 'chest X-ray', 'troponin'). "
            "Required when action_type is 'order_test'; ignored otherwise."
        ),
    )
    diagnosis: Optional[str] = Field(
        None,
        description=(
            "The agent's clinical diagnosis expressed as a free-text string "
            "(e.g. 'ST-elevation myocardial infarction'). "
            "Required when action_type is 'submit_diagnosis'; ignored otherwise."
        ),
    )
    triage_level: Optional[Literal["immediate", "urgent", "semi-urgent", "non-urgent"]] = Field(
        None,
        description=(
            "The Manchester Triage System priority level assigned alongside the diagnosis. "
            "'immediate' = resuscitation required; 'urgent' = high risk; "
            "'semi-urgent' = moderate risk; 'non-urgent' = low acuity. "
            "Required when action_type is 'submit_diagnosis'; ignored otherwise."
        ),
    )

    class Config:
        schema_extra = {
            "examples": [
                {
                    "action_type": "ask_question",
                    "question": "Can you describe the character and radiation of the chest pain?",
                    "test_name": None,
                    "diagnosis": None,
                    "triage_level": None,
                },
                {
                    "action_type": "order_test",
                    "question": None,
                    "test_name": "12-lead ECG",
                    "diagnosis": None,
                    "triage_level": None,
                },
                {
                    "action_type": "submit_diagnosis",
                    "question": None,
                    "test_name": None,
                    "diagnosis": "ST-elevation myocardial infarction",
                    "triage_level": "immediate",
                },
            ]
        }


class MedTriageObservation(Observation):
    """The environment's response after processing a ``MedTriageAction``.

    Carries the textual content returned by the simulation (patient reply,
    test result, or terminal message), the reward signals for the current
    step, and bookkeeping metadata that the agent may use for planning.
    """

    observation_type: Literal["patient_response", "test_result", "episode_end", "error"] = Field(
        ...,
        description=(
            "Categorises the content of this observation. "
            "'patient_response' is the reply to an asked question; "
            "'test_result' is the output of an ordered investigation; "
            "'episode_end' signals episode termination after a submitted diagnosis or step exhaustion; "
            "'error' indicates a malformed action was received."
        ),
    )
    content: str = Field(
        ...,
        description=(
            "The primary textual payload of the observation — the patient's verbatim answer, "
            "the formatted test report, the final scoring message, or an error description."
        ),
    )
    available_actions: List[str] = Field(
        ...,
        description=(
            "Actions that remain valid for the agent to take at the next step. "
            "Will be empty once the episode has ended."
        ),
    )
    step_reward: float = Field(
        0.0,
        description=(
            "Scalar reward assigned for the action that produced this observation. "
            "Negative penalties may apply for unnecessary tests or repeated questions."
        ),
    )
    cumulative_reward: float = Field(
        0.0,
        description="Running total of all step rewards accumulated since episode start.",
    )
    episode_done: bool = Field(
        False,
        description=(
            "True when the episode has terminated — either because the agent submitted a "
            "diagnosis or the maximum number of steps was reached."
        ),
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Auxiliary diagnostic metadata. Typical keys: "
            "'tests_ordered' (List[str]), "
            "'questions_asked' (List[str]), "
            "'steps_remaining' (int). "
            "Contents may vary by environment configuration."
        ),
    )

    class Config:
        schema_extra = {
            "example": {
                "observation_type": "test_result",
                "content": "ECG shows 2 mm ST elevation in leads II, III and aVF. Rate 88 bpm, sinus rhythm.",
                "available_actions": ["ask_question", "order_test", "submit_diagnosis"],
                "step_reward": -0.5,
                "cumulative_reward": 1.5,
                "episode_done": False,
                "info": {
                    "tests_ordered": ["troponin", "12-lead ECG"],
                    "questions_asked": ["Any history of cardiac disease?"],
                    "steps_remaining": 14,
                },
            }
        }


class MedTriageState(State):
    """Internal episode state maintained by the ``MedTriageEnv`` environment.

    This object is not exposed to the agent directly but may be serialised
    for logging, replay, or evaluation pipelines.  It tracks every
    significant event within a single diagnostic episode.
    """

    episode_id: str = Field(
        ...,
        description=(
            "Globally unique identifier for this episode instance, "
            "typically a UUID-4 string generated at environment reset."
        ),
    )
    case_id: str = Field(
        ...,
        description=(
            "Identifier of the clinical case definition loaded from the case library "
            "(corresponds to a record in cases.json or equivalent store)."
        ),
    )
    difficulty: str = Field(
        ...,
        description=(
            "Difficulty tier of the case as defined in the case library "
            "(e.g. 'easy', 'medium', 'hard', 'expert')."
        ),
    )
    step_count: int = Field(
        0,
        ge=0,
        description="Number of agent–environment interaction steps completed so far in this episode.",
    )
    max_steps: int = Field(
        20,
        gt=0,
        description=(
            "Maximum number of steps permitted before the episode is force-terminated "
            "and a partial score is assigned."
        ),
    )
    questions_asked: List[str] = Field(
        default_factory=list,
        description="Ordered list of questions the agent has posed to the patient during this episode.",
    )
    tests_ordered: List[str] = Field(
        default_factory=list,
        description="Ordered list of diagnostic test names the agent has ordered during this episode.",
    )
    diagnosis_submitted: bool = Field(
        False,
        description="True once the agent has called 'submit_diagnosis', preventing further actions.",
    )
    final_score: Optional[float] = Field(
        None,
        description=(
            "Normalised episode score in [0.0, 1.0] assigned at episode termination. "
            "None until the episode has ended."
        ),
    )

    class Config:
        schema_extra = {
            "example": {
                "episode_id": "3f7e2a1b-9c4d-4e8f-a2b6-1d0e5f3c7a9b",
                "case_id": "case_042",
                "difficulty": "hard",
                "step_count": 6,
                "max_steps": 20,
                "questions_asked": [
                    "Any history of cardiac disease?",
                    "Can you describe the character and radiation of the chest pain?",
                ],
                "tests_ordered": ["troponin", "12-lead ECG"],
                "diagnosis_submitted": False,
                "final_score": None,
            }
        }

