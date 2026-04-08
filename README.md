---
title: MedTriageEnv
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# MedTriageEnv

**MedTriageEnv** is an [OpenEnv](https://github.com/facebookresearch/OpenEnv)-compatible environment for **clinical triage and diagnostic reasoning**. Agents conduct a structured emergency-medicine episode: they interview a simulated patient, order investigations from a constrained menu, and submit a diagnosis with a Manchester-style triage level. The task couples **natural-language history-taking**, **tool use (lab/imaging)**, and **high-stakes acuity judgement**—a setting where mistakes are costly and where RL and agentic LLMs need reproducible, interpretable evaluation beyond single-turn QA.

---

## Environment overview

| Aspect | Description |
|--------|-------------|
| **Agent role** | Clinician agent: elicit history, order tests, synthesize findings, assign diagnosis + triage. |
| **Termination** | Episode ends on `submit_diagnosis` or when the step budget is exhausted. |

**Action space** (typed `MedTriageAction`):

- `ask_question` — free-text question to the patient (`question`).
- `order_test` — test name from the case’s `available_tests` (`test_name`).
- `submit_diagnosis` — final diagnosis string + triage level (`diagnosis`, `triage_level` ∈ `immediate` \| `urgent` \| `semi-urgent` \| `non-urgent`).

**Observation space** (`MedTriageObservation`):

- `patient_response` — answer to a question or initial presentation text.
- `test_result` — result string for an ordered test.
- `episode_end` — terminal message including **deterministic grading** (JSON breakdown + `total_score`).
- `error` — invalid action or environment error (episode may continue or end depending on context).

**Reward (episode score)** — computed at the end of the episode by a **pure-Python grader** (no LLM in the loop). The scalar **`total_score` ∈ [0, 1]** is a weighted sum of three components:

| Component | Weight | What it measures |
|-----------|--------|------------------|
| **Symptom** | **30%** | Coverage of case-defined **critical symptoms** via keyword-aligned questions (see grader logic / `scoring_hints`). |
| **Tests** | **30%** | Ordering **gold key tests** vs missing them; penalty for clearly unnecessary orders (relative to gold differential). |
| **Diagnosis** | **40%** | Match to **gold diagnosis** / keywords / differential; **triage** alignment adds a capped bonus. |

Formula: `total_score = 0.30 × symptom_score + 0.30 × test_score + 0.40 × diagnosis_score` (then clamped to [0, 1]).

During the episode, the environment may also emit **small step rewards** (e.g. for informative questions and appropriate tests); the **headline metric for leaderboards** is the terminal **`total_score`** above.

---

## Task descriptions (difficulty tiers)

Cases are tagged `easy`, `medium`, or `hard` in `data/cases.json`. Evaluation typically runs **all three** tiers; ranking may use the **sum** of episode scores across tiers.

| Tier | Clinical profile | Strong LLM agent — **expected `total_score` band** |
|------|------------------|-----------------------------------------------------|
| **Easy** | Classic, high-signal textbook presentations; often one or two decisive findings/tests. | **~0.75–0.95** (strong agents should land **≥ 0.80** consistently). |
| **Medium** | Overlapping differentials, atypical patterns, or multi-step interpretation of tests. | **~0.55–0.80** (competitive target **≥ 0.65**). |
| **Hard** | Rare or misleading presentations; triage errors hurt disproportionately. | **~0.35–0.65** (respectable **≥ 0.50**; frontier models still fail often). |

Bands are indicative; exact distributions depend on your case library and prompting.

---

## Quick start

Install OpenEnv core and this environment (replace placeholders with your Hugging Face username / repo):

```bash
pip install openenv-core
pip install git+https://huggingface.co/spaces/abhi-ag/medtriage_env
```

Minimal Python loop (`reset` returns an observation; `step` returns `StepResult`):

```python
from servers.environment import MedTriageEnvironment
from medtriage_env.models import MedTriageAction

env = MedTriageEnvironment()

obs = env.reset(difficulty="easy")
print(obs.observation_type, obs.content[:200], "...")
print("available_actions:", obs.available_actions)

out = env.step(
    MedTriageAction(
        action_type="ask_question",
        question="When did the pain start?",
    )
)
print(out.observation.observation_type, out.observation.content)
print("reward:", out.reward, "done:", out.done)

out = env.step(
    MedTriageAction(
        action_type="submit_diagnosis",
        diagnosis="Acute appendicitis",
        triage_level="urgent",
    )
)
o = out.observation
print(o.observation_type, o.content)  # episode_end: grading JSON + total_score
print("reward:", out.reward, "done:", out.done)
```

Adjust import paths if your install layout differs (e.g. run from repo root with `pip install -e .`).

---

## Running the baseline

Configure model endpoint and credentials, then run the provided baseline script:

```bash
export API_BASE_URL=...
export MODEL_NAME=...
export HF_TOKEN=...

python inference.py
```

(`inference.py` should read these env vars; set `API_BASE_URL` / `MODEL_NAME` to your OpenAI-compatible or provider-specific endpoint as documented in that script.)

---

## Why this environment matters

- **Structured clinical agency** — Evaluates multi-turn **information gathering**, **constrained tool use**, and **final decision quality** under a fixed case schema, not ad-hoc chat benchmarks.
- **Interpretable, reproducible scoring** — Terminal reward decomposes into **symptom / test / diagnosis** with JSON breakdowns suitable for error analysis and curriculum design.
- **Safety-relevant abstraction** — Triage and diagnosis are explicit outputs, supporting research on **calibration**, **over-investigation**, and **missed red flags** in agentic medical pipelines.

---

## Hugging Face Space

**Live demo / hosted build:** [https://abhi-ag-medtriage-env.hf.space](https://abhi-ag-medtriage-env.hf.space)



---

## License

See repository `LICENSE` if present; otherwise follow the hackathon / dataset terms you attach to the submission.

