from __future__ import annotations

"""
graders.py — Pure-Python, deterministic scoring logic for MedTriageEnv.

No LLM calls.  No randomness.  Same inputs → same score, always.

Score formula
─────────────
    total_score = (symptom_score * 0.30)
                + (test_score    * 0.30)
                + (diagnosis_score * 0.40)

Each component is described in detail below its section header.
"""

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Symptom keyword expansion table
# ---------------------------------------------------------------------------
# Maps each canonical critical_symptom token (as it appears in scoring_hints)
# to a set of natural-language substrings the agent might use in a question.
# Matching is case-insensitive substring containment — no fuzzy logic, no NLP.
#
# Rules for adding entries:
#   - Every synonym should be unambiguous (avoid single letters or stop-words).
#   - Prefer shorter substrings so they fire on a wider variety of phrasings.
#   - When a symptom token is already plain English, add its own words too.
# ---------------------------------------------------------------------------
SYMPTOM_KEYWORD_MAP: Dict[str, List[str]] = {
    # ── Abdominal / surgical ──────────────────────────────────────────────
    "RLQ_pain":               ["right lower", "rlq", "lower right", "iliac fossa", "right side"],
    "periumbilical_migration": ["belly button", "umbilical", "periumbilical", "navel", "started around the middle",
                                "migrat", "moved from"],
    "nausea_vomiting":        ["nausea", "vomit", "sick to", "throw up", "threw up", "queasy"],
    "anorexia":               ["appetite", "eating", "hungry", "food", "anorexia"],
    "rebound_tenderness":     ["rebound", "release", "let go", "peritoneal", "guarding"],

    # ── Respiratory ───────────────────────────────────────────────────────
    "productive_cough":       ["cough", "phlegm", "sputum", "mucus", "expectoration"],
    "pleuritic_chest_pain":   ["chest pain", "chest hurt", "side pain", "pleuritic", "breathe hurt",
                                "sharp when breath", "pain when inhale"],
    "low_oxygen_saturation":  ["oxygen", "spo2", "saturation", "o2", "breathless", "shortness of breath",
                                "short of breath", "winded", "dyspnea"],
    "tachycardia":            ["heart rate", "pulse", "racing heart", "palpitat", "fast heart"],
    "hemoptysis":             ["blood in sputum", "coughing blood", "bloody mucus", "hemoptysis"],

    # ── Urinary ───────────────────────────────────────────────────────────
    "dysuria":                ["burn", "sting", "pain when urinat", "painful urinat", "dysuria"],
    "urinary_frequency":      ["frequen", "how often", "urgency", "every few minutes", "urinat often"],
    "hematuria":              ["blood in urine", "hematuria", "pink urine", "red urine", "bloody urine"],
    "flank_pain":             ["flank", "back pain", "loin", "costovertebral", "kidney area"],

    # ── Cardiac ───────────────────────────────────────────────────────────
    "chest_pain":             ["chest pain", "chest hurt", "chest pressure", "chest tightness",
                                "squeezing", "chest discomfort"],
    "tearing_chest_pain":     ["tearing", "ripping", "tore", "back pain", "interscapular", "aorta"],
    "radiation":              ["radiat", "spread", "go to", "going to", "arm", "jaw", "shoulder",
                                "between shoulder"],
    "diaphoresis":            ["sweat", "diaphoresis", "clammy", "perspir"],
    "syncope":                ["faint", "syncope", "pass out", "lost consciousness", "blacked out",
                                "presyncopal", "near faint"],

    # ── Neurological ─────────────────────────────────────────────────────
    "headache":               ["headache", "head pain", "head hurt", "head ache"],
    "thunderclap_headache":   ["worst headache", "thunderclap", "sudden headache", "explosive headache",
                                "headache of my life"],
    "altered_consciousness":  ["confused", "confusion", "altered", "disoriented", "consciousness",
                                "unresponsive", "drowsy", "foggy"],
    "focal_deficits":         ["weakness", "numb", "droop", "slur", "vision", "paralysis", "facial"],
    "meningismus":            ["stiff neck", "neck stiffness", "meningismus", "photophobia", "light bother",
                                "phonophobia", "kernig", "brudzinski"],

    # ── Endocrine / metabolic ─────────────────────────────────────────────
    "profound_weakness":      ["weakness", "weak", "fatigue", "exhaustion", "tired", "lethargy"],
    "hypotension":            ["blood pressure", "bp", "low bp", "hypotension", "dizzy", "faint",
                                "lightheaded", "nearly fainted"],
    "hyponatremia":           ["sodium", "salt", "hyponatremia", "electrolyte"],
    "hyperkalemia":           ["potassium", "hyperkalemia", "electrolyte", "peaked t"],
    "known_Addisons":         ["addison", "adrenal", "cortisol", "steroid", "hydrocortisone",
                                "fludrocortisone", "medical history"],
    "inability_to_take_medications": ["medication", "medicine", "pills", "tablets", "doses",
                                       "could not take", "kept down", "vomiting medication"],

    # ── Obstetric / gynaecological ────────────────────────────────────────
    "pregnancy_risk":         ["pregnant", "pregnancy", "last period", "lmp", "missed period",
                                "sexually active", "contraception"],
    "vaginal_bleeding":       ["vaginal bleed", "bleeding", "spotting", "discharge"],
    "pelvic_pain":            ["pelvic", "lower abdomen", "lower belly", "pelvis", "groin"],

    # ── Constitutional / generic ──────────────────────────────────────────
    "fever":                  ["fever", "temperature", "hot", "febrile", "chills", "rigors",
                                "feverish"],
    "weight_loss":            ["weight loss", "losing weight", "lost weight"],
    "night_sweats":           ["night sweat", "sweating at night", "nocturnal"],
    "salt_craving":           ["salt crav", "crav salt", "salty", "craving"],
}


def _normalize(text: str) -> str:
    """Return lowercased, stripped version of *text* for matching."""
    return text.lower().strip()


def _symptom_covered(critical_symptom: str, questions: List[str]) -> bool:
    """Return True if any question in *questions* covers *critical_symptom*.

    Coverage is determined by:
      1. Looking up the symptom token in SYMPTOM_KEYWORD_MAP for a list of
         known substrings.  If any substring appears in any question → covered.
      2. Fallback: split the symptom token on underscores and check whether
         ALL resulting words appear (as substrings) in at least one question.
         This handles tokens not yet in the map without requiring a map update.

    Examples
    --------
    critical_symptom="RLQ_pain", question="Where exactly is your pain?"
        → not covered (no RLQ synonym matched)
    critical_symptom="RLQ_pain", question="Do you have pain in the lower right side?"
        → covered via "lower right" synonym
    """
    token = critical_symptom.lower()
    lowered_questions = [_normalize(q) for q in questions]

    # Strategy 1: keyword map lookup
    synonyms = SYMPTOM_KEYWORD_MAP.get(critical_symptom, [])
    for synonym in synonyms:
        syn_lower = synonym.lower()
        if any(syn_lower in q for q in lowered_questions):
            return True

    # Strategy 2: token word fallback (handles unmapped symptoms gracefully)
    words = [w for w in token.split("_") if len(w) > 2]  # skip short tokens like "of", "to"
    if words:
        for question in lowered_questions:
            if all(w in question for w in words):
                return True

    return False


# ---------------------------------------------------------------------------
# COMPONENT 1 — symptom_score
# ---------------------------------------------------------------------------
def _compute_symptom_score(
    questions_asked: List[str],
    critical_symptoms: List[str],
) -> tuple[float, dict]:
    """Score how thoroughly the agent probed the case's critical symptoms.

    Args:
        questions_asked:   All questions the agent posed during the episode.
        critical_symptoms: Canonical symptom tokens from scoring_hints.

    Returns:
        (symptom_score, breakdown_dict)

    Score logic
    ───────────
    symptom_score = covered_count / total_symptoms

    A symptom is "covered" if at least one asked question semantically matches
    it via SYMPTOM_KEYWORD_MAP (see _symptom_covered).  The score is 0.0 when
    no critical symptoms exist to avoid a ZeroDivisionError.
    """
    if not critical_symptoms:
        # No critical symptoms defined for this case → neutral full score
        return 1.0, {"covered": [], "missed": [], "total": 0, "covered_count": 0}

    covered = []
    missed = []
    for symptom in critical_symptoms:
        if _symptom_covered(symptom, questions_asked):
            covered.append(symptom)
        else:
            missed.append(symptom)

    score = len(covered) / len(critical_symptoms)
    breakdown = {
        "covered": covered,
        "missed": missed,
        "total": len(critical_symptoms),
        "covered_count": len(covered),
    }
    return score, breakdown


# ---------------------------------------------------------------------------
# COMPONENT 2 — test_score
# ---------------------------------------------------------------------------
# Penalty rationale: charging -0.1 per truly unnecessary test (not in gold_key_tests
# AND not in gold_differential) discourages shotgun ordering without punishing
# clinically plausible but non-gold investigations.
# ---------------------------------------------------------------------------
UNNECESSARY_TEST_PENALTY: float = 0.1


def _compute_test_score(
    tests_ordered: List[str],
    gold_key_tests: List[str],
    gold_differential: List[str],
) -> tuple[float, dict]:
    """Score the agent's test-ordering behaviour.

    Args:
        tests_ordered:    Tests the agent ordered (may contain duplicates; deduplicated below).
        gold_key_tests:   Tests that are essential for this diagnosis.
        gold_differential: Top differential diagnoses.  Any test name that appears
                           as a substring of a differential entry is considered
                           "clinically plausible" and exempt from the penalty.

    Score logic
    ───────────
        raw_score = (# gold_key_tests the agent ordered) / len(gold_key_tests)
        penalty   = UNNECESSARY_TEST_PENALTY * (# unnecessary tests)
        test_score = max(0.0, min(1.0, raw_score - penalty))

    "Unnecessary" = ordered by agent AND not in gold_key_tests AND name is not
    a substring of any gold_differential entry (case-insensitive).

    Deduplication: if the agent orders the same test twice, it counts once.
    """
    if not gold_key_tests:
        # Edge case: no key tests defined → full score, no penalty applicable
        return 1.0, {"ordered": tests_ordered, "gold_key_tests": [], "hit": [],
                     "missed": [], "unnecessary": [], "raw_score": 1.0, "penalty": 0.0}

    # Deduplicate, preserve case for display but compare lowercased
    ordered_unique = list({t.strip() for t in tests_ordered})
    gold_lower     = {t.lower() for t in gold_key_tests}
    diff_lower     = [d.lower() for d in gold_differential]

    # Tests the agent correctly ordered
    hit   = [t for t in ordered_unique if t.lower() in gold_lower]
    # Tests the agent missed
    missed = [g for g in gold_key_tests if g.lower() not in {t.lower() for t in ordered_unique}]

    raw_score = len(hit) / len(gold_key_tests)

    # Determine unnecessary tests:
    # A test is "clinically plausible" (not penalised) if its name appears as
    # a substring in any differential diagnosis string, e.g. "ECG" appearing
    # in a differential that mentions "arrhythmia + ECG".  This is intentionally
    # lenient — we only penalise clearly off-topic orders.
    unnecessary = []
    for t in ordered_unique:
        t_lower = t.lower()
        if t_lower in gold_lower:
            continue  # gold test — not unnecessary
        if any(t_lower in d for d in diff_lower):
            continue  # plausible given differential — not penalised
        unnecessary.append(t)

    penalty = UNNECESSARY_TEST_PENALTY * len(unnecessary)
    final_score = max(0.0, min(1.0, raw_score - penalty))

    breakdown = {
        "ordered": ordered_unique,
        "gold_key_tests": gold_key_tests,
        "hit": hit,
        "missed": missed,
        "unnecessary": unnecessary,
        "raw_score": round(raw_score, 4),
        "penalty": round(penalty, 4),
    }
    return final_score, breakdown


# ---------------------------------------------------------------------------
# COMPONENT 3 — diagnosis_score
# ---------------------------------------------------------------------------
# Scoring ladder:
#   1.0 — exact gold match OR any keyword from scoring_hints["diagnosis_keywords"]
#   0.5 — matches a differential diagnosis (partial credit: on the right track)
#   0.0 — no match at all
# Triage bonus:
#   +0.2 if triage_level == gold_triage_level (capped so total ≤ 1.0)
#   The bonus rewards correct acuity judgement independently of whether the
#   exact diagnosis label was used.
# ---------------------------------------------------------------------------
TRIAGE_BONUS: float = 0.2


def _compute_diagnosis_score(
    submitted_diagnosis: Optional[str],
    submitted_triage: Optional[str],
    gold_diagnosis: str,
    gold_triage_level: str,
    gold_differential: List[str],
    diagnosis_keywords: List[str],
) -> tuple[float, dict]:
    """Score the agent's submitted diagnosis and triage level assignment.

    Args:
        submitted_diagnosis: Free-text diagnosis the agent submitted.
        submitted_triage:    Triage level the agent assigned.
        gold_diagnosis:      The single correct diagnosis for this case.
        gold_triage_level:   The correct triage level.
        gold_differential:   Accepted partial-credit diagnoses.
        diagnosis_keywords:  Additional strings that count as a full match.

    Returns:
        (diagnosis_score, breakdown_dict)
    """
    if submitted_diagnosis is None:
        # Agent never submitted a diagnosis — zero credit
        return 0.0, {
            "submitted_diagnosis": None,
            "submitted_triage": submitted_triage,
            "base_score": 0.0,
            "triage_bonus": 0.0,
            "match_type": "none",
        }

    diag_lower = _normalize(submitted_diagnosis)
    gold_lower = _normalize(gold_diagnosis)
    keywords_lower = [_normalize(k) for k in diagnosis_keywords]
    diff_lower     = [_normalize(d) for d in gold_differential]

    # ── Determine base score ────────────────────────────────────────────────
    # Full match: exact token equality OR agent string equals a keyword OR
    # agent string contains the gold diagnosis as a substring OR
    # gold diagnosis contains the agent string as a substring
    # (handles "acute appendicitis" vs "appendicitis" gracefully).
    is_full_match = (
        diag_lower == gold_lower
        or diag_lower in keywords_lower
        or any(diag_lower in kw or kw in diag_lower for kw in keywords_lower)
        or gold_lower in diag_lower
        or diag_lower in gold_lower
    )

    if is_full_match:
        base_score  = 1.0
        match_type  = "full"
    else:
        # Partial match: submitted diagnosis appears in or overlaps with a differential entry
        is_partial = any(
            diag_lower == d or diag_lower in d or d in diag_lower
            for d in diff_lower
        )
        if is_partial:
            base_score = 0.5
            match_type = "partial_differential"
        else:
            base_score = 0.0
            match_type = "none"

    # ── Triage bonus ────────────────────────────────────────────────────────
    # Only awarded when a diagnosis was submitted (we already guard above).
    # We cap the total at 1.0 so the bonus never inflates beyond the ceiling.
    triage_correct = (
        submitted_triage is not None
        and _normalize(submitted_triage) == _normalize(gold_triage_level)
    )
    triage_bonus = TRIAGE_BONUS if triage_correct else 0.0
    final_score  = min(1.0, base_score + triage_bonus)

    breakdown = {
        "submitted_diagnosis": submitted_diagnosis,
        "submitted_triage": submitted_triage,
        "base_score": round(base_score, 4),
        "triage_bonus": round(triage_bonus, 4),
        "match_type": match_type,
        "triage_correct": triage_correct,
    }
    return final_score, breakdown


# ---------------------------------------------------------------------------
# Public grader class
# ---------------------------------------------------------------------------

class MedTriageGrader:
    """Grades a completed MedTriage episode against the case's gold standard.

    Usage
    -----
        grader = MedTriageGrader(case)
        result = grader.grade_episode(
            questions_asked=["Do you have chest pain?", "Does it radiate?"],
            tests_ordered=["ECG", "troponin"],
            submitted_diagnosis="STEMI",
            submitted_triage="immediate",
        )
        print(result["total_score"])   # e.g. 0.87

    The grader is stateless after construction — ``grade_episode`` can be
    called multiple times (e.g. for replay scoring) and always returns the
    same result for the same inputs.
    """

    # Weights must sum to 1.0
    WEIGHT_SYMPTOM   = 0.30
    WEIGHT_TEST      = 0.30
    WEIGHT_DIAGNOSIS = 0.40

    def __init__(self, case: dict) -> None:
        """Initialise the grader with a case definition dict.

        Args:
            case: A single case record as loaded from cases.json.  Must
                  contain keys: gold_diagnosis, gold_triage_level,
                  gold_key_tests, gold_differential, scoring_hints.
        """
        self.case = case

        # Pre-extract fields used in every grade_episode call
        self._gold_diagnosis:   str       = case["gold_diagnosis"]
        self._gold_triage:      str       = case["gold_triage_level"]
        self._gold_key_tests:   List[str] = case.get("gold_key_tests", [])
        self._gold_differential: List[str] = case.get("gold_differential", [])

        hints = case.get("scoring_hints", {})
        self._critical_symptoms:  List[str] = hints.get("critical_symptoms", [])
        self._diagnosis_keywords: List[str] = hints.get("diagnosis_keywords", [])

    def grade_episode(
        self,
        questions_asked: List[str],
        tests_ordered: List[str],
        submitted_diagnosis: Optional[str],
        submitted_triage: Optional[str],
    ) -> Dict[str, Any]:
        """Compute the final score for a completed episode.

        Args:
            questions_asked:      All natural-language questions the agent asked
                                  the patient during the episode, in order.
            tests_ordered:        All test names the agent ordered, in order.
            submitted_diagnosis:  The agent's final diagnosis string, or None if
                                  the episode ended without a submission.
            submitted_triage:     The agent's triage level ("immediate",
                                  "urgent", "semi-urgent", "non-urgent"), or
                                  None.

        Returns:
            A dict with keys:
              total_score      — float in [0.0, 1.0], the headline score
              symptom_score    — float in [0.0, 1.0]
              test_score       — float in [0.0, 1.0]
              diagnosis_score  — float in [0.0, 1.0]
              breakdown        — dict with per-component debug details
        """
        # ── Component scores ────────────────────────────────────────────────
        symptom_score,   symptom_bd   = _compute_symptom_score(
            questions_asked, self._critical_symptoms
        )
        test_score,      test_bd      = _compute_test_score(
            tests_ordered, self._gold_key_tests, self._gold_differential
        )
        diagnosis_score, diagnosis_bd = _compute_diagnosis_score(
            submitted_diagnosis,
            submitted_triage,
            self._gold_diagnosis,
            self._gold_triage,
            self._gold_differential,
            self._diagnosis_keywords,
        )

        # ── Weighted sum ────────────────────────────────────────────────────
        total_score = (
            symptom_score   * self.WEIGHT_SYMPTOM
            + test_score    * self.WEIGHT_TEST
            + diagnosis_score * self.WEIGHT_DIAGNOSIS
        )
        # Clamp to [0.0, 1.0] to guard against floating-point drift
        total_score = max(0.0, min(1.0, round(total_score, 6)))

        return {
            "total_score":     total_score,
            "symptom_score":   round(symptom_score,   4),
            "test_score":      round(test_score,       4),
            "diagnosis_score": round(diagnosis_score,  4),
            "breakdown": {
                "symptom":   symptom_bd,
                "test":      test_bd,
                "diagnosis": diagnosis_bd,
                # Convenience field: human-readable weight breakdown
                "weighted_contributions": {
                    "symptom":   round(symptom_score   * self.WEIGHT_SYMPTOM,   4),
                    "test":      round(test_score      * self.WEIGHT_TEST,      4),
                    "diagnosis": round(diagnosis_score * self.WEIGHT_DIAGNOSIS, 4),
                },
            },
        }


# ---------------------------------------------------------------------------
# Hackathon task list
# ---------------------------------------------------------------------------

def get_task_list() -> List[Dict[str, Any]]:
    """Return the three canonical tasks for the MedTriage hackathon.

    Each dict describes one difficulty tier.  Agents are evaluated on all
    three tiers and ranked by the sum of their total_scores.

    Returns:
        List of task descriptor dicts with keys:
          task_id     — slug used in API calls and leaderboard display
          description — human-readable task summary shown to participants
          difficulty  — mirrors the "difficulty" field in cases.json
    """
    return [
        {
            "task_id": "easy",
            "description": (
                "Diagnose and triage a patient presenting with a high-acuity but clinically "
                "clear-cut condition (e.g. acute appendicitis, community-acquired pneumonia, "
                "uncomplicated UTI).  Key symptoms are prominent and a single targeted test "
                "confirms the diagnosis.  Agents are expected to achieve a score ≥ 0.80."
            ),
            "difficulty": "easy",
        },
        {
            "task_id": "medium",
            "description": (
                "Diagnose and triage a patient whose presentation has moderate diagnostic "
                "ambiguity — overlapping differentials, atypical symptom patterns, or "
                "results that require synthesis across multiple investigations "
                "(e.g. pulmonary embolism, ectopic pregnancy, diabetic ketoacidosis).  "
                "Efficient test selection is rewarded.  Target score ≥ 0.65."
            ),
            "difficulty": "medium",
        },
        {
            "task_id": "hard",
            "description": (
                "Diagnose and triage a complex or rare presentation requiring clinical "
                "reasoning under high uncertainty — unusual chief complaints, misleading "
                "vitals, or diagnoses outside common patterns "
                "(e.g. adrenal crisis, aortic dissection, subarachnoid haemorrhage).  "
                "Incorrect triage level carries significant scoring consequences.  "
                "Target score ≥ 0.50."
            ),
            "difficulty": "hard",
        },
    ]
