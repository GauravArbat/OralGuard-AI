"""
OralGuard AI — Differential Diagnosis Engine

Generates ranked differential diagnoses based on clinical features,
questionnaire responses, and AI classification results.
Implements the differential diagnosis matrix from PRD Section 7.4.
"""


# ══════════════════════════════════════════════
# DIFFERENTIAL DIAGNOSIS KNOWLEDGE BASE
# ══════════════════════════════════════════════

DIFFERENTIAL_CONDITIONS = {
    "recurrent_aphthous_stomatitis": {
        "display_name": "Recurrent Aphthous Stomatitis",
        "icd10": "K12.0",
        "category": "primary",
        "key_features": {
            "recurrent": True,
            "painful": True,
            "border_type": "regular",
            "vesicle_preceded": False,
            "non_keratinized": True,
            "heals_within_2_weeks": True,
            "induration": False,
        },
        "supporting_clues": [
            "Round or oval shape",
            "Yellowish-gray pseudomembranous center",
            "Erythematous halo",
            "Non-keratinized mucosa location",
            "Recurrent episodes",
            "Self-limiting (heals 7-14 days)",
        ],
        "opposing_clues": [
            "Duration > 3 weeks",
            "Keratinized mucosa location",
            "Induration/hardness",
            "Vesicular onset",
            "Associated systemic features",
        ],
    },
    "oscc": {
        "display_name": "Oral Squamous Cell Carcinoma",
        "icd10": "C06.9",
        "category": "primary",
        "key_features": {
            "persistent_over_3_weeks": True,
            "induration": True,
            "growing": True,
            "irregular_borders": True,
            "non_healing": True,
            "painless_initially": True,
        },
        "supporting_clues": [
            "Persistent ulcer > 3 weeks",
            "Indurated/firm edges",
            "Irregular or rolled borders",
            "Enlarging over time",
            "High-risk site (lateral tongue, floor of mouth)",
            "Tobacco/alcohol/betel use",
            "Age > 40",
            "Lymphadenopathy",
            "Exophytic or fungating mass",
        ],
        "opposing_clues": [
            "Recurrent pattern (heals and returns)",
            "Self-limiting episodes",
            "Young patient without risk factors",
            "Regular, well-defined borders",
            "Clearly painful from onset",
        ],
    },
    "traumatic_ulcer": {
        "display_name": "Traumatic Ulcer",
        "icd10": "K12.1",
        "category": "differential",
        "key_features": {
            "trauma_history": True,
            "corresponds_to_trauma_site": True,
            "single_ulcer": True,
            "heals_after_trauma_removed": True,
        },
        "supporting_clues": [
            "History of biting cheek/lip/tongue",
            "Sharp or broken tooth nearby",
            "Denture or orthodontic appliance",
            "Location matches trauma source",
            "Heals after cause removed",
        ],
        "opposing_clues": [
            "No identifiable trauma source",
            "Recurrent at different sites",
            "Multiple ulcers",
        ],
    },
    "recurrent_herpes": {
        "display_name": "Recurrent Intraoral Herpes",
        "icd10": "B00.2",
        "category": "differential",
        "key_features": {
            "keratinized_mucosa": True,
            "vesicle_preceded": True,
            "clustered_ulcers": True,
        },
        "supporting_clues": [
            "Keratinized mucosa (hard palate, attached gingiva)",
            "Vesicles/blisters preceded ulcers",
            "Clustered small ulcers",
            "Prodromal tingling/burning",
            "Stress or illness triggered",
        ],
        "opposing_clues": [
            "Non-keratinized mucosa",
            "No vesicular history",
            "Single large ulcer",
            "No prodromal symptoms",
        ],
    },
    "erythema_multiforme": {
        "display_name": "Erythema Multiforme",
        "icd10": "L51",
        "category": "differential",
        "key_features": {
            "multiple_ulcers": True,
            "acute_onset": True,
            "lip_crusting": True,
            "skin_lesions": True,
        },
        "supporting_clues": [
            "Acute onset of multiple oral ulcers",
            "Hemorrhagic crusting of lips",
            "Skin target lesions",
            "Preceded by herpes or drug exposure",
        ],
        "opposing_clues": [
            "Chronic/recurrent pattern",
            "Single ulcer",
            "No skin involvement",
        ],
    },
    "pemphigus_vulgaris": {
        "display_name": "Pemphigus Vulgaris",
        "icd10": "L10.0",
        "category": "differential",
        "key_features": {
            "multiple_erosions": True,
            "fragile_bullae": True,
            "widespread": True,
        },
        "supporting_clues": [
            "Multiple erosions/ulcers",
            "Fragile bullae that rupture easily",
            "Positive Nikolsky sign",
            "Widespread oral involvement",
            "Skin bullae may follow",
        ],
        "opposing_clues": [
            "Single well-defined ulcer",
            "Self-limiting episodes",
            "No bullae history",
        ],
    },
    "behcet_disease": {
        "display_name": "Behcet Disease",
        "icd10": "M35.2",
        "category": "differential",
        "key_features": {
            "recurrent_aphthae": True,
            "genital_ulcers": True,
            "ocular_involvement": True,
        },
        "supporting_clues": [
            "Recurrent oral aphthous ulcers (major criterion)",
            "Genital ulceration",
            "Ocular inflammation (uveitis)",
            "Skin lesions (erythema nodosum)",
            "Pathergy phenomenon",
        ],
        "opposing_clues": [
            "Oral ulcers only, no other manifestations",
            "No genital or ocular involvement",
        ],
    },
    "persistent_traumatic_ulcer": {
        "display_name": "Persistent Traumatic Ulcer / TUGSE",
        "icd10": "K12.1",
        "category": "differential",
        "key_features": {
            "persistent": True,
            "indurated": True,
            "mimics_malignancy": True,
        },
        "supporting_clues": [
            "Solitary persistent ulcer",
            "Often indurated margins",
            "May mimic OSCC clinically",
            "History of chronic trauma",
            "Biopsy shows eosinophilic infiltrate",
        ],
        "opposing_clues": [
            "No trauma history",
            "Multiple ulcers",
            "Tobacco/alcohol risk factors present",
        ],
    },
}


def generate_differential_diagnosis(
    classification_result: dict,
    clinical_features: dict,
    questionnaire_responses: dict,
) -> list:
    """
    Generate ranked differential diagnoses.

    Algorithm:
    1. Start with classification probabilities as base
    2. Score each condition based on feature matches
    3. Adjust based on questionnaire red flags
    4. Rank and return top differentials

    Returns:
        list of differential dicts sorted by probability
    """
    scores = {}
    details = {}

    for condition_id, condition in DIFFERENTIAL_CONDITIONS.items():
        score = 0.0
        supporting = []
        opposing = []

        # ── Base score from classification ──
        cls_probs = {
            p["label"]: p["probability"]
            for p in classification_result.get("probabilities", [])
        }

        if condition_id == "recurrent_aphthous_stomatitis":
            score += cls_probs.get("aphthous_ulcer", 0) * 50
        elif condition_id == "oscc":
            score += cls_probs.get("oscc", 0) * 50
        else:
            score += cls_probs.get("other", 0) * 10

        # ── Score from clinical features ──
        features = clinical_features or {}

        # Border type
        if condition["key_features"].get("irregular_borders"):
            if features.get("border_type") in ("irregular", "rolled"):
                score += 15
                supporting.append("Irregular/rolled borders detected")
            else:
                score -= 5
                opposing.append("Regular borders (atypical for this condition)")

        if condition["key_features"].get("border_type") == "regular":
            if features.get("border_type") == "regular":
                score += 10
                supporting.append("Regular, well-defined borders")

        # Induration
        if condition["key_features"].get("induration"):
            if questionnaire_responses.get("induration_present"):
                score += 20
                supporting.append("Induration/hardness present")

        # Exophytic growth
        if features.get("exophytic_growth") and condition_id == "oscc":
            score += 15
            supporting.append("Exophytic growth detected")

        # ── Score from questionnaire ──
        responses = questionnaire_responses or {}

        # Duration
        duration = responses.get("onset_duration_days", 7)
        if condition["key_features"].get("persistent_over_3_weeks"):
            if duration > 21:
                score += 20
                supporting.append(f"Present for {duration} days (>3 weeks)")
            else:
                score -= 10
                opposing.append(f"Only {duration} days duration")
        elif condition["key_features"].get("heals_within_2_weeks"):
            if duration <= 14:
                score += 10
                supporting.append("Duration consistent with self-limiting ulcer")

        # Recurrence
        if condition["key_features"].get("recurrent"):
            if responses.get("recurrent"):
                score += 15
                supporting.append("History of recurrent ulcers")
            else:
                score -= 8

        # Pain
        if condition["key_features"].get("painful"):
            if responses.get("pain_present"):
                score += 8
                supporting.append("Pain present (characteristic)")

        if condition["key_features"].get("painless_initially"):
            if not responses.get("pain_present"):
                score += 10
                supporting.append("Painless presentation (concerning)")

        # Vesicle
        if condition["key_features"].get("vesicle_preceded"):
            if responses.get("vesicle_preceded"):
                score += 15
                supporting.append("Vesicular onset")
            else:
                score -= 10

        # Trauma & Habits (support both multiselect 'habits' and 'trauma_history')
        habits = responses.get("habits", [])
        if isinstance(habits, str):
            habits = [habits]
        trauma = responses.get("trauma_history", "none")
        has_trauma = (
            trauma != "none"
            or "sharp_tooth_denture" in habits
            or "cheek_lip_biting" in habits
        )
        if condition["key_features"].get("trauma_history"):
            if has_trauma:
                score += 20
                cause = trauma if trauma != "none" else ("denture/sharp tooth" if "sharp_tooth_denture" in habits else "cheek/lip biting")
                supporting.append(f"Trauma etiology identified ({cause})")
            else:
                score -= 10

        # Anatomical Location (Aphthae: non-keratinized; Herpes: keratinized; OSCC: lateral tongue / floor of mouth)
        raw_loc = responses.get("location", [])
        locations = raw_loc if isinstance(raw_loc, list) else ([raw_loc] if raw_loc else [])
        non_keratinized = {"labial_mucosa", "buccal_mucosa", "ventral_tongue", "floor_of_mouth", "soft_palate"}
        keratinized = {"hard_palate", "gingiva", "dorsal_tongue"}
        high_risk_oscc_sites = {"lateral_tongue", "floor_of_mouth", "ventral_tongue", "retromolar"}

        has_non_keratinized = any(loc in non_keratinized for loc in locations)
        has_keratinized = any(loc in keratinized for loc in locations)
        has_high_risk_site = any(loc in high_risk_oscc_sites for loc in locations)

        if condition_id == "recurrent_aphthous_stomatitis":
            if has_non_keratinized:
                score += 15
                supporting.append("Located on non-keratinized mucosa (classic aphthous presentation)")
            if has_keratinized and not has_non_keratinized:
                score -= 20
                opposing.append("Ulcer located on keratinized mucosa (hard palate/gums - atypical for aphthae)")

        if condition_id == "recurrent_herpes":
            if has_keratinized:
                score += 25
                supporting.append("Ulcer on keratinized mucosa (hard palate/gingiva - classic site for intraoral herpes)")
            elif has_non_keratinized and not has_keratinized:
                score -= 15
                opposing.append("Non-keratinized site (atypical for recurrent herpes)")

        if condition_id == "oscc":
            if has_high_risk_site:
                score += 20
                supporting.append("High-risk site for OSCC (lateral border of tongue / floor of mouth)")

        # Systemic features
        systemic = responses.get("systemic_symptoms", [])
        if isinstance(systemic, str):
            systemic = [systemic]
        if condition_id == "behcet_disease":
            if "genital_ulcers" in systemic:
                score += 25
                supporting.append("Genital ulcers present (Behcet criterion)")
            if "eye_problems" in systemic:
                score += 15
                supporting.append("Ocular involvement (Behcet criterion)")
        if condition_id == "erythema_multiforme":
            if "skin_lesions" in systemic:
                score += 20
                supporting.append("Skin lesions present")

        # Medical Conditions (Anemia, Celiac, IBD, Autoimmune)
        med_conditions = responses.get("medical_conditions", [])
        if isinstance(med_conditions, str):
            med_conditions = [med_conditions]
        if condition_id == "recurrent_aphthous_stomatitis":
            predisposing = [c for c in med_conditions if c in ("anemia", "celiac", "ibd")]
            if predisposing:
                score += 15
                supporting.append(f"Underlying systemic condition predisposing to aphthae ({', '.join(predisposing)})")
        if condition_id == "behcet_disease" and "autoimmune" in med_conditions:
            score += 15
            supporting.append("Autoimmune history predisposing to aphthae")

        # Medications (Nicorandil, Methotrexate, NSAIDs, Bisphosphonates)
        meds = responses.get("medications", [])
        if isinstance(meds, str):
            meds = [meds]
        if "nicorandil" in meds:
            if condition_id == "persistent_traumatic_ulcer":
                score += 25
                supporting.append("Taking Nicorandil (notoriously causes persistent deep oral ulcerations)")
        if any(m in meds for m in ("methotrexate", "chemotherapy")):
            if condition_id in ("persistent_traumatic_ulcer", "recurrent_aphthous_stomatitis"):
                supporting.append("Taking cytotoxic/immunosuppressive drug predisposing to oral ulceration")

        # Tobacco, Betel nut & Alcohol Habits (unified 'habits' + legacy fields)
        if condition_id == "oscc":
            tobacco_legacy = responses.get("tobacco_use", "none")
            has_tobacco = (
                "smoking" in habits
                or "smokeless_tobacco" in habits
                or tobacco_legacy in ("smoking_current", "smokeless_current")
            )
            has_betel = "betel_nut" in habits or bool(responses.get("betel_quid_use"))
            has_alcohol = "alcohol" in habits or responses.get("alcohol_use") in ("moderate", "heavy")

            if has_tobacco:
                score += 15
                supporting.append("Tobacco exposure (major oncogenic risk factor)")
            if has_betel:
                score += 20
                supporting.append("Betel quid / paan masala / areca nut chewing (IARC Group 1 carcinogen)")
            if has_alcohol:
                score += 10
                supporting.append("Regular alcohol consumption")
            if (has_tobacco or has_betel) and has_alcohol:
                score += 10
                supporting.append("Synergistic tobacco/areca nut + alcohol multiplier")

        # Growth pattern
        healing = responses.get("healing_trend", "")
        if condition_id == "oscc" and healing == "growing":
            score += 20
            supporting.append("Lesion is enlarging")
        elif condition["key_features"].get("heals_within_2_weeks") and healing == "healing":
            score += 10
            supporting.append("Showing signs of healing")

        # Normalize score to 0-1 probability
        scores[condition_id] = max(0, score)
        details[condition_id] = {
            "supporting": supporting,
            "opposing": opposing,
        }

    # ── Normalize to probabilities ──
    total = sum(scores.values())
    if total == 0:
        total = 1

    differentials = []
    for condition_id, raw_score in sorted(
        scores.items(), key=lambda x: x[1], reverse=True
    ):
        condition = DIFFERENTIAL_CONDITIONS[condition_id]
        prob = raw_score / total

        differentials.append({
            "condition": condition_id,
            "display_name": condition["display_name"],
            "probability": round(prob, 4),
            "icd10_code": condition["icd10"],
            "supporting_features": details[condition_id]["supporting"],
            "opposing_features": details[condition_id]["opposing"],
        })

    # Add rank
    for i, d in enumerate(differentials):
        d["rank"] = i + 1

    return differentials
