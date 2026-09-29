"""
OralGuard AI — Dynamic Clinical Questionnaire Engine

Generates targeted clinical questions based on the PRD's
20-question Aphthous Ulcer pathway and OSCC pathway.
Questions are dynamically ordered and filtered based on
AI initial observations and previous answers.
"""


# ══════════════════════════════════════════════
# APHTHOUS ULCER PATHWAY — 20 Questions
# ══════════════════════════════════════════════

APHTHOUS_QUESTIONS = [
    {
        "id": "onset_duration_days",
        "question": "When did the ulcer first appear? (number of days ago)",
        "question_type": "number",
        "required": True,
        "category": "onset",
        "clinical_rationale": "Duration helps distinguish aphthae (heal in 7-14 days) from persistent ulcers that may indicate malignancy.",
    },
    {
        "id": "recurrent",
        "question": "Have you had similar ulcers before?",
        "question_type": "boolean",
        "required": True,
        "category": "recurrence",
        "clinical_rationale": "Recurrence strongly supports recurrent aphthous stomatitis diagnosis.",
    },
    {
        "id": "recurrence_frequency",
        "question": "How frequently do the ulcers recur?",
        "question_type": "select",
        "options": [
            {"value": "first_time", "label": "This is the first time"},
            {"value": "rarely", "label": "Less than once a year"},
            {"value": "few_per_year", "label": "A few times a year"},
            {"value": "monthly", "label": "Monthly or more"},
            {"value": "almost_continuous", "label": "Almost continuous"},
        ],
        "required": False,
        "category": "recurrence",
        "clinical_rationale": "Helps establish recurrent pattern and severity classification.",
        "depends_on": {"recurrent": True},
    },
    {
        "id": "pain_present",
        "question": "Is the ulcer painful?",
        "question_type": "boolean",
        "required": True,
        "category": "symptoms",
        "clinical_rationale": "Aphthae are characteristically painful. Painless ulcers raise suspicion for malignancy.",
    },
    {
        "id": "pain_level",
        "question": "Rate the pain severity (1 = mild, 10 = severe)",
        "question_type": "scale",
        "options": [{"value": str(i), "label": str(i)} for i in range(1, 11)],
        "required": False,
        "category": "symptoms",
        "clinical_rationale": "Pain severity helps gauge impact and guides symptomatic management.",
        "depends_on": {"pain_present": True},
    },
    {
        "id": "vesicle_preceded",
        "question": "Did you notice a blister or vesicle before the ulcer formed?",
        "question_type": "boolean",
        "required": True,
        "category": "onset",
        "clinical_rationale": "Vesicular onset suggests herpes (recurrent intraoral herpes) rather than aphthous ulcer.",
    },
    {
        "id": "ulcer_count",
        "question": "How many ulcers are currently present?",
        "question_type": "select",
        "options": [
            {"value": "1", "label": "1 (single)"},
            {"value": "2-3", "label": "2-3"},
            {"value": "4-10", "label": "4-10"},
            {"value": "10+", "label": "More than 10"},
        ],
        "required": True,
        "category": "morphology",
        "clinical_rationale": "Helps classify: minor (1-5, <1cm), major (1-3, >1cm), herpetiform (10-100, tiny).",
    },
    {
        "id": "ulcer_size_mm",
        "question": "What is the approximate size of the largest ulcer? (mm)",
        "question_type": "select",
        "options": [
            {"value": "under_5", "label": "Less than 5 mm"},
            {"value": "5_to_10", "label": "5-10 mm"},
            {"value": "10_to_20", "label": "10-20 mm"},
            {"value": "over_20", "label": "More than 20 mm"},
        ],
        "required": True,
        "category": "morphology",
        "clinical_rationale": "Minor aphthae <10mm, major aphthae >10mm. Large persistent ulcers need biopsy rule-out.",
    },
    {
        "id": "location",
        "question": "Where exactly is the ulcer located?",
        "question_type": "select",
        "options": [
            {"value": "labial_mucosa", "label": "Inside of lip"},
            {"value": "buccal_mucosa", "label": "Inside of cheek"},
            {"value": "lateral_tongue", "label": "Side of tongue"},
            {"value": "ventral_tongue", "label": "Under the tongue"},
            {"value": "dorsal_tongue", "label": "Top of tongue"},
            {"value": "floor_of_mouth", "label": "Floor of mouth"},
            {"value": "soft_palate", "label": "Soft palate"},
            {"value": "hard_palate", "label": "Hard palate (roof of mouth)"},
            {"value": "gingiva", "label": "Gums"},
            {"value": "retromolar", "label": "Behind back teeth"},
        ],
        "required": True,
        "category": "morphology",
        "clinical_rationale": "Aphthae occur on non-keratinized mucosa. Keratinized site ulcers suggest herpes. Lateral tongue/floor of mouth = high OSCC risk.",
    },
    {
        "id": "dysphagia",
        "question": "Do you have difficulty or pain while eating or swallowing?",
        "question_type": "boolean",
        "required": True,
        "category": "symptoms",
        "clinical_rationale": "Functional impact indicates severity. Persistent dysphagia may suggest deeper pathology.",
    },
    {
        "id": "systemic_symptoms",
        "question": "Do you have any of these symptoms?",
        "question_type": "multiselect",
        "options": [
            {"value": "fever", "label": "Fever"},
            {"value": "malaise", "label": "General feeling of illness"},
            {"value": "weight_loss", "label": "Unintentional weight loss"},
            {"value": "joint_pain", "label": "Joint pain"},
            {"value": "skin_lesions", "label": "Skin rashes or sores"},
            {"value": "genital_ulcers", "label": "Genital ulcers"},
            {"value": "eye_problems", "label": "Eye redness or pain"},
            {"value": "gi_problems", "label": "Chronic diarrhea or stomach issues"},
            {"value": "none", "label": "None of the above"},
        ],
        "required": True,
        "category": "systemic",
        "clinical_rationale": "Genital ulcers + eye issues = Behcet. GI symptoms = Crohn/celiac. Skin = erythema multiforme. Weight loss = malignancy concern.",
    },
    {
        "id": "tobacco_use",
        "question": "Do you use any form of tobacco?",
        "question_type": "select",
        "options": [
            {"value": "none", "label": "Never used"},
            {"value": "smoking_current", "label": "Currently smoke (cigarettes/bidi)"},
            {"value": "smoking_former", "label": "Former smoker"},
            {"value": "smokeless_current", "label": "Currently use smokeless tobacco (gutkha/khaini/zarda)"},
            {"value": "smokeless_former", "label": "Former smokeless tobacco user"},
        ],
        "required": True,
        "category": "risk_factors",
        "clinical_rationale": "Tobacco is the strongest risk factor for OSCC. Smokeless tobacco extremely common in India.",
    },
    {
        "id": "betel_quid_use",
        "question": "Do you chew betel nut (supari), paan, or paan masala?",
        "question_type": "boolean",
        "required": True,
        "category": "risk_factors",
        "clinical_rationale": "Betel quid/areca nut is a Group 1 carcinogen. Extremely important risk factor in Indian population.",
    },
    {
        "id": "alcohol_use",
        "question": "How much alcohol do you consume?",
        "question_type": "select",
        "options": [
            {"value": "none", "label": "None / rarely"},
            {"value": "light", "label": "Light (1-2 drinks/week)"},
            {"value": "moderate", "label": "Moderate (3-7 drinks/week)"},
            {"value": "heavy", "label": "Heavy (>7 drinks/week)"},
        ],
        "required": True,
        "category": "risk_factors",
        "clinical_rationale": "Alcohol + tobacco synergistically increase OSCC risk by 15x.",
    },
    {
        "id": "medical_conditions",
        "question": "Do you have any of these medical conditions?",
        "question_type": "multiselect",
        "options": [
            {"value": "diabetes", "label": "Diabetes"},
            {"value": "hiv", "label": "HIV/AIDS"},
            {"value": "autoimmune", "label": "Autoimmune disease (lupus, rheumatoid arthritis)"},
            {"value": "ibd", "label": "Inflammatory bowel disease (Crohn's, UC)"},
            {"value": "celiac", "label": "Celiac disease"},
            {"value": "anemia", "label": "Anemia or iron/B12/folate deficiency"},
            {"value": "immunosuppressed", "label": "Taking immunosuppressive medications"},
            {"value": "cancer_history", "label": "Previous cancer diagnosis"},
            {"value": "none", "label": "None of the above"},
        ],
        "required": True,
        "category": "systemic",
        "clinical_rationale": "Systemic diseases cause recurrent aphthae. Immunosuppression increases infection and malignancy risk.",
    },
    {
        "id": "medications",
        "question": "Are you currently taking any of these medications?",
        "question_type": "multiselect",
        "options": [
            {"value": "nsaids", "label": "NSAIDs (ibuprofen, aspirin)"},
            {"value": "methotrexate", "label": "Methotrexate"},
            {"value": "nicorandil", "label": "Nicorandil"},
            {"value": "bisphosphonates", "label": "Bisphosphonates"},
            {"value": "immunosuppressants", "label": "Immunosuppressants"},
            {"value": "chemotherapy", "label": "Chemotherapy drugs"},
            {"value": "none", "label": "None of the above"},
        ],
        "required": True,
        "category": "systemic",
        "clinical_rationale": "Certain drugs (NSAIDs, nicorandil, methotrexate) can cause drug-induced oral ulceration.",
    },
    {
        "id": "trauma_history",
        "question": "Could the ulcer be due to trauma?",
        "question_type": "select",
        "options": [
            {"value": "none", "label": "No obvious trauma"},
            {"value": "biting", "label": "I bit my cheek/lip/tongue"},
            {"value": "sharp_tooth", "label": "Sharp or broken tooth"},
            {"value": "denture", "label": "Denture or orthodontic appliance"},
            {"value": "hot_food", "label": "Burn from hot food/drink"},
        ],
        "required": True,
        "category": "onset",
        "clinical_rationale": "Trauma history points to traumatic ulcer rather than aphthae. Important for differential.",
    },
    {
        "id": "induration_present",
        "question": "Does the ulcer feel hard or firm when you touch it?",
        "question_type": "boolean",
        "required": True,
        "category": "red_flags",
        "clinical_rationale": "Induration (hardness) is a RED FLAG for malignancy. Aphthae are NOT indurated.",
    },
    {
        "id": "lymphadenopathy",
        "question": "Have you noticed any swollen lymph nodes in your neck?",
        "question_type": "boolean",
        "required": True,
        "category": "red_flags",
        "clinical_rationale": "Regional lymphadenopathy may indicate metastatic spread of OSCC.",
    },
    {
        "id": "healing_trend",
        "question": "Is the ulcer showing signs of healing (getting smaller)?",
        "question_type": "select",
        "options": [
            {"value": "healing", "label": "Yes, it's getting smaller"},
            {"value": "stable", "label": "No change"},
            {"value": "growing", "label": "It's getting bigger"},
            {"value": "unsure", "label": "Not sure"},
        ],
        "required": True,
        "category": "progression",
        "clinical_rationale": "Growing/non-healing ulcers are a major RED FLAG for malignancy.",
    },
]


# ══════════════════════════════════════════════
# OSCC PATHWAY — Additional Questions
# (Asked when AI suspects malignancy)
# ══════════════════════════════════════════════

OSCC_ADDITIONAL_QUESTIONS = [
    {
        "id": "numbness_paraesthesia",
        "question": "Do you have any numbness or tingling in your lip, tongue, or face?",
        "question_type": "boolean",
        "required": True,
        "category": "red_flags",
        "clinical_rationale": "Numbness/paraesthesia may indicate nerve invasion by malignancy.",
    },
    {
        "id": "trismus",
        "question": "Do you have difficulty opening your mouth fully?",
        "question_type": "boolean",
        "required": True,
        "category": "red_flags",
        "clinical_rationale": "Trismus (restricted mouth opening) may indicate deep tissue invasion or OSMF.",
    },
    {
        "id": "loose_teeth",
        "question": "Have any teeth become loose recently without obvious cause?",
        "question_type": "boolean",
        "required": True,
        "category": "red_flags",
        "clinical_rationale": "Unexplained tooth mobility may indicate bone invasion by tumor.",
    },
    {
        "id": "ear_pain",
        "question": "Do you have unexplained pain in your ear?",
        "question_type": "boolean",
        "required": True,
        "category": "red_flags",
        "clinical_rationale": "Referred otalgia is common in oral/oropharyngeal malignancy.",
    },
    {
        "id": "voice_changes",
        "question": "Have you noticed any changes in your voice?",
        "question_type": "boolean",
        "required": True,
        "category": "red_flags",
        "clinical_rationale": "Voice changes may indicate laryngeal or deep oropharyngeal involvement.",
    },
    {
        "id": "family_cancer_history",
        "question": "Does anyone in your family have a history of oral or head-and-neck cancer?",
        "question_type": "boolean",
        "required": True,
        "category": "risk_factors",
        "clinical_rationale": "Family history may indicate genetic susceptibility.",
    },
]


def generate_questionnaire(
    initial_classification: str = None,
    oscc_probability: float = 0.0,
    clinical_features: dict = None,
) -> list:
    """
    Generate a dynamic questionnaire based on AI initial assessment.

    Args:
        initial_classification: Initial AI classification (aphthous_ulcer/oscc/other)
        oscc_probability: Initial OSCC probability from classifier
        clinical_features: Extracted visual features

    Returns:
        list of question dicts to present to the user
    """
    questions = list(APHTHOUS_QUESTIONS)  # Always start with base questions

    # If OSCC probability > 15%, add OSCC-specific questions
    if oscc_probability > 0.15 or initial_classification == "oscc":
        questions.extend(OSCC_ADDITIONAL_QUESTIONS)

    # If features suggest high-risk, add OSCC questions
    if clinical_features:
        high_risk_features = (
            clinical_features.get("exophytic_growth", False)
            or clinical_features.get("necrotic_surface", False)
            or clinical_features.get("border_type") in ("irregular", "rolled")
        )
        if high_risk_features and OSCC_ADDITIONAL_QUESTIONS[0] not in questions:
            questions.extend(OSCC_ADDITIONAL_QUESTIONS)

    return questions


def filter_dependent_questions(
    questions: list, current_responses: dict
) -> list:
    """
    Filter questions based on dependency conditions.
    Only return questions whose dependencies are met.
    """
    filtered = []
    for q in questions:
        depends = q.get("depends_on")
        if depends is None:
            filtered.append(q)
        else:
            # Check if all dependency conditions are met
            all_met = all(
                current_responses.get(dep_key) == dep_val
                for dep_key, dep_val in depends.items()
            )
            if all_met:
                filtered.append(q)
    return filtered
