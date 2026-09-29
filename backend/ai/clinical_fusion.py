"""
OralGuard AI — Clinical Fusion Model

Stage 4 of the AI pipeline:
Combines image-based features with questionnaire responses
and risk factors to produce final diagnosis probabilities
and risk score.

Architecture:  XGBoost / lightweight Transformer
Inputs:        Image features (2048-d) + 8 visual features + questionnaire
Output:        Final probability distribution + risk score (0-100)
"""

import numpy as np
from loguru import logger

from config import settings

try:
    import xgboost as xgb
except ImportError:
    xgb = None
    logger.warning("xgboost not installed; fusion model uses rule-based fallback.")


class ClinicalFusionModel:
    """
    Multimodal fusion model combining:
    - Image classification features (2048-dim from EfficientNet-B4)
    - 8 extracted visual clinical features
    - 20-32 questionnaire responses (encoded)
    - Risk factors (tobacco/alcohol/medical history)

    Output: Final risk score (0-100) and adjusted probabilities.
    """

    # Risk factor weights from clinical literature
    RISK_WEIGHTS = {
        # Tobacco-related
        "smoking_current": 15,
        "smoking_former": 8,
        "smokeless_tobacco_current": 20,  # High risk in India
        "smokeless_tobacco_former": 10,
        "betel_quid": 25,  # Very high risk factor in India

        # Alcohol
        "alcohol_heavy": 12,
        "alcohol_moderate": 5,

        # Clinical red flags
        "duration_over_3_weeks": 15,
        "non_healing": 20,
        "induration_present": 18,
        "irregular_borders": 15,
        "exophytic_growth": 20,
        "fixation_to_underlying": 22,
        "lymphadenopathy": 18,
        "weight_loss_unexplained": 12,
        "age_over_40": 5,
        "male_gender": 3,

        # Location risk (high-risk sites for OSCC)
        "lateral_tongue": 8,
        "floor_of_mouth": 8,
        "ventral_tongue": 7,
        "retromolar_area": 6,

        # Protective / benign indicators (negative weight = reduces risk)
        "recurrent_history": -10,
        "heals_within_2_weeks": -15,
        "pain_present": -5,  # OSCC often painless initially
        "vesicle_preceded": -8,  # Suggests herpes, not malignancy
        "regular_borders": -10,
        "non_keratinized_site": -3,
    }

    def __init__(self):
        self.xgb_model = None
        self._loaded = False

    def load(self, model_path: str = None):
        """Load fusion model (XGBoost)."""
        path = model_path or settings.FUSION_MODEL_PATH

        if xgb and path and Path(path).exists():
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(path)
            self._loaded = True
            logger.info(f"Loaded XGBoost fusion model from: {path}")
        else:
            logger.info("Using rule-based fusion (no trained XGBoost model)")
            self._loaded = True

    def fuse(
        self,
        image_features: np.ndarray,
        clinical_features: dict,
        questionnaire_responses: dict,
        classification_result: dict,
    ) -> dict:
        """
        Fuse all inputs into final risk score and adjusted diagnosis.

        Args:
            image_features: 2048-dim feature vector from classifier
            clinical_features: Extracted visual features dict
            questionnaire_responses: Patient questionnaire answers
            classification_result: Initial classification probabilities

        Returns:
            dict with:
                - risk_score (0-100)
                - risk_level (low/medium/high/urgent)
                - adjusted_probabilities
                - contributing_factors
                - image_risk, questionnaire_risk, clinical_features_risk
        """
        if not self._loaded:
            self.load()

        # ── Compute sub-scores ──

        # 1. Image-based risk (from classification)
        image_risk = self._compute_image_risk(classification_result)

        # 2. Clinical features risk
        features_risk = self._compute_features_risk(clinical_features)

        # 3. Questionnaire / history risk
        questionnaire_risk, risk_factors = self._compute_questionnaire_risk(
            questionnaire_responses
        )

        # ── Weighted fusion ──
        # PRD weights: Image 40%, Clinical features 25%, Questionnaire 35%
        risk_score = int(
            0.40 * image_risk
            + 0.25 * features_risk
            + 0.35 * questionnaire_risk
        )
        risk_score = max(0, min(100, risk_score))

        # ── Risk level mapping ──
        if risk_score <= settings.RISK_LOW_MAX:
            risk_level = "low"
        elif risk_score <= settings.RISK_MEDIUM_MAX:
            risk_level = "medium"
        elif risk_score <= settings.RISK_HIGH_MAX:
            risk_level = "high"
        else:
            risk_level = "urgent"

        # ── Adjust classification probabilities ──
        adjusted_probs = self._adjust_probabilities(
            classification_result, questionnaire_risk, features_risk
        )

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "image_risk": float(image_risk),
            "questionnaire_risk": float(questionnaire_risk),
            "clinical_features_risk": float(features_risk),
            "adjusted_probabilities": adjusted_probs,
            "contributing_factors": risk_factors,
        }

    def _compute_image_risk(self, classification_result: dict) -> float:
        """Compute risk based on classification model output."""
        probs = {
            p["label"]: p["probability"]
            for p in classification_result.get("probabilities", [])
        }

        oscc_prob = probs.get("oscc", 0)
        other_prob = probs.get("other", 0)

        # OSCC probability directly maps to high risk
        # Scale: oscc_prob 0.5+ = very high risk
        risk = oscc_prob * 100

        # "Other" carries moderate uncertainty risk
        risk += other_prob * 30

        return min(100, risk)

    def _compute_features_risk(self, features: dict) -> float:
        """Compute risk from extracted clinical visual features."""
        risk = 0

        # High-risk visual indicators
        if features.get("border_type") in ("irregular", "rolled"):
            risk += 20
        if features.get("exophytic_growth"):
            risk += 25
        if features.get("necrotic_surface"):
            risk += 20
        if features.get("mixed_red_white"):
            risk += 10
        if features.get("red_component") in ("moderate", "severe"):
            risk += 8
        if features.get("white_component") in ("moderate", "severe"):
            risk += 8

        # Low-risk indicators
        if features.get("border_type") == "regular":
            risk -= 10
        if features.get("ulceration") and not features.get("exophytic_growth"):
            risk -= 5  # Ulcer without growth is more likely benign

        # High-risk anatomical locations
        high_risk_locations = [
            "lateral_tongue_right", "lateral_tongue_left",
            "ventral_tongue", "floor_of_mouth", "retromolar_area",
        ]
        if features.get("anatomical_location") in high_risk_locations:
            risk += 10

        return max(0, min(100, risk))

    def _compute_questionnaire_risk(self, responses: dict) -> tuple:
        """
        Compute risk from questionnaire responses.

        Returns:
            tuple: (risk_score, list_of_contributing_factors)
        """
        risk = 0
        factors = []

        if not responses:
            return 25, ["Questionnaire not completed"]

        # Duration
        duration_days = responses.get("onset_duration_days", 0)
        if duration_days > 21:
            risk += self.RISK_WEIGHTS["duration_over_3_weeks"]
            factors.append(f"Lesion present for {duration_days} days (>3 weeks)")
        if duration_days > 14 and not responses.get("healing_trend", False):
            risk += self.RISK_WEIGHTS["non_healing"]
            factors.append("Non-healing lesion >2 weeks")

        # Recurrence
        if responses.get("recurrent", False):
            risk += self.RISK_WEIGHTS["recurrent_history"]
            factors.append("History of recurrent ulcers (favors aphthous)")

        # Healing pattern
        if responses.get("heals_within_2_weeks", False):
            risk += self.RISK_WEIGHTS["heals_within_2_weeks"]
            factors.append("Ulcers typically heal within 2 weeks (favors aphthous)")

        # Pain
        if responses.get("pain_present", False):
            risk += self.RISK_WEIGHTS["pain_present"]
            factors.append("Pain present (more common in aphthous)")

        # Vesicle history
        if responses.get("vesicle_preceded", False):
            risk += self.RISK_WEIGHTS["vesicle_preceded"]
            factors.append("Vesicle preceded ulcer (suggests herpes)")

        # Tobacco
        tobacco = responses.get("tobacco_use", "none")
        if tobacco == "smoking_current":
            risk += self.RISK_WEIGHTS["smoking_current"]
            factors.append("Current smoker")
        elif tobacco == "smokeless_current":
            risk += self.RISK_WEIGHTS["smokeless_tobacco_current"]
            factors.append("Current smokeless tobacco user")

        # Betel quid / paan
        if responses.get("betel_quid_use", False):
            risk += self.RISK_WEIGHTS["betel_quid"]
            factors.append("Betel quid / paan masala use (high OSCC risk)")

        # Alcohol
        alcohol = responses.get("alcohol_use", "none")
        if alcohol == "heavy":
            risk += self.RISK_WEIGHTS["alcohol_heavy"]
            factors.append("Heavy alcohol consumption")
        elif alcohol == "moderate":
            risk += self.RISK_WEIGHTS["alcohol_moderate"]
            factors.append("Moderate alcohol consumption")

        # Induration
        if responses.get("induration_present", False):
            risk += self.RISK_WEIGHTS["induration_present"]
            factors.append("Induration/hardness present (red flag)")

        # Lymphadenopathy
        if responses.get("lymphadenopathy", False):
            risk += self.RISK_WEIGHTS["lymphadenopathy"]
            factors.append("Regional lymph node swelling")

        # Weight loss
        if responses.get("weight_loss", False):
            risk += self.RISK_WEIGHTS["weight_loss_unexplained"]
            factors.append("Unexplained weight loss")

        # Age
        age = responses.get("age", 0)
        if age >= 40:
            risk += self.RISK_WEIGHTS["age_over_40"]
            factors.append(f"Age {age} (OSCC risk increases with age)")

        # Gender
        if responses.get("gender", "").lower() == "male":
            risk += self.RISK_WEIGHTS["male_gender"]

        return max(0, min(100, risk)), factors

    def _adjust_probabilities(
        self,
        classification_result: dict,
        questionnaire_risk: float,
        features_risk: float,
    ) -> list:
        """
        Adjust initial classification probabilities based on
        questionnaire and clinical features.
        """
        probs = {}
        for p in classification_result.get("probabilities", []):
            probs[p["label"]] = p["probability"]

        aphthous_prob = probs.get("aphthous_ulcer", 0.5)
        oscc_prob = probs.get("oscc", 0.1)
        other_prob = probs.get("other", 0.4)

        # Questionnaire-based adjustment
        combined_risk = (questionnaire_risk + features_risk) / 2

        if combined_risk > 60:
            # High risk: boost OSCC probability
            oscc_boost = min(0.3, (combined_risk - 60) / 100)
            oscc_prob += oscc_boost
            aphthous_prob -= oscc_boost * 0.7
            other_prob -= oscc_boost * 0.3
        elif combined_risk < 20:
            # Low risk: boost aphthous probability
            aphthous_boost = min(0.2, (20 - combined_risk) / 100)
            aphthous_prob += aphthous_boost
            oscc_prob -= aphthous_boost * 0.5
            other_prob -= aphthous_boost * 0.5

        # Normalize
        total = aphthous_prob + oscc_prob + other_prob
        if total > 0:
            aphthous_prob /= total
            oscc_prob /= total
            other_prob /= total

        return [
            {"label": "aphthous_ulcer", "probability": max(0, aphthous_prob),
             "display_name": "Recurrent Aphthous Ulcer"},
            {"label": "oscc", "probability": max(0, oscc_prob),
             "display_name": "Oral Squamous Cell Carcinoma"},
            {"label": "other", "probability": max(0, other_prob),
             "display_name": "Other / Uncertain"},
        ]


# Need this import
from pathlib import Path

# Singleton
clinical_fusion = ClinicalFusionModel()
