"""
OralGuard AI — Clinical Decision Engine

The central brain that orchestrates:
1. AI pipeline execution
2. Questionnaire generation
3. Differential diagnosis
4. Recommendations & referral logic

This is the main entry point for the clinical workflow.
"""

from loguru import logger
from PIL import Image

from clinical.questionnaire import (
    generate_questionnaire,
    filter_dependent_questions,
)
from clinical.differential import generate_differential_diagnosis
from config import settings


class ClinicalDecisionEngine:
    """
    Master clinical decision engine.

    Workflow:
    1. Image upload → AI pipeline (detection, segmentation, classification)
    2. Generate dynamic questionnaire based on AI observations
    3. After questionnaire → differential diagnosis + risk scoring
    4. Generate recommendations and referral urgency
    """

    # ── Referral urgency rules (from PRD Section 9) ──
    REFERRAL_RULES = {
        "urgent": {
            "conditions": [
                "risk_score >= 76",
                "oscc_probability >= 0.50",
                "induration_present",
                "growing_lesion",
                "lymphadenopathy",
            ],
            "timeframe": "Within 24-48 hours",
            "action": "Urgent specialist referral — see oral surgeon or oncologist",
            "color": "red",
        },
        "high": {
            "conditions": [
                "risk_score 51-75",
                "oscc_probability 0.20-0.49",
                "duration_over_3_weeks",
                "suspicious_features",
            ],
            "timeframe": "Within 1-2 weeks",
            "action": "Priority referral — biopsy recommended",
            "color": "orange",
        },
        "medium": {
            "conditions": [
                "risk_score 26-50",
                "mixed_features",
                "uncertain_diagnosis",
            ],
            "timeframe": "Within 2-4 weeks",
            "action": "Routine specialist review recommended",
            "color": "yellow",
        },
        "low": {
            "conditions": [
                "risk_score 0-25",
                "classic_aphthous_presentation",
                "no_red_flags",
            ],
            "timeframe": "Reassess if not healed in 2 weeks",
            "action": "Self-care with follow-up monitoring",
            "color": "green",
        },
    }

    @property
    def pipeline(self):
        from ai.pipeline import inference_pipeline
        return inference_pipeline

    def process_image(self, image: Image.Image, save_dir: str = None) -> dict:
        """
        Phase 1: Process uploaded image through AI pipeline.

        Returns initial AI assessment before questionnaire.
        """
        logger.info("Clinical Decision Engine: Processing image")

        results = self.pipeline.run(
            image=image,
            questionnaire_responses=None,
            save_dir=save_dir,
        )

        # Generate initial questionnaire based on AI observations
        classification = results["summary"]
        oscc_prob = 0.0
        for p in results["stages"]["classification"]["probabilities"]:
            if p["label"] == "oscc":
                oscc_prob = p["probability"]

        questionnaire = generate_questionnaire(
            initial_classification=classification["primary_diagnosis"],
            oscc_probability=oscc_prob,
            clinical_features=classification["features"],
        )

        results["questionnaire"] = questionnaire
        results["phase"] = "awaiting_questionnaire"

        return results

    def process_questionnaire(
        self,
        initial_results: dict,
        questionnaire_responses: dict,
    ) -> dict:
        """
        Phase 2: Process questionnaire responses and generate final diagnosis.
        """
        logger.info("Clinical Decision Engine: Processing questionnaire")

        # Re-run fusion with questionnaire data
        from ai.clinical_fusion import clinical_fusion

        classification_result = initial_results["stages"]["classification"]
        clinical_features = initial_results["stages"]["features"]

        import numpy as np
        feature_vector = np.zeros(2048)  # Would come from stored initial run

        fusion_result = clinical_fusion.fuse(
            image_features=feature_vector,
            clinical_features=clinical_features,
            questionnaire_responses=questionnaire_responses,
            classification_result=classification_result,
        )

        # Generate differential diagnosis
        differentials = generate_differential_diagnosis(
            classification_result=classification_result,
            clinical_features=clinical_features,
            questionnaire_responses=questionnaire_responses,
        )

        # Generate recommendations
        recommendations = self.generate_recommendations(
            risk_level=fusion_result["risk_level"],
            risk_score=fusion_result["risk_score"],
            primary_diagnosis=classification_result["primary_class"],
            differentials=differentials,
            questionnaire_responses=questionnaire_responses,
            clinical_features=clinical_features,
        )

        # Determine referral urgency
        referral_urgency = self._determine_referral_urgency(
            risk_score=fusion_result["risk_score"],
            risk_level=fusion_result["risk_level"],
            questionnaire_responses=questionnaire_responses,
            classification_result=classification_result,
        )

        # Compile final results
        final_results = {
            **initial_results,
            "stages": {
                **initial_results["stages"],
                "fusion": fusion_result,
            },
            "summary": {
                **initial_results["summary"],
                "risk_score": fusion_result["risk_score"],
                "risk_level": fusion_result["risk_level"],
                "contributing_factors": fusion_result["contributing_factors"],
                "adjusted_probabilities": fusion_result["adjusted_probabilities"],
            },
            "differentials": differentials,
            "recommendations": recommendations,
            "referral_urgency": referral_urgency,
            "phase": "completed",
        }

        return final_results

    def generate_recommendations(
        self,
        risk_level: str,
        risk_score: int,
        primary_diagnosis: str,
        differentials: list,
        questionnaire_responses: dict,
        clinical_features: dict,
    ) -> list:
        """
        Generate clinical recommendations based on all available data.
        """
        recommendations = []

        # ── Risk-level based recommendations ──

        if risk_level == "urgent":
            recommendations.extend([
                "URGENT: Immediate referral to oral medicine specialist or oral surgeon.",
                "Incisional biopsy strongly recommended to rule out malignancy.",
                "Complete blood count and regional imaging (CT/MRI) may be indicated.",
                "Do NOT delay — schedule specialist appointment within 48 hours.",
            ])

        elif risk_level == "high":
            recommendations.extend([
                "Priority referral to dental specialist recommended within 1-2 weeks.",
                "Biopsy may be indicated based on clinical examination.",
                "Monitor for any changes in size, color, or symptoms.",
                "Avoid tobacco, alcohol, and betel products.",
            ])

        elif risk_level == "medium":
            recommendations.extend([
                "Clinical evaluation by a dentist recommended within 2-4 weeks.",
                "Reassess if the lesion does not show improvement.",
                "Document current appearance for comparison at follow-up.",
            ])

        elif risk_level == "low":
            recommendations.extend([
                "Clinical evaluation recommended as routine follow-up.",
                "Reassess if the ulcer does not heal within 2 weeks.",
            ])

        # ── Diagnosis-specific recommendations ──

        if primary_diagnosis == "aphthous_ulcer":
            recommendations.extend([
                "Symptomatic management: topical analgesic gel (e.g., lignocaine 2% gel).",
                "Antiseptic mouthwash (chlorhexidine 0.12%) twice daily.",
                "Avoid spicy, acidic, or abrasive foods until healed.",
                "Maintain good oral hygiene with a soft-bristled toothbrush.",
            ])

            # Subtype-specific
            responses = questionnaire_responses or {}
            size = responses.get("ulcer_size_mm", "")
            if size == "over_20" or size == "10_to_20":
                recommendations.append(
                    "Large aphthous ulcer (major type): topical corticosteroid "
                    "(triamcinolone acetonide 0.1%) may be prescribed by your dentist."
                )

            # Recurrence advice
            if responses.get("recurrent"):
                recommendations.append(
                    "For frequent recurrences: consider screening for nutritional "
                    "deficiencies (iron, B12, folate) and celiac disease."
                )

        elif primary_diagnosis == "oscc":
            recommendations.extend([
                "This screening suggests possible malignancy — this must be confirmed by biopsy.",
                "DO NOT IGNORE: Early detection significantly improves treatment outcomes.",
                "Cessation of all tobacco and betel products is critical.",
                "Alcohol cessation or reduction strongly recommended.",
            ])

        # ── Risk factor specific advice ──
        responses = questionnaire_responses or {}

        if responses.get("tobacco_use") in ("smoking_current", "smokeless_current"):
            recommendations.append(
                "Tobacco cessation counseling strongly recommended. "
                "Contact National Tobacco Quitline: 1800-11-2356."
            )

        if responses.get("betel_quid_use"):
            recommendations.append(
                "Stop betel nut/paan masala use immediately — "
                "this is a proven oral cancer risk factor."
            )

        return recommendations

    def _determine_referral_urgency(
        self,
        risk_score: int,
        risk_level: str,
        questionnaire_responses: dict,
        classification_result: dict,
    ) -> dict:
        """Determine referral urgency based on risk level and red flags."""
        responses = questionnaire_responses or {}
        red_flags = []

        # Check for red flags
        if responses.get("induration_present"):
            red_flags.append("Induration present")
        if responses.get("lymphadenopathy"):
            red_flags.append("Lymphadenopathy detected")
        if responses.get("healing_trend") == "growing":
            red_flags.append("Lesion is enlarging")
        if responses.get("weight_loss"):
            red_flags.append("Unexplained weight loss")
        if responses.get("numbness_paraesthesia"):
            red_flags.append("Numbness/paraesthesia")

        duration = responses.get("onset_duration_days", 0)
        if duration > 21:
            red_flags.append(f"Duration > 3 weeks ({duration} days)")

        # Override risk level if critical red flags present
        if len(red_flags) >= 3 and risk_level in ("medium", "high"):
            risk_level = "urgent"

        referral = self.REFERRAL_RULES[risk_level].copy()
        referral["red_flags"] = red_flags
        referral["risk_level"] = risk_level
        referral["risk_score"] = risk_score

        return referral


# Singleton
clinical_engine = ClinicalDecisionEngine()
