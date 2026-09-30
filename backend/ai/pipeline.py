"""
OralGuard AI — End-to-End Inference Pipeline

Orchestrates the 4-stage AI pipeline:
  Stage 1: YOLOv8 detection (lesion localization)
  Stage 2: HF-UNet segmentation (pixel mask)
  Stage 3: EfficientNet-B4 classification + ResNet-50 feature extraction
  Stage 4: Clinical fusion (image + questionnaire → risk score)
"""

import time
import gc
import numpy as np
import cv2
import torch
from PIL import Image
from pathlib import Path
from loguru import logger

from config import settings
from ai.image_preprocessor import image_preprocessor
from ai.lesion_detector import lesion_detector
from ai.lesion_classifier import lesion_classifier
from ai.lesion_segmenter import lesion_segmenter
from ai.feature_extractor import feature_extractor
from ai.clinical_fusion import clinical_fusion
from ai.gradcam import generate_gradcam_for_image


class InferencePipeline:
    """
    Master inference pipeline that chains all AI models.

    Flow:
        Image → Preprocessing → Detection → Crop →
        Classification + Segmentation + Feature Extraction →
        Grad-CAM → Clinical Fusion → Results
    """

    def __init__(self):
        self._models_loaded = False

    def load_all_models(self):
        """Pre-load all models into memory."""
        logger.info("Loading all AI models...")
        start = time.time()

        lesion_detector.load()
        lesion_classifier.load()
        lesion_segmenter.load()
        feature_extractor.load()
        clinical_fusion.load()

        elapsed = time.time() - start
        logger.info(f"All models loaded in {elapsed:.1f}s")
        self._models_loaded = True

    def run(
        self,
        image: Image.Image,
        questionnaire_responses: dict = None,
        save_dir: str = None,
    ) -> dict:
        """
        Run the full inference pipeline on an image.

        Args:
            image: PIL Image (RGB)
            questionnaire_responses: Patient questionnaire answers (optional)
            save_dir: Directory to save output images (optional)

        Returns:
            Comprehensive result dict with all pipeline outputs
        """
        start_time = time.time()
        original_np = np.array(image)

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        results = {
            "status": "processing",
            "stages": {},
        }

        # ──────────────────────────────────────
        # STAGE 1: Lesion Detection (YOLOv8)
        # ──────────────────────────────────────
        logger.info("Stage 1: Lesion Detection")
        detection_input = image_preprocessor.preprocess_for_detection(image)
        detection_result = lesion_detector.detect(
            (detection_input * 255).astype(np.uint8)
        )
        results["stages"]["detection"] = detection_result

        # Crop lesion region for subsequent stages
        cropped_image = self._crop_lesion(image, detection_result)

        # ──────────────────────────────────────
        # STAGE 2: Lesion Segmentation (HF-UNet)
        # ──────────────────────────────────────
        logger.info("Stage 2: Lesion Segmentation")
        seg_input = image_preprocessor.preprocess_for_segmentation(cropped_image)
        segmentation_result = lesion_segmenter.segment(seg_input)

        # Save segmentation mask
        seg_mask_path = None
        if save_dir:
            seg_mask_path = str(Path(save_dir) / "segmentation_mask.png")
            mask_vis = (segmentation_result["mask"] * 255).astype(np.uint8)
            cv2.imwrite(seg_mask_path, mask_vis)

        results["stages"]["segmentation"] = {
            "lesion_area_pixels": segmentation_result["lesion_area_pixels"],
            "lesion_area_percentage": segmentation_result["lesion_area_percentage"],
            "mask_path": seg_mask_path,
        }

        # ──────────────────────────────────────
        # STAGE 3a: Classification (EfficientNet-B4)
        # ──────────────────────────────────────
        logger.info("Stage 3a: Classification")
        cls_input = image_preprocessor.preprocess_for_classification(cropped_image)
        classification_result = lesion_classifier.classify(cls_input)

        results["stages"]["classification"] = {
            "primary_class": classification_result["primary_class"],
            "primary_class_display": classification_result["primary_class_display"],
            "confidence": classification_result["confidence"],
            "subtype": classification_result["subtype"],
            "subtype_display": classification_result["subtype_display"],
            "probabilities": classification_result["probabilities"],
        }

        # ──────────────────────────────────────
        # STAGE 3b: Feature Extraction (ResNet-50)
        # ──────────────────────────────────────
        logger.info("Stage 3b: Feature Extraction")
        feat_input = image_preprocessor.preprocess_for_classification(cropped_image)
        raw_features = feature_extractor.extract(feat_input)
        formatted_features = feature_extractor.get_formatted_features(raw_features)

        results["stages"]["features"] = formatted_features

        # ──────────────────────────────────────
        # STAGE 3c: Grad-CAM / Attention Heatmap
        # ──────────────────────────────────────
        logger.info("Stage 3c: Generating attention heatmap from trained HF-UNet segmenter")
        gradcam_path = None
        try:
            cropped_np = np.array(cropped_image.resize(
                (settings.CLASSIFICATION_INPUT_SIZE, settings.CLASSIFICATION_INPUT_SIZE)
            ))

            # Use the trained HF-UNet segmenter model (trained on real Autooral data)
            # as the primary attention source — its raw probability output IS the heatmap.
            # Pass seg_input (256x256) which is what the HF-UNet was trained on.
            heatmap, overlaid = generate_gradcam_for_image(
                model=lesion_classifier.model,
                image_tensor=seg_input,       # 256x256 for HF-UNet segmenter
                original_image=cropped_np,
                segmenter_model=lesion_segmenter.model,
            )

            if save_dir:
                gradcam_path = str(Path(save_dir) / "gradcam_overlay.jpg")
                cv2.imwrite(
                    gradcam_path,
                    cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR),
                )
        except Exception as e:
            logger.warning(f"Attention heatmap generation failed: {e}")

        results["stages"]["gradcam"] = {"path": gradcam_path}

        # ──────────────────────────────────────
        # STAGE 4: Clinical Fusion
        # ──────────────────────────────────────
        logger.info("Stage 4: Clinical Fusion")
        fusion_result = clinical_fusion.fuse(
            image_features=classification_result.get("feature_vector", np.zeros(2048)),
            clinical_features=formatted_features,
            questionnaire_responses=questionnaire_responses or {},
            classification_result=classification_result,
        )

        results["stages"]["fusion"] = fusion_result

        # ──────────────────────────────────────
        # COMPILE FINAL RESULTS
        # ──────────────────────────────────────
        processing_time = int((time.time() - start_time) * 1000)

        results["status"] = "completed"
        results["processing_time_ms"] = processing_time

        # Flatten key results for easy access
        results["summary"] = {
            "primary_diagnosis": classification_result["primary_class"],
            "primary_diagnosis_display": classification_result["primary_class_display"],
            "confidence": classification_result["confidence"],
            "subtype": classification_result["subtype"],
            "subtype_display": classification_result["subtype_display"],
            "risk_score": fusion_result["risk_score"],
            "risk_level": fusion_result["risk_level"],
            "features": formatted_features,
            "contributing_factors": fusion_result["contributing_factors"],
            "adjusted_probabilities": fusion_result["adjusted_probabilities"],
            "gradcam_path": gradcam_path,
            "segmentation_mask_path": seg_mask_path,
        }

        logger.info(
            f"Pipeline complete: {classification_result['primary_class_display']} "
            f"(conf={classification_result['confidence']:.2f}, "
            f"risk={fusion_result['risk_score']}/100) "
            f"in {processing_time}ms"
        )

        # Free memory after inference (critical for Render free tier 512MB)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def _crop_lesion(
        self, image: Image.Image, detection_result: dict
    ) -> Image.Image:
        """
        Crop the detected lesion region from the full image.
        Adds padding around the bounding box for context.
        """
        w, h = image.size

        if not detection_result.get("lesion_detected") or not detection_result.get("bounding_boxes"):
            # Return full image if no detection
            return image

        # Use the highest-confidence bounding box
        bbox = max(
            detection_result["bounding_boxes"],
            key=lambda b: b["confidence"],
        )

        # Add 15% padding
        pad_x = bbox["width"] * 0.15
        pad_y = bbox["height"] * 0.15

        x1 = max(0, int(bbox["x"] - pad_x))
        y1 = max(0, int(bbox["y"] - pad_y))
        x2 = min(w, int(bbox["x"] + bbox["width"] + pad_x))
        y2 = min(h, int(bbox["y"] + bbox["height"] + pad_y))

        # Ensure minimum crop size
        if (x2 - x1) < 100 or (y2 - y1) < 100:
            return image

        return image.crop((x1, y1, x2, y2))


# Singleton
inference_pipeline = InferencePipeline()
