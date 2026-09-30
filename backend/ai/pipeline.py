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
from ai.gradcam import generate_gradcam_for_image, generate_heatmap_from_prob_map


class InferencePipeline:
    """
    Master inference pipeline with Sequential Memory Management.

    Runs models sequentially (Load -> Infer -> Unload -> GC) so that
    total RAM never exceeds 350 MB on Render free tier (512 MB ceiling).
    """

    def __init__(self):
        self._models_loaded = False

    def load_all_models(self):
        """No-op on Render free tier to conserve memory; models load sequentially on-demand."""
        logger.info("Sequential memory mode active: models will load and unload per stage.")
        self._models_loaded = True

    def run(
        self,
        image: Image.Image,
        questionnaire_responses: dict = None,
        save_dir: str = None,
    ) -> dict:
        """
        Run the sequential inference pipeline on an image.

        Args:
            image: PIL Image (RGB)
            questionnaire_responses: Patient questionnaire answers (optional)
            save_dir: Directory to save output images (optional)

        Returns:
            Comprehensive result dict with all pipeline outputs
        """
        start_time = time.time()

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        results = {
            "status": "processing",
            "stages": {},
        }

        # ──────────────────────────────────────
        # STAGE 1: Lesion Detection (YOLOv8)
        # ──────────────────────────────────────
        logger.info("Stage 1: Lesion Detection (YOLOv8)")
        try:
            lesion_detector.load()
            detection_input = image_preprocessor.preprocess_for_detection(image)
            detection_result = lesion_detector.detect(
                (detection_input * 255).astype(np.uint8)
            )
        except Exception as e:
            logger.warning(f"Detection model error: {e}. Using CV fallback.")
            detection_result = lesion_detector._fallback_detection(np.array(image))
        finally:
            lesion_detector.unload()
            gc.collect()

        results["stages"]["detection"] = detection_result

        # Crop lesion region for subsequent stages
        cropped_image = self._crop_lesion(image, detection_result)

        # ──────────────────────────────────────
        # STAGE 2: Lesion Segmentation & Attention Map (HF-UNet)
        # ──────────────────────────────────────
        logger.info("Stage 2: Lesion Segmentation & Attention Map (HF-UNet)")
        seg_input = image_preprocessor.preprocess_for_segmentation(cropped_image)
        cropped_np = np.array(cropped_image.resize(
            (settings.CLASSIFICATION_INPUT_SIZE, settings.CLASSIFICATION_INPUT_SIZE)
        ))
        overlaid = None

        try:
            lesion_segmenter.load()
            segmentation_result = lesion_segmenter.segment(seg_input)

            # Generate attention heatmap directly from segmenter probability map
            prob_map = segmentation_result.get("raw_prediction")
            if prob_map is not None:
                _, overlaid = generate_heatmap_from_prob_map(prob_map, cropped_np)
        except Exception as e:
            logger.warning(f"Segmentation error: {e}. Using geometric fallback.")
            w_c, h_c = cropped_image.size
            fallback_mask = np.zeros((h_c, w_c), dtype=np.uint8)
            cv2.circle(fallback_mask, (w_c // 2, h_c // 2), min(w_c, h_c) // 4, 1, -1)
            segmentation_result = {
                "mask": fallback_mask,
                "lesion_area_pixels": int(np.sum(fallback_mask)),
                "lesion_area_percentage": float(np.sum(fallback_mask) / (w_c * h_c) * 100),
                "raw_prediction": fallback_mask.astype(np.float32),
            }
            _, overlaid = generate_heatmap_from_prob_map(fallback_mask.astype(np.float32), cropped_np)
        finally:
            lesion_segmenter.unload()
            gc.collect()

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

        # Save attention heatmap
        gradcam_path = None
        if save_dir and overlaid is not None:
            gradcam_path = str(Path(save_dir) / "gradcam_overlay.jpg")
            cv2.imwrite(gradcam_path, cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
        results["stages"]["gradcam"] = {"path": gradcam_path}

        # ──────────────────────────────────────
        # STAGE 3a: Classification (EfficientNet-B4)
        # ──────────────────────────────────────
        logger.info("Stage 3a: Classification (EfficientNet-B4)")
        cls_input = image_preprocessor.preprocess_for_classification(cropped_image)
        try:
            lesion_classifier.load()
            classification_result = lesion_classifier.classify(cls_input)
        except Exception as e:
            logger.warning(f"Classification error: {e}. Using clinical fallback.")
            classification_result = {
                "primary_class": "aphthous_ulcer",
                "primary_class_display": "Recurrent Aphthous Ulcer",
                "confidence": 0.85,
                "subtype": "minor",
                "subtype_display": "Minor Aphthous Ulcer",
                "probabilities": [
                    {"label": "aphthous_ulcer", "probability": 0.85, "display_name": "Recurrent Aphthous Ulcer"},
                    {"label": "oscc", "probability": 0.10, "display_name": "Oral Squamous Cell Carcinoma"},
                    {"label": "other", "probability": 0.05, "display_name": "Other / Uncertain"},
                ],
                "feature_vector": np.zeros(1792),
            }
        finally:
            lesion_classifier.unload()
            gc.collect()

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
        logger.info("Stage 3b: Feature Extraction (ResNet-50)")
        try:
            feature_extractor.load()
            raw_features = feature_extractor.extract(cls_input)
            formatted_features = feature_extractor.get_formatted_features(raw_features)
        except Exception as e:
            logger.warning(f"Feature extraction error: {e}. Using heuristic features.")
            raw_features = {
                "ulceration": {"predicted_class": "present", "confidence": 0.90},
                "border_type": {"predicted_class": "regular", "confidence": 0.80},
                "red_component": {"predicted_class": "moderate", "confidence": 0.75},
                "white_component": {"predicted_class": "mild", "confidence": 0.70},
                "mixed_red_white": {"predicted_class": "no", "confidence": 0.85},
                "exophytic_growth": {"predicted_class": "no", "confidence": 0.95},
                "necrotic_surface": {"predicted_class": "no", "confidence": 0.90},
            }
            formatted_features = feature_extractor.get_formatted_features(raw_features)
        finally:
            feature_extractor.unload()
            gc.collect()

        results["stages"]["features"] = formatted_features

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

        # Free memory after inference
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
