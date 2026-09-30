"""
OralGuard AI — Clinical Feature Extraction Model

Stage 3b of the AI pipeline:
Extracts 8 clinical visual features from the lesion image
using a multi-task ResNet-50 with custom classification heads.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger

from config import settings

try:
    import torchvision.models as tv_models
except ImportError:
    tv_models = None


# ── Feature definitions ──

FEATURE_DEFINITIONS = {
    "ulceration": {
        "type": "binary",
        "classes": ["absent", "present"],
        "description": "Ulceration present or absent",
    },
    "border_type": {
        "type": "multiclass",
        "classes": ["regular", "irregular", "rolled", "diffuse"],
        "description": "Border morphology of the lesion",
    },
    "red_component": {
        "type": "ordinal",
        "classes": ["none", "mild", "moderate", "severe"],
        "description": "Degree of erythema/redness",
    },
    "white_component": {
        "type": "ordinal",
        "classes": ["none", "mild", "moderate", "severe"],
        "description": "Degree of white/keratotic component",
    },
    "mixed_red_white": {
        "type": "binary",
        "classes": ["no", "yes"],
        "description": "Mixed red and white pattern",
    },
    "exophytic_growth": {
        "type": "binary",
        "classes": ["no", "yes"],
        "description": "Exophytic or fungating growth present",
    },
    "necrotic_surface": {
        "type": "binary",
        "classes": ["no", "yes"],
        "description": "Necrotic or granular surface",
    },
    "anatomical_location": {
        "type": "multiclass",
        "classes": [
            "labial_mucosa_upper", "labial_mucosa_lower",
            "buccal_mucosa_right", "buccal_mucosa_left",
            "lateral_tongue_right", "lateral_tongue_left",
            "ventral_tongue", "dorsal_tongue",
            "floor_of_mouth", "soft_palate",
            "hard_palate", "attached_gingiva",
            "retromolar_area", "lip_vermilion",
        ],
        "description": "Anatomical site of the lesion",
    },
}


class MultiTaskFeatureExtractor(nn.Module):
    """
    Multi-task ResNet-50 for simultaneous prediction of 8 clinical features.

    Each feature has its own classification head branching from
    the shared ResNet-50 backbone.
    """

    def __init__(self):
        super().__init__()

        # Shared backbone
        if tv_models is not None:
            backbone = tv_models.resnet50(weights="IMAGENET1K_V2")
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            feature_dim = 2048
        else:
            self.features = None
            feature_dim = 2048

        self.flatten = nn.Flatten()

        # Per-feature classification heads
        self.heads = nn.ModuleDict()
        for feat_name, feat_def in FEATURE_DEFINITIONS.items():
            num_classes = len(feat_def["classes"])
            self.heads[feat_name] = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )

    def forward(self, x):
        """
        Forward pass returns dict of logits per feature.
        """
        if self.features is not None:
            feat = self.features(x)
        else:
            feat = torch.randn(x.size(0), 2048, 1, 1)

        feat = self.flatten(feat)

        outputs = {}
        for feat_name, head in self.heads.items():
            outputs[feat_name] = head(feat)

        return outputs


class FeatureExtractorService:
    """Service wrapper for the multi-task feature extraction model."""

    def __init__(self):
        self.model = None
        self.device = torch.device(settings.DEVICE)
        self._loaded = False

    def load(self, model_path: str = None):
        """Load feature extraction model weights."""
        self.model = MultiTaskFeatureExtractor()

        path = model_path or settings.FEATURE_MODEL_PATH

        if path and Path(path).exists():
            logger.info(f"Loading feature model from: {path}")
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
        else:
            logger.info("Using randomly initialized feature extractor (no weights)")

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True

    def extract(self, image_tensor: np.ndarray) -> dict:
        """
        Extract 8 clinical features from a preprocessed image.

        Returns:
            dict mapping feature names to predicted values
        """
        if not self._loaded:
            self.load()

        with torch.no_grad():
            tensor = torch.from_numpy(image_tensor).unsqueeze(0).float().to(self.device)
            outputs = self.model(tensor)

            results = {}
            for feat_name, logits in outputs.items():
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_idx = int(np.argmax(probs))
                classes = FEATURE_DEFINITIONS[feat_name]["classes"]

                results[feat_name] = {
                    "predicted_class": classes[pred_idx],
                    "confidence": float(probs[pred_idx]),
                    "probabilities": {
                        cls: float(probs[i]) for i, cls in enumerate(classes)
                    },
                }

            return results

    def get_formatted_features(self, raw_results: dict) -> dict:
        """
        Convert raw extraction results to the ClinicalFeatures format.
        """
        return {
            "ulceration": raw_results["ulceration"]["predicted_class"] == "present",
            "border_type": raw_results["border_type"]["predicted_class"],
            "red_component": raw_results["red_component"]["predicted_class"],
            "white_component": raw_results["white_component"]["predicted_class"],
            "mixed_red_white": raw_results["mixed_red_white"]["predicted_class"] == "yes",
            "exophytic_growth": raw_results["exophytic_growth"]["predicted_class"] == "yes",
            "necrotic_surface": raw_results["necrotic_surface"]["predicted_class"] == "yes",
            "anatomical_location": raw_results["anatomical_location"]["predicted_class"],
        }


# Singleton
feature_extractor = FeatureExtractorService()
