"""
OralGuard AI — Lesion Classification Model (EfficientNet-B4)

Stage 3 of the AI pipeline:
Classifies detected oral lesions into primary categories
(Aphthous Ulcer vs OSCC vs Other) with subtype classification.
"""

import gc
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger

from config import settings

try:
    import timm
except ImportError:
    timm = None
    logger.warning("timm not installed; classification model unavailable.")


class OralLesionClassifier(nn.Module):
    """
    EfficientNet-B4 based multi-task classifier.

    Architecture:    EfficientNet-B4 (primary) + optional DenseNet-169 ensemble
    Input Size:      380 x 380 px
    Primary Task:    3-class classification (aphthous, oscc, other)
    Secondary Task:  Subtype classification
    Output:          Softmax probabilities + subtype prediction

    Training Strategy: ImageNet pretrained -> fine-tune on oral lesion datasets
    """

    # Class and subtype definitions
    PRIMARY_CLASSES = ["aphthous_ulcer", "oscc", "other"]
    PRIMARY_DISPLAY = {
        "aphthous_ulcer": "Recurrent Aphthous Ulcer",
        "oscc": "Oral Squamous Cell Carcinoma",
        "other": "Other / Uncertain",
    }

    APHTHOUS_SUBTYPES = ["minor", "major", "herpetiform"]
    APHTHOUS_DISPLAY = {
        "minor": "Minor Aphthous Ulcer",
        "major": "Major Aphthous Ulcer",
        "herpetiform": "Herpetiform Aphthous Ulcer",
    }

    OSCC_SUBTYPES = ["endophytic", "exophytic", "verrucous"]
    OSCC_DISPLAY = {
        "endophytic": "Endophytic OSCC",
        "exophytic": "Exophytic OSCC",
        "verrucous": "Verrucous Carcinoma",
    }

    def __init__(self, num_primary=3, num_aphthous_sub=3, num_oscc_sub=3):
        super().__init__()
        self.num_primary = num_primary
        self.num_aphthous_sub = num_aphthous_sub
        self.num_oscc_sub = num_oscc_sub

        # Backbone: EfficientNet-B4
        if timm is not None:
            self.backbone = timm.create_model(
                "efficientnet_b4",
                pretrained=False,
                num_classes=0,  # Remove classifier head
                global_pool="avg",
            )
            feature_dim = self.backbone.num_features  # 1792 for B4
        else:
            feature_dim = 1792
            self.backbone = None

        # Classification heads
        self.dropout = nn.Dropout(0.3)

        # Primary classifier: Aphthous vs OSCC vs Other
        self.primary_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_primary),
        )

        # Aphthous subtype head: Minor vs Major vs Herpetiform
        self.aphthous_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_aphthous_sub),
        )

        # OSCC subtype head: Endophytic vs Exophytic vs Verrucous
        self.oscc_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_oscc_sub),
        )

    def forward(self, x):
        """
        Forward pass.

        Returns:
            tuple: (primary_logits, aphthous_logits, oscc_logits, features)
        """
        if self.backbone is not None:
            features = self.backbone(x)
        else:
            # Fallback: random features for testing
            features = torch.randn(x.size(0), 1792)

        features = self.dropout(features)

        primary_logits = self.primary_head(features)
        aphthous_logits = self.aphthous_head(features)
        oscc_logits = self.oscc_head(features)

        return primary_logits, aphthous_logits, oscc_logits, features

    def get_feature_vector(self, x):
        """Extract 1792-dim feature vector for fusion model."""
        if self.backbone is not None:
            return self.backbone(x)
        return torch.randn(x.size(0), 1792)


class LesionClassifierService:
    """
    Service wrapper for the classification model.
    Handles model loading, inference, and result formatting.
    """

    def __init__(self):
        self.model = None
        self.device = torch.device(settings.DEVICE)
        self._loaded = False

    def load(self, model_path: str = None):
        """Load classification model weights."""
        self.model = OralLesionClassifier()

        path = model_path or settings.CLASSIFICATION_MODEL_PATH

        if path and Path(path).exists():
            logger.info(f"Loading classification weights from: {path}")
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
        else:
            logger.info("Using randomly initialized classification model (no weights)")

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info("Classification model loaded")

    def unload(self):
        """Unload classification model from memory to free RAM."""
        self.model = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def classify(self, image_tensor: np.ndarray) -> dict:
        """
        Run classification on preprocessed image.

        Args:
            image_tensor: numpy array (C, H, W) normalized for EfficientNet-B4

        Returns:
            dict with classification results
        """
        if not self._loaded:
            self.load()

        with torch.no_grad():
            tensor = torch.from_numpy(image_tensor).unsqueeze(0).float().to(self.device)

            primary_logits, aphthous_logits, oscc_logits, features = self.model(tensor)

            # Softmax probabilities
            primary_probs = torch.softmax(primary_logits, dim=1)[0].cpu().numpy()
            aphthous_probs = torch.softmax(aphthous_logits, dim=1)[0].cpu().numpy()
            oscc_probs = torch.softmax(oscc_logits, dim=1)[0].cpu().numpy()

            # Primary prediction
            primary_idx = int(np.argmax(primary_probs))
            primary_class = OralLesionClassifier.PRIMARY_CLASSES[primary_idx]
            primary_conf = float(primary_probs[primary_idx])

            # Subtype prediction
            subtype = None
            subtype_display = None

            if primary_class == "aphthous_ulcer":
                sub_idx = int(np.argmax(aphthous_probs))
                subtype = OralLesionClassifier.APHTHOUS_SUBTYPES[sub_idx]
                subtype_display = OralLesionClassifier.APHTHOUS_DISPLAY[subtype]
            elif primary_class == "oscc":
                sub_idx = int(np.argmax(oscc_probs))
                subtype = OralLesionClassifier.OSCC_SUBTYPES[sub_idx]
                subtype_display = OralLesionClassifier.OSCC_DISPLAY[subtype]

            # Build probability list
            probabilities = []
            for i, cls in enumerate(OralLesionClassifier.PRIMARY_CLASSES):
                probabilities.append({
                    "label": cls,
                    "probability": float(primary_probs[i]),
                    "display_name": OralLesionClassifier.PRIMARY_DISPLAY[cls],
                })

            # Feature vector for fusion model
            feature_vector = features[0].cpu().numpy()

            return {
                "primary_class": primary_class,
                "primary_class_display": OralLesionClassifier.PRIMARY_DISPLAY[primary_class],
                "confidence": primary_conf,
                "subtype": subtype,
                "subtype_display": subtype_display,
                "probabilities": probabilities,
                "feature_vector": feature_vector,
                "aphthous_subtype_probs": {
                    s: float(aphthous_probs[i])
                    for i, s in enumerate(OralLesionClassifier.APHTHOUS_SUBTYPES)
                },
                "oscc_subtype_probs": {
                    s: float(oscc_probs[i])
                    for i, s in enumerate(OralLesionClassifier.OSCC_SUBTYPES)
                },
            }


# Singleton
lesion_classifier = LesionClassifierService()
