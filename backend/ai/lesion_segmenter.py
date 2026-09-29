"""
OralGuard AI — Lesion Segmentation Model (HF-UNet)

Stage 2 of the AI pipeline:
Pixel-level segmentation of oral lesions to compute precise
lesion boundaries, area, and shape features.

Reference: Wu et al., "A high-order focus interaction model and
oral ulcer dataset for oral ulcer segmentation" (Nature Portfolio)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from loguru import logger

from config import settings


class DoubleConvBlock(nn.Module):
    """Standard double convolution block for UNet."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class HighOrderFocusBlock(nn.Module):
    """
    High-order focus interaction block (HF block).
    Captures multi-scale contextual features for improved segmentation
    of oral ulcer boundaries.
    """

    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 5, padding=2),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
        )
        self.branch_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.branch_pool(x)
        bp = F.interpolate(bp, size=x.shape[2:], mode="bilinear", align_corners=False)

        cat = torch.cat([b1, b3, b5, bp], dim=1)
        attention = self.fusion(cat)
        return x * attention + x


class HFUNet(nn.Module):
    """
    HF-UNet: U-Net with High-order Focus interaction blocks.

    Architecture: Encoder-Decoder with skip connections and HF blocks
    Input:  512 x 512 x 3
    Output: 512 x 512 x 1 (binary mask)
    Loss:   Dice Loss + Binary Cross-Entropy
    Target Dice Score: >= 0.82
    Target mIoU:       >= 0.78
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConvBlock(in_channels, 64)
        self.enc2 = DoubleConvBlock(64, 128)
        self.enc3 = DoubleConvBlock(128, 256)
        self.enc4 = DoubleConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck with HF block
        self.bottleneck = nn.Sequential(
            DoubleConvBlock(512, 1024),
            HighOrderFocusBlock(1024),
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.hf4 = HighOrderFocusBlock(512)
        self.dec4 = DoubleConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.hf3 = HighOrderFocusBlock(256)
        self.dec3 = DoubleConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.hf2 = HighOrderFocusBlock(128)
        self.dec2 = DoubleConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.hf1 = HighOrderFocusBlock(64)
        self.dec1 = DoubleConvBlock(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections and HF blocks
        d4 = self.up4(b)
        e4 = self.hf4(e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self.hf3(e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self.hf2(e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.hf1(e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))


class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross-Entropy loss for segmentation training."""

    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        # Dice loss
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = 1 - (2 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        # BCE loss
        bce = self.bce(pred, target)

        return self.dice_weight * dice + self.bce_weight * bce


class LesionSegmenterService:
    """
    Service wrapper for the HF-UNet segmentation model.
    """

    def __init__(self):
        self.model = None
        self.device = torch.device(settings.DEVICE)
        self._loaded = False

    def load(self, model_path: str = None):
        """Load segmentation model weights."""
        self.model = HFUNet(in_channels=3, out_channels=1)

        path = model_path or settings.SEGMENTATION_MODEL_PATH

        if path and Path(path).exists():
            logger.info(f"Loading segmentation weights from: {path}")
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            logger.info("Using randomly initialized HF-UNet (no weights)")

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info("Segmentation model loaded")

    def segment(self, image_tensor: np.ndarray) -> dict:
        """
        Run lesion segmentation on preprocessed image.

        Args:
            image_tensor: numpy array (C, H, W) normalized

        Returns:
            dict with segmentation results
        """
        if not self._loaded:
            self.load()

        with torch.no_grad():
            tensor = torch.from_numpy(image_tensor).unsqueeze(0).float().to(self.device)
            mask_pred = self.model(tensor)
            mask = (mask_pred[0, 0] > 0.5).cpu().numpy().astype(np.uint8)

            total_pixels = mask.shape[0] * mask.shape[1]
            lesion_pixels = int(np.sum(mask))
            area_percentage = (lesion_pixels / total_pixels) * 100

            return {
                "mask": mask,
                "lesion_area_pixels": lesion_pixels,
                "lesion_area_percentage": float(area_percentage),
                "raw_prediction": mask_pred[0, 0].cpu().numpy(),
            }

    @staticmethod
    def compute_dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Compute Dice coefficient between predicted and ground truth masks."""
        smooth = 1e-6
        intersection = np.sum(pred_mask * gt_mask)
        return (2 * intersection + smooth) / (
            np.sum(pred_mask) + np.sum(gt_mask) + smooth
        )

    @staticmethod
    def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Compute IoU between predicted and ground truth masks."""
        smooth = 1e-6
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask) - intersection
        return (intersection + smooth) / (union + smooth)


# Singleton
lesion_segmenter = LesionSegmenterService()
