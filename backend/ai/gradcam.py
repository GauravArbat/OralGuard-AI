"""
OralGuard AI — Attention Heatmap Module

Uses ONLY the trained HF-UNet segmentation model output.
No color heuristics, no saliency analysis — pure model-based.

The model outputs a sigmoid probability map (0-1) for each pixel.
We threshold at 0.5 (the trained decision boundary), clean up noise
with morphology, and create a smooth heatmap from the result.
"""

import numpy as np
import cv2
import torch
from loguru import logger


def generate_segmentation_heatmap(
    segmenter_model,
    image_tensor: np.ndarray,
    original_image: np.ndarray,
) -> tuple:
    """
    Generate heatmap using ONLY the trained HF-UNet model output.

    Steps:
      1. Run model → get raw sigmoid probability map
      2. Threshold at 0.5 to get binary mask (what model considers lesion)
      3. Morphological cleanup (remove tiny noise dots)
      4. Keep only significant connected components
      5. Create smooth heatmap from cleaned mask
      6. Overlay on original image
    """
    try:
        device = next(segmenter_model.parameters()).device
        tensor = torch.from_numpy(image_tensor).unsqueeze(0).float().to(device)

        with torch.no_grad():
            raw_pred = segmenter_model(tensor)

        # Raw sigmoid probability map from model
        prob_map = raw_pred[0, 0].cpu().numpy()  # (256, 256), values in [0, 1]

        h, w = original_image.shape[:2]

        # ── Step 1: Binary mask at model's trained threshold ──
        binary = (prob_map > 0.5).astype(np.uint8) * 255

        # ── Step 2: Morphological cleanup on model resolution ──
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Remove small noise specks
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=2)
        # Fill small holes inside detected regions
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_med, iterations=2)

        # ── Step 3: Keep only significant connected components ──
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        clean_mask = np.zeros_like(binary)
        if num_labels > 1:
            # Get areas of all components (skip label 0 = background)
            areas = stats[1:, cv2.CC_STAT_AREA]
            total_area = binary.shape[0] * binary.shape[1]

            for i in range(len(areas)):
                # Keep components that are at least 0.3% of the image
                # (filters out tiny noise while keeping real ulcers)
                if areas[i] > total_area * 0.003:
                    clean_mask[labels == (i + 1)] = 255

        # ── Step 4: Resize clean mask to original image size ──
        clean_resized = cv2.resize(clean_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # ── Step 5: Create smooth heatmap from clean mask ──
        # Use the SOFT probability values inside the clean regions only
        prob_resized = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_float = clean_resized.astype(np.float32) / 255.0

        # Slightly dilate and blur for a smooth halo/glow effect
        glow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask_glow = cv2.dilate(mask_float, glow_kernel, iterations=2)
        mask_glow = cv2.GaussianBlur(mask_glow, (31, 31), 0)

        # Final heatmap: soft model probabilities gated by clean mask with glow
        heatmap = prob_resized * mask_glow

        # Normalize to [0, 1]
        hm_max = heatmap.max()
        if hm_max > 1e-6:
            heatmap = heatmap / hm_max
        else:
            # Model detected nothing — return empty (dark) heatmap
            logger.warning("Model detected no lesion regions above threshold")
            heatmap = np.zeros((h, w), dtype=np.float32)

        overlaid = overlay_heatmap(original_image, heatmap, alpha=0.5)

        lesion_pct = float(np.sum(mask_float > 0.5) / mask_float.size * 100)
        logger.info(
            f"Model-only heatmap: detected {np.sum(clean_mask > 0)} pixels, "
            f"lesion area={lesion_pct:.1f}%"
        )

        return heatmap, overlaid

    except Exception as e:
        logger.error(f"Segmentation heatmap failed: {e}")
        # Return empty heatmap on failure — no fallback heuristics
        h, w = original_image.shape[:2]
        empty = np.zeros((h, w), dtype=np.float32)
        return empty, original_image.copy()


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay heatmap on original image using JET colormap."""
    h, w = original_image.shape[:2]

    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap.astype(np.float32), (w, h))

    heatmap_u8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_u8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlaid = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlaid


def generate_gradcam_for_image(
    model,
    image_tensor: np.ndarray,
    original_image: np.ndarray,
    target_class: int = None,
    segmenter_model=None,
) -> tuple:
    """
    Generate attention heatmap using ONLY the trained segmenter model.
    Kept compatible with pipeline.py call signature.
    """
    if segmenter_model is not None:
        return generate_segmentation_heatmap(
            segmenter_model, image_tensor, original_image
        )

    # No model available — return empty
    h, w = original_image.shape[:2]
    return np.zeros((h, w), dtype=np.float32), original_image.copy()
