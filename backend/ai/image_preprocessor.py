"""
OralGuard AI — Image Preprocessing Pipeline

Handles image validation, quality assessment, and preprocessing
before AI model inference.
"""

import io
import numpy as np
from PIL import Image, ImageFilter, ImageStat
from loguru import logger

from config import settings


class ImagePreprocessor:
    """
    Validates and preprocesses intraoral photographs.

    Pipeline:
        1. Format validation (JPEG/PNG/HEIC/WebP)
        2. Resolution check
        3. Quality scoring (resolution + focus + lighting + FoV)
        4. Orientation correction (EXIF)
        5. Resize for model input
        6. Normalization
    """

    # ImageNet normalization constants
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self):
        self.max_size_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
        self.allowed_types = settings.ALLOWED_IMAGE_TYPES

    def validate_image(self, image_bytes: bytes, mime_type: str) -> dict:
        """
        Validate uploaded image file.

        Returns:
            dict with 'valid' (bool), 'message' (str), and 'image' (PIL.Image or None)
        """
        # Check file size
        if len(image_bytes) > self.max_size_bytes:
            return {
                "valid": False,
                "message": f"Image exceeds maximum size of {settings.MAX_IMAGE_SIZE_MB}MB",
                "image": None,
            }

        # Check MIME type
        if mime_type not in self.allowed_types:
            return {
                "valid": False,
                "message": f"Unsupported image format: {mime_type}. Use JPEG, PNG, or WebP.",
                "image": None,
            }

        # Try to open image
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
            # Re-open after verify (verify closes the file)
            img = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return {
                "valid": False,
                "message": f"Corrupted or unreadable image file: {str(e)}",
                "image": None,
            }

        # Convert to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Fix EXIF orientation
        img = self._fix_orientation(img)

        # Minimum resolution check & auto-upscaling for small or cropped images
        w, h = img.size
        if w < 64 or h < 64:
            return {
                "valid": False,
                "message": f"Image too small ({w}x{h}). Minimum 64x64 required.",
                "image": None,
            }

        # Auto-upscale images smaller than 320px (e.g. 256x256 dataset crops) so models receive optimal input
        if w < 320 or h < 320:
            scale = max(320.0 / w, 320.0 / h)
            new_w = max(320, int(w * scale))
            new_h = max(320, int(h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)

        return {"valid": True, "message": "Image accepted", "image": img}

    def compute_quality_score(self, img: Image.Image) -> dict:
        """
        Compute image quality score (0-100) based on PRD Appendix D rubric.

        Scores 4 parameters: resolution, focus, lighting, field-of-view.
        """
        w, h = img.size
        arr = np.array(img)

        # 1. Resolution score (0-25)
        max_dim = max(w, h)
        if max_dim >= 1920:
            resolution_score = 25
        elif max_dim >= 1280:
            resolution_score = 20
        elif max_dim >= 720:
            resolution_score = 16
        elif max_dim >= 480:
            resolution_score = 14
        elif max_dim >= 250:
            resolution_score = 12
        else:
            resolution_score = 10

        # 2. Focus/Sharpness score (0-25) — Laplacian variance
        focus_score = self._compute_focus_score(arr)

        # 3. Lighting score (0-25) — Exposure analysis
        lighting_score = self._compute_lighting_score(arr)

        # 4. Field of View score (0-25) — Heuristic
        fov_score = self._compute_fov_score(arr)

        total = resolution_score + focus_score + lighting_score + fov_score

        return {
            "resolution": resolution_score,
            "focus": focus_score,
            "lighting": lighting_score,
            "field_of_view": fov_score,
            "total": total,
            "acceptable": total >= 50,
        }

    def preprocess_for_detection(self, img: Image.Image) -> np.ndarray:
        """Resize and normalize for YOLOv8 detection (640x640)."""
        size = settings.DETECTION_INPUT_SIZE
        img_resized = img.resize((size, size), Image.LANCZOS)
        arr = np.array(img_resized).astype(np.float32) / 255.0
        return arr

    def preprocess_for_classification(self, img: Image.Image) -> np.ndarray:
        """Resize and normalize for EfficientNet-B4 classification (380x380)."""
        size = settings.CLASSIFICATION_INPUT_SIZE
        img_resized = img.resize((size, size), Image.LANCZOS)
        arr = np.array(img_resized).astype(np.float32) / 255.0

        # ImageNet normalization
        for c in range(3):
            arr[:, :, c] = (arr[:, :, c] - self.MEAN[c]) / self.STD[c]

        # HWC -> CHW for PyTorch
        arr = np.transpose(arr, (2, 0, 1))
        return arr

    def preprocess_for_segmentation(self, img: Image.Image) -> np.ndarray:
        """Resize and normalize for HF-UNet segmentation (256x256)."""
        # HF-UNet was specifically trained on 256x256 images
        size = 256 
        img_resized = img.resize((size, size), Image.LANCZOS)
        
        # Training script only divided by 255.0, NO ImageNet normalization was applied
        arr = np.array(img_resized).astype(np.float32) / 255.0

        arr = np.transpose(arr, (2, 0, 1))
        return arr

    # ── Private Methods ──

    def _fix_orientation(self, img: Image.Image) -> Image.Image:
        """Correct image orientation using EXIF data."""
        try:
            from PIL.ExifTags import Base as ExifBase
            exif = img.getexif()
            orientation = exif.get(0x0112)  # Orientation tag

            transforms = {
                2: Image.FLIP_LEFT_RIGHT,
                3: Image.ROTATE_180,
                4: Image.FLIP_TOP_BOTTOM,
                5: Image.TRANSPOSE,
                6: Image.ROTATE_270,
                7: Image.TRANSVERSE,
                8: Image.ROTATE_90,
            }

            if orientation in transforms:
                img = img.transpose(transforms[orientation])
        except Exception:
            pass
        return img

    def _compute_focus_score(self, arr: np.ndarray) -> int:
        """Laplacian variance method for sharpness detection."""
        gray = np.mean(arr, axis=2).astype(np.float64)

        # Laplacian kernel
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

        from scipy.signal import convolve2d
        try:
            response = convolve2d(gray, laplacian, mode="same", boundary="symm")
            variance = np.var(response)
        except ImportError:
            # Fallback: simple gradient magnitude
            gx = np.diff(gray, axis=1)
            gy = np.diff(gray, axis=0)
            variance = np.mean(gx ** 2) + np.mean(gy ** 2)

        if variance > 500:
            return 25
        elif variance > 200:
            return 20
        elif variance > 100:
            return 15
        elif variance > 50:
            return 10
        else:
            return 5

    def _compute_lighting_score(self, arr: np.ndarray) -> int:
        """Evaluate image exposure quality."""
        brightness = np.mean(arr)  # 0-255 range

        if 80 <= brightness <= 200:
            return 25
        elif 60 <= brightness <= 220:
            return 20
        elif 40 <= brightness <= 240:
            return 15
        else:
            return 5

    def _compute_fov_score(self, arr: np.ndarray) -> int:
        """
        Heuristic check for field-of-view adequacy.
        Checks if there's meaningful content (not blank/dark edges).
        """
        h, w = arr.shape[:2]

        # Check that center region has different content from edges
        center = arr[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        center_mean = np.mean(center)
        edge_mean = np.mean(arr)

        # If center is significantly different from overall, content is present
        diff = abs(center_mean - edge_mean)

        if diff > 20:
            return 25
        elif diff > 10:
            return 20
        elif diff > 5:
            return 15
        else:
            return 10  # Likely uniform, but not necessarily bad


# Singleton
image_preprocessor = ImagePreprocessor()
