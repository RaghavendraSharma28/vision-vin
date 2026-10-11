from __future__ import annotations

from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None


class ImagePreprocessor:
    def __init__(self, max_width: int = 1800) -> None:
        self.max_width = max_width

    def preprocess(self, image_path: str | Path):
        cv = _get_cv2()
        image = cv.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image = self._resize(image)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        denoised = cv.fastNlMeansDenoising(gray, h=10)
        enhanced = cv.equalizeHist(denoised)
        sharpened = cv.addWeighted(enhanced, 1.5, cv.GaussianBlur(enhanced, (0, 0), 1), -0.5, 0)
        return sharpened

    def _resize(self, image):
        height, width = image.shape[:2]
        if width <= self.max_width:
            return image
        scale = self.max_width / width
        return cv2.resize(image, (self.max_width, int(height * scale)), interpolation=cv2.INTER_AREA)


def preprocess_for_ocr(image_path: str) -> str:
    """Save a lightly resized OCR input image and return its new path."""
    cv = _get_cv2()
    image_path_obj = Path(image_path)
    image = cv.imread(str(image_path_obj))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    min_width = 1200
    if width < min_width:
        scale = min_width / width
        image = cv.resize(image, (min_width, int(height * scale)), interpolation=cv.INTER_CUBIC)

    output_dir = Path("outputs") / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / image_path_obj.name

    if not cv.imwrite(str(output_path), image):
        raise ValueError(f"Could not write preprocessed image: {output_path}")

    return str(output_path)


def _get_cv2():
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed. Install it with: pip install opencv-python")
    return cv2
