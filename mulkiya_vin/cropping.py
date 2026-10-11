from __future__ import annotations

import re
from pathlib import Path


CHASSIS_LABEL_VARIANTS = {
    "chassis",
    "chassis no",
    "chassis number",
    "chasis",
    "chasis no",
    "chaassis",
    "cbaus no",
    "cbaus",
    "bau no",
    "cbau no",
}


def find_chassis_label_line(ocr_lines: list[dict]) -> dict | None:
    """Return the OCR line most likely to contain the chassis label."""
    best_line = None
    best_score = -1

    for line in ocr_lines:
        text = str(line.get("text", "") or "")
        normalized = _normalize_label_text(text)
        score = _label_match_score(normalized)
        if score > best_score:
            best_score = score
            best_line = line

    return best_line if best_score > 0 else None


def crop_chassis_region(
    image_path: str,
    ocr_lines: list[dict],
    output_dir: str = "outputs/qwen_crops",
    padding_x: int = 80,
    padding_y: int = 50,
) -> str | None:
    """Crop a horizontal band around the chassis label row and save it."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is not installed. Install it with: pip install pillow") from exc

    label_line = find_chassis_label_line(ocr_lines)
    if label_line is None:
        return None

    bbox = label_line.get("bbox")
    if not bbox:
        return None

    x_values, y_values = _extract_bbox_points(bbox)
    if not x_values or not y_values:
        return None

    image_path_obj = Path(image_path)
    image = Image.open(image_path_obj).convert("RGB")
    image_width, image_height = image.size

    label_x_min = min(x_values)
    label_x_max = max(x_values)
    label_y_min = min(y_values)
    label_y_max = max(y_values)

    crop_left = max(0, int(label_x_min - padding_x))
    crop_top = max(0, int(label_y_min - padding_y))
    crop_right = min(image_width, int(max(label_x_max + padding_x, image_width - padding_x)))
    crop_bottom = min(image_height, int(label_y_max + padding_y))

    if crop_right <= crop_left or crop_bottom <= crop_top:
        return None

    cropped = image.crop((crop_left, crop_top, crop_right, crop_bottom))

    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_obj / f"{image_path_obj.stem}_chassis_crop.jpg"
    cropped.save(output_path, format="JPEG")
    return str(output_path)


def _normalize_label_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _label_match_score(text: str) -> int:
    if not text:
        return 0
    if text in CHASSIS_LABEL_VARIANTS:
        return 3
    if any(variant in text for variant in CHASSIS_LABEL_VARIANTS):
        return 2
    if "chassis" in text or "chasis" in text or "cbaus" in text or "cbau" in text:
        return 1
    return 0


def _extract_bbox_points(bbox) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []

    if not isinstance(bbox, (list, tuple)):
        return x_values, y_values

    for point in bbox:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            x_values.append(float(point[0]))
            y_values.append(float(point[1]))
        except (TypeError, ValueError):
            continue

    return x_values, y_values
