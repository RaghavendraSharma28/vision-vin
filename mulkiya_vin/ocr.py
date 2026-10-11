from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .types import OcrResult
from .vin import clean_chassis_no, validate_chassis_no


class OcrEngine:
    def extract(self, image_path: str | Path) -> list[OcrResult]:
        raise NotImplementedError


class PaddleOcrEngine(OcrEngine):
    def __init__(self, lang: str = "en") -> None:
        self._ocr = get_ocr_engine(engine_name="paddle", lang=lang)

    def extract(self, image_path: str | Path) -> list[OcrResult]:
        return [
            OcrResult(text=line["text"], confidence=line["confidence"])
            for line in extract_ocr_lines(str(image_path), self._ocr)
        ]


class NullOcrEngine(OcrEngine):
    def extract(self, image_path: str | Path) -> list[OcrResult]:
        return []


def get_ocr_engine(engine_name: str = "easyocr", lang: str = "en"):
    """Create an OCR engine for English Mulkiya backside text."""
    engine_name = engine_name.lower().strip()

    if engine_name == "easyocr":
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError("EasyOCR is not installed. Install it with: pip install easyocr") from exc
        return easyocr.Reader(["en"], gpu=False)

    if engine_name == "rapidocr":
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError("RapidOCR is not installed. Install it with: pip install rapidocr-onnxruntime") from exc
        return RapidOCR()

    if engine_name != "paddle":
        raise ValueError(f"Unsupported OCR engine: {engine_name}")

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError("PaddleOCR is not installed. Install it with: pip install paddleocr paddlepaddle") from exc
    return PaddleOCR(use_angle_cls=True, lang=lang)


def extract_ocr_lines(image_path: str, ocr_engine: Any | None = None) -> list[dict]:
    """Run OCR and return simple line dictionaries with text, confidence, and bbox."""
    engine = ocr_engine or get_ocr_engine()
    if not _is_image_valid_for_ocr(image_path):
        return []

    if hasattr(engine, "readtext"):
        try:
            raw = engine.readtext(str(image_path))
        except Exception as exc:
            print(f"WARNING: OCR failed for {image_path}: {exc}")
            return []
        engine_label = "EasyOCR"
    elif hasattr(engine, "ocr"):
        try:
            raw = engine.ocr(str(image_path))
        except Exception as exc:
            print(f"WARNING: OCR failed for {image_path}: {exc}")
            return []
        engine_label = "PaddleOCR"
    else:
        try:
            raw = engine(str(image_path))
        except Exception as exc:
            print(f"WARNING: OCR failed for {image_path}: {exc}")
            return []
        engine_label = "RapidOCR"

    lines: list[dict] = []
    try:
        iterator = _iter_rapidocr_items(raw) if engine_label == "RapidOCR" else _iter_ocr_items(raw)
        for item in iterator:
            parsed = _parse_rapidocr_item(item) if engine_label == "RapidOCR" else _parse_ocr_item(item)
            if parsed is not None:
                lines.append(parsed)
    except Exception as exc:
        raise RuntimeError(f"Could not parse {engine_label} output. Raw output preview: {_preview_raw_ocr(raw)}") from exc
    return lines


def extract_chassis_candidates_from_text(text: str) -> list[str]:
    """Extract cleaned 17-character chassis candidates from OCR text."""
    if not text:
        return []

    normalized = str(text).upper()
    candidates: list[str] = []

    for chunk in re.findall(r"[A-Z0-9]{17,}", clean_chassis_no(normalized)):
        _add_exact_length_windows(chunk, candidates)

    tokens = re.findall(r"[A-Z0-9]+", normalized)
    for start in range(len(tokens)):
        combined = ""
        for token in tokens[start:]:
            combined += clean_chassis_no(token)
            if len(combined) > 20:
                break
            if len(combined) == 17:
                candidates.append(combined)

    for match in re.finditer(r"[A-Z0-9][A-Z0-9\s\-_/.:]{10,28}[A-Z0-9]", normalized):
        cleaned = clean_chassis_no(match.group(0))
        if 12 <= len(cleaned) <= 20:
            _add_exact_length_windows(cleaned, candidates)

    compact = clean_chassis_no(normalized)
    if 12 <= len(compact) <= 20:
        _add_exact_length_windows(compact, candidates)

    return list(dict.fromkeys(candidates))


def extract_chassis_candidates_from_ocr_lines(ocr_lines: list[dict]) -> list[dict]:
    """Extract chassis candidates from OCR line dictionaries."""
    candidates_by_value: dict[str, dict] = {}

    for line in ocr_lines:
        source_text = str(line.get("text", ""))
        confidence = float(line.get("confidence", 0.0) or 0.0)
        _add_candidates_from_source(candidates_by_value, source_text, confidence)

    joined_with_spaces = " ".join(str(line.get("text", "")) for line in ocr_lines)
    joined_without_spaces = "".join(str(line.get("text", "")) for line in ocr_lines)
    joined_confidence = _average_confidence(ocr_lines)
    _add_candidates_from_source(candidates_by_value, joined_with_spaces, joined_confidence)
    _add_candidates_from_source(candidates_by_value, joined_without_spaces, joined_confidence)

    for source_text, confidence in _chassis_neighborhoods(ocr_lines):
        _add_candidates_from_source(candidates_by_value, source_text, confidence)

    return list(candidates_by_value.values())


def _add_exact_length_windows(value: str, candidates: list[str]) -> None:
    if len(value) == 17:
        candidates.append(value)
        return

    if len(value) > 17:
        for start in range(0, len(value) - 16):
            candidates.append(value[start : start + 17])


def _add_candidates_from_source(candidates_by_value: dict[str, dict], source_text: str, confidence: float) -> None:
    for candidate in extract_chassis_candidates_from_text(source_text):
        validation = validate_chassis_no(candidate)
        candidate_row = {
            "candidate": candidate,
            "source_text": source_text,
            "ocr_confidence": confidence,
            "format_valid": bool(validation["format_valid"]),
        }
        existing = candidates_by_value.get(candidate)
        if existing is None or confidence > existing["ocr_confidence"]:
            candidates_by_value[candidate] = candidate_row


def _average_confidence(ocr_lines: list[dict]) -> float:
    confidences = [float(line.get("confidence", 0.0) or 0.0) for line in ocr_lines]
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


def _chassis_neighborhoods(ocr_lines: list[dict]) -> list[tuple[str, float]]:
    neighborhoods: list[tuple[str, float]] = []
    for index, line in enumerate(ocr_lines):
        text = str(line.get("text", ""))
        if not _looks_like_chassis_label(text):
            continue

        nearby_lines = ocr_lines[index : index + 4]
        nearby_text = " ".join(str(item.get("text", "")) for item in nearby_lines)
        nearby_confidences = [float(item.get("confidence", 0.0) or 0.0) for item in nearby_lines]
        confidence = max(nearby_confidences) if nearby_confidences else 0.0
        neighborhoods.append((nearby_text, confidence))
        neighborhoods.append((clean_chassis_no(nearby_text), confidence))

    return neighborhoods


def _looks_like_chassis_label(text: str) -> bool:
    cleaned = clean_chassis_no(text)
    return any(label in cleaned for label in ("CHASSISNO", "CHASSIS", "CHASIS"))


def _iter_ocr_items(raw: Any):
    if raw is None:
        return

    if isinstance(raw, dict):
        texts = raw.get("rec_texts") or raw.get("texts") or []
        scores = raw.get("rec_scores") or raw.get("scores") or []
        boxes = raw.get("rec_boxes") or raw.get("dt_polys") or raw.get("boxes") or []
        for index, text in enumerate(texts):
            yield {
                "text": text,
                "confidence": scores[index] if index < len(scores) else 0.0,
                "bbox": boxes[index] if index < len(boxes) else None,
            }
        return

    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"Unsupported OCR output type: {type(raw).__name__}")

    for item in raw:
        if isinstance(item, dict):
            if any(key in item for key in ("rec_texts", "texts")):
                yield from _iter_ocr_items(item)
            else:
                yield item
            continue

        if _looks_like_ocr_item(item):
            yield item
            continue

        if isinstance(item, (list, tuple)):
            for nested in item:
                if _looks_like_ocr_item(nested) or isinstance(nested, dict):
                    yield nested


def _parse_ocr_item(item: Any) -> dict | None:
    if isinstance(item, dict):
        text = item.get("text") or item.get("rec_text") or item.get("label")
        confidence = item.get("confidence", item.get("score", item.get("rec_score", 0.0)))
        bbox = item.get("bbox", item.get("box", item.get("points", item.get("dt_poly"))))
        if text is None:
            return None
        return {"text": str(text), "confidence": float(confidence or 0.0), "bbox": bbox}

    if not _looks_like_ocr_item(item):
        return None

    bbox = item[0]
    text_info = item[1]
    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
        text = text_info[0]
        confidence = text_info[1]
    else:
        text = text_info
        confidence = item[2] if len(item) >= 3 else 0.0

    return {"text": str(text), "confidence": float(confidence or 0.0), "bbox": bbox}


def _looks_like_ocr_item(item: Any) -> bool:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return False

    text_info = item[1]
    if isinstance(text_info, str):
        return True

    if isinstance(text_info, (list, tuple)) and text_info and isinstance(text_info[0], str):
        return True

    return False


def _preview_raw_ocr(raw: Any) -> str:
    text = repr(raw)
    if len(text) > 1000:
        return text[:1000] + "... <truncated>"
    return text


def _iter_rapidocr_items(raw: Any):
    if raw is None:
        return

    result = raw[0] if isinstance(raw, tuple) and raw else raw
    if result is None:
        return

    if not isinstance(result, (list, tuple)):
        raise TypeError(f"Unsupported RapidOCR output type: {type(result).__name__}")

    for item in result:
        yield item


def _parse_rapidocr_item(item: Any) -> dict | None:
    if not isinstance(item, (list, tuple)) or len(item) < 3:
        return None

    bbox = item[0]
    text = item[1]
    confidence = item[2]
    return {
        "text": str(text),
        "confidence": float(confidence or 0.0),
        "bbox": bbox,
    }


def _is_image_valid_for_ocr(image_path: str) -> bool:
    try:
        import cv2
    except ImportError:
        return True

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"WARNING: OCR failed for {image_path}: image could not be read")
        return False

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        print(f"WARNING: OCR failed for {image_path}: image has empty dimensions")
        return False

    return True
