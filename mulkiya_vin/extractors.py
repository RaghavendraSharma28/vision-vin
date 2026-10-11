from __future__ import annotations

from pathlib import Path

from .ocr import extract_chassis_candidates_from_ocr_lines, extract_ocr_lines, get_ocr_engine
from .vlm import (
    QWEN_MULKIYA_CROP_PROMPT,
    QWEN_MULKIYA_RETRY_PROMPT,
    extract_chassis_candidate_from_qwen_result,
    extract_with_qwen,
    load_qwen_model,
)
from .vin import is_valid_chassis_format


class VisionExtractor:
    """Small abstraction for image-to-candidate extractors."""

    def extract_candidates(self, image_path: str | Path, **kwargs) -> dict:
        raise NotImplementedError


class RapidOCRExtractor(VisionExtractor):
    """Run a local OCR engine and normalize chassis candidates."""

    def __init__(self, engine_name: str = "rapidocr") -> None:
        self.engine_name = engine_name
        self.engine = get_ocr_engine(engine_name=engine_name)

    def extract_candidates(self, image_path: str | Path, **kwargs) -> dict:
        ocr_lines = extract_ocr_lines(str(image_path), self.engine)
        candidates = extract_chassis_candidates_from_ocr_lines(ocr_lines)
        return {
            "engine_name": self.engine_name,
            "ocr_lines": ocr_lines,
            "candidates": candidates,
            "status": "success" if ocr_lines else "failed",
        }


class QwenExtractor(VisionExtractor):
    """Wrap Qwen loading and per-image extraction behind one interface."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is None or self._processor is None:
            self._model, self._processor = load_qwen_model(
                model_name=self.model_name,
                device=self.device,
            )

    def extract_candidates(
        self,
        image_path: str | Path,
        mode: str = "full",
        prompt: str | None = None,
        **kwargs,
    ) -> dict:
        self._ensure_loaded()

        if prompt is None:
            if mode == "crop":
                prompt = QWEN_MULKIYA_CROP_PROMPT
            elif mode == "retry":
                prompt = QWEN_MULKIYA_RETRY_PROMPT

        result = extract_with_qwen(
            image_path=str(image_path),
            model=self._model,
            processor=self._processor,
            prompt=prompt,
        )
        cleaned_candidate = extract_chassis_candidate_from_qwen_result(result)
        candidates = []
        if cleaned_candidate:
            candidates.append(
                {
                    "candidate": cleaned_candidate,
                    "source_text": str(result.get("chassis_no") or ""),
                    "ocr_confidence": 1.0,
                    "format_valid": is_valid_chassis_format(cleaned_candidate),
                    "source": f"qwen_{mode}",
                }
            )

        return {
            "mode": mode,
            "result": result,
            "candidates": candidates,
        }


class FutureFineTunedExtractor(VisionExtractor):
    """Placeholder for a future fine-tuned fallback model."""

    def extract_candidates(self, image_path: str | Path, **kwargs) -> dict:
        raise NotImplementedError("Future fine-tuned extractor is not implemented yet.")
