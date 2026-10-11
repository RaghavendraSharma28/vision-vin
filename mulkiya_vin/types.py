from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float = 0.0


@dataclass(frozen=True)
class VehicleMetadata:
    vin_number: str | None = None
    make: str | None = None
    vehicle_type: str | None = None
    year: str | int | None = None
    origin: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class VinValidationResult:
    vin: str
    is_valid: bool
    score: float
    reasons: tuple[str, ...] = ()
    manufacturer: str | None = None
    model_year: int | None = None


@dataclass(frozen=True)
class VinCandidate:
    vin: str
    source: str
    confidence: float
    validation: VinValidationResult


@dataclass(frozen=True)
class PipelineResult:
    request_id: str
    image_name: str
    ocr_results: list[OcrResult]
    vlm_output: VehicleMetadata
    candidates: list[VinCandidate]
    final_vin: str | None
    lookup_result: dict[str, Any] | None
    overall_confidence: float
    failure_reason: str | None = None
    human_review_required: bool = False
    audit: dict[str, Any] = field(default_factory=dict)
