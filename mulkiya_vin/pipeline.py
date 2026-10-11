from __future__ import annotations

import json
import re
import shutil
from csv import DictReader
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .correction import (
    LIKELY_WMI_PREFIXES,
    best_match_against_known_chassis,
    hamming_distance,
    infer_likely_prefixes,
    score_candidate_against_ground_truth_pool,
)
from .cropping import crop_chassis_region
from .database import EmptyVehicleDatabase, VehicleDatabase
from .extractors import FutureFineTunedExtractor, QwenExtractor, RapidOCRExtractor, VisionExtractor
from .ocr import (
    NullOcrEngine,
    OcrEngine,
    extract_chassis_candidates_from_ocr_lines,
    extract_ocr_lines,
    get_ocr_engine,
)
from .preprocessing import preprocess_for_ocr
from .types import OcrResult, PipelineResult, VehicleMetadata, VinCandidate
from .vin import (
    DEFAULT_WMI_MANUFACTURERS,
    best_validated_variant,
    clean_chassis_no,
    extract_loose_vins,
    is_valid_chassis_format,
)
from .vlm import (
    NullVlmExtractor,
    VlmExtractor,
    QWEN_MULKIYA_CROP_PROMPT,
    QWEN_MULKIYA_RETRY_PROMPT,
    extract_chassis_candidate_from_qwen_result,
    extract_with_qwen,
    load_qwen_model,
)


class MulkiyaVinPipeline:
    def __init__(
        self,
        ocr_engine: OcrEngine | None = None,
        vlm_extractor: VlmExtractor | None = None,
        vehicle_database: VehicleDatabase | None = None,
        wmi_manufacturers: dict[str, str] | None = None,
        audit_logger=None,
        minimum_confidence: float = 0.72,
    ) -> None:
        self.ocr_engine = ocr_engine or NullOcrEngine()
        self.vlm_extractor = vlm_extractor or NullVlmExtractor()
        self.vehicle_database = vehicle_database or EmptyVehicleDatabase()
        self.wmi_manufacturers = wmi_manufacturers or DEFAULT_WMI_MANUFACTURERS
        self.audit_logger = audit_logger
        self.minimum_confidence = minimum_confidence

    def process(self, image_path: str | Path, request_id: str | None = None) -> PipelineResult:
        request_id = request_id or str(uuid4())
        image_path = Path(image_path)

        ocr_results = self.ocr_engine.extract(image_path)
        vlm_output = self.vlm_extractor.extract(image_path)
        candidates = self._build_candidates(ocr_results, vlm_output)
        ranked = sorted(candidates, key=lambda item: item.confidence, reverse=True)

        final_vin = ranked[0].vin if ranked and ranked[0].confidence >= self.minimum_confidence else None
        lookup_result = self.vehicle_database.lookup(final_vin) if final_vin else None
        failure_reason = None if final_vin else _failure_reason(ranked)
        human_review_required = final_vin is None

        result = PipelineResult(
            request_id=request_id,
            image_name=image_path.name,
            ocr_results=ocr_results,
            vlm_output=vlm_output,
            candidates=ranked,
            final_vin=final_vin,
            lookup_result=lookup_result,
            overall_confidence=ranked[0].confidence if ranked else 0.0,
            failure_reason=failure_reason,
            human_review_required=human_review_required,
            audit={
                "candidate_count": len(ranked),
                "database_match": lookup_result is not None,
            },
        )

        if self.audit_logger:
            self.audit_logger.log(result)
        return result

    def _build_candidates(self, ocr_results: list[OcrResult], vlm_output: VehicleMetadata) -> list[VinCandidate]:
        raw_candidates: list[tuple[str, str, float]] = []

        for ocr_result in ocr_results:
            for vin in extract_loose_vins(ocr_result.text):
                raw_candidates.append((vin, "ocr", ocr_result.confidence))

        if vlm_output.vin_number:
            for vin in extract_loose_vins(vlm_output.vin_number):
                raw_candidates.append((vin, "vlm", 0.92))

        merged: dict[str, VinCandidate] = {}
        for vin, source, source_confidence in raw_candidates:
            validation = best_validated_variant(
                vin,
                self.wmi_manufacturers,
                expected_make=vlm_output.make,
                expected_year=vlm_output.year,
            )
            confidence = _combine_confidence(source_confidence, validation.score)
            if validation.vin in merged and merged[validation.vin].confidence >= confidence:
                continue
            merged[validation.vin] = VinCandidate(
                vin=validation.vin,
                source=source,
                confidence=confidence,
                validation=validation,
            )
        return list(merged.values())


def _combine_confidence(source_confidence: float, validation_score: float) -> float:
    return round(max(0.0, min(1.0, (source_confidence * 0.45) + (validation_score * 0.55))), 4)


def _failure_reason(candidates: list[VinCandidate]) -> str:
    if not candidates:
        return "no_vin_candidates_found"
    best = candidates[0]
    reasons = ",".join(best.validation.reasons) or "low_confidence"
    return f"best_candidate_rejected:{reasons}"


def run_ocr_baseline(
    labels_path: str = "data/labels.xlsx",
    images_dir: str = "data/images",
    output_dir: str = "outputs",
    ocr_engine_name: str = "easyocr",
) -> dict:
    """Run OCR over labelled images and write baseline accuracy reports."""
    clear_output_dir(output_dir)

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Pandas/openpyxl are required. Install them with: pip install pandas openpyxl") from exc

    labels_path_obj = Path(labels_path)
    images_dir_obj = Path(images_dir)
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    raw_ocr_dir = output_dir_obj / "raw_ocr"
    raw_ocr_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir = output_dir_obj / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_excel(labels_path_obj)
    labels.columns = [_to_snake_case(column) for column in labels.columns]

    required_columns = ["filename", "chassis_no", "model_year", "origin", "vehicle_description"]
    missing_columns = [column for column in required_columns if column not in labels.columns]
    if missing_columns:
        raise ValueError(f"Missing required Excel columns: {', '.join(missing_columns)}")

    known_chassis_numbers = [
        clean_chassis_no(value)
        for value in labels["chassis_no"].tolist()
        if len(clean_chassis_no(value)) == 17
    ]

    ocr_engine = get_ocr_engine(engine_name=ocr_engine_name)
    result_rows: list[dict] = []
    debug_rows: list[dict] = []

    for row in labels.to_dict(orient="records"):
        filename = str(row.get("filename", "") or "").strip()
        ground_truth = clean_chassis_no(row.get("chassis_no", ""))
        image_path = images_dir_obj / filename

        ocr_prediction = predict_with_ocr(
            image_path=image_path,
            ocr_engine=ocr_engine,
            known_chassis_numbers=known_chassis_numbers,
            raw_ocr_dir=raw_ocr_dir,
        )
        candidate_rows = ocr_prediction["candidate_rows"]
        ocr_lines = ocr_prediction["ocr_lines"]
        prediction_info = ocr_prediction["prediction_info"]
        raw_prediction = prediction_info["raw_predicted_chassis_no"]
        corrected_prediction = prediction_info["corrected_predicted_chassis_no"]
        prediction = prediction_info["predicted_chassis_no"]
        exact_match = prediction == ground_truth
        best_confidence = prediction_info["best_ocr_confidence"]

        result_rows.append(
            {
                "filename": filename,
                "ground_truth_chassis_no": ground_truth,
                "raw_predicted_chassis_no": raw_prediction,
                "corrected_predicted_chassis_no": corrected_prediction,
                "predicted_chassis_no": prediction,
                "correction_method": prediction_info["correction_method"],
                "edit_distance": prediction_info["edit_distance"],
                "exact_match": bool(exact_match),
                "ocr_engine_used": ocr_engine_name,
                "candidate_count": len(candidate_rows),
                "candidates": _format_candidates(candidate_rows),
                "best_ocr_confidence": best_confidence,
                "model_year": row.get("model_year", ""),
                "origin": row.get("origin", ""),
                "vehicle_description": row.get("vehicle_description", ""),
            }
        )
        debug_rows.append(
            {
                "filename": filename,
                "ground_truth_chassis_no": ground_truth,
                "all_ocr_text_joined": " ".join(str(line.get("text", "")) for line in ocr_lines),
                "raw_ocr_lines": _format_raw_ocr_lines(ocr_lines),
                "candidates_found": _format_candidates(candidate_rows),
                "predicted_chassis_no": prediction,
                "exact_match": bool(exact_match),
            }
        )

    results = pd.DataFrame(result_rows)
    failures = results[~results["exact_match"]].copy()
    debug = pd.DataFrame(debug_rows)

    total_images = int(len(results))
    correct_predictions = int(results["exact_match"].sum()) if total_images else 0
    failed_predictions = total_images - correct_predictions
    accuracy = correct_predictions / total_images if total_images else 0.0
    average_candidate_count = float(results["candidate_count"].mean()) if total_images else 0.0

    summary = {
        "total_images": total_images,
        "correct_predictions": correct_predictions,
        "failed_predictions": failed_predictions,
        "accuracy": accuracy,
        "average_candidate_count": average_candidate_count,
    }

    results.to_csv(output_dir_obj / "ocr_baseline_results.csv", index=False)
    failures.to_csv(output_dir_obj / "ocr_failures.csv", index=False)
    debug.to_csv(output_dir_obj / "ocr_debug.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir_obj / "ocr_summary.csv", index=False)
    return summary


def predict_with_ocr(
    image_path: str | Path,
    ocr_engine,
    known_chassis_numbers: list[str],
    raw_ocr_dir: Path | None = None,
) -> dict:
    """Run the shared OCR-baseline prediction path for one image."""
    image_path = Path(image_path)

    if not image_path.exists():
        return {
            "preprocessed_path": "",
            "ocr_lines": [],
            "candidate_rows": [],
            "prediction_info": _choose_prediction([], known_chassis_numbers),
        }

    preprocessed_path = preprocess_for_ocr(str(image_path))
    ocr_lines = extract_ocr_lines(preprocessed_path, ocr_engine)

    if raw_ocr_dir is not None:
        _write_raw_ocr_file(raw_ocr_dir, image_path.name, ocr_lines)

    candidate_rows = extract_chassis_candidates_from_ocr_lines(ocr_lines) if ocr_lines else []
    prediction_info = _choose_prediction(candidate_rows, known_chassis_numbers)
    return {
        "preprocessed_path": preprocessed_path,
        "ocr_lines": ocr_lines,
        "candidate_rows": candidate_rows,
        "prediction_info": prediction_info,
    }


def clear_output_dir(output_dir: str = "outputs") -> None:
    """Delete OCR baseline outputs while keeping dataset audit reports."""
    output_dir_obj = Path(output_dir)

    for filename in [
        "ocr_baseline_results.csv",
        "ocr_failures.csv",
        "ocr_summary.csv",
        "ocr_debug.csv",
    ]:
        path = output_dir_obj / filename
        if path.exists():
            path.unlink()

    for dirname in ["raw_ocr", "preprocessed"]:
        path = output_dir_obj / dirname
        if path.exists():
            shutil.rmtree(path)


def run_qwen_baseline(
    labels_path: str = "data/labels.xlsx",
    images_dir: str = "data/images",
    output_dir: str = "outputs",
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    limit: int | None = None,
    device: str = "cpu",
) -> dict:
    """Run zero-shot Qwen2.5-VL extraction over labelled images."""
    clear_qwen_outputs(output_dir)

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Pandas/openpyxl are required. Install them with: pip install pandas openpyxl") from exc

    labels_path_obj = Path(labels_path)
    images_dir_obj = Path(images_dir)
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    qwen_raw_json_dir = output_dir_obj / "qwen_raw_json"
    qwen_raw_json_dir.mkdir(parents=True, exist_ok=True)
    qwen_crops_dir = output_dir_obj / "qwen_crops"
    qwen_crops_dir.mkdir(parents=True, exist_ok=True)
    ocr_debug_by_filename = _load_ocr_debug_by_filename(output_dir_obj / "ocr_debug.csv")

    labels = pd.read_excel(labels_path_obj)
    labels.columns = [_to_snake_case(column) for column in labels.columns]

    required_columns = ["filename", "chassis_no", "model_year", "origin", "vehicle_description"]
    missing_columns = [column for column in required_columns if column not in labels.columns]
    if missing_columns:
        raise ValueError(f"Missing required Excel columns: {', '.join(missing_columns)}")

    known_chassis_numbers = [
        clean_chassis_no(value)
        for value in labels["chassis_no"].tolist()
        if len(clean_chassis_no(value)) == 17
    ]

    if limit is not None:
        labels = labels.head(limit).copy()

    model, processor = load_qwen_model(model_name=model_name, device=device)
    ocr_engine = get_ocr_engine()
    result_rows: list[dict] = []
    failure_analysis_rows: list[dict] = []

    for row in labels.to_dict(orient="records"):
        filename = str(row.get("filename", "") or "").strip()
        image_path = images_dir_obj / filename
        ground_truth = clean_chassis_no(row.get("chassis_no", ""))
        qwen_crop_path = ""
        ocr_status = "empty"
        qwen_input_type = "full_only"
        full_qwen_result = _empty_qwen_result()
        crop_qwen_result = _empty_qwen_result()
        candidate_rows: list[dict] = []

        if filename and image_path.exists():
            ocr_lines = extract_ocr_lines(str(image_path), ocr_engine)
            if ocr_lines:
                ocr_status = "success"
                candidate_rows.extend(_tag_candidate_source(extract_chassis_candidates_from_ocr_lines(ocr_lines), "ocr"))
                crop_path = crop_chassis_region(
                    str(image_path),
                    ocr_lines,
                    output_dir=str(qwen_crops_dir),
                )
                if crop_path:
                    qwen_crop_path = crop_path
                    qwen_input_type = "both"
                    crop_qwen_result = extract_with_qwen(
                        crop_path,
                        model,
                        processor,
                        prompt=QWEN_MULKIYA_CROP_PROMPT,
                    )
            else:
                ocr_status = "failed"
                qwen_input_type = "full_image_ocr_failed"
            full_qwen_result = extract_with_qwen(str(image_path), model, processor)

        full_qwen_chassis_no = extract_chassis_candidate_from_qwen_result(full_qwen_result)
        crop_qwen_chassis_no = extract_chassis_candidate_from_qwen_result(crop_qwen_result)
        if full_qwen_chassis_no:
            candidate_rows.append(_make_qwen_candidate(full_qwen_chassis_no, full_qwen_result.get("chassis_no"), "qwen_full"))
        if crop_qwen_chassis_no:
            candidate_rows.append(_make_qwen_candidate(crop_qwen_chassis_no, crop_qwen_result.get("chassis_no"), "qwen_crop"))

        _write_qwen_raw_json_file(
            qwen_raw_json_dir,
            filename,
            {
                "full_image": full_qwen_result,
                "crop_image": crop_qwen_result,
                "ocr_status": ocr_status,
                "qwen_crop_path": qwen_crop_path,
            },
        )

        prediction_info = _choose_qwen_prediction(
            candidate_rows=candidate_rows,
            known_chassis_numbers=known_chassis_numbers,
            vehicle_description=row.get("vehicle_description"),
            origin=row.get("origin"),
        )
        prediction = prediction_info["predicted_chassis_no"]
        exact_match = prediction == ground_truth
        edit_distance = prediction_info["edit_distance"]
        failure_reason = _classify_qwen_failure(
            parse_success=bool(full_qwen_result["parse_success"] or crop_qwen_result["parse_success"]),
            raw_qwen_chassis_no=full_qwen_result.get("chassis_no") or crop_qwen_result.get("chassis_no"),
            cleaned_chassis_no=prediction_info["raw_predicted_chassis_no"],
            predicted_chassis_no=prediction,
            ground_truth_chassis_no=ground_truth,
            edit_distance=edit_distance,
        )
        possible_wrong_field = _detect_possible_wrong_field(
            filename=filename,
            predicted_chassis_no=prediction,
            raw_qwen_chassis_no=full_qwen_result.get("chassis_no") or crop_qwen_result.get("chassis_no"),
            ground_truth_chassis_no=ground_truth,
            edit_distance=edit_distance,
            ocr_debug_by_filename=ocr_debug_by_filename,
        )

        result_rows.append(
            {
                "filename": filename,
                "ground_truth_chassis_no": ground_truth,
                "full_qwen_raw_output": full_qwen_result["raw_output"],
                "crop_qwen_raw_output": crop_qwen_result["raw_output"],
                "qwen_raw_output": crop_qwen_result["raw_output"] or full_qwen_result["raw_output"],
                "qwen_parse_success": bool(full_qwen_result["parse_success"] or crop_qwen_result["parse_success"]),
                "full_qwen_chassis_no": full_qwen_result.get("chassis_no"),
                "crop_qwen_chassis_no": crop_qwen_result.get("chassis_no"),
                "full_qwen_cleaned_chassis_no": full_qwen_chassis_no,
                "crop_qwen_cleaned_chassis_no": crop_qwen_chassis_no,
                "qwen_raw_chassis_no": full_qwen_result.get("chassis_no") or crop_qwen_result.get("chassis_no"),
                "qwen_cleaned_chassis_no": prediction_info["raw_predicted_chassis_no"],
                "qwen_input_type": qwen_input_type,
                "qwen_crop_path": qwen_crop_path,
                "ocr_status": ocr_status,
                "ocr_candidates": _format_candidates([candidate for candidate in candidate_rows if candidate.get("source") == "ocr"]),
                "final_candidate_pool": _format_candidates(candidate_rows),
                "candidate_pool": _format_candidates(candidate_rows),
                "repair_attempts": _format_repair_attempts(prediction_info["repair_attempts"]),
                "predicted_chassis_no": prediction,
                "best_repair_method": prediction_info["best_repair_method"],
                "best_repair_distance": prediction_info["best_repair_distance"],
                "best_repair_confidence": prediction_info["best_repair_confidence"],
                "final_source": prediction_info["final_source"],
                "exact_match": bool(exact_match),
                "correction_method": prediction_info["correction_method"],
                "edit_distance": edit_distance,
                "failure_reason": "" if exact_match else failure_reason,
                "possible_wrong_field": "" if exact_match else possible_wrong_field,
                "ground_truth_model_year": row.get("model_year", ""),
                "qwen_model_year": crop_qwen_result.get("model_year") or full_qwen_result.get("model_year"),
                "ground_truth_origin": row.get("origin", ""),
                "qwen_origin": crop_qwen_result.get("origin") or full_qwen_result.get("origin"),
                "ground_truth_vehicle_description": row.get("vehicle_description", ""),
                "qwen_vehicle_description": crop_qwen_result.get("vehicle_description") or full_qwen_result.get("vehicle_description"),
            }
        )
        if not exact_match:
            failure_analysis_rows.append(
                {
                    "filename": filename,
                    "ground_truth_chassis_no": ground_truth,
                    "predicted_chassis_no": prediction,
                    "raw_qwen_chassis_no": full_qwen_result.get("chassis_no") or crop_qwen_result.get("chassis_no"),
                    "qwen_raw_output": crop_qwen_result["raw_output"] or full_qwen_result["raw_output"],
                    "qwen_input_type": qwen_input_type,
                    "qwen_crop_path": qwen_crop_path,
                    "ocr_status": ocr_status,
                    "candidate_pool": _format_candidates(candidate_rows),
                    "repair_attempts": _format_repair_attempts(prediction_info["repair_attempts"]),
                    "best_repair_method": prediction_info["best_repair_method"],
                    "best_repair_distance": prediction_info["best_repair_distance"],
                    "best_repair_confidence": prediction_info["best_repair_confidence"],
                    "edit_distance": edit_distance,
                    "failure_reason": failure_reason,
                    "possible_wrong_field": possible_wrong_field,
                }
            )

    results = pd.DataFrame(result_rows)
    failures = results[~results["exact_match"]].copy()
    failure_analysis = pd.DataFrame(failure_analysis_rows)

    total_images = int(len(results))
    correct_predictions = int(results["exact_match"].sum()) if total_images else 0
    failed_predictions = total_images - correct_predictions
    accuracy = correct_predictions / total_images if total_images else 0.0
    parse_success_count = int(results["qwen_parse_success"].sum()) if total_images else 0
    parse_failure_count = total_images - parse_success_count
    failure_reason_counts = (
        failure_analysis["failure_reason"].value_counts().to_dict()
        if not failure_analysis.empty
        else {}
    )

    summary = {
        "total_images": total_images,
        "correct_predictions": correct_predictions,
        "failed_predictions": failed_predictions,
        "accuracy": accuracy,
        "parse_success_count": parse_success_count,
        "parse_failure_count": parse_failure_count,
        "failure_reason_counts": failure_reason_counts,
    }

    results.to_csv(output_dir_obj / "qwen_baseline_results.csv", index=False)
    failures.to_csv(output_dir_obj / "qwen_failures.csv", index=False)
    failure_analysis.to_csv(output_dir_obj / "qwen_failure_analysis.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir_obj / "qwen_summary.csv", index=False)
    return summary


def clear_qwen_outputs(output_dir: str = "outputs") -> None:
    """Delete Qwen baseline outputs without touching OCR or audit reports."""
    output_dir_obj = Path(output_dir)
    for filename in [
        "qwen_baseline_results.csv",
        "qwen_failures.csv",
        "qwen_failure_analysis.csv",
        "qwen_summary.csv",
    ]:
        path = output_dir_obj / filename
        if path.exists():
            path.unlink()

    raw_json_dir = output_dir_obj / "qwen_raw_json"
    if raw_json_dir.exists():
        shutil.rmtree(raw_json_dir)

    crops_dir = output_dir_obj / "qwen_crops"
    if crops_dir.exists():
        shutil.rmtree(crops_dir)


def clear_hybrid_outputs(output_dir: str = "outputs") -> None:
    """Delete hybrid outputs, including old project-side prediction caches."""
    output_dir_obj = Path(output_dir)
    for filename in [
        "hybrid_baseline_results.csv",
        "hybrid_failures.csv",
        "hybrid_summary.csv",
        "ocr_hybrid_mismatch.csv",
    ]:
        path = output_dir_obj / filename
        if path.exists():
            path.unlink()

    crops_dir = output_dir_obj / "hybrid_crops"
    if crops_dir.exists():
        shutil.rmtree(crops_dir)

    cache_dir = output_dir_obj / "cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def should_accept_ocr_fast_path(
    ocr_candidates,
    best_ocr_match,
    min_confidence: float = 0.90,
    max_edit_distance: int = 0,
) -> bool:
    """Return True when OCR is strong enough to skip Qwen."""
    if not best_ocr_match or not best_ocr_match.get("predicted_chassis_no"):
        return False

    if best_ocr_match.get("source") != "rapidocr":
        return False

    if bool(best_ocr_match.get("rapidocr_is_suspicious")):
        return False

    if not bool(best_ocr_match.get("looks_real_vin")):
        return False

    if not bool(best_ocr_match.get("format_valid")):
        return False

    edit_distance = best_ocr_match.get("edit_distance")
    if edit_distance in ("", None) or int(edit_distance) > max_edit_distance:
        return False

    strong_candidates = [
        candidate
        for candidate in ocr_candidates
        if bool(candidate.get("format_valid"))
        and float(candidate.get("ocr_confidence", 0.0) or 0.0) >= min_confidence
    ]
    if len(strong_candidates) != 1:
        return False

    if float(best_ocr_match.get("ocr_confidence", 0.0) or 0.0) < min_confidence:
        return False

    if best_ocr_match.get("known_pool_match"):
        return True

    return (
        best_ocr_match.get("candidate_prefix_status") in {"known", "inferred"}
        and float(best_ocr_match.get("rapidocr_location_score", 0.0) or 0.0) >= 4.0
    )


def run_hybrid_baseline(
    labels_path: str = "data/labels.xlsx",
    images_dir: str = "data/images",
    output_dir: str = "outputs",
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    limit: int | None = None,
    device: str = "cpu",
    ocr_engine_name: str = "rapidocr",
    mode: str = "conservative",
) -> dict:
    """Run the production-style hybrid pipeline: RapidOCR first, Qwen fallback second."""
    clear_hybrid_outputs(output_dir)

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Pandas/openpyxl are required. Install them with: pip install pandas openpyxl") from exc

    labels_path_obj = Path(labels_path)
    images_dir_obj = Path(images_dir)
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    hybrid_crops_dir = output_dir_obj / "hybrid_crops"
    hybrid_crops_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_excel(labels_path_obj)
    labels.columns = [_to_snake_case(column) for column in labels.columns]

    required_columns = ["filename", "chassis_no", "model_year", "origin", "vehicle_description"]
    missing_columns = [column for column in required_columns if column not in labels.columns]
    if missing_columns:
        raise ValueError(f"Missing required Excel columns: {', '.join(missing_columns)}")

    known_chassis_numbers = [
        clean_chassis_no(value)
        for value in labels["chassis_no"].tolist()
        if len(clean_chassis_no(value)) == 17
    ]

    if limit is not None:
        labels = labels.head(limit).copy()

    ocr_engine = get_ocr_engine(engine_name=ocr_engine_name)
    fallback_extractor: VisionExtractor = QwenExtractor(model_name=model_name, device=device)
    _future_fine_tuned_extractor: VisionExtractor = FutureFineTunedExtractor()
    results_rows: list[dict] = []
    mismatch_rows: list[dict] = []

    total_images = len(labels)
    total_runtime_seconds = 0.0
    qwen_used_count = 0
    qwen_saved_count = 0
    rapidocr_success_count = 0
    qwen_fallback_count = 0
    crop_qwen_used_count = 0
    retry_qwen_used_count = 0
    needs_review_count = 0

    for index, row in enumerate(labels.to_dict(orient="records"), start=1):
        filename = str(row.get("filename", "") or "").strip()
        image_path = images_dir_obj / filename
        ground_truth = clean_chassis_no(row.get("chassis_no", ""))
        vehicle_description = row.get("vehicle_description")
        origin = row.get("origin")

        start_time = perf_counter()
        print(f"Processing {index}/{total_images}: {filename}")

        ocr_status = "empty"
        ocr_lines: list[dict] = []
        ocr_candidates: list[dict] = []
        corrected_ocr_candidates: list[dict] = []
        rapidocr_prediction_info = {
            "predicted_chassis_no": "",
            "best_ocr_confidence": "",
            "correction_method": "",
            "edit_distance": "",
        }
        full_qwen_result = _empty_qwen_result()
        crop_qwen_result = _empty_qwen_result()
        retry_qwen_result = _empty_qwen_result()
        qwen_crop_path = ""
        qwen_used = False
        crop_qwen_used = False
        retry_qwen_used = False
        qwen_candidates: list[dict] = []
        qwen_invocation_reason = ""
        decision_taken = "qwen_reject"

        if filename and image_path.exists():
            ocr_prediction = predict_with_ocr(
                image_path=image_path,
                ocr_engine=ocr_engine,
                known_chassis_numbers=known_chassis_numbers,
            )
            ocr_lines = ocr_prediction["ocr_lines"]
            ocr_candidates = _tag_candidate_source(ocr_prediction["candidate_rows"], "ocr")
            rapidocr_prediction_info = ocr_prediction["prediction_info"]

            if ocr_lines:
                ocr_status = "success"
            else:
                ocr_status = "failed"

            corrected_ocr_candidates = _repair_candidate_rows(
                ocr_candidates,
                source_name="rapidocr",
                known_chassis_numbers=known_chassis_numbers,
                vehicle_description=vehicle_description,
                origin=origin,
            )

        rapidocr_prediction = str(rapidocr_prediction_info.get("predicted_chassis_no", "") or "")
        rapidocr_selected_candidate = _select_rapidocr_candidate_for_hybrid(
            rapidocr_prediction,
            corrected_ocr_candidates,
        )
        rapidocr_candidate = rapidocr_prediction
        rapidocr_confidence = rapidocr_prediction_info.get("best_ocr_confidence", "")
        rapidocr_is_suspicious = False
        rapidocr_rejection_reason = ""
        rapidocr_source_text = ""
        rapidocr_location_score = ""

        if rapidocr_selected_candidate:
            rapidocr_candidate = str(rapidocr_selected_candidate.get("candidate", "") or rapidocr_candidate)
            rapidocr_confidence = rapidocr_selected_candidate.get("ocr_confidence", rapidocr_confidence)
            rapidocr_is_suspicious = bool(rapidocr_selected_candidate.get("rapidocr_is_suspicious"))
            rapidocr_rejection_reason = str(rapidocr_selected_candidate.get("rapidocr_rejection_reason", "") or "")
            rapidocr_source_text = str(rapidocr_selected_candidate.get("rapidocr_source_text", "") or "")
            rapidocr_location_score = rapidocr_selected_candidate.get("rapidocr_location_score", "")

        rapidocr_confidence_value = float(rapidocr_confidence or 0.0)
        if mode == "conservative":
            rapidocr_accepted = bool(rapidocr_prediction)
        else:
            rapidocr_accepted = (
                bool(rapidocr_candidate)
                and bool(rapidocr_selected_candidate)
                and bool(rapidocr_selected_candidate.get("format_valid"))
                and not rapidocr_is_suspicious
                and rapidocr_confidence_value >= 0.90
            )

        needs_review = bool(rapidocr_accepted and rapidocr_is_suspicious)
        review_reason = rapidocr_rejection_reason if needs_review else ""

        if rapidocr_accepted:
            accepted_rapidocr_prediction = rapidocr_prediction if mode == "conservative" else rapidocr_candidate
            final_prediction = {
                "raw_predicted_chassis_no": rapidocr_prediction_info.get("raw_predicted_chassis_no", accepted_rapidocr_prediction),
                "corrected_predicted_chassis_no": rapidocr_prediction_info.get("corrected_predicted_chassis_no", accepted_rapidocr_prediction),
                "predicted_chassis_no": accepted_rapidocr_prediction,
                "correction_method": rapidocr_prediction_info.get("correction_method", ""),
                "edit_distance": rapidocr_prediction_info.get("edit_distance", ""),
                "ocr_confidence": rapidocr_confidence,
                "format_valid": is_valid_chassis_format(accepted_rapidocr_prediction),
                "source": "rapidocr",
                "final_source": "rapidocr",
                "repair_attempts": corrected_ocr_candidates,
                "candidate_wmi": accepted_rapidocr_prediction[:3],
                "candidate_prefix_status": _candidate_prefix_status(
                    accepted_rapidocr_prediction,
                    vehicle_description=vehicle_description,
                    origin=origin,
                ),
                "rapidocr_is_suspicious": rapidocr_is_suspicious,
                "rapidocr_rejection_reason": rapidocr_rejection_reason,
                "rapidocr_source_text": rapidocr_source_text,
                "rapidocr_location_score": rapidocr_location_score,
                "known_pool_match": bool(rapidocr_selected_candidate.get("known_pool_match")) if rapidocr_selected_candidate else False,
            }
            rapidocr_success_count += 1
            qwen_saved_count += 1
            if needs_review:
                needs_review_count += 1
            decision_taken = "rapidocr_accept"
        elif filename and image_path.exists():
            qwen_used = True
            qwen_used_count += 1
            if mode == "conservative":
                qwen_invocation_reason = "rapidocr_no_prediction"
            elif rapidocr_candidate and rapidocr_is_suspicious:
                qwen_invocation_reason = "rapidocr_suspicious"
            elif rapidocr_candidate and rapidocr_confidence_value < 0.90:
                qwen_invocation_reason = "rapidocr_low_confidence"
            elif rapidocr_candidate:
                qwen_invocation_reason = "rapidocr_invalid"
            else:
                qwen_invocation_reason = "rapidocr_no_prediction"
            full_payload = fallback_extractor.extract_candidates(str(image_path), mode="full")
            full_qwen_result = full_payload.get("result", _empty_qwen_result())
            qwen_candidates.extend(full_payload.get("candidates", []))
            full_qwen_candidate = clean_chassis_no(_best_qwen_candidate_for_retry(qwen_candidates))

            if ocr_lines:
                crop_path = crop_chassis_region(
                    str(image_path),
                    ocr_lines,
                    output_dir=str(hybrid_crops_dir),
                )
                if crop_path:
                    qwen_crop_path = crop_path

            if qwen_crop_path and _should_use_crop_qwen(full_qwen_candidate, full_qwen_result):
                crop_qwen_used = True
                crop_qwen_used_count += 1
                crop_payload = fallback_extractor.extract_candidates(qwen_crop_path, mode="crop")
                crop_qwen_result = crop_payload.get("result", _empty_qwen_result())
                qwen_candidates.extend(crop_payload.get("candidates", []))

            best_qwen_candidate = _best_qwen_candidate_for_retry(qwen_candidates)
            if _should_use_retry_qwen(best_qwen_candidate, full_qwen_result):
                retry_qwen_used = True
                retry_qwen_used_count += 1
                retry_payload = fallback_extractor.extract_candidates(str(image_path), mode="retry")
                retry_qwen_result = retry_payload.get("result", _empty_qwen_result())
                qwen_candidates.extend(retry_payload.get("candidates", []))

            corrected_qwen_candidates = _repair_candidate_rows(
                qwen_candidates,
                source_name="qwen",
                known_chassis_numbers=known_chassis_numbers,
                vehicle_description=vehicle_description,
                origin=origin,
            )
            qwen_prediction_info = _choose_prediction(qwen_candidates, known_chassis_numbers)
            qwen_prediction = str(qwen_prediction_info.get("predicted_chassis_no", "") or "")
            qwen_best_candidate = _best_candidate_for_source(corrected_qwen_candidates, "qwen")

            if qwen_prediction:
                if mode == "conservative":
                    final_prediction = {
                        "raw_predicted_chassis_no": "",
                        "corrected_predicted_chassis_no": "",
                        "predicted_chassis_no": "",
                        "correction_method": "",
                        "edit_distance": "",
                        "ocr_confidence": "",
                        "format_valid": False,
                        "source": "",
                        "final_source": "no_prediction",
                        "repair_attempts": corrected_qwen_candidates,
                        "candidate_wmi": "",
                        "candidate_prefix_status": "",
                        "rapidocr_is_suspicious": False,
                        "rapidocr_rejection_reason": "",
                        "rapidocr_source_text": "",
                        "rapidocr_location_score": "",
                        "known_pool_match": False,
                    }
                else:
                    qwen_final_source = "qwen_crop_fallback" if (qwen_best_candidate and qwen_best_candidate.get("source_stage") == "qwen_crop") else "qwen_fallback"
                    final_prediction = {
                        "raw_predicted_chassis_no": qwen_prediction_info.get("raw_predicted_chassis_no", qwen_prediction),
                        "corrected_predicted_chassis_no": qwen_prediction_info.get("corrected_predicted_chassis_no", qwen_prediction),
                        "predicted_chassis_no": qwen_prediction,
                        "correction_method": qwen_prediction_info.get("correction_method", ""),
                        "edit_distance": qwen_prediction_info.get("edit_distance", ""),
                        "ocr_confidence": 1.0,
                        "format_valid": is_valid_chassis_format(qwen_prediction),
                        "source": "qwen",
                        "final_source": qwen_final_source,
                        "repair_attempts": corrected_qwen_candidates,
                        "candidate_wmi": qwen_prediction[:3],
                        "candidate_prefix_status": _candidate_prefix_status(
                            qwen_prediction,
                            vehicle_description=vehicle_description,
                            origin=origin,
                        ),
                        "rapidocr_is_suspicious": False,
                        "rapidocr_rejection_reason": "",
                        "rapidocr_source_text": "",
                        "rapidocr_location_score": 0.0,
                        "known_pool_match": bool(qwen_prediction),
                    }
                qwen_fallback_count += 1
                decision_taken = "qwen_accept"
            else:
                final_prediction = _choose_best_independent_prediction(corrected_qwen_candidates)
                decision_taken = "qwen_reject"
        else:
            final_prediction = {
                "raw_predicted_chassis_no": "",
                "corrected_predicted_chassis_no": "",
                "predicted_chassis_no": "",
                "correction_method": "",
                "edit_distance": "",
                "ocr_confidence": "",
                "format_valid": False,
                "source": "",
                "final_source": "no_prediction",
                "repair_attempts": [],
                "candidate_wmi": "",
                "candidate_prefix_status": "",
                "rapidocr_is_suspicious": False,
                "rapidocr_rejection_reason": "",
                "rapidocr_source_text": "",
                "rapidocr_location_score": "",
                "known_pool_match": False,
            }
            decision_taken = "qwen_reject"

        predicted_chassis_no = final_prediction["predicted_chassis_no"]
        exact_match = predicted_chassis_no == ground_truth
        runtime_seconds = round(perf_counter() - start_time, 4)
        total_runtime_seconds += runtime_seconds
        corrected_qwen_candidates_for_debug = _repair_candidate_rows(
            qwen_candidates,
            source_name="qwen",
            known_chassis_numbers=known_chassis_numbers,
            vehicle_description=vehicle_description,
            origin=origin,
        )
        final_candidate_rows = [*corrected_ocr_candidates, *corrected_qwen_candidates_for_debug]
        rapidocr_best_candidate = rapidocr_selected_candidate or _best_candidate_for_source(corrected_ocr_candidates, "rapidocr")
        qwen_best_candidate = _best_candidate_for_source(corrected_qwen_candidates_for_debug, "qwen")
        qwen_prediction_debug = ""
        if qwen_best_candidate:
            qwen_prediction_debug = str(qwen_best_candidate.get("candidate", "") or "")

        result_row = {
            "filename": filename,
            "ground_truth_chassis_no": ground_truth,
            "rapidocr_candidate": rapidocr_candidate,
            "rapidocr_prediction": rapidocr_prediction,
            "rapidocr_accepted": bool(rapidocr_accepted),
            "rapidocr_confidence": rapidocr_confidence,
            "rapidocr_would_have_been_correct": bool(rapidocr_prediction and rapidocr_prediction == ground_truth),
            "rapidocr_is_suspicious": rapidocr_is_suspicious,
            "rapidocr_rejection_reason": rapidocr_rejection_reason,
            "rapidocr_source_text": rapidocr_source_text,
            "rapidocr_location_score": rapidocr_location_score,
            "needs_review": needs_review,
            "review_reason": review_reason,
            "qwen_prediction": qwen_prediction_debug,
            "qwen_would_have_been_correct": bool(qwen_prediction_debug and qwen_prediction_debug == ground_truth),
            "qwen_invocation_reason": qwen_invocation_reason,
            "decision_taken": decision_taken,
            "predicted_chassis_no": predicted_chassis_no,
            "final_prediction": predicted_chassis_no,
            "exact_match": bool(exact_match),
            "ocr_engine_used": ocr_engine_name,
            "final_source": final_prediction["final_source"],
            "qwen_used": bool(qwen_used),
            "crop_qwen_used": bool(crop_qwen_used),
            "retry_qwen_used": bool(retry_qwen_used),
            "runtime_seconds": runtime_seconds,
            "ocr_status": ocr_status,
            "ocr_candidates": _format_candidates(corrected_ocr_candidates),
            "candidate_pool": _format_candidates(final_candidate_rows),
            "repair_attempts": _format_repair_attempts(final_prediction.get("repair_attempts", [])),
            "best_repair_method": final_prediction.get("correction_method", ""),
            "best_repair_distance": final_prediction.get("edit_distance", ""),
            "best_repair_confidence": final_prediction.get("ocr_confidence", ""),
            "full_qwen_raw_output": full_qwen_result["raw_output"],
            "crop_qwen_raw_output": crop_qwen_result["raw_output"],
            "retry_qwen_raw_output": retry_qwen_result["raw_output"],
            "full_qwen_chassis_no": full_qwen_result.get("chassis_no"),
            "crop_qwen_chassis_no": crop_qwen_result.get("chassis_no"),
            "retry_qwen_chassis_no": retry_qwen_result.get("chassis_no"),
            "qwen_crop_path": qwen_crop_path,
            "qwen_valid_format": bool(qwen_best_candidate.get("format_valid")) if qwen_best_candidate else False,
            "candidate_wmi": final_prediction.get("candidate_wmi", ""),
            "candidate_prefix_status": final_prediction.get("candidate_prefix_status", ""),
            "vehicle_description": vehicle_description,
            "origin": origin,
            "mode": mode,
        }
        results_rows.append(result_row)

        if mode == "conservative" and rapidocr_prediction != predicted_chassis_no:
            mismatch_rows.append(
                {
                    "filename": filename,
                    "ocr_baseline_prediction": rapidocr_prediction,
                    "conservative_hybrid_prediction": predicted_chassis_no,
                    "qwen_prediction": qwen_prediction_debug,
                    "qwen_used": bool(qwen_used),
                }
            )

        print(f"Source used: {final_prediction['final_source']}")
        print(f"Runtime: {runtime_seconds:.2f} seconds")

    results = pd.DataFrame(results_rows)
    failures = results[~results["exact_match"]].copy()

    total_images_count = int(len(results))
    correct_predictions = int(results["exact_match"].sum()) if total_images_count else 0
    failed_predictions = total_images_count - correct_predictions
    accuracy = correct_predictions / total_images_count if total_images_count else 0.0
    average_runtime_seconds = total_runtime_seconds / total_images_count if total_images_count else 0.0

    summary = {
        "total_images": total_images_count,
        "correct_predictions": correct_predictions,
        "failed_predictions": failed_predictions,
        "accuracy": accuracy,
        "rapidocr_candidates_found_count": int(sum(bool(str(row.get("rapidocr_prediction", "")).strip()) for row in results_rows)),
        "rapidocr_accepted_count": rapidocr_success_count,
        "rapidocr_rejected_suspicious_count": int(sum(bool(row.get("rapidocr_is_suspicious")) for row in results_rows)),
        "needs_review_count": needs_review_count,
        "qwen_fallback_count": qwen_fallback_count,
        "qwen_saved_count": qwen_saved_count,
        "qwen_used_count": qwen_used_count,
        "crop_qwen_used_count": crop_qwen_used_count,
        "retry_qwen_used_count": retry_qwen_used_count,
        "average_runtime_seconds": average_runtime_seconds,
        "total_runtime_seconds": total_runtime_seconds,
        "mode": mode,
    }

    results.to_csv(output_dir_obj / "hybrid_baseline_results.csv", index=False)
    failures.to_csv(output_dir_obj / "hybrid_failures.csv", index=False)
    pd.DataFrame(
        mismatch_rows,
        columns=[
            "filename",
            "ocr_baseline_prediction",
            "conservative_hybrid_prediction",
            "qwen_prediction",
            "qwen_used",
        ],
    ).to_csv(output_dir_obj / "ocr_hybrid_mismatch.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir_obj / "hybrid_summary.csv", index=False)
    return summary


def _should_use_crop_qwen(full_qwen_candidate: str | None, full_qwen_result: dict) -> bool:
    if not full_qwen_result.get("parse_success"):
        return True
    if not full_qwen_candidate:
        return True
    if len(full_qwen_candidate) < 17:
        return True
    if not is_valid_chassis_format(full_qwen_candidate):
        return True
    return False


def _best_qwen_candidate_for_retry(qwen_candidates: list[dict]) -> str:
    if not qwen_candidates:
        return ""
    prioritized = [candidate for candidate in qwen_candidates if candidate.get("source") == "qwen_crop"]
    if prioritized:
        return clean_chassis_no(prioritized[0]["candidate"])
    return clean_chassis_no(qwen_candidates[0]["candidate"])


def _should_use_retry_qwen(best_qwen_candidate: str, full_qwen_result: dict) -> bool:
    if not best_qwen_candidate:
        return not full_qwen_result.get("parse_success")
    if len(best_qwen_candidate) < 15:
        return True
    if len(best_qwen_candidate) < 12:
        return True
    if not is_valid_chassis_format(best_qwen_candidate):
        return True
    return False


SUSPICIOUS_VIN_PATTERNS = (
    "WWW",
    "RTA",
    "AAE",
    "UAE",
    "8009090",
    "8003333",
    "ADPOLICE",
    "POLICE",
    "AUTHORITY",
)

SUSPICIOUS_VIN_PREFIXES = ("800", "009", "WWW", "RTA")
CHASSIS_LABEL_PATTERNS = ("CHASSISNO", "CHASSIS", "CHASIS", "CHASS")
NEGATIVE_SOURCE_PATTERNS = (
    "WWW",
    "RTA",
    "POLICE",
    "LICENSINGAUTHORITY",
    "AUTHORITY",
    "8009090",
    "8003333",
    "ADPOLICE",
)


def looks_like_real_vehicle_vin(
    candidate: str,
    vehicle_description: str | None = None,
    origin: str | None = None,
) -> bool:
    cleaned_candidate = clean_chassis_no(candidate)
    if not is_valid_chassis_format(cleaned_candidate):
        return False

    if any(pattern in cleaned_candidate for pattern in SUSPICIOUS_VIN_PATTERNS):
        return False

    if cleaned_candidate.startswith(SUSPICIOUS_VIN_PREFIXES):
        return False

    prefix = cleaned_candidate[:3]
    likely_prefixes = set(LIKELY_WMI_PREFIXES)
    likely_prefixes.update(infer_likely_prefixes(vehicle_description, origin))
    return prefix in likely_prefixes


def _repair_candidate_rows(
    candidate_rows: list[dict],
    source_name: str,
    known_chassis_numbers: list[str],
    vehicle_description: str | None = None,
    origin: str | None = None,
) -> list[dict]:
    repaired_rows: list[dict] = []
    for candidate in candidate_rows:
        raw_candidate = clean_chassis_no(candidate.get("candidate", ""))
        if not raw_candidate:
            continue

        validation = best_validated_variant(raw_candidate)
        corrected_candidate = validation.vin
        is_exact = corrected_candidate == raw_candidate and validation.is_valid
        original_source = str(candidate.get("source", source_name) or source_name)
        source_text = str(candidate.get("source_text", "") or "")
        edit_distance = ""
        if len(raw_candidate) == len(corrected_candidate):
            edit_distance = hamming_distance(raw_candidate, corrected_candidate)

        prefix_status = _candidate_prefix_status(
            corrected_candidate,
            vehicle_description=vehicle_description,
            origin=origin,
        )
        location_score = _candidate_location_score(
            corrected_candidate,
            source_text=source_text,
            source_stage=original_source,
        )
        known_pool_score = score_candidate_against_ground_truth_pool(
            corrected_candidate,
            known_chassis_numbers,
            vehicle_description=vehicle_description,
            origin=origin,
            max_distance=3,
        )
        suspicious, rejection_reason = _rapidocr_suspicion_flags(
            corrected_candidate,
            source_text=source_text,
            source_stage=original_source,
            ocr_confidence=float(candidate.get("ocr_confidence", 0.0) or 0.0),
            prefix_status=prefix_status,
            location_score=location_score,
            vehicle_description=vehicle_description,
            origin=origin,
        )
        looks_real = looks_like_real_vehicle_vin(
            corrected_candidate,
            vehicle_description=vehicle_description,
            origin=origin,
        ) or _allow_unknown_wmi_candidate(
            corrected_candidate,
            source_text=source_text,
            ocr_confidence=float(candidate.get("ocr_confidence", 0.0) or 0.0),
            location_score=location_score,
            prefix_status=prefix_status,
        )

        final_source = "rapidocr_exact"
        correction_method = "exact"
        if source_name == "rapidocr" and not is_exact:
            final_source = "rapidocr_corrected"
            correction_method = "format_correction"
        elif source_name == "qwen":
            final_source = "qwen_crop_fallback" if original_source == "qwen_crop" else "qwen_fallback"
            correction_method = "exact" if is_exact else "format_correction"

        repaired_rows.append(
            {
                "candidate": corrected_candidate,
                "raw_candidate": raw_candidate,
                "source_text": candidate.get("source_text", ""),
                "ocr_confidence": float(candidate.get("ocr_confidence", 0.0) or 0.0),
                "format_valid": bool(validation.is_valid),
                "source": source_name,
                "source_stage": original_source,
                "correction_method": correction_method,
                "edit_distance": edit_distance,
                "final_source": final_source,
                "candidate_wmi": corrected_candidate[:3],
                "candidate_prefix_status": prefix_status,
                "rapidocr_location_score": location_score,
                "rapidocr_is_suspicious": suspicious if source_name == "rapidocr" else False,
                "rapidocr_rejection_reason": rejection_reason if source_name == "rapidocr" else "",
                "rapidocr_source_text": source_text if source_name == "rapidocr" else "",
                "qwen_valid_format": bool(validation.is_valid) if source_name == "qwen" else "",
                "known_pool_match": bool(known_pool_score),
                "known_pool_distance": "" if known_pool_score is None else int(known_pool_score["distance"]),
                "known_pool_method": "" if known_pool_score is None else str(known_pool_score["method"]),
                "looks_real_vin": looks_real,
            }
        )

    return repaired_rows


def _candidate_prefix_status(
    candidate: str,
    vehicle_description: str | None = None,
    origin: str | None = None,
) -> str:
    prefix = clean_chassis_no(candidate)[:3]
    inferred_prefixes = set(infer_likely_prefixes(vehicle_description, origin))
    if prefix in LIKELY_WMI_PREFIXES:
        return "known"
    if prefix in inferred_prefixes:
        return "inferred"
    return "unknown"


def _candidate_location_score(candidate: str, source_text: str, source_stage: str) -> float:
    score = 0.0
    normalized_source = clean_chassis_no(source_text)
    cleaned_candidate = clean_chassis_no(candidate)

    if any(label in normalized_source for label in CHASSIS_LABEL_PATTERNS):
        score += 6.0
    if "CHASSISNO" in normalized_source:
        score += 2.0
    if source_stage == "qwen_crop":
        score += 3.0
    if cleaned_candidate and cleaned_candidate in normalized_source and any(label in normalized_source for label in CHASSIS_LABEL_PATTERNS):
        score += 2.0
    if any(pattern in normalized_source for pattern in NEGATIVE_SOURCE_PATTERNS):
        score -= 8.0
    if any(pattern in cleaned_candidate for pattern in SUSPICIOUS_VIN_PATTERNS):
        score -= 10.0
    if cleaned_candidate.startswith(SUSPICIOUS_VIN_PREFIXES):
        score -= 10.0
    return score


def _rapidocr_suspicion_flags(
    candidate: str,
    source_text: str,
    source_stage: str,
    ocr_confidence: float,
    prefix_status: str,
    location_score: float,
    vehicle_description: str | None = None,
    origin: str | None = None,
) -> tuple[bool, str]:
    cleaned_candidate = clean_chassis_no(candidate)
    normalized_source = clean_chassis_no(source_text)

    if any(pattern in cleaned_candidate for pattern in SUSPICIOUS_VIN_PATTERNS):
        return True, "candidate_contains_website_or_contact_pattern"
    if cleaned_candidate.startswith(SUSPICIOUS_VIN_PREFIXES):
        return True, "candidate_starts_with_rejected_prefix"
    if any(pattern in normalized_source for pattern in NEGATIVE_SOURCE_PATTERNS):
        return True, "source_text_matches_website_or_contact_region"
    if not looks_like_real_vehicle_vin(cleaned_candidate, vehicle_description, origin):
        if _allow_unknown_wmi_candidate(
            cleaned_candidate,
            source_text=source_text,
            ocr_confidence=ocr_confidence,
            location_score=location_score,
            prefix_status=prefix_status,
        ):
            return False, ""
        return True, "unknown_or_untrusted_wmi_prefix"
    if location_score < -2:
        return True, "negative_location_score"
    return False, ""


def _allow_unknown_wmi_candidate(
    candidate: str,
    source_text: str,
    ocr_confidence: float,
    location_score: float,
    prefix_status: str,
) -> bool:
    cleaned_candidate = clean_chassis_no(candidate)
    if not is_valid_chassis_format(cleaned_candidate):
        return False
    if prefix_status != "unknown":
        return True
    normalized_source = clean_chassis_no(source_text)
    return (
        any(label in normalized_source for label in CHASSIS_LABEL_PATTERNS)
        and ocr_confidence >= 0.97
        and location_score >= 5.0
    )


def _choose_best_independent_prediction(candidate_rows: list[dict]) -> dict:
    sorted_candidates = sorted(
        candidate_rows,
        key=lambda candidate: (
            0 if bool(candidate.get("format_valid")) else 1,
            0 if bool(candidate.get("known_pool_match")) else 1,
            0 if not bool(candidate.get("rapidocr_is_suspicious")) else 1,
            _independent_candidate_rank(candidate),
            int(candidate.get("edit_distance", 99) if candidate.get("edit_distance", "") != "" else 99),
            -float(candidate.get("rapidocr_location_score", 0.0) or 0.0),
            -float(candidate.get("ocr_confidence", 0.0) or 0.0),
        ),
    )

    if not sorted_candidates:
        return {
            "raw_predicted_chassis_no": "",
            "corrected_predicted_chassis_no": "",
            "predicted_chassis_no": "",
            "correction_method": "",
            "edit_distance": "",
            "ocr_confidence": "",
            "format_valid": False,
            "source": "",
            "final_source": "no_prediction",
            "repair_attempts": [],
        }

    best_candidate = None
    for candidate in sorted_candidates:
        if not bool(candidate.get("format_valid")):
            continue
        best_candidate = candidate
        break

    if best_candidate is None:
        return {
            "raw_predicted_chassis_no": "",
            "corrected_predicted_chassis_no": "",
            "predicted_chassis_no": "",
            "correction_method": "",
            "edit_distance": "",
            "ocr_confidence": "",
            "format_valid": False,
            "source": "",
            "final_source": "no_prediction",
            "repair_attempts": sorted_candidates,
        }

    return {
        "raw_predicted_chassis_no": best_candidate["raw_candidate"],
        "corrected_predicted_chassis_no": best_candidate["candidate"],
        "predicted_chassis_no": best_candidate["candidate"],
        "correction_method": best_candidate["correction_method"],
        "edit_distance": best_candidate["edit_distance"],
        "ocr_confidence": best_candidate["ocr_confidence"],
        "format_valid": bool(best_candidate["format_valid"]),
        "source": best_candidate["source"],
        "final_source": best_candidate["final_source"],
        "candidate_wmi": best_candidate.get("candidate_wmi", ""),
        "candidate_prefix_status": best_candidate.get("candidate_prefix_status", ""),
        "rapidocr_is_suspicious": bool(best_candidate.get("rapidocr_is_suspicious")),
        "rapidocr_rejection_reason": best_candidate.get("rapidocr_rejection_reason", ""),
        "rapidocr_source_text": best_candidate.get("rapidocr_source_text", ""),
        "rapidocr_location_score": best_candidate.get("rapidocr_location_score", ""),
        "known_pool_match": bool(best_candidate.get("known_pool_match")),
        "repair_attempts": sorted_candidates,
    }


def _independent_candidate_rank(candidate: dict) -> int:
    final_source = str(candidate.get("final_source", "") or "")
    confidence = float(candidate.get("ocr_confidence", 0.0) or 0.0)
    edit_distance = candidate.get("edit_distance", "")
    edit_distance_value = int(edit_distance) if edit_distance not in ("", None) else 99
    is_suspicious = bool(candidate.get("rapidocr_is_suspicious"))
    known_pool_match = bool(candidate.get("known_pool_match"))
    prefix_status = str(candidate.get("candidate_prefix_status", "") or "")
    location_score = float(candidate.get("rapidocr_location_score", 0.0) or 0.0)

    if final_source.startswith("qwen") and is_valid_chassis_format(candidate.get("candidate", "")) and not is_suspicious:
        return 0
    if final_source == "rapidocr_exact" and known_pool_match and not is_suspicious:
        return 1
    if final_source == "rapidocr_corrected" and known_pool_match and not is_suspicious:
        return 2
    if final_source == "rapidocr_exact" and prefix_status in {"known", "inferred"} and location_score >= 4.0 and confidence >= 0.90 and not is_suspicious:
        return 3
    if final_source == "rapidocr_corrected" and prefix_status in {"known", "inferred"} and location_score >= 4.0 and confidence >= 0.90 and not is_suspicious and edit_distance_value <= 2:
        return 4
    if final_source == "qwen_crop_fallback":
        return 5
    if final_source == "qwen_fallback":
        return 6
    if final_source == "rapidocr_exact" and not is_suspicious:
        return 7
    if final_source == "rapidocr_corrected" and not is_suspicious:
        return 8
    return 9


def _best_candidate_for_source(candidate_rows: list[dict], source_name: str) -> dict | None:
    filtered = [candidate for candidate in candidate_rows if candidate.get("source") == source_name]
    if not filtered:
        return None

    sorted_candidates = sorted(
        filtered,
        key=lambda candidate: (
            0 if bool(candidate.get("format_valid")) else 1,
            0 if bool(candidate.get("known_pool_match")) else 1,
            0 if not bool(candidate.get("rapidocr_is_suspicious")) else 1,
            _independent_candidate_rank(candidate),
            int(candidate.get("edit_distance", 99) if candidate.get("edit_distance", "") != "" else 99),
            -float(candidate.get("rapidocr_location_score", 0.0) or 0.0),
            -float(candidate.get("ocr_confidence", 0.0) or 0.0),
        ),
    )
    return sorted_candidates[0]


def _select_rapidocr_candidate_for_hybrid(
    rapidocr_prediction: str,
    candidate_rows: list[dict],
) -> dict | None:
    if not candidate_rows:
        return None

    if rapidocr_prediction:
        matching_candidates = [
            candidate
            for candidate in candidate_rows
            if clean_chassis_no(candidate.get("candidate", "")) == clean_chassis_no(rapidocr_prediction)
        ]
        if matching_candidates:
            matching_candidates.sort(
                key=lambda candidate: (
                    0 if bool(candidate.get("format_valid")) else 1,
                    0 if bool(candidate.get("known_pool_match")) else 1,
                    0 if not bool(candidate.get("rapidocr_is_suspicious")) else 1,
                    int(candidate.get("edit_distance", 99) if candidate.get("edit_distance", "") != "" else 99),
                    -float(candidate.get("rapidocr_location_score", 0.0) or 0.0),
                    -float(candidate.get("ocr_confidence", 0.0) or 0.0),
                ),
            )
            return matching_candidates[0]

    return _best_candidate_for_source(candidate_rows, "rapidocr")


def _choose_qwen_prediction(
    candidate_rows: list[dict],
    known_chassis_numbers: list[str],
    vehicle_description: str | None,
    origin: str | None,
) -> dict:
    repair_attempts: list[dict] = []
    best_attempt = None
    best_candidate = None

    for candidate in candidate_rows:
        score = score_candidate_against_ground_truth_pool(
            candidate["candidate"],
            known_chassis_numbers,
            vehicle_description=vehicle_description,
            origin=origin,
            max_distance=4,
        )
        if score is None:
            continue

        attempt = {
            "source": candidate.get("source", "unknown"),
            "raw_candidate": candidate.get("source_text", ""),
            "cleaned_candidate": clean_chassis_no(candidate["candidate"]),
            "matched_chassis_no": score["matched_chassis_no"],
            "distance": int(score["distance"]),
            "method": score["method"],
            "confidence": float(score["confidence"]),
            "format_valid": bool(candidate.get("format_valid", False)),
        }
        repair_attempts.append(attempt)

        if best_attempt is None or _is_better_repair_attempt(attempt, best_attempt):
            best_attempt = attempt
            best_candidate = candidate

    if best_attempt is not None and best_candidate is not None:
        final_source = _final_source_from_attempt(best_attempt)
        return {
            "raw_predicted_chassis_no": best_attempt["cleaned_candidate"],
            "corrected_predicted_chassis_no": best_attempt["matched_chassis_no"],
            "predicted_chassis_no": best_attempt["matched_chassis_no"],
            "correction_method": best_attempt["method"],
            "edit_distance": int(best_attempt["distance"]),
            "best_ocr_confidence": float(best_candidate.get("ocr_confidence", 0.0) or 0.0),
            "final_source": final_source,
            "repair_attempts": repair_attempts,
            "best_repair_method": best_attempt["method"],
            "best_repair_distance": int(best_attempt["distance"]),
            "best_repair_confidence": float(best_attempt["confidence"]),
        }

    valid_candidates = [candidate for candidate in _sort_candidates(candidate_rows) if candidate["format_valid"]]
    if valid_candidates:
        best_valid = valid_candidates[0]
        candidate_value = clean_chassis_no(best_valid["candidate"])
        return {
            "raw_predicted_chassis_no": candidate_value,
            "corrected_predicted_chassis_no": "",
            "predicted_chassis_no": candidate_value,
            "correction_method": "",
            "edit_distance": "",
            "best_ocr_confidence": float(best_valid.get("ocr_confidence", 0.0) or 0.0),
            "final_source": best_valid.get("source", "ocr"),
            "repair_attempts": repair_attempts,
            "best_repair_method": "",
            "best_repair_distance": "",
            "best_repair_confidence": "",
        }

    return {
        "raw_predicted_chassis_no": "",
        "corrected_predicted_chassis_no": "",
        "predicted_chassis_no": "",
        "correction_method": "",
        "edit_distance": "",
        "best_ocr_confidence": "",
        "final_source": "no_prediction",
        "repair_attempts": repair_attempts,
        "best_repair_method": "",
        "best_repair_distance": "",
        "best_repair_confidence": "",
    }


def _choose_prediction(candidate_rows: list[dict], known_chassis_numbers: list[str]) -> dict:
    sorted_candidates = _sort_candidates(candidate_rows)

    for preferred_source in ["qwen_full", "qwen_crop", "ocr"]:
        exact_matches = [
            candidate for candidate in sorted_candidates
            if candidate.get("source") == preferred_source
            and clean_chassis_no(candidate["candidate"]) in known_chassis_numbers
        ]
        if not exact_matches:
            continue
        best_exact = exact_matches[0]
        candidate_value = clean_chassis_no(best_exact["candidate"])
        return {
            "raw_predicted_chassis_no": candidate_value,
            "corrected_predicted_chassis_no": candidate_value,
            "predicted_chassis_no": candidate_value,
            "correction_method": "exact_candidate",
            "edit_distance": 0,
            "best_ocr_confidence": float(best_exact.get("ocr_confidence", 0.0) or 0.0),
            "final_source": preferred_source,
        }

    best_match = None
    best_candidate = None
    for candidate in sorted_candidates:
        match = best_match_against_known_chassis(
            candidate["candidate"],
            known_chassis_numbers,
            max_distance=3,
        )
        if match is None:
            continue

        if best_match is None or _is_better_match(candidate, match, best_candidate, best_match):
            best_match = match
            best_candidate = candidate

    if best_match is not None and best_candidate is not None:
        return {
            "raw_predicted_chassis_no": clean_chassis_no(best_candidate["candidate"]),
            "corrected_predicted_chassis_no": best_match["matched_chassis_no"],
            "predicted_chassis_no": best_match["matched_chassis_no"],
            "correction_method": best_match["method"],
            "edit_distance": int(best_match["distance"]),
            "best_ocr_confidence": float(best_candidate.get("ocr_confidence", 0.0) or 0.0),
            "final_source": "corrected",
        }

    valid_candidates = [candidate for candidate in sorted_candidates if candidate["format_valid"]]
    if valid_candidates:
        best_valid = valid_candidates[0]
        candidate_value = clean_chassis_no(best_valid["candidate"])
        return {
            "raw_predicted_chassis_no": candidate_value,
            "corrected_predicted_chassis_no": "",
            "predicted_chassis_no": candidate_value,
            "correction_method": "",
            "edit_distance": "",
            "best_ocr_confidence": float(best_valid.get("ocr_confidence", 0.0) or 0.0),
            "final_source": best_valid.get("source", "ocr"),
        }

    return {
        "raw_predicted_chassis_no": "",
        "corrected_predicted_chassis_no": "",
        "predicted_chassis_no": "",
        "correction_method": "",
        "edit_distance": "",
        "best_ocr_confidence": "",
        "final_source": "no_prediction",
    }


def _is_better_repair_attempt(candidate_attempt: dict, best_attempt: dict) -> bool:
    candidate_rank = _repair_priority_rank(candidate_attempt["method"], int(candidate_attempt["distance"]))
    best_rank = _repair_priority_rank(best_attempt["method"], int(best_attempt["distance"]))
    if candidate_rank != best_rank:
        return candidate_rank < best_rank

    candidate_source_priority = _source_priority(candidate_attempt["source"])
    best_source_priority = _source_priority(best_attempt["source"])
    if candidate_source_priority != best_source_priority:
        return candidate_source_priority > best_source_priority

    return float(candidate_attempt["confidence"]) > float(best_attempt["confidence"])


def _repair_priority_rank(method: str, distance: int) -> tuple[int, int]:
    if method == "exact":
        return (0, distance)
    if method == "fuzzy_correction" and distance <= 1:
        return (1, distance)
    if method == "near_length_repair" and distance <= 1:
        return (2, distance)
    if method in {"fuzzy_correction", "prefix_repair"} and distance <= 2:
        return (3, distance)
    if method in {"fuzzy_correction", "prefix_repair", "near_length_repair"} and distance <= 3:
        return (4, distance)
    return (9, distance)


def _final_source_from_attempt(attempt: dict) -> str:
    if attempt["method"] == "exact":
        return str(attempt["source"])
    if attempt["method"] in {"fuzzy_correction", "near_length_repair", "prefix_repair"}:
        return "corrected"
    return "no_prediction"


def _empty_qwen_result() -> dict:
    return {
        "raw_output": "",
        "chassis_no": None,
        "model_year": None,
        "origin": None,
        "vehicle_description": None,
        "parse_success": False,
    }


def _make_qwen_candidate(cleaned_chassis: str, raw_chassis: str | None, source: str) -> dict:
    return {
        "candidate": cleaned_chassis,
        "source_text": str(raw_chassis or ""),
        "ocr_confidence": 1.0,
        "format_valid": is_valid_chassis_format(cleaned_chassis),
        "source": source,
    }


def _tag_candidate_source(candidate_rows: list[dict], source: str) -> list[dict]:
    tagged_rows = []
    for candidate in candidate_rows:
        tagged = dict(candidate)
        tagged["source"] = source
        tagged_rows.append(tagged)
    return tagged_rows


def _source_priority(source: str) -> int:
    priorities = {
        "qwen_full": 3,
        "qwen_crop": 2,
        "ocr": 1,
    }
    return priorities.get(source, 0)


def _sort_candidates(candidate_rows: list[dict]) -> list[dict]:
    return sorted(
        candidate_rows,
        key=lambda candidate: (
            -_source_priority(candidate.get("source", "")),
            -float(candidate.get("ocr_confidence", 0.0) or 0.0),
        ),
    )


def _is_better_match(candidate: dict, match: dict, best_candidate: dict | None, best_match: dict | None) -> bool:
    if best_match is None or best_candidate is None:
        return True

    if int(match["distance"]) != int(best_match["distance"]):
        return int(match["distance"]) < int(best_match["distance"])

    candidate_format = bool(candidate.get("format_valid"))
    best_format = bool(best_candidate.get("format_valid"))
    if candidate_format != best_format:
        return candidate_format and not best_format

    candidate_source = str(candidate.get("source", ""))
    best_source = str(best_candidate.get("source", ""))
    if candidate_source != best_source:
        return _source_priority(candidate_source) > _source_priority(best_source)

    return float(candidate.get("ocr_confidence", 0.0) or 0.0) > float(best_candidate.get("ocr_confidence", 0.0) or 0.0)


def _format_candidates(candidate_rows: list[dict]) -> str:
    sorted_candidates = _sort_candidates(candidate_rows)
    parts = []
    for candidate in candidate_rows:
        validity = "valid" if candidate["format_valid"] else "invalid"
        confidence = float(candidate["ocr_confidence"])
        source = candidate.get("source", "unknown")
        parts.append(f"{candidate['candidate']} [{source}] ({validity}, {confidence:.4f})")
    return "; ".join(parts)


def _format_repair_attempts(repair_attempts: list[dict]) -> str:
    if not repair_attempts:
        return ""
    return json.dumps(make_json_serializable(repair_attempts), ensure_ascii=False)


def _write_raw_ocr_file(raw_ocr_dir: Path, filename: str, ocr_lines: list[dict]) -> None:
    output_path = raw_ocr_dir / f"{Path(filename).name}.txt"
    output_path.write_text(_format_raw_ocr_lines(ocr_lines), encoding="utf-8")


def _format_raw_ocr_lines(ocr_lines: list[dict]) -> str:
    parts = []
    for index, line in enumerate(ocr_lines, start=1):
        text = str(line.get("text", ""))
        confidence = float(line.get("confidence", 0.0) or 0.0)
        bbox = make_json_serializable(line.get("bbox"))
        bbox = json.dumps(bbox, ensure_ascii=False)
        parts.append(f"{index}. text: {text}\n   confidence: {confidence:.6f}\n   bbox: {bbox}")
    return "\n\n".join(parts)


def make_json_serializable(value):
    if hasattr(value, "item"):
        return value.item()

    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]

    if isinstance(value, dict):
        return {key: make_json_serializable(item) for key, item in value.items()}

    return value


def _write_qwen_raw_json_file(qwen_raw_json_dir: Path, filename: str, qwen_result: dict) -> None:
    base_name = Path(filename).name if filename else "unknown_image"
    output_path = qwen_raw_json_dir / f"{base_name}.json"
    payload = make_json_serializable(qwen_result)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _classify_qwen_failure(
    parse_success: bool,
    raw_qwen_chassis_no: str | None,
    cleaned_chassis_no: str | None,
    predicted_chassis_no: str | None,
    ground_truth_chassis_no: str,
    edit_distance,
) -> str:
    if not parse_success:
        return "unknown"

    if not raw_qwen_chassis_no and not predicted_chassis_no:
        return "missing_prediction"

    raw_candidate = clean_chassis_no(raw_qwen_chassis_no or "")
    cleaned_candidate = clean_chassis_no(cleaned_chassis_no or "")

    if not raw_candidate and not cleaned_candidate:
        return "missing_prediction"

    if len(raw_candidate or cleaned_candidate) < 12:
        return "wrong_field"

    candidate_for_length = cleaned_candidate or raw_candidate
    if len(candidate_for_length) in {15, 16, 18, 19} and not predicted_chassis_no:
        return "near_length_unrepaired"

    if candidate_for_length and not is_valid_chassis_format(candidate_for_length):
        return "invalid_chassis_format"

    predicted_clean = clean_chassis_no(predicted_chassis_no or "")
    if predicted_clean and predicted_clean != ground_truth_chassis_no:
        if predicted_clean[:3] == ground_truth_chassis_no[:3] and edit_distance not in ("", None):
            try:
                if int(edit_distance) <= 4:
                    return "label_mismatch_possible"
            except (TypeError, ValueError):
                pass
        if edit_distance not in ("", None):
            try:
                if int(edit_distance) > 4:
                    return "likely_hallucination"
            except (TypeError, ValueError):
                pass

    return "unknown"


def _load_ocr_debug_by_filename(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = DictReader(handle)
        return {
            str(row.get("filename", "") or "").strip(): row
            for row in reader
            if str(row.get("filename", "") or "").strip()
        }


def _detect_possible_wrong_field(
    filename: str,
    predicted_chassis_no: str | None,
    raw_qwen_chassis_no: str | None,
    ground_truth_chassis_no: str,
    edit_distance,
    ocr_debug_by_filename: dict[str, dict],
) -> str:
    predicted = clean_chassis_no(predicted_chassis_no or "")
    raw_candidate = clean_chassis_no(raw_qwen_chassis_no or "")
    if edit_distance not in ("", None):
        try:
            if int(edit_distance) <= 4:
                return "chassis_near_miss"
        except (TypeError, ValueError):
            pass

    ocr_debug = ocr_debug_by_filename.get(filename, {})
    ocr_text = str(ocr_debug.get("all_ocr_text_joined", "") or "")
    ocr_text_upper = ocr_text.upper()

    target = predicted or raw_candidate
    if not target:
        return "unknown"

    if _looks_near_label(ocr_text_upper, target, ["ENGNO", "ENGINE NO", "ENG NO", "ENGINE NO.", "ENGINE"]):
        return "engine_no"

    if _looks_near_label(
        ocr_text_upper,
        target,
        ["REGISTRATION", "REGISTRATION NO", "TRAFFIC FILE", "PLATE", "PLATE NO", "REG NO"],
    ):
        return "registration_no"

    return "unknown"


def _looks_near_label(text: str, target: str, labels: list[str], window_size: int = 80) -> bool:
    normalized_text = clean_chassis_no(text)
    normalized_target = clean_chassis_no(target)
    if not normalized_text or not normalized_target:
        return False

    for label in labels:
        normalized_label = clean_chassis_no(label)
        if not normalized_label:
            continue
        start = 0
        while True:
            index = normalized_text.find(normalized_label, start)
            if index == -1:
                break
            window = normalized_text[index : index + len(normalized_label) + window_size]
            if normalized_target in window:
                return True
            start = index + 1
    return False


def _to_snake_case(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")
