from __future__ import annotations

import json
import re
from csv import DictWriter
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .vin import clean_chassis_no, validate_chassis_no

REQUIRED_COLUMNS = [
    "filename",
    "chassis_no",
    "model_year",
    "origin",
    "vehicle_description",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def audit_dataset(
    labels_path: str = "data/labels.xlsx",
    images_dir: str = "data/images",
    output_dir: str = "outputs",
) -> dict:
    """Audit label/image consistency and write CSV reports."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install database extras with: pip install -e .[db]") from exc

    labels_path = Path(labels_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_excel(labels_path)
    labels.columns = [_to_snake_case(column) for column in labels.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in labels.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required Excel columns: {missing_text}")

    labels["filename"] = labels["filename"].fillna("").astype(str).str.strip()
    labels["chassis_no_clean"] = labels["chassis_no"].map(clean_chassis_no)

    validation_rows = labels["chassis_no"].map(validate_chassis_no)
    validation = pd.DataFrame(validation_rows.tolist(), index=labels.index)
    labels = pd.concat([labels, validation.add_prefix("chassis_")], axis=1)

    image_names = _list_image_names(images_dir)
    excel_filenames = set(labels["filename"])

    missing_images = labels[~labels["filename"].isin(image_names)].copy()
    unlabelled_images = pd.DataFrame(
        {"filename": sorted(image_names - excel_filenames)}
    )

    duplicate_filenames = labels[
        labels["filename"].ne("") & labels["filename"].duplicated(keep=False)
    ].copy()
    duplicate_chassis_numbers = labels[
        labels["chassis_no_clean"].ne("")
        & labels["chassis_no_clean"].duplicated(keep=False)
    ].copy()
    invalid_chassis_numbers = labels[~labels["chassis_format_valid"]].copy()

    missing_images.to_csv(output_dir / "missing_images.csv", index=False)
    unlabelled_images.to_csv(output_dir / "unlabelled_images.csv", index=False)
    duplicate_filenames.to_csv(output_dir / "duplicate_filenames.csv", index=False)
    duplicate_chassis_numbers.to_csv(output_dir / "duplicate_chassis_numbers.csv", index=False)
    invalid_chassis_numbers.to_csv(output_dir / "invalid_chassis_numbers.csv", index=False)

    summary = {
        "total_excel_rows": int(len(labels)),
        "total_images": int(len(image_names)),
        "missing_images_count": int(len(missing_images)),
        "unlabelled_images_count": int(len(unlabelled_images)),
        "duplicate_filename_count": int(len(duplicate_filenames)),
        "duplicate_chassis_count": int(len(duplicate_chassis_numbers)),
        "invalid_chassis_count": int(len(invalid_chassis_numbers)),
    }

    pd.DataFrame([summary]).to_csv(output_dir / "audit_summary.csv", index=False)
    return summary


def cleanup_unlabelled_images(
    labels_path: str = "data/labels.xlsx",
    images_dir: str = "data/images",
    output_dir: str = "outputs",
) -> dict:
    """Move images not referenced by the Excel file into a recycle-bin folder."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install database extras with: pip install -e .[db]") from exc

    labels_path_obj = Path(labels_path)
    images_dir_obj = Path(images_dir)
    output_dir_obj = Path(output_dir)
    recycle_dir = output_dir_obj / "deleted_unlabelled"
    recycle_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_excel(labels_path_obj)
    labels.columns = [_to_snake_case(column) for column in labels.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in labels.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required Excel columns: {missing_text}")

    valid_filenames = set(labels["filename"].fillna("").astype(str).str.strip())
    valid_filenames.discard("")

    image_paths = _list_image_paths(images_dir_obj)
    unlabelled_paths = [
        path
        for path in image_paths
        if path.name.strip() not in valid_filenames
    ]

    summary = {
        "images_in_folder": int(len(image_paths)),
        "images_in_excel": int(len(valid_filenames)),
        "unlabelled_images_count": int(len(unlabelled_paths)),
        "unlabelled_paths": unlabelled_paths,
        "recycle_dir": recycle_dir,
    }
    return summary


def move_unlabelled_images(cleanup_summary: dict) -> int:
    """Move the computed unlabelled images and append a CSV log."""
    recycle_dir = Path(cleanup_summary["recycle_dir"])
    recycle_dir.mkdir(parents=True, exist_ok=True)
    log_path = recycle_dir / "deleted_files.csv"
    moved_count = 0

    fieldnames = ["filename", "original_path", "moved_to", "timestamp"]
    file_exists = log_path.exists()
    with log_path.open("a", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for original_path in cleanup_summary["unlabelled_paths"]:
            original_path = Path(original_path)
            if not original_path.exists():
                continue

            destination_path = _unique_destination(recycle_dir / original_path.name)
            original_path.replace(destination_path)
            timestamp = datetime.now(timezone.utc).isoformat()
            writer.writerow(
                {
                    "filename": original_path.name,
                    "original_path": original_path.as_posix(),
                    "moved_to": destination_path.as_posix(),
                    "timestamp": timestamp,
                }
            )
            moved_count += 1

    return moved_count


def _to_snake_case(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _list_image_names(images_dir: Path) -> set[str]:
    if not images_dir.exists():
        return set()
    return {
        path.name.strip()
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def _list_image_paths(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


class JsonlAuditLogger:
    def __init__(self, audit_path: str | Path, failures_path: str | Path | None = None) -> None:
        self.audit_path = Path(audit_path)
        self.failures_path = Path(failures_path) if failures_path else self.audit_path.with_name("failures.jsonl")
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.failures_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, result: Any) -> None:
        payload = _to_jsonable(result)
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._append(self.audit_path, payload)
        if payload.get("human_review_required"):
            self._append(self.failures_path, payload)

    @staticmethod
    def _append(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
