from __future__ import annotations

import json
import random
from pathlib import Path

from .audit import IMAGE_EXTENSIONS, REQUIRED_COLUMNS, _to_snake_case
from .vin import clean_chassis_no, validate_chassis_no


def prepare_dataset(
    labels_path: str = "data/labels.xlsx",
    images_dir: str = "data/images",
    output_dir: str = "outputs/dataset",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """Prepare clean train/val/test exports and fine-tuning files."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Pandas/openpyxl are required. Install them with: pip install pandas openpyxl") from exc

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required. Install it with: pip install pillow") from exc

    _validate_split_ratios(train_ratio, val_ratio, test_ratio)

    labels_path_obj = Path(labels_path)
    images_dir_obj = Path(images_dir)
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    labels = pd.read_excel(labels_path_obj)
    labels.columns = [_to_snake_case(column) for column in labels.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in labels.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required Excel columns: {missing_text}")

    labels["filename"] = labels["filename"].fillna("").astype(str).str.strip()
    labels["chassis_no_clean"] = labels["chassis_no"].map(clean_chassis_no)
    labels["origin_clean"] = labels["origin"].map(_clean_text_value)
    labels["vehicle_description_clean"] = labels["vehicle_description"].map(_clean_text_value)
    labels["model_year_clean"] = labels["model_year"].map(_clean_model_year)

    validation_rows = labels["chassis_no"].map(validate_chassis_no)
    validation = pd.DataFrame(validation_rows.tolist(), index=labels.index)
    labels = pd.concat([labels, validation.add_prefix("chassis_")], axis=1)

    image_inventory = _scan_images(images_dir_obj, Image)
    image_inventory_df = pd.DataFrame(image_inventory)
    image_quality_report = image_inventory_df.copy()
    if image_quality_report.empty:
        image_quality_report = pd.DataFrame(
            columns=[
                "filename",
                "image_path",
                "extension",
                "supported_extension",
                "readable",
                "width",
                "height",
                "file_size_bytes",
                "error",
            ]
        )

    supported_filenames = {
        row["filename"]
        for row in image_inventory
        if row["supported_extension"]
    }
    readable_filenames = {
        row["filename"]
        for row in image_inventory
        if row["supported_extension"] and row["readable"]
    }
    unsupported_rows = [row for row in image_inventory if not row["supported_extension"]]
    unreadable_rows = [row for row in image_inventory if row["supported_extension"] and not row["readable"]]

    excel_filenames = set(labels["filename"])
    missing_images = labels[~labels["filename"].isin(supported_filenames)].copy()
    unlabelled_images = pd.DataFrame(
        {"filename": sorted(supported_filenames - excel_filenames)}
    )
    duplicate_filenames = labels[
        labels["filename"].ne("") & labels["filename"].duplicated(keep=False)
    ].copy()
    duplicate_chassis = labels[
        labels["chassis_no_clean"].ne("")
        & labels["chassis_no_clean"].duplicated(keep=False)
    ].copy()
    invalid_chassis = labels[~labels["chassis_format_valid"]].copy()
    unreadable_images = pd.DataFrame(unreadable_rows)
    if unreadable_images.empty:
        unreadable_images = pd.DataFrame(columns=image_quality_report.columns)

    supported_extension_by_filename = {
        row["filename"]: bool(row["supported_extension"])
        for row in image_inventory
    }
    readable_by_filename = {
        row["filename"]: bool(row["readable"])
        for row in image_inventory
    }

    labels["image_path"] = labels["filename"].map(lambda name: _build_image_path(images_dir_obj, name))
    labels["image_readable"] = labels["filename"].map(lambda name: readable_by_filename.get(name, False))
    labels["supported_extension"] = labels["filename"].map(
        lambda name: supported_extension_by_filename.get(name, False)
    )
    labels["has_duplicate_filename"] = labels["filename"].isin(set(duplicate_filenames["filename"]))
    labels["has_duplicate_chassis"] = labels["chassis_no_clean"].isin(set(duplicate_chassis["chassis_no_clean"]))

    clean_labels = labels[
        labels["filename"].ne("")
        & labels["chassis_format_valid"]
        & labels["supported_extension"]
        & labels["image_readable"]
        & ~labels["has_duplicate_filename"]
        & ~labels["has_duplicate_chassis"]
    ].copy()

    export_columns = [
        "filename",
        "image_path",
        "chassis_no",
        "model_year",
        "origin",
        "vehicle_description",
        "chassis_no_clean",
        "origin_clean",
        "vehicle_description_clean",
    ]
    clean_labels = clean_labels.assign(
        chassis_no=clean_labels["chassis_no"].map(_clean_text_value),
        model_year=clean_labels["model_year"].map(_clean_model_year),
        origin=clean_labels["origin"].map(_clean_text_value),
        vehicle_description=clean_labels["vehicle_description"].map(_clean_text_value),
    )
    clean_labels_export = clean_labels[export_columns].copy()

    train_df, val_df, test_df = _split_dataframe(
        clean_labels_export,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    _write_jsonl(train_df, output_dir_obj / "finetune_train.jsonl")
    _write_jsonl(val_df, output_dir_obj / "finetune_val.jsonl")
    _write_jsonl(test_df, output_dir_obj / "finetune_test.jsonl")
    _write_ocr_targets(clean_labels_export, output_dir_obj / "ocr_targets.csv")
    _write_finetuning_plan(output_dir_obj / "FINETUNING_PLAN.md")

    clean_labels_export.to_csv(output_dir_obj / "clean_labels.csv", index=False)
    train_df.to_csv(output_dir_obj / "train.csv", index=False)
    val_df.to_csv(output_dir_obj / "val.csv", index=False)
    test_df.to_csv(output_dir_obj / "test.csv", index=False)
    image_quality_report.to_csv(output_dir_obj / "image_quality_report.csv", index=False)

    missing_images.to_csv(output_dir_obj / "missing_images.csv", index=False)
    unlabelled_images.to_csv(output_dir_obj / "unlabelled_images.csv", index=False)
    duplicate_filenames.to_csv(output_dir_obj / "duplicate_filenames.csv", index=False)
    duplicate_chassis.to_csv(output_dir_obj / "duplicate_chassis.csv", index=False)
    invalid_chassis.to_csv(output_dir_obj / "invalid_chassis.csv", index=False)
    unreadable_images.to_csv(output_dir_obj / "unreadable_images.csv", index=False)

    summary = {
        "total_excel_rows": int(len(labels)),
        "total_image_files": int(len(image_inventory)),
        "supported_image_files": int(sum(1 for row in image_inventory if row["supported_extension"])),
        "unsupported_file_count": int(len(unsupported_rows)),
        "missing_images_count": int(len(missing_images)),
        "unlabelled_images_count": int(len(unlabelled_images)),
        "duplicate_filename_count": int(len(duplicate_filenames)),
        "duplicate_chassis_count": int(len(duplicate_chassis)),
        "invalid_chassis_count": int(len(invalid_chassis)),
        "unreadable_images_count": int(len(unreadable_rows)),
        "clean_rows_count": int(len(clean_labels_export)),
        "train_count": int(len(train_df)),
        "val_count": int(len(val_df)),
        "test_count": int(len(test_df)),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
    }
    pd.DataFrame([summary]).to_csv(output_dir_obj / "dataset_summary.csv", index=False)
    return summary


def _validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be zero or positive.")
    total_ratio = sum(ratios)
    if total_ratio <= 0:
        raise ValueError("Split ratios must add up to more than zero.")
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must add up to 1.0.")


def _scan_images(images_dir: Path, image_class) -> list[dict]:
    if not images_dir.exists():
        return []

    rows: list[dict] = []
    for path in sorted(images_dir.iterdir()):
        if not path.is_file():
            continue

        supported_extension = path.suffix.lower() in IMAGE_EXTENSIONS
        readable = False
        width = ""
        height = ""
        error = ""

        if supported_extension:
            try:
                with image_class.open(path) as image:
                    image.load()
                    width, height = image.size
                readable = True
            except Exception as exc:  # pragma: no cover - depends on image data
                error = str(exc)

        rows.append(
            {
                "filename": path.name,
                "image_path": path.as_posix(),
                "extension": path.suffix.lower(),
                "supported_extension": supported_extension,
                "readable": readable,
                "width": width,
                "height": height,
                "file_size_bytes": path.stat().st_size,
                "error": error,
            }
        )
    return rows


def _split_dataframe(
    dataframe,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    shuffled_indexes = list(dataframe.index)
    random.Random(seed).shuffle(shuffled_indexes)
    shuffled = dataframe.loc[shuffled_indexes].reset_index(drop=True)

    total_rows = len(shuffled)
    train_count = int(total_rows * train_ratio)
    val_count = int(total_rows * val_ratio)
    test_count = total_rows - train_count - val_count

    train_df = shuffled.iloc[:train_count].copy()
    val_df = shuffled.iloc[train_count : train_count + val_count].copy()
    test_df = shuffled.iloc[train_count + val_count : train_count + val_count + test_count].copy()
    return train_df, val_df, test_df


def _write_jsonl(dataframe, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for row in dataframe.to_dict(orient="records"):
            assistant_payload = {
                "chassis_no": row["chassis_no_clean"] or None,
                "model_year": row["model_year"] or None,
                "origin": row["origin_clean"] or None,
                "vehicle_description": row["vehicle_description_clean"] or None,
            }
            item = {
                "image": row["image_path"],
                "messages": [
                    {
                        "role": "user",
                        "content": "Extract the chassis number, model year, origin, and vehicle description from this UAE Mulkiya image. Return only valid JSON.",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(assistant_payload, ensure_ascii=False),
                    },
                ],
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _write_ocr_targets(dataframe, output_path: Path) -> None:
    rows = []
    for row in dataframe.to_dict(orient="records"):
        target_payload = {
            "chassis_no": row["chassis_no_clean"],
            "model_year": row["model_year"],
            "origin": row["origin_clean"],
            "vehicle_description": row["vehicle_description_clean"],
        }
        rows.append(
            {
                "filename": row["filename"],
                "image_path": row["image_path"],
                "target_chassis_no": row["chassis_no_clean"],
                "target_text_json": json.dumps(target_payload, ensure_ascii=False),
            }
        )

    import pandas as pd

    pd.DataFrame(rows).to_csv(output_path, index=False)


def _write_finetuning_plan(output_path: Path) -> None:
    content = """# Fine-Tuning Plan

## Recommended Model
Start with `Qwen/Qwen2.5-VL-3B-Instruct` or another small vision-language model in the same size range.

## Why Fine-Tuning
Fine-tuning is being prepared so the project can learn Mulkiya-specific layout patterns and field wording from a larger internal dataset.

## Training Objective
The model should read a UAE Mulkiya image and return only valid JSON containing:
- `chassis_no`
- `model_year`
- `origin`
- `vehicle_description`

## Dataset Format
This dataset export includes:
- `train.csv`
- `val.csv`
- `test.csv`
- `finetune_train.jsonl`
- `finetune_val.jsonl`
- `finetune_test.jsonl`

Each JSONL row contains an image path and a user/assistant message pair.

## Train / Validation / Test Split
The dataset is split into train, validation, and test sets so the future fine-tuned model can be trained and evaluated consistently.

## Evaluation Metrics
Recommended evaluation metrics:
- exact chassis number match accuracy
- edit distance for chassis number prediction
- model year exact match
- origin exact match
- vehicle description exact match or normalized string match

## Expected Output Format
The fine-tuned model should return only valid JSON:

```json
{"chassis_no":"...","model_year":"...","origin":"...","vehicle_description":"..."}
```

## Production Warning
Fine-tuning should not replace the existing validation and correction logic.

The current RapidOCR pipeline remains the production baseline.

The first deployment target for a fine-tuned model should be fallback handling for failed or review-flagged cases, not direct replacement of the production OCR flow.
"""
    output_path.write_text(content, encoding="utf-8")


def _clean_text_value(value: object) -> str:
    text = " ".join(str(value or "").strip().split())
    return text


def _clean_model_year(value: object) -> str:
    text = _clean_text_value(value)
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _build_image_path(images_dir: Path, filename: str) -> str:
    if not filename:
        return ""
    return (images_dir / filename).as_posix()
