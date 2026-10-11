from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .audit import (
    JsonlAuditLogger,
    audit_dataset,
    cleanup_unlabelled_images,
    move_unlabelled_images,
)
from .database import DataFrameVehicleDatabase, EmptyVehicleDatabase
from .dataset_prep import prepare_dataset
from .ocr import PaddleOcrEngine
from .pipeline import MulkiyaVinPipeline, run_hybrid_baseline, run_ocr_baseline, run_qwen_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="UAE Mulkiya VIN tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Audit labels.xlsx against image files.")
    audit_parser.add_argument("--labels", default="data/labels.xlsx", help="Path to labels Excel file.")
    audit_parser.add_argument("--images", default="data/images", help="Path to image directory.")
    audit_parser.add_argument("--output", default="outputs", help="Directory for audit CSV reports.")

    cleanup_parser = subparsers.add_parser("cleanup-dataset", help="Move unlabelled images out of the image folder, then rerun the audit.")
    cleanup_parser.add_argument("--labels", default="data/labels.xlsx", help="Path to labels Excel file.")
    cleanup_parser.add_argument("--images", default="data/images", help="Path to image directory.")
    cleanup_parser.add_argument("--output", default="outputs", help="Directory for cleanup and audit outputs.")
    cleanup_parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt and move files immediately.")

    baseline_parser = subparsers.add_parser("ocr-baseline", help="Run the OCR baseline over labels.xlsx.")
    baseline_parser.add_argument("--labels", default="data/labels.xlsx", help="Path to labels Excel file.")
    baseline_parser.add_argument("--images", default="data/images", help="Path to image directory.")
    baseline_parser.add_argument("--output", default="outputs", help="Directory for OCR baseline CSV reports.")
    baseline_parser.add_argument("--ocr-engine", default="easyocr", choices=["easyocr", "rapidocr"], help="Local OCR engine to use.")

    qwen_parser = subparsers.add_parser("qwen-baseline", help="Run the Qwen VLM baseline over labels.xlsx.")
    qwen_parser.add_argument("--labels", default="data/labels.xlsx", help="Path to labels Excel file.")
    qwen_parser.add_argument("--images", default="data/images", help="Path to image directory.")
    qwen_parser.add_argument("--output", default="outputs", help="Directory for Qwen baseline CSV reports.")
    qwen_parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct", help="Qwen model name.")
    qwen_parser.add_argument("--limit", type=int, help="Only process the first N labelled images.")
    qwen_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="Device mode for Qwen inference.")

    hybrid_parser = subparsers.add_parser("hybrid-baseline", help="Run the staged OCR-first hybrid baseline over labels.xlsx.")
    hybrid_parser.add_argument("--labels", default="data/labels.xlsx", help="Path to labels Excel file.")
    hybrid_parser.add_argument("--images", default="data/images", help="Path to image directory.")
    hybrid_parser.add_argument("--output", default="outputs", help="Directory for hybrid baseline CSV reports.")
    hybrid_parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct", help="Qwen model name.")
    hybrid_parser.add_argument("--limit", type=int, help="Only process the first N labelled images.")
    hybrid_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="Device mode for Qwen inference.")
    hybrid_parser.add_argument("--ocr-engine", default="rapidocr", choices=["easyocr", "rapidocr"], help="Local OCR engine to use.")
    hybrid_parser.add_argument("--mode", default="conservative", choices=["conservative", "strict"], help="Hybrid routing mode. Conservative keeps any OCR-baseline prediction and only falls back to Qwen when OCR returns no prediction.")

    prepare_dataset_parser = subparsers.add_parser("prepare-dataset", help="Audit, clean, split, and export dataset files for fine-tuning preparation.")
    prepare_dataset_parser.add_argument("--labels", default="data/labels.xlsx", help="Path to labels Excel file.")
    prepare_dataset_parser.add_argument("--images-dir", default="data/images", help="Path to image directory.")
    prepare_dataset_parser.add_argument("--output-dir", default="outputs/dataset", help="Directory for dataset preparation outputs.")
    prepare_dataset_parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio.")
    prepare_dataset_parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    prepare_dataset_parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    prepare_dataset_parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting.")

    show_failures_parser = subparsers.add_parser("show-qwen-failures", help="Print existing Qwen failures from CSV.")
    show_failures_parser.add_argument("--output", default="outputs", help="Directory containing Qwen output CSV files.")
    show_failures_parser.add_argument("--limit", type=int, default=10, help="Number of failures to print.")

    extract_parser = subparsers.add_parser("extract", help="Run the extraction pipeline for one image.")
    extract_parser.add_argument("image", type=Path, help="Path to a JPG/PNG Mulkiya image.")
    extract_parser.add_argument("--database", type=Path, help="Excel or CSV vehicle database/labels file.")
    extract_parser.add_argument("--vin-column", default="VIN Number", help="VIN column name in the database file.")
    extract_parser.add_argument("--audit-log", type=Path, default=Path("logs/audit.jsonl"), help="JSONL audit log path.")
    extract_parser.add_argument("--no-ocr", action="store_true", help="Skip OCR. Useful for integration tests.")
    args = parser.parse_args()

    if args.command == "audit":
        summary = audit_dataset(
            labels_path=args.labels,
            images_dir=args.images,
            output_dir=args.output,
        )
        print("Dataset audit complete")
        print(f"Total Excel rows: {summary['total_excel_rows']}")
        print(f"Total images: {summary['total_images']}")
        print(f"Missing images: {summary['missing_images_count']}")
        print(f"Unlabelled images: {summary['unlabelled_images_count']}")
        print(f"Duplicate filenames: {summary['duplicate_filename_count']}")
        print(f"Duplicate chassis numbers: {summary['duplicate_chassis_count']}")
        print(f"Invalid chassis numbers: {summary['invalid_chassis_count']}")
        print(f"Reports written to: {Path(args.output)}")
        return

    if args.command == "cleanup-dataset":
        cleanup_summary = cleanup_unlabelled_images(
            labels_path=args.labels,
            images_dir=args.images,
            output_dir=args.output,
        )
        print(f"Images in folder: {cleanup_summary['images_in_folder']}")
        print(f"Images in Excel: {cleanup_summary['images_in_excel']}")
        print(f"Unlabelled images: {cleanup_summary['unlabelled_images_count']}")

        if cleanup_summary["unlabelled_images_count"] == 0:
            print("No unlabelled images found.")
            updated_summary = audit_dataset(
                labels_path=args.labels,
                images_dir=args.images,
                output_dir=args.output,
            )
            print("Updated dataset summary:")
            print(f"Total Excel rows: {updated_summary['total_excel_rows']}")
            print(f"Total images: {updated_summary['total_images']}")
            print(f"Missing images: {updated_summary['missing_images_count']}")
            print(f"Unlabelled images: {updated_summary['unlabelled_images_count']}")
            print(f"Duplicate filenames: {updated_summary['duplicate_filename_count']}")
            print(f"Duplicate chassis numbers: {updated_summary['duplicate_chassis_count']}")
            print(f"Invalid chassis numbers: {updated_summary['invalid_chassis_count']}")
            return

        should_continue = args.yes
        if not should_continue:
            prompt = (
                f"Move {cleanup_summary['unlabelled_images_count']} unlabelled images "
                "to outputs/deleted_unlabelled? (y/n) "
            )
            should_continue = input(prompt).strip().lower() == "y"

        if not should_continue:
            print("Cleanup cancelled.")
            return

        moved_count = move_unlabelled_images(cleanup_summary)
        print(f"Moved {moved_count} unlabelled images.")

        updated_summary = audit_dataset(
            labels_path=args.labels,
            images_dir=args.images,
            output_dir=args.output,
        )
        print("Updated dataset summary:")
        print(f"Total Excel rows: {updated_summary['total_excel_rows']}")
        print(f"Total images: {updated_summary['total_images']}")
        print(f"Missing images: {updated_summary['missing_images_count']}")
        print(f"Unlabelled images: {updated_summary['unlabelled_images_count']}")
        print(f"Duplicate filenames: {updated_summary['duplicate_filename_count']}")
        print(f"Duplicate chassis numbers: {updated_summary['duplicate_chassis_count']}")
        print(f"Invalid chassis numbers: {updated_summary['invalid_chassis_count']}")
        return

    if args.command == "ocr-baseline":
        summary = run_ocr_baseline(
            labels_path=args.labels,
            images_dir=args.images,
            output_dir=args.output,
            ocr_engine_name=args.ocr_engine,
        )
        print("Cleared previous OCR outputs.")
        print(f"Total images: {summary['total_images']}")
        print(f"Correct: {summary['correct_predictions']}")
        print(f"Failed: {summary['failed_predictions']}")
        print(f"Accuracy: {summary['accuracy']:.4f}")
        return

    if args.command == "qwen-baseline":
        summary = run_qwen_baseline(
            labels_path=args.labels,
            images_dir=args.images,
            output_dir=args.output,
            model_name=args.model,
            limit=args.limit,
            device=args.device,
        )
        print(f"Total images: {summary['total_images']}")
        print(f"Correct: {summary['correct_predictions']}")
        print(f"Failed: {summary['failed_predictions']}")
        print(f"Accuracy: {summary['accuracy']:.4f}")
        print(f"Parse success: {summary['parse_success_count']}")
        print(f"Parse failed: {summary['parse_failure_count']}")
        failure_reason_counts = summary.get("failure_reason_counts", {})
        print("Top failure reasons:")
        print(f"- {failure_reason_counts.get('parse_failure', 0)} parse failures")
        print(f"- {failure_reason_counts.get('wrong_character_substitution', 0)} wrong substitutions")
        print(f"- {failure_reason_counts.get('truncated_prediction', 0)} truncations")
        print(f"- {failure_reason_counts.get('invalid_chassis_format', 0)} invalid formats")
        return

    if args.command == "hybrid-baseline":
        summary = run_hybrid_baseline(
            labels_path=args.labels,
            images_dir=args.images,
            output_dir=args.output,
            model_name=args.model,
            limit=args.limit,
            device=args.device,
            ocr_engine_name=args.ocr_engine,
            mode=args.mode,
        )
        print(f"Mode: {summary['mode']}")
        print(f"Total images: {summary['total_images']}")
        print(f"Correct: {summary['correct_predictions']}")
        print(f"Failed: {summary['failed_predictions']}")
        print(f"Accuracy: {summary['accuracy']:.4f}")
        print(f"RapidOCR candidates found: {summary['rapidocr_candidates_found_count']}")
        print(f"RapidOCR accepted: {summary['rapidocr_accepted_count']}")
        print(f"RapidOCR rejected as suspicious: {summary['rapidocr_rejected_suspicious_count']}")
        print(f"Needs review: {summary['needs_review_count']}")
        print(f"Qwen fallbacks: {summary['qwen_fallback_count']}")
        print(f"Qwen saved: {summary['qwen_saved_count']}")
        print(f"Qwen used: {summary['qwen_used_count']}")
        print(f"Crop Qwen used: {summary['crop_qwen_used_count']}")
        print(f"Retry Qwen used: {summary['retry_qwen_used_count']}")
        print(f"Average runtime: {summary['average_runtime_seconds']:.4f}")
        print(f"Total runtime: {summary['total_runtime_seconds']:.4f}")
        return

    if args.command == "prepare-dataset":
        summary = prepare_dataset(
            labels_path=args.labels,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        print("Dataset preparation complete")
        print(f"Total Excel rows: {summary['total_excel_rows']}")
        print(f"Total image files: {summary['total_image_files']}")
        print(f"Supported image files: {summary['supported_image_files']}")
        print(f"Missing images: {summary['missing_images_count']}")
        print(f"Unlabelled images: {summary['unlabelled_images_count']}")
        print(f"Duplicate filenames: {summary['duplicate_filename_count']}")
        print(f"Duplicate chassis numbers: {summary['duplicate_chassis_count']}")
        print(f"Invalid chassis numbers: {summary['invalid_chassis_count']}")
        print(f"Unreadable images: {summary['unreadable_images_count']}")
        print(f"Unsupported files: {summary['unsupported_file_count']}")
        print(f"Clean rows: {summary['clean_rows_count']}")
        print(f"Train: {summary['train_count']}")
        print(f"Validation: {summary['val_count']}")
        print(f"Test: {summary['test_count']}")
        print(f"Outputs written to: {Path(args.output_dir)}")
        return

    if args.command == "show-qwen-failures":
        failure_path = Path(args.output) / "qwen_failure_analysis.csv"
        if not failure_path.exists():
            raise FileNotFoundError(f"Qwen failure analysis file not found: {failure_path}")

        with failure_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)[: args.limit]

        for index, row in enumerate(rows, start=1):
            print(f"[{index}] {row.get('filename', '')}")
            print(f"ground_truth_chassis_no: {row.get('ground_truth_chassis_no', '')}")
            print(f"predicted_chassis_no: {row.get('predicted_chassis_no', '')}")
            print(f"qwen_raw_output: {row.get('qwen_raw_output', '')}")
            print(f"edit_distance: {row.get('edit_distance', '')}")
            print(f"failure_reason: {row.get('failure_reason', '')}")
            print(f"possible_wrong_field: {row.get('possible_wrong_field', '')}")
            if index != len(rows):
                print()
        return

    database = (
        DataFrameVehicleDatabase.from_file(args.database, vin_column=args.vin_column)
        if args.database
        else EmptyVehicleDatabase()
    )
    ocr_engine = None if args.no_ocr else PaddleOcrEngine()
    pipeline = MulkiyaVinPipeline(
        ocr_engine=ocr_engine,
        vehicle_database=database,
        audit_logger=JsonlAuditLogger(args.audit_log),
    )
    result = pipeline.process(args.image)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
