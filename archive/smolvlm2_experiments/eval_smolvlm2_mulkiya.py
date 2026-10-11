from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "Extract the chassis number, model year, origin, and vehicle description "
    "from this UAE Mulkiya image. Return only valid JSON."
)


def _require_dependency(module_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing required package '{module_name}'. Install it with: {install_hint}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned SmolVLM2 model on the Mulkiya test JSONL."
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Path to a fine-tuned model directory or LoRA adapter directory. "
            "If omitted, --model-name is loaded directly from HuggingFace Hub (zero-shot eval)."
        ),
    )
    parser.add_argument(
        "--test-file",
        default="outputs/dataset/finetune_test.jsonl",
        help="Test JSONL file (from prepare-dataset).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/smolvlm2_eval",
        help="Directory to write evaluation CSVs.",
    )
    parser.add_argument(
        "--model-name",
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        help=(
            "HuggingFace Hub model name. "
            "Used as the base when --model is a LoRA adapter, "
            "or as the standalone model when --model is omitted (zero-shot)."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit the number of test samples (for quick sanity checks).",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=512,
        help="Resize images so the longest side is at most this many pixels.",
    )
    parser.add_argument(
        "--baseline-csv",
        default="outputs/ocr_baseline_results.csv",
        help="RapidOCR baseline results CSV to compare against.",
    )
    return parser


def load_test_data(test_file: str, max_samples: int | None = None) -> list[dict]:
    path = Path(test_file)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")

    rows: list[dict] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            image_path = Path(str(row.get("image", "")))
            if not image_path.exists():
                print(f"WARNING: Skipping missing image: {image_path}")
                skipped += 1
                continue
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break

    if skipped:
        print(f"Skipped {skipped} rows with missing images.")
    return rows


def _list_available_models(search_dir: str = "outputs/models") -> list[str]:
    """Return paths of sub-directories under search_dir, sorted alphabetically."""
    search_path = Path(search_dir)
    if not search_path.exists():
        return []
    return [str(d) for d in sorted(search_path.iterdir()) if d.is_dir()]


def _print_model_not_found_error(model_path: str) -> None:
    print(f"ERROR: Model directory not found: {model_path}")
    print()
    print("No trained model found. Run training first:")
    print("  python scripts/finetune_smolvlm2.py --config configs/finetune_smolvlm2.yaml")
    print()
    available = _list_available_models()
    if available:
        print("Available model folders under outputs/models/:")
        for folder in available:
            print(f"  {folder}")
    else:
        print("No model folders found under outputs/models/")
    print()
    print("To evaluate the base model without fine-tuning (zero-shot), omit --model:")
    print(
        "  python scripts/eval_smolvlm2_mulkiya.py"
        " --model-name HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        " --test-file outputs/dataset/finetune_test.jsonl"
    )


def load_model_and_processor(model_path: str | None, base_model_name: str, dtype):
    """Load model and processor from a local fine-tuned directory or HuggingFace Hub.

    When model_path is None, base_model_name is loaded directly from the Hub
    for zero-shot (base model) evaluation.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if model_path is None:
        # Zero-shot / base-model evaluation — load directly from the Hub
        processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        return model, processor

    model_path_obj = Path(model_path)
    processor = AutoProcessor.from_pretrained(str(model_path_obj), trust_remote_code=True)

    if (model_path_obj / "adapter_config.json").exists():
        # LoRA adapter — load base model first, then apply adapter weights
        _require_dependency("peft", "pip install peft")
        from peft import PeftModel

        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(model_path_obj))
    else:
        # Full fine-tuned model saved with trainer.save_model()
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_path_obj),
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    return model, processor


def _resize_image(image, max_size: int):
    width, height = image.size
    longest = max(width, height)
    if longest <= max_size:
        return image.copy()
    scale = max_size / float(longest)
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))))


def run_inference(model, processor, image, device: str, max_new_tokens: int) -> str:
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": DEFAULT_PROMPT},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[prompt_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)

    prompt_len = inputs["input_ids"].shape[1]
    new_ids = generated[:, prompt_len:]
    return processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def parse_json_output(raw_output: str) -> dict[str, Any]:
    """Try to extract a JSON dict from the model output."""
    raw = raw_output.strip()

    # Fenced code block
    if "```" in raw:
        for block in raw.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                result = json.loads(block)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

    # Direct parse
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Find the first { ... } pair
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(raw[start : end + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return {}


def _normalize_chassis(value: str | None) -> str:
    if not value:
        return ""
    return "".join(c for c in str(value).upper() if c.isalnum())


def load_baseline_by_filename(baseline_csv: str) -> dict[str, dict]:
    path = Path(baseline_csv)
    if not path.exists():
        return {}
    result: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            filename = str(row.get("filename", "") or "").strip()
            if filename:
                result[filename] = row
    return result


def main() -> None:
    args = build_parser().parse_args()

    _require_dependency("torch", "pip install torch torchvision")
    _require_dependency("transformers", "pip install transformers sentencepiece")
    pil_image = _require_dependency("PIL.Image", "pip install pillow")

    import torch

    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        print(f"GPU: {props.name}")
        print(f"VRAM: {round(props.total_memory / (1024**3), 2)} GB")

    device = "cuda" if cuda_available else "cpu"
    dtype = (
        torch.bfloat16
        if cuda_available
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
        else (torch.float16 if cuda_available else torch.float32)
    )

    print(f"Loading test data from: {args.test_file}")
    test_rows = load_test_data(args.test_file, args.max_samples)
    print(f"Loaded {len(test_rows)} test samples")
    if not test_rows:
        raise RuntimeError("No test samples found. Check that the JSONL file and image paths exist.")

    # Validate the model path before doing anything expensive
    if args.model is not None and not Path(args.model).exists():
        _print_model_not_found_error(args.model)
        sys.exit(1)

    if args.model is not None:
        print(f"Loading fine-tuned model from: {args.model}")
    else:
        print(f"No --model path given. Evaluating base model (zero-shot): {args.model_name}")

    model, processor = load_model_and_processor(args.model, args.model_name, dtype)
    if cuda_available:
        model = model.to(device)
    model.eval()

    baseline_by_filename = load_baseline_by_filename(args.baseline_csv)
    if baseline_by_filename:
        print(f"Loaded baseline results for {len(baseline_by_filename)} images.")
    else:
        print(f"No baseline CSV found at: {args.baseline_csv} (comparison will be skipped).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict] = []
    chassis_correct = 0
    json_parse_ok = 0
    model_year_correct = 0
    origin_correct = 0
    vd_correct = 0
    total_runtime = 0.0
    total = len(test_rows)

    for index, row in enumerate(test_rows, start=1):
        image_path = Path(row["image"])
        filename = image_path.name

        # Ground truth comes from the assistant message in the JSONL
        gt_json: dict[str, Any] = {}
        messages = row.get("messages", [])
        if len(messages) >= 2:
            try:
                gt_json = json.loads(str(messages[-1].get("content", "{}") or "{}"))
            except json.JSONDecodeError:
                pass

        gt_chassis = _normalize_chassis(gt_json.get("chassis_no"))
        gt_year = str(gt_json.get("model_year") or "").strip()
        gt_origin = str(gt_json.get("origin") or "").strip().upper()
        gt_vd = str(gt_json.get("vehicle_description") or "").strip().upper()

        print(f"[{index}/{total}] {filename}")
        t0 = time.perf_counter()

        with pil_image.open(image_path) as img:
            rgb = _resize_image(img.convert("RGB"), args.max_image_size)

        raw_output = run_inference(model, processor, rgb, device, args.max_new_tokens)
        runtime = round(time.perf_counter() - t0, 4)
        total_runtime += runtime

        parsed = parse_json_output(raw_output)
        parse_ok = bool(parsed)

        pred_chassis = _normalize_chassis(parsed.get("chassis_no"))
        pred_year = str(parsed.get("model_year") or "").strip()
        pred_origin = str(parsed.get("origin") or "").strip().upper()
        pred_vd = str(parsed.get("vehicle_description") or "").strip().upper()

        chassis_match = bool(gt_chassis and pred_chassis == gt_chassis)
        year_match = pred_year == gt_year
        origin_match = pred_origin == gt_origin
        vd_match = pred_vd == gt_vd

        if parse_ok:
            json_parse_ok += 1
        if chassis_match:
            chassis_correct += 1
        if year_match:
            model_year_correct += 1
        if origin_match:
            origin_correct += 1
        if vd_match:
            vd_correct += 1

        baseline_row = baseline_by_filename.get(filename, {})
        baseline_chassis = _normalize_chassis(
            baseline_row.get("predicted_chassis_no") or baseline_row.get("corrected_predicted_chassis_no")
        )
        baseline_correct = bool(gt_chassis and baseline_chassis == gt_chassis)

        result_rows.append(
            {
                "filename": filename,
                "ground_truth_chassis_no": gt_chassis,
                "predicted_chassis_no": pred_chassis,
                "chassis_exact_match": chassis_match,
                "ground_truth_model_year": gt_year,
                "predicted_model_year": pred_year,
                "model_year_match": year_match,
                "ground_truth_origin": gt_origin,
                "predicted_origin": pred_origin,
                "origin_match": origin_match,
                "ground_truth_vehicle_description": gt_vd,
                "predicted_vehicle_description": pred_vd,
                "vehicle_description_match": vd_match,
                "json_parse_success": parse_ok,
                "raw_output": raw_output,
                "runtime_seconds": runtime,
                "baseline_chassis_no": baseline_chassis,
                "baseline_correct": baseline_correct,
            }
        )

        status = "OK" if chassis_match else "FAIL"
        print(f"  [{status}] GT={gt_chassis} | Pred={pred_chassis} | {runtime:.2f}s")

    avg_runtime = total_runtime / total if total else 0.0

    summary = {
        "total_samples": total,
        "json_parse_success": json_parse_ok,
        "json_parse_rate": round(json_parse_ok / total, 4) if total else 0.0,
        "chassis_exact_matches": chassis_correct,
        "chassis_accuracy": round(chassis_correct / total, 4) if total else 0.0,
        "model_year_exact_matches": model_year_correct,
        "model_year_accuracy": round(model_year_correct / total, 4) if total else 0.0,
        "origin_exact_matches": origin_correct,
        "origin_accuracy": round(origin_correct / total, 4) if total else 0.0,
        "vehicle_description_exact_matches": vd_correct,
        "vehicle_description_accuracy": round(vd_correct / total, 4) if total else 0.0,
        "average_runtime_seconds": round(avg_runtime, 4),
        "total_runtime_seconds": round(total_runtime, 4),
        "model_path": args.model if args.model is not None else args.model_name,
        "eval_mode": "fine_tuned" if args.model is not None else "zero_shot_base",
        "test_file": args.test_file,
    }

    if baseline_by_filename:
        baseline_chassis_correct = sum(1 for r in result_rows if r["baseline_correct"])
        summary["baseline_chassis_correct"] = baseline_chassis_correct
        summary["baseline_chassis_accuracy"] = round(baseline_chassis_correct / total, 4) if total else 0.0

    try:
        import pandas as pd

        pd.DataFrame(result_rows).to_csv(output_dir / "smolvlm2_eval_results.csv", index=False)
        pd.DataFrame([summary]).to_csv(output_dir / "smolvlm2_eval_summary.csv", index=False)
        failures = [r for r in result_rows if not r["chassis_exact_match"]]
        pd.DataFrame(failures).to_csv(output_dir / "smolvlm2_eval_failures.csv", index=False)
    except ImportError:
        # Fall back to csv module if pandas is not installed
        _write_csv(output_dir / "smolvlm2_eval_results.csv", result_rows)
        _write_csv(output_dir / "smolvlm2_eval_summary.csv", [summary])
        _write_csv(
            output_dir / "smolvlm2_eval_failures.csv",
            [r for r in result_rows if not r["chassis_exact_match"]],
        )

    print("\n=== SmolVLM2 Evaluation Results ===")
    print(f"Total samples:            {total}")
    print(f"JSON parse success:       {json_parse_ok}/{total}  ({summary['json_parse_rate']*100:.1f}%)")
    print(f"Chassis accuracy:         {chassis_correct}/{total}  ({summary['chassis_accuracy']*100:.1f}%)")
    print(f"Model year accuracy:      {model_year_correct}/{total}  ({summary['model_year_accuracy']*100:.1f}%)")
    print(f"Origin accuracy:          {origin_correct}/{total}  ({summary['origin_accuracy']*100:.1f}%)")
    print(f"Vehicle desc accuracy:    {vd_correct}/{total}  ({summary['vehicle_description_accuracy']*100:.1f}%)")
    print(f"Average runtime:          {avg_runtime:.2f}s / image")
    if "baseline_chassis_accuracy" in summary:
        print(
            f"RapidOCR baseline chassis:{summary['baseline_chassis_correct']}/{total}"
            f"  ({summary['baseline_chassis_accuracy']*100:.1f}%)"
        )
    print(f"\nResults saved to: {output_dir}")

    if chassis_correct < total:
        print("\nFailure examples (first 5):")
        failures = [r for r in result_rows if not r["chassis_exact_match"]]
        for fail in failures[:5]:
            print(f"  {fail['filename']}: GT={fail['ground_truth_chassis_no']} | Pred={fail['predicted_chassis_no']}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
