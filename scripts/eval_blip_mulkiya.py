"""
Evaluate a fine-tuned (or zero-shot base) BLIP model on the UAE Mulkiya test set.

Metrics reported:
  - JSON parse rate        — % of outputs that are valid JSON
  - Chassis exact accuracy — case-insensitive exact match
  - Model year accuracy    — exact match
  - Origin accuracy        — exact match
  - Vehicle desc accuracy  — exact match
  - Runtime per image      — seconds

Baseline for comparison: RapidOCR production pipeline = 109/110 (99.09%) chassis accuracy.

Usage:
    # Fine-tuned model
    python scripts\\eval_blip_mulkiya.py ^
        --model outputs\\models\\blip_mulkiya ^
        --test-file outputs\\dataset\\finetune_test.jsonl

    # Zero-shot base model (no training required)
    python scripts\\eval_blip_mulkiya.py ^
        --model-name Salesforce/blip-image-captioning-base ^
        --test-file outputs\\dataset\\finetune_test.jsonl
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

BASE_MODEL_NAME = "Salesforce/blip-image-captioning-base"
DEFAULT_PROMPT = (
    "Extract the chassis number, model year, origin, and vehicle description "
    "from this UAE Mulkiya image. Return only valid JSON."
)
RAPIDOCR_BASELINE = {"chassis_accuracy": 0.9909, "correct": 109, "total": 110}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_dependency(module_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing required package '{module_name}'. Install it with: {install_hint}"
        ) from exc


def _list_available_models(search_dir: str = "outputs/models") -> list[str]:
    base = Path(search_dir)
    if not base.exists():
        return []
    return [str(p) for p in sorted(base.iterdir()) if p.is_dir()]


def _print_model_not_found_error(model_path: str) -> None:
    print(f"\nERROR: Model not found at: {model_path}")
    print()
    available = _list_available_models()
    if available:
        print("Available models in outputs/models/:")
        for m in available:
            print(f"  {m}")
    else:
        print("No trained models found in outputs/models/")
    print()
    print("To train BLIP first:")
    print("  python scripts\\finetune_blip_mulkiya.py --config configs\\finetune_blip_mulkiya.yaml")
    print()
    print("To evaluate the zero-shot base model (no training required):")
    print("  python scripts\\eval_blip_mulkiya.py \\")
    print(f"      --model-name {BASE_MODEL_NAME} \\")
    print("      --test-file outputs\\dataset\\finetune_test.jsonl")


def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from raw generated text."""
    text = text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: find first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 3: extract key:value pairs with regex
    chassis = re.search(r'"?chassis_no"?\s*:\s*"([^"]*)"', text)
    year = re.search(r'"?model_year"?\s*:\s*"([^"]*)"', text)
    origin = re.search(r'"?origin"?\s*:\s*"([^"]*)"', text)
    desc = re.search(r'"?vehicle_description"?\s*:\s*"([^"]*)"', text)

    if any([chassis, year, origin, desc]):
        return {
            "chassis_no": chassis.group(1) if chassis else "",
            "model_year": year.group(1) if year else "",
            "origin": origin.group(1) if origin else "",
            "vehicle_description": desc.group(1) if desc else "",
        }

    return None


def _normalize(s: str) -> str:
    return str(s).strip().upper()


def _exact_match(pred: str, gt: str) -> bool:
    return _normalize(pred) == _normalize(gt)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_and_processor(
    model_path: str | None,
    base_model_name: str,
    torch_dtype,
    device: str,
):
    _require_dependency("transformers", "pip install transformers")
    from transformers import BlipForConditionalGeneration, BlipProcessor

    if model_path is None:
        # Zero-shot evaluation using the base model directly
        print(f"Loading base model (zero-shot): {base_model_name}")
        model = BlipForConditionalGeneration.from_pretrained(
            base_model_name, torch_dtype=torch_dtype
        )
        processor = BlipProcessor.from_pretrained(base_model_name)
        model = model.to(device)
        return model, processor

    model_dir = Path(model_path)

    if (model_dir / "adapter_config.json").exists():
        # LoRA adapter — load base model then merge adapter
        print(f"Loading LoRA adapter from: {model_path}")
        _require_dependency("peft", "pip install peft")
        from peft import PeftModel
        base = BlipForConditionalGeneration.from_pretrained(
            base_model_name, torch_dtype=torch_dtype
        )
        model = PeftModel.from_pretrained(base, str(model_dir))
        # Try to load processor from the adapter dir, fall back to base
        proc_source = str(model_dir) if (model_dir / "tokenizer_config.json").exists() else base_model_name
        processor = BlipProcessor.from_pretrained(proc_source)
    else:
        # Full fine-tuned model
        print(f"Loading fine-tuned model from: {model_path}")
        model = BlipForConditionalGeneration.from_pretrained(
            str(model_dir), torch_dtype=torch_dtype
        )
        proc_source = str(model_dir) if (model_dir / "tokenizer_config.json").exists() else base_model_name
        processor = BlipProcessor.from_pretrained(proc_source)

    model = model.to(device)
    return model, processor


# ── Inference ──────────────────────────────────────────────────────────────────

def _run_inference(
    model,
    processor,
    image,
    device: str,
    max_new_tokens: int = 128,
    prompt_prefix: str = "JSON:",
    num_beams: int = 1,
) -> str:
    """Generate structured JSON from an image using a short text prefix.

    A non-empty prompt_prefix is required: passing zero-length input_ids to
    model.generate() causes a reshape crash on CPU when num_beams > 1.
    'JSON:' is short enough not to pollute the output and still seeds the
    decoder so BLIP knows it should produce structured text.

    num_beams=1 (greedy) avoids beam-search shape issues on CPU.
    Returns an empty string on any generation error rather than crashing.
    """
    import torch

    try:
        inputs = processor(images=image, text=prompt_prefix, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
            )

        return processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    except Exception as exc:
        print(f"    [WARN] generation failed: {type(exc).__name__}: {exc}")
        return ""


# ── Evaluation loop ────────────────────────────────────────────────────────────

def evaluate(
    model_path: str | None,
    base_model_name: str,
    test_file: str,
    output_dir: str,
    device: str,
    max_samples: int | None,
    max_new_tokens: int,
    prompt_prefix: str = "JSON:",
    num_beams: int = 1,
) -> dict[str, Any]:
    import torch
    _require_dependency("PIL.Image", "pip install pillow")
    from PIL import Image

    use_cuda = device.startswith("cuda")
    torch_dtype = torch.float16 if use_cuda else torch.float32

    model, processor = load_model_and_processor(model_path, base_model_name, torch_dtype, device)
    model.eval()

    test_path = Path(test_file)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    rows: list[dict] = []
    with test_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))

    if max_samples is not None:
        rows = rows[:max_samples]

    results: list[dict] = []
    raw_preds: list[dict] = []   # 5-column debug view saved to blip_raw_predictions.csv
    total = len(rows)

    print(f"\nEvaluating {total} samples ...")
    print(f"Device        : {device}")
    print(f"prompt_prefix : {prompt_prefix!r}")
    print(f"num_beams     : {num_beams}")
    if use_cuda:
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print()

    for i, row in enumerate(rows, 1):
        image_path = Path(str(row.get("image", "")))
        messages = row.get("messages", [])
        gt_raw = str(messages[-1].get("content", "{}")) if len(messages) >= 2 else "{}"
        gt = _extract_json(gt_raw) or {}

        if not image_path.exists():
            print(f"  [{i}/{total}] SKIP (missing image): {image_path.name}")
            raw_preds.append({
                "filename": image_path.name,
                "ground_truth_json": gt_raw,
                "raw_generation": "",
                "parsed_json": "",
                "json_parse_success": False,
            })
            results.append({
                "filename": image_path.name,
                "status": "missing_image",
                "generated_raw": "",
                "json_parsed": False,
                "gt_chassis": gt.get("chassis_no", ""),
                "pred_chassis": "",
                "chassis_match": False,
                "gt_model_year": gt.get("model_year", ""),
                "pred_model_year": "",
                "year_match": False,
                "gt_origin": gt.get("origin", ""),
                "pred_origin": "",
                "origin_match": False,
                "gt_vehicle_desc": gt.get("vehicle_description", ""),
                "pred_vehicle_desc": "",
                "desc_match": False,
                "runtime_s": 0.0,
            })
            continue

        with Image.open(image_path) as img:
            pil_image = img.convert("RGB").copy()

        t0 = time.perf_counter()
        generated_text = _run_inference(
            model, processor, pil_image, device,
            max_new_tokens=max_new_tokens,
            prompt_prefix=prompt_prefix,
            num_beams=num_beams,
        )
        runtime_s = time.perf_counter() - t0

        if i <= 5:
            print(f"  [DEBUG {i}] raw_generation: {generated_text!r}")

        pred = _extract_json(generated_text) or {}
        json_parsed = bool(pred)

        raw_preds.append({
            "filename": image_path.name,
            "ground_truth_json": gt_raw,
            "raw_generation": generated_text,
            "parsed_json": json.dumps(pred) if pred else "",
            "json_parse_success": json_parsed,
        })

        pred_chassis = pred.get("chassis_no", "")
        pred_year = pred.get("model_year", "")
        pred_origin = pred.get("origin", "")
        pred_desc = pred.get("vehicle_description", "")

        chassis_match = _exact_match(pred_chassis, gt.get("chassis_no", ""))
        year_match = _exact_match(pred_year, gt.get("model_year", ""))
        origin_match = _exact_match(pred_origin, gt.get("origin", ""))
        desc_match = _exact_match(pred_desc, gt.get("vehicle_description", ""))

        status_sym = "OK" if chassis_match else "!!"
        print(
            f"  [{status_sym}] {i:>3}/{total}  {image_path.name:<30}  "
            f"chassis={'Y' if chassis_match else 'N'}  "
            f"year={'Y' if year_match else 'N'}  "
            f"origin={'Y' if origin_match else 'N'}  "
            f"({runtime_s:.1f}s)"
        )

        results.append({
            "filename": image_path.name,
            "status": "ok",
            "generated_raw": generated_text,
            "json_parsed": json_parsed,
            "gt_chassis": gt.get("chassis_no", ""),
            "pred_chassis": pred_chassis,
            "chassis_match": chassis_match,
            "gt_model_year": gt.get("model_year", ""),
            "pred_model_year": pred_year,
            "year_match": year_match,
            "gt_origin": gt.get("origin", ""),
            "pred_origin": pred_origin,
            "origin_match": origin_match,
            "gt_vehicle_desc": gt.get("vehicle_description", ""),
            "pred_vehicle_desc": pred_desc,
            "desc_match": desc_match,
            "runtime_s": round(runtime_s, 3),
        })

    # Print first 10 raw outputs before computing metrics — useful for diagnosing
    # 0% parse rate without needing to open the CSV.
    generated_preds = [rp for rp in raw_preds if rp["raw_generation"] != ""]
    print("\n" + "=" * 60)
    print(f"  RAW MODEL OUTPUTS (first {min(10, len(generated_preds))} of {len(generated_preds)})")
    print("=" * 60)
    for idx, rp in enumerate(generated_preds[:10], 1):
        print(f"\n[{idx}] {rp['filename']}")
        print(f"  GT  : {rp['ground_truth_json']}")
        print(f"  RAW : {rp['raw_generation']}")
        print(f"  OK  : {rp['json_parse_success']}")
    print("=" * 60)

    # Aggregate metrics
    evaluated = [r for r in results if r["status"] == "ok"]
    n = len(evaluated)

    if n == 0:
        raise RuntimeError("No samples could be evaluated (all images missing?).")

    json_parse_rate = sum(1 for r in evaluated if r["json_parsed"]) / n
    chassis_acc = sum(1 for r in evaluated if r["chassis_match"]) / n
    year_acc = sum(1 for r in evaluated if r["year_match"]) / n
    origin_acc = sum(1 for r in evaluated if r["origin_match"]) / n
    desc_acc = sum(1 for r in evaluated if r["desc_match"]) / n
    avg_runtime = sum(r["runtime_s"] for r in evaluated) / n

    summary = {
        "model": model_path or base_model_name,
        "test_samples_evaluated": n,
        "test_samples_skipped": len(results) - n,
        "json_parse_rate": round(json_parse_rate, 4),
        "chassis_exact_accuracy": round(chassis_acc, 4),
        "model_year_accuracy": round(year_acc, 4),
        "origin_accuracy": round(origin_acc, 4),
        "vehicle_desc_accuracy": round(desc_acc, 4),
        "avg_runtime_per_image_s": round(avg_runtime, 3),
        "rapidocr_baseline_chassis": RAPIDOCR_BASELINE["chassis_accuracy"],
    }

    # Save outputs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results_path = out / "blip_eval_results.csv"
    summary_path = out / "blip_eval_summary.csv"
    failures_path = out / "blip_eval_failures.csv"

    fieldnames = list(results[0].keys()) if results else []
    with results_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    failures = [r for r in evaluated if not r["chassis_match"]]
    with failures_path.open("w", encoding="utf-8", newline="") as fh:
        if failures:
            writer = csv.DictWriter(fh, fieldnames=list(failures[0].keys()))
            writer.writeheader()
            writer.writerows(failures)

    raw_path = out / "blip_raw_predictions.csv"
    raw_fieldnames = ["filename", "ground_truth_json", "raw_generation", "parsed_json", "json_parse_success"]
    with raw_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=raw_fieldnames)
        writer.writeheader()
        writer.writerows(raw_preds)

    return summary


# ── Entry point ────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate fine-tuned BLIP on UAE Mulkiya test set.")
    p.add_argument("--model", default=None,
                   help="Path to fine-tuned model directory.  Omit for zero-shot base eval.")
    p.add_argument("--model-name", default=BASE_MODEL_NAME,
                   help=f"Base model name for zero-shot eval or LoRA base.  Default: {BASE_MODEL_NAME}")
    p.add_argument("--test-file", default="outputs/dataset/finetune_test.jsonl",
                   help="Test JSONL file.")
    p.add_argument("--output-dir", default="outputs/eval",
                   help="Directory to write CSV reports.  Default: outputs/eval")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None,
                   help="Inference device.  Default: auto-detect.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit number of test samples (for quick checks).")
    p.add_argument("--max-new-tokens", type=int, default=128,
                   help="Max new tokens to generate per image.  Default: 128.")
    p.add_argument("--prompt-prefix", default="JSON:",
                   help="Short text prefix passed to the BLIP processor before generation.  "
                        "Must be non-empty to avoid reshape crash on CPU.  Default: 'JSON:'")
    p.add_argument("--num-beams", type=int, default=1,
                   help="Number of beams for generation.  1 = greedy (safe on CPU).  Default: 1.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Validate model existence
    if args.model is not None and not Path(args.model).exists():
        _print_model_not_found_error(args.model)
        sys.exit(1)

    # Resolve device
    if args.device:
        device = args.device
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("BLIP Evaluation — UAE Mulkiya VIN extraction")
    print(f"  Model         : {args.model or args.model_name + ' (zero-shot)'}")
    print(f"  Test file     : {args.test_file}")
    print(f"  Device        : {device}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"  prompt_prefix : {args.prompt_prefix!r}")
    print(f"  num_beams     : {args.num_beams}")

    summary = evaluate(
        model_path=args.model,
        base_model_name=args.model_name,
        test_file=args.test_file,
        output_dir=args.output_dir,
        device=device,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        prompt_prefix=args.prompt_prefix,
        num_beams=args.num_beams,
    )

    print()
    print("=" * 52)
    print("  RESULTS")
    print("=" * 52)
    print(f"  Samples evaluated    : {summary['test_samples_evaluated']}")
    print(f"  JSON parse rate      : {summary['json_parse_rate']:.1%}")
    print(f"  Chassis accuracy     : {summary['chassis_exact_accuracy']:.1%}  "
          f"({int(summary['chassis_exact_accuracy'] * summary['test_samples_evaluated'])}"
          f"/{summary['test_samples_evaluated']})")
    print(f"  Model year accuracy  : {summary['model_year_accuracy']:.1%}")
    print(f"  Origin accuracy      : {summary['origin_accuracy']:.1%}")
    print(f"  Vehicle desc accuracy: {summary['vehicle_desc_accuracy']:.1%}")
    print(f"  Avg runtime/image    : {summary['avg_runtime_per_image_s']:.2f}s")
    print()
    baseline = RAPIDOCR_BASELINE["chassis_accuracy"]
    blip_chassis = summary["chassis_exact_accuracy"]
    delta = blip_chassis - baseline
    delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
    print(f"  RapidOCR baseline    : {baseline:.1%}  (delta: {delta_str})")
    print("=" * 52)
    print()
    print(f"  Reports saved to: {args.output_dir}/")
    print(f"    blip_raw_predictions.csv  ← raw model outputs for inspection")
    print(f"    blip_eval_results.csv")
    print(f"    blip_eval_summary.csv")
    print(f"    blip_eval_failures.csv")


if __name__ == "__main__":
    main()
