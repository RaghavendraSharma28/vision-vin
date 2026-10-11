"""
Minimal SmolVLM2 GPU smoke test.

Tests model load → processor → forward pass → loss → backward pass without the
Trainer or dataset.  Run this BEFORE full training to confirm that the crash is
not in the model or CUDA itself.

If this passes but full training crashes, the issue is in Trainer / Accelerate /
LoRA setup — not in the model or CUDA kernel.

If this crashes, the issue is in the model forward/backward path — usually the
attention implementation.  Try --attn-impl eager (safest) and --attn-impl sdpa.

Usage:
    python scripts/smoke_test_vlm_gpu.py
    python scripts/smoke_test_vlm_gpu.py --attn-impl eager
    python scripts/smoke_test_vlm_gpu.py --attn-impl sdpa
    python scripts/smoke_test_vlm_gpu.py --dtype fp32 --attn-impl eager
"""
from __future__ import annotations

import argparse
import sys
import traceback


DEFAULT_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
PROMPT = "Extract the chassis number from this image. Return only valid JSON."


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SmolVLM2 GPU smoke test (no dataset needed).")
    p.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name or local path.  Default: {DEFAULT_MODEL}",
    )
    p.add_argument(
        "--attn-impl",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
        help="Attention implementation to test.  Default: eager (safest on Windows).",
    )
    p.add_argument(
        "--dtype",
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
        help="Model dtype.  Default: fp16.",
    )
    return p


def _step(label: str, fn):
    """Run fn(), print PASS or FAIL with a traceback on failure.

    Returns (result, success_bool).
    """
    print(f"  {label} ...", end=" ", flush=True)
    try:
        result = fn()
        print("PASS")
        return result, True
    except Exception as exc:
        print(f"FAIL")
        print(f"    {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None, False


def main() -> None:
    args = _build_parser().parse_args()
    sep = "=" * 64

    print(sep)
    print("  SmolVLM2 GPU SMOKE TEST")
    print(sep)
    print(f"  model     : {args.model_name}")
    print(f"  attn_impl : {args.attn_impl}")
    print(f"  dtype     : {args.dtype}")

    import torch

    if not torch.cuda.is_available():
        print("\nCUDA is not available.  This smoke test requires a GPU.")
        print("Run 'python scripts/gpu_diagnostics.py' for details.")
        sys.exit(1)

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]
    device = torch.device("cuda")

    gpu_props = torch.cuda.get_device_properties(0)
    print(f"\n  GPU       : {gpu_props.name}")
    print(f"  VRAM      : {gpu_props.total_memory / 1024**3:.1f} GB")
    print(f"  Compute   : {gpu_props.major}.{gpu_props.minor}")
    print()

    # ── Step 1: processor ─────────────────────────────────────────
    processor = None

    def _load_processor():
        nonlocal processor
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    _, ok = _step("1. Load processor", _load_processor)
    if not ok:
        sys.exit(1)

    # ── Step 2: model ─────────────────────────────────────────────
    model = None

    def _load_model():
        nonlocal model
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=None,
            attn_implementation=args.attn_impl,
        )
        model = model.to(device)
        model.config.use_cache = False
        model.enable_input_require_grads()

    _, ok = _step("2. Load model", _load_model)
    if not ok:
        sys.exit(1)

    vram_after_load = torch.cuda.memory_allocated() / 1024**3
    print(f"     VRAM after load : {vram_after_load:.2f} GB")

    # ── Step 3: build processor inputs ────────────────────────────
    inputs: dict | None = None

    def _build_inputs():
        nonlocal inputs
        from PIL import Image

        # Synthetic image — no dataset dependency
        image = Image.new("RGB", (256, 256), color=(0, 100, 180))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        batch = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in batch.items()}

    _, ok = _step("3. Build processor inputs", _build_inputs)
    if not ok:
        sys.exit(1)

    # ── Step 4: forward (no_grad) ─────────────────────────────────
    def _forward_no_grad():
        torch.cuda.synchronize()
        with torch.no_grad():
            out = model(**inputs)
        torch.cuda.synchronize()
        return out.logits

    _, ok = _step("4. Forward pass (no_grad)", _forward_no_grad)
    if not ok:
        sys.exit(1)

    # ── Step 5: forward with loss ─────────────────────────────────
    loss_tensor = None

    def _forward_with_loss():
        nonlocal loss_tensor
        labels = inputs["input_ids"].clone()
        # Predict only the last non-padding token to keep it simple
        labels[:, :-1] = -100
        labels[inputs["attention_mask"] == 0] = -100

        torch.cuda.synchronize()
        out = model(**inputs, labels=labels)
        torch.cuda.synchronize()
        loss_tensor = out.loss
        return loss_tensor

    loss_obj, ok = _step("5. Forward pass (with loss)", _forward_with_loss)
    if not ok:
        sys.exit(1)

    # ── Step 6: backward ──────────────────────────────────────────
    def _backward():
        torch.cuda.synchronize()
        loss_obj.backward()
        torch.cuda.synchronize()

    _, ok = _step("6. Backward pass", _backward)
    if not ok:
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────
    vram_peak = torch.cuda.max_memory_allocated() / 1024**3
    print()
    print(sep)
    print("  RESULT: ALL STEPS PASSED")
    print(f"  Peak VRAM  : {vram_peak:.2f} GB")
    print(f"  Loss value : {loss_obj.item():.4f}")
    print()
    print("  The VLM forward and backward passes work with this attn_impl.")
    print("  If full training still crashes, the issue is Trainer / Accelerate")
    print("  config, LoRA setup, or the data collator — not the model itself.")
    print()
    print("  Next step:")
    print(f"    python scripts/finetune_smolvlm2.py \\")
    print(f"        --config configs/finetune_smolvlm2.yaml \\")
    print(f"        --attention-implementation {args.attn_impl} \\")
    print(f"        --max-samples 1 --no-lora")
    print(sep)


if __name__ == "__main__":
    main()
