"""
Fine-tune Salesforce/blip-image-captioning-base for UAE Mulkiya VIN extraction.

BLIP is the Windows/CUDA-compatible fallback for SmolVLM2, which crashes on
Windows RTX hardware due to VLM-specific CUDA kernels.  BLIP uses only standard
PyTorch attention and is verified to run on RTX 4050 (sm_89) without the
'illegal instruction' errors that affect SmolVLM2.

Target output format (same JSON as SmolVLM2):
    {"chassis_no": "...", "model_year": "...", "origin": "...", "vehicle_description": "..."}

Usage:
    # GPU training (smoke test runs automatically before training starts)
    python scripts\\finetune_blip_mulkiya.py --config configs\\finetune_blip_mulkiya.yaml

    # GPU smoke test only (verify BLIP works on this GPU without loading the dataset)
    python scripts\\finetune_blip_mulkiya.py --config configs\\finetune_blip_mulkiya.yaml --smoke-test

    # CPU smoke test (verify data pipeline and training loop)
    python scripts\\finetune_blip_mulkiya.py --config configs\\finetune_blip_mulkiya.yaml ^
        --device cpu --max-samples 2

    # LoRA fine-tuning (smaller memory footprint, optional)
    python scripts\\finetune_blip_mulkiya.py --config configs\\finetune_blip_mulkiya.yaml ^
        --use-lora
"""
from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "Extract the chassis number, model year, origin, and vehicle description "
    "from this UAE Mulkiya image. Return only valid JSON."
)
BASE_MODEL_NAME = "Salesforce/blip-image-captioning-base"


# ── Dependency helper ──────────────────────────────────────────────────────────

def _require_dependency(module_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing required package '{module_name}'. Install it with: {install_hint}"
        ) from exc


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tune BLIP-base for UAE Mulkiya structured JSON extraction."
    )
    p.add_argument("--config", default="configs/finetune_blip_mulkiya.yaml",
                   help="Path to YAML config.")
    p.add_argument("--train-file", help="Override training JSONL.")
    p.add_argument("--val-file", help="Override validation JSONL.")
    p.add_argument("--output-dir", help="Override output directory.")
    p.add_argument("--model-name", help="Override base model name.")
    p.add_argument("--epochs", type=int, help="Override training epochs.")
    p.add_argument("--batch-size", type=int, help="Override per-device batch size.")
    p.add_argument("--learning-rate", type=float, help="Override learning rate.")
    p.add_argument("--max-samples", type=int, help="Limit train/val rows (smoke tests).")
    p.add_argument("--max-image-size", type=int, help="Longest side in pixels before processor resize.")
    p.add_argument("--gradient-accumulation-steps", type=int,
                   help="Override gradient accumulation steps.")
    p.add_argument("--max-label-length", type=int, help="Max decoder label tokens.")
    p.add_argument("--use-lora", dest="use_lora", action="store_true")
    p.add_argument("--no-lora", dest="use_lora", action="store_false")
    p.set_defaults(use_lora=None)
    p.add_argument("--device", choices=["cuda", "cpu"], default=None,
                   help="Training device.  Default: auto-detect.")
    p.add_argument("--fp32", dest="fp32", action="store_true",
                   help=(
                       "Force fp32 training — disables fp16 and bf16.  "
                       "Recommended for Windows RTX GPU stability when fp16 triggers "
                       "illegal-memory-access errors in causal mask creation."
                   ))
    p.add_argument("--fp16", dest="fp16", action="store_true", help="Force FP16.")
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--bf16", dest="bf16", action="store_true", help="Force BF16.")
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.set_defaults(fp16=None, bf16=None, fp32=False)
    p.add_argument(
        "--resume-from-checkpoint",
        dest="resume_from_checkpoint",
        default=None,
        metavar="CHECKPOINT",
        help="Resume from a checkpoint path or 'latest' to auto-detect the highest checkpoint in output_dir.",
    )
    p.add_argument("--smoke-test", action="store_true",
                   help="Run GPU smoke test then exit without training.")
    return p


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class BlipTrainConfig:
    model_name: str
    train_file: str
    val_file: str
    output_dir: str
    epochs: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    max_image_size: int
    max_label_length: int
    use_lora: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    max_samples: int | None
    device: str
    fp16: bool | None
    bf16: bool | None
    fp32: bool  # True → float32, overrides fp16/bf16
    resume_from_checkpoint: str | None


def load_config(config_path: str, cli_args: argparse.Namespace) -> BlipTrainConfig:
    yaml = _require_dependency("yaml", "pip install pyyaml")
    with Path(config_path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    def pick(name: str, default: Any) -> Any:
        val = getattr(cli_args, name.replace("-", "_"), None)
        if val is not None:
            return val
        return raw.get(name, default)

    use_lora = cli_args.use_lora
    if use_lora is None:
        use_lora = bool(raw.get("use_lora", False))

    device_arg = getattr(cli_args, "device", None)
    device = device_arg if device_arg is not None else str(raw.get("device", "auto"))

    def _tristate(attr: str) -> bool | None:
        val = getattr(cli_args, attr, None)
        if val is not None:
            return val
        r = raw.get(attr)
        return bool(r) if r is not None else None

    return BlipTrainConfig(
        model_name=pick("model_name", BASE_MODEL_NAME),
        train_file=pick("train_file", "outputs/dataset/finetune_train.jsonl"),
        val_file=pick("val_file", "outputs/dataset/finetune_val.jsonl"),
        output_dir=pick("output_dir", "outputs/models/blip_mulkiya"),
        epochs=int(pick("epochs", 5)),
        learning_rate=float(pick("learning_rate", 5e-5)),
        batch_size=int(pick("batch_size", 1)),
        gradient_accumulation_steps=int(pick("gradient_accumulation_steps", 8)),
        max_image_size=int(pick("max_image_size", 384)),
        max_label_length=int(pick("max_label_length", 128)),
        use_lora=bool(use_lora),
        lora_r=int(pick("lora_r", 8)),
        lora_alpha=int(pick("lora_alpha", 16)),
        lora_dropout=float(pick("lora_dropout", 0.05)),
        max_samples=cli_args.max_samples,
        device=device,
        fp16=_tristate("fp16"),
        bf16=_tristate("bf16"),
        fp32=bool(getattr(cli_args, "fp32", False)),
        resume_from_checkpoint=getattr(cli_args, "resume_from_checkpoint", None),
    )


# ── Dataset ────────────────────────────────────────────────────────────────────

class MulkiyaBlipDataset:
    """Reads the same finetune_*.jsonl files used by SmolVLM2 training."""

    def __init__(
        self,
        jsonl_path: str | Path,
        split_name: str,
        max_samples: int | None = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.jsonl_path}")

        self.split_name = split_name
        self.rows: list[dict] = []
        self.skipped_rows: list[dict] = []
        self.requested_count = 0

        with self.jsonl_path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                row["__line_number__"] = lineno
                self.requested_count += 1
                image_path = Path(str(row.get("image", "") or ""))
                if not image_path.exists():
                    self.skipped_rows.append({
                        "filename": image_path.name,
                        "expected_path": str(image_path).replace("\\", "/"),
                        "split": split_name,
                    })
                    continue
                self.rows.append(row)

        if max_samples is not None:
            self.rows = self.rows[:max_samples]
            self.requested_count = min(self.requested_count, max_samples)

    @property
    def loaded_count(self) -> int:
        return len(self.rows)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


# ── Collator ───────────────────────────────────────────────────────────────────

class BlipDataCollator:
    """
    Converts a batch of JSONL rows into BLIP training tensors.

    Teacher-forcing setup for BlipForConditionalGeneration:
        full_ids  = tokenize(target_json)           shape: (B, L)
        input_ids = full_ids[:, :-1]                shape: (B, L-1)  decoder input
        labels    = full_ids[:, 1:],  pad→-100      shape: (B, L-1)  decoder target

    Both input_ids and labels are length L-1, so the cross-entropy loss never
    sees a batch-size / sequence-length mismatch.

    model(pixel_values=..., input_ids=input_ids,
          attention_mask=attn_mask, labels=labels)
    """

    def __init__(
        self,
        processor,
        max_image_size: int = 384,
        max_label_length: int = 128,
    ) -> None:
        self.processor = processor
        self.max_image_size = max_image_size
        self.max_label_length = max_label_length
        self._debug_printed = False  # print first-sample info once per training run

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        from PIL import Image

        images: list = []
        targets: list[str] = []

        for feature in features:
            image_path = Path(feature["image"])
            with Image.open(image_path) as img:
                rgb = self._resize(img.convert("RGB"))

            messages = feature.get("messages", [])
            # The label is the ASSISTANT's JSON response — the last message content.
            # The USER's instruction prompt is intentionally excluded from labels.
            target = str(messages[-1].get("content", "") or "{}") if len(messages) >= 2 else "{}"

            images.append(rgb)
            targets.append(target)

        # Debug: print the first training sample once to confirm labels are JSON
        if not self._debug_printed:
            self._debug_printed = True
            first_feature = features[0]
            first_target = targets[0]
            print(f"\n[DEBUG] First training sample:")
            print(f"  image path        : {first_feature.get('image', '')}")
            print(f"  target_text       : {first_target[:120]}{'...' if len(first_target) > 120 else ''}")
            print(f"  starts with '{{' : {first_target.startswith('{')}")
            if not first_target.startswith("{"):
                print("  WARNING: target does not start with '{{' — check JSONL message order!")
            print()

        # pixel_values — vision encoder input
        image_inputs = self.processor(images=images, return_tensors="pt", padding=True)

        # Tokenize the full target JSON (with special tokens [CLS]…[SEP]).
        # max_length+1 so that after the :-1 / 1: slices we have max_label_length tokens.
        target_enc = self.processor.tokenizer(
            targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_label_length + 1,
            add_special_tokens=True,
        )
        full_ids = target_enc.input_ids          # (B, L)
        full_mask = target_enc.attention_mask    # (B, L)

        # Teacher forcing: decoder sees tokens 0..L-2, must predict tokens 1..L-1
        decoder_input_ids = full_ids[:, :-1].contiguous()       # (B, L-1)
        decoder_attn_mask = full_mask[:, :-1].contiguous()      # (B, L-1)
        labels = full_ids[:, 1:].clone().contiguous()            # (B, L-1)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": image_inputs.pixel_values,
            "input_ids": decoder_input_ids,
            "attention_mask": decoder_attn_mask,
            "labels": labels,
        }

    def _resize(self, image):
        w, h = image.size
        longest = max(w, h)
        if longest <= self.max_image_size:
            return image.copy()
        scale = self.max_image_size / longest
        return image.resize((max(1, int(w * scale)), max(1, int(h * scale))))


# ── GPU helpers ────────────────────────────────────────────────────────────────

def _resolve_dtype(config: BlipTrainConfig, use_cuda: bool, torch_module):
    """Return (torch_dtype, use_fp16, use_bf16) consistent for model load and Trainer.

    Priority order:
      1. --fp32            → float32, fp16=False, bf16=False  (Windows RTX safe mode)
      2. --bf16            → bfloat16
      3. --no-fp16         → float32
      4. auto / --fp16     → float16  (default on CUDA)
      CPU always → float32 regardless of flags.
    """
    if not use_cuda:
        return torch_module.float32, False, False
    if config.fp32:
        return torch_module.float32, False, False
    if config.bf16 is True:
        return torch_module.bfloat16, False, True
    if config.fp16 is False:
        return torch_module.float32, False, False
    # Default: fp16 — safe on BERT-style models, matches BLIP's typical training setup
    return torch_module.float16, True, False


def _blip_gpu_smoke_test(model, processor, device: str) -> None:
    """
    One synthetic forward pass using teacher-forcing.

    Tokenizes the target JSON, slices into decoder_input_ids ([:, :-1]) and
    labels ([:, 1:]) so that both have the same length — no cross_entropy
    batch-size mismatch.

    Raises RuntimeError on failure so the caller can catch and report cleanly.
    Runs automatically before any CUDA training starts.
    """
    import torch
    from PIL import Image

    print("BLIP GPU smoke test ...")

    # pixel_values — solid-colour synthetic image
    image = Image.new("RGB", (256, 256), color=(30, 100, 180))
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    # Target JSON matching the training label format
    target_text = (
        '{"chassis_no":"TEST123456789000","model_year":"2025",'
        '"origin":"JAPAN","vehicle_description":"NISSAN"}'
    )
    target_enc = processor.tokenizer(
        target_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=129,          # +1 so slices yield max 128 tokens
        add_special_tokens=True,
    )
    full_ids  = target_enc.input_ids.to(device)     # (1, L)
    full_mask = target_enc.attention_mask.to(device) # (1, L)

    decoder_input_ids = full_ids[:, :-1].contiguous()    # (1, L-1)
    decoder_attn_mask = full_mask[:, :-1].contiguous()   # (1, L-1)
    labels = full_ids[:, 1:].clone().contiguous()         # (1, L-1)
    labels[labels == processor.tokenizer.pad_token_id] = -100

    print(f"  pixel_values shape      : {tuple(pixel_values.shape)}")
    print(f"  decoder_input_ids shape : {tuple(decoder_input_ids.shape)}")
    print(f"  labels shape            : {tuple(labels.shape)}")

    torch.cuda.synchronize()
    with torch.no_grad():
        out = model(
            pixel_values=pixel_values,
            input_ids=decoder_input_ids,
            attention_mask=decoder_attn_mask,
            labels=labels,
        )
    torch.cuda.synchronize()

    print(f"  loss                    : {out.loss.item():.4f}")
    print("PASS")


# ── Model builder ──────────────────────────────────────────────────────────────

def build_blip_model_and_processor(config: BlipTrainConfig):
    torch = _require_dependency("torch", "pip install torch torchvision")
    _require_dependency("accelerate", "pip install accelerate")
    transformers = _require_dependency("transformers", "pip install transformers")

    from transformers import BlipForConditionalGeneration, BlipProcessor

    # Resolve device
    if config.device == "cpu":
        device = "cpu"
    elif config.device == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: --device cuda requested but CUDA unavailable. Falling back to cpu.")
            device = "cpu"
        else:
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    use_cuda = device == "cuda"
    torch_dtype, use_fp16, use_bf16 = _resolve_dtype(config, use_cuda, torch)

    if use_cuda:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print(f"GPU: {props.name}  CC: {props.major}.{props.minor}  "
              f"VRAM: {props.total_memory / 1024**3:.1f} GB")
    print(f"Device: {device}  dtype: {torch_dtype}  fp16: {use_fp16}  bf16: {use_bf16}")

    processor = BlipProcessor.from_pretrained(config.model_name, trust_remote_code=True)
    model = BlipForConditionalGeneration.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model = model.to(device)

    lora_enabled = False
    if config.use_lora:
        _require_dependency("peft", "pip install peft")
        from peft import LoraConfig, get_peft_model

        target_modules = _find_blip_lora_targets(model)
        if not target_modules:
            print("WARNING: Could not auto-detect LoRA target modules. Defaulting to ['query', 'value'].")
            target_modules = ["query", "value"]

        # No task_type: passing None lets PEFT apply LoRA without task-specific
        # handling, which is correct for BLIP's hybrid ViT + BERT decoder architecture.
        # SEQ_2_SEQ_LM would confuse PEFT because BLIP is not a standard T5/BART model.
        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)

        # BLIP's ViT (vision_model) and BERT decoder (text_decoder) both have
        # nn.Linear layers named 'query' and 'value', so PEFT applies LoRA adapters
        # to both.  Freeze the vision model adapters so only text decoder adapters
        # remain trainable — this is where the language generation learning happens.
        vision_lora_frozen = 0
        for name, param in model.named_parameters():
            if "vision_model" in name and param.requires_grad:
                param.requires_grad = False
                vision_lora_frozen += 1
        if vision_lora_frozen:
            print(f"  Froze {vision_lora_frozen} vision-model LoRA tensors (text decoder only).")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pct = 100.0 * trainable_params / total_params if total_params else 0.0
        print(f"  Trainable params : {trainable_params:,} / {total_params:,} ({pct:.2f}%)")
        if pct > 5.0:
            print(
                f"  WARNING: {pct:.1f}% of params are trainable — higher than expected. "
                "Reduce lora_r or check target_modules if VRAM is tight."
            )
        lora_enabled = True

    return model, processor, torch, lora_enabled, transformers.__version__, device, use_fp16, use_bf16


def _find_blip_lora_targets(model) -> list[str]:
    """Return LoRA target leaf module names for BLIP.

    Both the vision_model (ViT BlipAttention) and text_decoder (BERT
    BlipTextSelfAttention) use nn.Linear layers named 'query' and 'value'.
    We return both names here; the LoRA adapters added to the vision model
    are then frozen immediately after get_peft_model() so that only the text
    decoder adapters are actually trained.
    """
    import torch.nn as nn
    preferred = {"query", "value"}
    found = []
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in preferred and isinstance(module, nn.Linear) and leaf not in found:
            found.append(leaf)
    return found


# ── TrainingArguments builder ──────────────────────────────────────────────────

def _build_training_kwargs(
    training_arguments_class,
    use_cpu: bool = False,
    **base_kwargs,
) -> dict[str, Any]:
    sig = inspect.signature(training_arguments_class.__init__)
    supported = sig.parameters

    kwargs = dict(base_kwargs)

    if "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in supported:
        kwargs["eval_strategy"] = "epoch"

    if "dataloader_num_workers" in supported:
        kwargs.setdefault("dataloader_num_workers", 0)

    if use_cpu:
        if "use_cpu" in supported:
            kwargs["use_cpu"] = True
        elif "no_cuda" in supported:
            kwargs["no_cuda"] = True

    return kwargs


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_missing_image_report(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "expected_path", "split"])
        writer.writeheader()
        writer.writerows(rows)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


# ── Checkpoint resume ──────────────────────────────────────────────────────────

def _resolve_checkpoint(resume_arg: str | None, output_dir: str) -> str | None:
    """Resolve --resume-from-checkpoint to a concrete path, or None to start fresh."""
    if resume_arg is None:
        return None
    if resume_arg != "latest":
        checkpoint_path = Path(resume_arg)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "Remove --resume-from-checkpoint to start a new training run."
            )
        return str(checkpoint_path)
    # Scan output_dir for checkpoint-N directories and return the highest one
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    best_step = -1
    best_path: Path | None = None
    for candidate in output_path.iterdir():
        if candidate.is_dir() and candidate.name.startswith("checkpoint-"):
            try:
                step = int(candidate.name.split("-", 1)[1])
                if step > best_step:
                    best_step = step
                    best_path = candidate
            except (ValueError, IndexError):
                pass
    return str(best_path) if best_path is not None else None


# ── Training ───────────────────────────────────────────────────────────────────

def train(config: BlipTrainConfig) -> dict[str, Any]:
    _require_dependency("PIL.Image", "pip install pillow")
    from transformers import Trainer, TrainingArguments

    model, processor, torch, lora_enabled, transformers_version, device, use_fp16, use_bf16 = (
        build_blip_model_and_processor(config)
    )
    use_cuda = device == "cuda"

    # Smoke test before loading the full dataset — fails fast if the GPU is incompatible.
    if use_cuda:
        try:
            _blip_gpu_smoke_test(model, processor, device)
        except Exception as exc:
            raise RuntimeError(
                f"BLIP GPU smoke test failed on this CUDA environment:\n  {exc}\n\n"
                "This means BLIP's forward pass is not compatible with the current GPU/driver stack.\n"
                "Switch to CPU training: remove --device cuda from your command\n"
                "(CPU is the default — just omit --device entirely)."
            ) from exc

    use_fp32 = not use_fp16 and not use_bf16

    print("\n[BLIP Training Configuration]")
    print(f"  Model          : {config.model_name}")
    print(f"  Device         : {device}")
    print(f"  fp32           : {use_fp32}")
    print(f"  fp16           : {use_fp16}")
    print(f"  bf16           : {use_bf16}")
    print(f"  LoRA           : {lora_enabled}")
    print(f"  Epochs         : {config.epochs}")
    print(f"  Batch size     : {config.batch_size}")
    print(f"  Grad accum     : {config.gradient_accumulation_steps}")
    print(f"  Effective batch: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning rate  : {config.learning_rate}")
    print(f"  Max label len  : {config.max_label_length}")
    print(f"  Transformers   : {transformers_version}")
    print()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_checkpoint = _resolve_checkpoint(config.resume_from_checkpoint, config.output_dir)
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    else:
        print("Starting new training run.")

    image_module = _require_dependency("PIL.Image", "pip install pillow")
    train_dataset = MulkiyaBlipDataset(config.train_file, "train", config.max_samples)
    val_dataset = MulkiyaBlipDataset(config.val_file, "validation", config.max_samples)
    collator = BlipDataCollator(processor, config.max_image_size, config.max_label_length)

    _write_missing_image_report(
        output_dir / "missing_training_images.csv",
        [*train_dataset.skipped_rows, *val_dataset.skipped_rows],
    )

    print(f"Train : {train_dataset.loaded_count} loaded, {train_dataset.skipped_count} skipped")
    print(f"Val   : {val_dataset.loaded_count} loaded, {val_dataset.skipped_count} skipped")

    if train_dataset.loaded_count == 0:
        raise RuntimeError("No training samples loaded after skipping missing images.")
    if val_dataset.loaded_count == 0:
        raise RuntimeError("No validation samples loaded after skipping missing images.")

    training_arguments = TrainingArguments(
        **_build_training_kwargs(
            TrainingArguments,
            use_cpu=not use_cuda,
            output_dir=str(output_dir),
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=0.05,
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
            report_to=[],
            fp16=use_fp16,
            bf16=use_bf16,
            dataloader_pin_memory=False,
        )
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    eval_metrics = trainer.evaluate()

    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    training_log = {
        "config": asdict(config),
        "transformers_version": transformers_version,
        "lora_enabled": lora_enabled,
        "device": device,
        "use_fp16": use_fp16,
        "use_bf16": use_bf16,
        "train_metrics": _to_jsonable(train_result.metrics),
        "eval_metrics": _to_jsonable(eval_metrics),
        "log_history": _to_jsonable(trainer.state.log_history),
    }
    with (output_dir / "training_log.json").open("w", encoding="utf-8") as fh:
        json.dump(training_log, fh, indent=2)

    return training_log


def _run_smoke_test_only(config: BlipTrainConfig) -> None:
    """Load the model and run one synthetic forward pass, then exit."""
    print("=== BLIP GPU smoke test (--smoke-test mode) ===")
    model, processor, torch, _, _, device, _, _ = build_blip_model_and_processor(config)

    if device == "cpu":
        print("Running on CPU — CUDA smoke test skipped.")
        print("PASS: model loads and can move to CPU.")
        return

    _blip_gpu_smoke_test(model, processor, device)

    vram_used = torch.cuda.memory_allocated() / 1024**3
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM used: {vram_used:.2f} / {vram_total:.1f} GB")
    print()
    print("BLIP is compatible with this GPU.")
    print("Next step:")
    print("  python scripts\\finetune_blip_mulkiya.py --config configs\\finetune_blip_mulkiya.yaml")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # CPU is the default and stable training mode.
    # CUDA is only activated when --device cuda is explicitly passed.
    # Setting CUDA_VISIBLE_DEVICES="" before any torch import hides the GPU from
    # every library so Accelerate cannot accidentally use it.
    if args.device != "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("CPU training mode enabled.")
    else:
        print("CUDA mode enabled (experimental — CPU is recommended for stable training).")

    config = load_config(args.config, args)

    print("BLIP fine-tuning — UAE Mulkiya VIN extraction")
    print(f"  Model      : {config.model_name}")
    print(f"  Train      : {config.train_file}")
    print(f"  Val        : {config.val_file}")
    print(f"  Output     : {config.output_dir}")
    print(f"  LoRA       : {config.use_lora}")
    print(f"  Device     : {config.device}")

    if args.smoke_test:
        _run_smoke_test_only(config)
        return

    training_log = train(config)
    print("\nTraining complete.")
    print(json.dumps(training_log["eval_metrics"], indent=2))


if __name__ == "__main__":
    main()
