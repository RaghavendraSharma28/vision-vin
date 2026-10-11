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


def _require_dependency(module_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            f"Missing required package '{module_name}'. Install it with: {install_hint}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolVLM2-500M for UAE Mulkiya structured JSON extraction."
    )
    parser.add_argument(
        "--config",
        default="configs/finetune_smolvlm2.yaml",
        help="Path to YAML training config.",
    )
    parser.add_argument("--train-file", help="Override training JSONL file.")
    parser.add_argument("--val-file", help="Override validation JSONL file.")
    parser.add_argument("--model-name", help="Override base model name.")
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument("--epochs", type=int, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, help="Override per-device batch size.")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate.")
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optionally limit train and validation rows for smoke tests.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        help="Resize images so the longest side is at most this many pixels.",
    )
    parser.add_argument(
        "--use-lora",
        dest="use_lora",
        action="store_true",
        help="Enable LoRA adapters.",
    )
    parser.add_argument(
        "--no-lora",
        dest="use_lora",
        action="store_false",
        help="Disable LoRA and run full fine-tuning.",
    )
    parser.set_defaults(use_lora=None)
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Training device. Default: auto-detect (cuda if available, else cpu).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        dest="resume_from_checkpoint",
        default=None,
        metavar="CHECKPOINT",
        help=(
            "Resume training from a checkpoint. Pass a path to a specific checkpoint "
            "directory, or 'latest' to automatically use the highest checkpoint in output_dir."
        ),
    )

    # ── GPU configuration ─────────────────────────────────────────────────────
    parser.add_argument("--fp16", dest="fp16", action="store_true",
                        help="Force FP16 mixed-precision training.")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false",
                        help="Disable FP16.")
    parser.add_argument("--bf16", dest="bf16", action="store_true",
                        help="Force BF16 mixed-precision training (requires Ampere+).")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false",
                        help="Disable BF16.")
    parser.add_argument("--tf32", dest="tf32", action="store_true",
                        help="Enable TF32 on Ampere+ GPUs (free throughput gain, minor precision change).")
    parser.add_argument("--no-tf32", dest="tf32", action="store_false",
                        help="Disable TF32.")
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save VRAM (slower training).")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false",
                        help="Disable gradient checkpointing (default).")
    parser.add_argument(
        "--attention-implementation",
        dest="attention_implementation",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default="auto",
        help=(
            "Attention kernel to use.  Default: auto (resolves to 'eager' on CUDA for safety)."
            "  'eager' = standard PyTorch, no special kernels;"
            "  'sdpa' = torch.nn.functional.scaled_dot_product_attention;"
            "  'flash_attention_2' = requires flash-attn package."
        ),
    )
    # Shorthand aliases so --eager-attention / --sdpa / --flash-attention also work
    parser.add_argument("--eager-attention", dest="attention_implementation",
                        action="store_const", const="eager",
                        help="Shorthand for --attention-implementation eager.")
    parser.add_argument("--sdpa", dest="attention_implementation",
                        action="store_const", const="sdpa",
                        help="Shorthand for --attention-implementation sdpa.")
    parser.add_argument("--flash-attention", dest="attention_implementation",
                        action="store_const", const="flash_attention_2",
                        help="Shorthand for --attention-implementation flash_attention_2.")

    parser.set_defaults(fp16=None, bf16=None, tf32=None, gradient_checkpointing=None)
    return parser


@dataclass
class TrainConfig:
    model_name: str
    train_file: str
    val_file: str
    output_dir: str
    epochs: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    max_image_size: int
    use_lora: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    max_samples: int | None
    device: str  # "cuda", "cpu", or "auto"
    resume_from_checkpoint: str | None
    fp16: bool | None  # None = auto-resolve
    bf16: bool | None  # None = auto-resolve
    tf32: bool | None  # None = auto-resolve (enabled on Ampere+ when None)
    gradient_checkpointing_enabled: bool | None  # None = auto-resolve (disabled by default)
    attention_implementation: str  # "auto", "eager", "sdpa", "flash_attention_2"


def load_config(config_path: str, cli_args: argparse.Namespace) -> TrainConfig:
    yaml = _require_dependency("yaml", "pip install pyyaml")

    with Path(config_path).open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    def pick(name: str, default: Any) -> Any:
        value = getattr(cli_args, name.replace("-", "_"), None)
        if value is not None:
            return value
        return raw_config.get(name, default)

    use_lora = cli_args.use_lora
    if use_lora is None:
        use_lora = bool(raw_config.get("use_lora", True))

    device_arg = getattr(cli_args, "device", None)
    device = device_arg if device_arg is not None else str(raw_config.get("device", "auto"))

    resume_from_checkpoint = getattr(cli_args, "resume_from_checkpoint", None)

    # Three-state GPU flags: CLI value wins; then YAML; then None (auto).
    def _tristate(attr: str) -> bool | None:
        val = getattr(cli_args, attr, None)
        if val is not None:
            return val
        raw = raw_config.get(attr)
        return bool(raw) if raw is not None else None

    return TrainConfig(
        model_name=pick("model_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"),
        train_file=pick("train_file", "outputs/dataset/finetune_train.jsonl"),
        val_file=pick("val_file", "outputs/dataset/finetune_val.jsonl"),
        output_dir=pick("output_dir", "outputs/models/smolvlm2_mulkiya"),
        epochs=int(pick("epochs", 3)),
        learning_rate=float(pick("learning_rate", 1e-4)),
        batch_size=int(pick("batch_size", 1)),
        gradient_accumulation_steps=int(pick("gradient_accumulation_steps", 8)),
        max_image_size=int(pick("max_image_size", 512)),
        use_lora=bool(use_lora),
        lora_r=int(pick("lora_r", 16)),
        lora_alpha=int(pick("lora_alpha", 32)),
        lora_dropout=float(pick("lora_dropout", 0.05)),
        max_samples=cli_args.max_samples,
        device=device,
        resume_from_checkpoint=resume_from_checkpoint,
        fp16=_tristate("fp16"),
        bf16=_tristate("bf16"),
        tf32=_tristate("tf32"),
        gradient_checkpointing_enabled=_tristate("gradient_checkpointing"),
        attention_implementation=str(
            getattr(cli_args, "attention_implementation", None)
            or raw_config.get("attention_implementation", "auto")
        ),
    )


def print_gpu_info(torch_module, effective_device: str = "cuda") -> dict[str, Any]:
    if effective_device == "cpu":
        print("Device: cpu  (fp16/bf16 and gradient checkpointing disabled)")
        return {"cuda_available": False, "gpu_name": "cpu", "total_vram_gb": 0.0, "effective_device": "cpu"}

    cuda_available = bool(torch_module.cuda.is_available())
    gpu_name = ""
    total_vram_gb = 0.0

    if cuda_available:
        device_index = torch_module.cuda.current_device()
        properties = torch_module.cuda.get_device_properties(device_index)
        gpu_name = properties.name
        total_vram_gb = round(properties.total_memory / (1024**3), 2)

    print(f"torch.cuda.is_available(): {cuda_available}")
    print(f"GPU name: {gpu_name or 'N/A'}")
    print(f"Total VRAM (GB): {total_vram_gb}")

    return {
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "total_vram_gb": total_vram_gb,
        "effective_device": effective_device,
    }


class MulkiyaVisionJsonlDataset:
    def __init__(
        self,
        jsonl_path: str | Path,
        split_name: str,
        max_samples: int | None = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Training file not found: {self.jsonl_path}")

        self.rows = []
        self.split_name = split_name
        self.requested_count = 0
        self.skipped_rows: list[dict[str, str]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                row["__line_number__"] = line_number
                self.requested_count += 1
                image_path = Path(str(row.get("image", "") or ""))
                if not image_path.exists():
                    print("WARNING: Missing image skipped:")
                    print(str(image_path))
                    self.skipped_rows.append(
                        {
                            "filename": image_path.name,
                            "expected_path": str(image_path).replace("\\", "/"),
                            "split": split_name,
                        }
                    )
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

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class SmolVLMDataCollator:
    def __init__(self, processor, image_class, max_image_size: int = 512) -> None:
        self.processor = processor
        self.image_class = image_class
        self.max_image_size = max_image_size

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        full_texts = []
        prompt_texts = []
        images = []

        for feature in features:
            image_path = Path(feature["image"])
            with self.image_class.open(image_path) as image:
                rgb_image = self._resize_image(image.convert("RGB"))

            source_messages = feature.get("messages", [])
            if len(source_messages) < 2:
                raise ValueError(
                    f"Each JSONL row must contain at least a user and assistant message. Bad row: {image_path}"
                )

            user_prompt = str(source_messages[0].get("content", "") or DEFAULT_PROMPT)
            assistant_text = str(source_messages[-1].get("content", "") or "")

            full_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                },
            ]
            # Prompt-only template (with generation prompt) to find where the answer starts
            prompt_only_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            full_text = self.processor.apply_chat_template(
                full_messages,
                add_generation_prompt=False,
                tokenize=False,
            )
            prompt_text = self.processor.apply_chat_template(
                prompt_only_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            full_texts.append(full_text)
            prompt_texts.append(prompt_text)
            images.append(rgb_image)

        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        # Mask the prompt portion so the model only learns to predict the assistant answer.
        # We call the processor once more per sample on the prompt-only text to get the
        # exact token count after image expansion — this is the safest approach for VLMs
        # where image tokens change the effective sequence length unpredictably.
        seq_len = batch["input_ids"].shape[1]
        for i, (prompt_text, rgb_image) in enumerate(zip(prompt_texts, images)):
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=[rgb_image],
                return_tensors="pt",
            )
            prompt_len = min(int(prompt_inputs["input_ids"].shape[1]), seq_len)
            labels[i, :prompt_len] = -100

        batch["labels"] = labels
        return batch

    def _resize_image(self, image):
        width, height = image.size
        longest_side = max(width, height)
        if longest_side <= self.max_image_size:
            return image.copy()

        scale = self.max_image_size / float(longest_side)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        return image.resize((new_width, new_height))


def build_model_and_processor(
    config: TrainConfig,
    enable_gradient_checkpointing: bool = False,
    attn_implementation: str = "eager",
):
    torch = _require_dependency("torch", "pip install torch torchvision")
    _require_dependency("accelerate", "pip install accelerate")
    _require_dependency("PIL.Image", "pip install pillow")
    transformers = _require_dependency(
        "transformers",
        "pip install transformers sentencepiece",
    )

    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = _effective_device(config.device, torch)
    use_cuda = device == "cuda"

    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)

    # device_map=None prevents Accelerate from auto-placing layers on GPU;
    # we move the model ourselves so we control exactly where it lands.
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=_resolve_torch_dtype(torch, use_cuda),
        device_map=None,
        attn_implementation=attn_implementation,
    )
    if not use_cuda:
        model = model.to("cpu")

    model.config.use_cache = False
    model.enable_input_require_grads()
    if enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_enabled = False
    if config.use_lora:
        _require_dependency("peft", "pip install peft")
        from peft import LoraConfig, get_peft_model

        target_modules = _find_lora_target_modules(model)
        if not target_modules:
            raise RuntimeError(
                "Could not find LoRA target modules for SmolVLM2. "
                "Retry with --no-lora for full fine-tuning."
            )
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        lora_enabled = True

    return model, processor, torch, lora_enabled, transformers.__version__, device


def _effective_device(config_device: str, torch_module) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu'; warn if 'cuda' was requested but unavailable."""
    if config_device == "cpu":
        return "cpu"
    if config_device == "cuda":
        if not torch_module.cuda.is_available():
            print("WARNING: --device cuda requested but CUDA is not available. Falling back to cpu.")
            return "cpu"
        return "cuda"
    # "auto"
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _resolve_torch_dtype(torch_module, use_cuda: bool):
    if not use_cuda:
        return torch_module.float32
    if _supports_bf16(torch_module):
        return torch_module.bfloat16
    return torch_module.float16


def _supports_bf16(torch_module) -> bool:
    return bool(
        torch_module.cuda.is_available()
        and hasattr(torch_module.cuda, "is_bf16_supported")
        and torch_module.cuda.is_bf16_supported()
    )


def _find_lora_target_modules(model) -> list[str]:
    preferred_suffixes = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    found = []
    for module_name, _module in model.named_modules():
        leaf_name = module_name.split(".")[-1]
        if leaf_name in preferred_suffixes and leaf_name not in found:
            found.append(leaf_name)
    return found


@dataclass
class ResolvedGPUConfig:
    fp16: bool
    bf16: bool
    tf32: bool
    gradient_checkpointing: bool
    attention_implementation: str
    gpu_name: str
    compute_capability: str
    vram_gb: float


def _resolve_gpu_config(config: TrainConfig, use_cuda: bool, torch_module) -> ResolvedGPUConfig:
    """Resolve None / 'auto' GPU settings to concrete values.

    Safe defaults for Windows RTX-class hardware:
      fp16=True, bf16=False, tf32=auto(Ampere+), gradient_checkpointing=False,
      attention_implementation='eager'  (no special CUDA kernels — avoids illegal-instruction
      crashes caused by attention packages compiled for a different compute capability).
    """
    if not use_cuda:
        return ResolvedGPUConfig(
            fp16=False, bf16=False, tf32=False,
            gradient_checkpointing=False,
            attention_implementation="eager",
            gpu_name="cpu", compute_capability="", vram_gb=0.0,
        )

    idx = torch_module.cuda.current_device()
    props = torch_module.cuda.get_device_properties(idx)
    compute_major = props.major
    cc = f"{props.major}.{props.minor}"
    vram_gb = round(props.total_memory / 1024**3, 2)
    gpu_name = props.name

    # Attention implementation
    requested_attn = config.attention_implementation
    if requested_attn == "auto":
        # "eager" avoids all attention-specific CUDA kernels.  It is the safest
        # choice on Windows where flash_attention_2 / SDPA can produce
        # "illegal instruction" errors if compiled for a mismatched arch.
        attn_impl = "eager"
    elif requested_attn == "flash_attention_2":
        try:
            importlib.import_module("flash_attn")
            attn_impl = "flash_attention_2"
        except ImportError:
            print(
                "WARNING: --attention-implementation flash_attention_2 requested "
                "but flash-attn is not installed.  Falling back to sdpa."
            )
            attn_impl = "sdpa"
    else:
        attn_impl = requested_attn  # "eager" or "sdpa" passed explicitly

    # Precision: explicit flag wins; auto default is fp16=True (safer than bf16 on Windows)
    if config.bf16 is True:
        use_bf16, use_fp16 = True, False
    elif config.fp16 is False and config.bf16 is not True:
        use_fp16, use_bf16 = False, False
    elif config.fp16 is True:
        use_fp16, use_bf16 = True, False
    else:
        # Auto: fp16 is the safe default on Windows for Ada Lovelace and older
        use_fp16, use_bf16 = True, False

    # TF32: auto-enable on Ampere (sm_8x) and newer — free throughput gain
    use_tf32 = compute_major >= 8 if config.tf32 is None else bool(config.tf32)

    # Gradient checkpointing: OFF by default (PEFT + GC has had compatibility issues)
    use_gc = False if config.gradient_checkpointing_enabled is None else bool(config.gradient_checkpointing_enabled)

    return ResolvedGPUConfig(
        fp16=use_fp16,
        bf16=use_bf16,
        tf32=use_tf32,
        gradient_checkpointing=use_gc,
        attention_implementation=attn_impl,
        gpu_name=gpu_name,
        compute_capability=cc,
        vram_gb=vram_gb,
    )


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


def train(config: TrainConfig) -> dict[str, Any]:
    _require_dependency("datasets", "pip install datasets")
    _require_dependency("num2words", "pip install num2words")
    _require_dependency("PIL.Image", "pip install pillow")

    from transformers import Trainer, TrainingArguments

    # Import torch early so we can resolve GPU config BEFORE the model is loaded.
    # This lets us set TF32 globally and choose attn_implementation for from_pretrained().
    torch = _require_dependency("torch", "pip install torch torchvision")
    effective_device = _effective_device(config.device, torch)
    use_cuda = effective_device == "cuda"

    gpu_config = _resolve_gpu_config(config, use_cuda, torch)

    # TF32 must be set before any matmul — it affects the global PyTorch state.
    if gpu_config.tf32 and use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model, processor, torch, lora_enabled, transformers_version, effective_device = build_model_and_processor(
        config,
        enable_gradient_checkpointing=gpu_config.gradient_checkpointing,
        attn_implementation=gpu_config.attention_implementation,
    )
    gpu_info = print_gpu_info(torch, effective_device)

    print("\n[Training Configuration]")
    print(f"  GPU                 : {gpu_config.gpu_name or 'cpu'}")
    print(f"  Compute capability  : {gpu_config.compute_capability or 'N/A'}")
    print(f"  VRAM                : {gpu_config.vram_gb} GB")
    print(f"  Attention impl      : {gpu_config.attention_implementation}")
    print(f"  fp16                : {gpu_config.fp16}")
    print(f"  bf16                : {gpu_config.bf16}")
    print(f"  tf32                : {gpu_config.tf32}")
    print(f"  Gradient checkpoint : {gpu_config.gradient_checkpointing}")
    print(f"  LoRA                : {config.use_lora}")
    print(f"  Epochs              : {config.epochs}")
    print(f"  Batch size          : {config.batch_size}")
    print(f"  Grad accumulation   : {config.gradient_accumulation_steps}")
    print(f"  Effective batch     : {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning rate       : {config.learning_rate}")
    print(f"  Transformers        : {transformers_version}")
    print()

    # Right-padding is required so that prompt_len masking in the collator is correct
    # (prompt tokens occupy the first N positions; answer tokens follow immediately).
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "padding_side"):
        processor.tokenizer.padding_side = "right"

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_checkpoint = _resolve_checkpoint(config.resume_from_checkpoint, config.output_dir)
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    else:
        print("Starting new training run.")

    image_module = _require_dependency("PIL.Image", "pip install pillow")
    train_dataset = MulkiyaVisionJsonlDataset(
        config.train_file,
        split_name="train",
        max_samples=config.max_samples,
    )
    val_dataset = MulkiyaVisionJsonlDataset(
        config.val_file,
        split_name="validation",
        max_samples=config.max_samples,
    )
    collator = SmolVLMDataCollator(
        processor=processor,
        image_class=image_module,
        max_image_size=config.max_image_size,
    )

    _write_missing_image_report(
        output_dir / "missing_training_images.csv",
        [*train_dataset.skipped_rows, *val_dataset.skipped_rows],
    )

    print(f"Training samples requested: {train_dataset.requested_count}")
    print(f"Training samples loaded: {train_dataset.loaded_count}")
    print(f"Training samples skipped: {train_dataset.skipped_count}")
    print(f"Validation requested: {val_dataset.requested_count}")
    print(f"Validation loaded: {val_dataset.loaded_count}")
    print(f"Validation skipped: {val_dataset.skipped_count}")

    if train_dataset.loaded_count == 0:
        raise RuntimeError("No training samples were loaded after skipping missing images.")
    if val_dataset.loaded_count == 0:
        raise RuntimeError("No validation samples were loaded after skipping missing images.")

    # Confirm the model is actually on the expected device before handing it to Trainer
    actual_device = str(next(model.parameters()).device)
    print(f"Model device check: {actual_device}")
    if not use_cuda and not actual_device.startswith("cpu"):
        raise RuntimeError(
            f"--device cpu was requested but model is on '{actual_device}'. "
            "This should not happen — please report this as a bug."
        )

    training_arguments = TrainingArguments(
        **_build_training_arguments_kwargs(
            TrainingArguments,
            use_cpu=not use_cuda,
            output_dir=str(output_dir),
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
            report_to=[],
            fp16=gpu_config.fp16,
            bf16=gpu_config.bf16,
            gradient_checkpointing=gpu_config.gradient_checkpointing,
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
        "gpu_info": gpu_info,
        "transformers_version": transformers_version,
        "lora_enabled": lora_enabled,
        "train_metrics": _to_jsonable(train_result.metrics),
        "eval_metrics": _to_jsonable(eval_metrics),
        "log_history": _to_jsonable(trainer.state.log_history),
    }
    with (output_dir / "training_log.json").open("w", encoding="utf-8") as handle:
        json.dump(training_log, handle, indent=2)

    return training_log


def _build_training_arguments_kwargs(
    training_arguments_class,
    use_cpu: bool = False,
    **base_kwargs,
) -> dict[str, Any]:
    signature = inspect.signature(training_arguments_class.__init__)
    supported_parameters = signature.parameters

    kwargs = dict(base_kwargs)
    if "evaluation_strategy" in supported_parameters:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in supported_parameters:
        kwargs["eval_strategy"] = "epoch"

    # use_reentrant=False avoids a gradient-checkpointing incompatibility with newer PEFT
    if "gradient_checkpointing_kwargs" in supported_parameters:
        kwargs.setdefault("gradient_checkpointing_kwargs", {"use_reentrant": False})

    # Windows multiprocessing for DataLoader workers is unreliable; 0 = main process only
    if "dataloader_num_workers" in supported_parameters:
        kwargs.setdefault("dataloader_num_workers", 0)

    # Tell the Trainer / Accelerate not to use CUDA even if the runtime can see a GPU.
    # "use_cpu" is the modern name (transformers >= 4.36); "no_cuda" is the legacy name.
    if use_cpu:
        if "use_cpu" in supported_parameters:
            kwargs["use_cpu"] = True
        elif "no_cuda" in supported_parameters:
            kwargs["no_cuda"] = True

    return kwargs


def _write_missing_image_report(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "expected_path", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Hide CUDA from every library (torch, accelerate, transformers) BEFORE any import
    # that might touch the CUDA runtime.  This is the only reliable way to force CPU on
    # a machine that has a GPU — setting TrainingArguments flags alone is not enough
    # because Accelerate reads device availability at import time.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("CPU mode enabled. CUDA disabled for this run.")

    config = load_config(args.config, args)

    print("Starting SmolVLM2 fine-tuning")
    print(f"Model: {config.model_name}")
    print(f"Train file: {config.train_file}")
    print(f"Validation file: {config.val_file}")
    print(f"Output dir: {config.output_dir}")
    print(f"LoRA enabled: {config.use_lora}")
    print(f"Device: {config.device}")

    training_log = train(config)

    print("Training complete")
    print(json.dumps(training_log["eval_metrics"], indent=2))


if __name__ == "__main__":
    main()
