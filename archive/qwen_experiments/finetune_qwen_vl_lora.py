from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import inspect
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _require_dependency(module_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            f"Missing required package '{module_name}'. Install it with: {install_hint}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experimental: fine-tune Qwen2.5-VL with LoRA or QLoRA for UAE Mulkiya extraction."
    )
    parser.add_argument(
        "--config",
        default="configs/finetune_qwen_vl_lora.yaml",
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
        "--use-qlora",
        dest="use_qlora",
        action="store_true",
        help="Force QLoRA when bitsandbytes is available.",
    )
    parser.add_argument(
        "--no-qlora",
        dest="use_qlora",
        action="store_false",
        help="Disable QLoRA and use standard LoRA.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optionally limit train and validation rows for smoke tests.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend to use when loading the model.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        help="Resize images so the longest side is at most this many pixels.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing for stability testing.",
    )
    parser.set_defaults(use_qlora=None)
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
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    max_image_size: int
    use_qlora: bool
    max_samples: int | None
    attn_implementation: str
    gradient_checkpointing: bool


def load_config(config_path: str, cli_args: argparse.Namespace) -> TrainConfig:
    yaml = _require_dependency("yaml", "pip install pyyaml")

    with Path(config_path).open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    def pick(name: str, default: Any) -> Any:
        value = getattr(cli_args, name.replace("-", "_"), None)
        if value is not None:
            return value
        return raw_config.get(name, default)

    use_qlora = cli_args.use_qlora
    if use_qlora is None:
        use_qlora = bool(raw_config.get("use_qlora", True))

    return TrainConfig(
        model_name=pick("model_name", "Qwen/Qwen2.5-VL-3B-Instruct"),
        train_file=pick("train_file", "outputs/dataset/finetune_train.jsonl"),
        val_file=pick("val_file", "outputs/dataset/finetune_val.jsonl"),
        output_dir=pick("output_dir", "outputs/models/qwen_mulkiya_lora"),
        epochs=int(pick("epochs", 3)),
        learning_rate=float(pick("learning_rate", 2e-4)),
        batch_size=int(pick("batch_size", 1)),
        gradient_accumulation_steps=int(pick("gradient_accumulation_steps", 8)),
        lora_r=int(pick("lora_r", 16)),
        lora_alpha=int(pick("lora_alpha", 32)),
        lora_dropout=float(pick("lora_dropout", 0.05)),
        max_image_size=int(pick("max_image_size", 768)),
        use_qlora=bool(use_qlora),
        max_samples=cli_args.max_samples,
        attn_implementation=pick("attn_implementation", "eager"),
        gradient_checkpointing=not bool(cli_args.no_gradient_checkpointing),
    )


def print_gpu_info(torch_module) -> dict[str, Any]:
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


class QwenVisionDataCollator:
    def __init__(self, processor, image_class, max_image_size: int = 1024) -> None:
        self.processor = processor
        self.image_class = image_class
        self.max_image_size = max_image_size
        self._debug_printed = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        qwen_vl_utils = _require_dependency(
            "qwen_vl_utils",
            "pip install qwen-vl-utils",
        )
        process_vision_info = qwen_vl_utils.process_vision_info

        rendered_texts = []
        image_inputs_batch = []
        video_inputs_batch = []

        for feature in features:
            image_path = Path(feature["image"])
            with self.image_class.open(image_path) as image:
                resized = self._resize_image(image.convert("RGB"))

            source_messages = feature["messages"]
            if len(source_messages) < 2:
                raise ValueError(
                    f"Each JSONL row must contain at least a user and assistant message. Bad row: {image_path}"
                )

            user_prompt = str(source_messages[0].get("content", "") or "")
            assistant_text = str(source_messages[-1].get("content", "") or "")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": resized},
                        {"type": "text", "text": user_prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": assistant_text,
                },
            ]

            rendered_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            image_inputs, video_inputs = process_vision_info(messages)

            rendered_texts.append(rendered_text)
            if isinstance(image_inputs, list) and len(image_inputs) == 1:
                image_inputs_batch.append(image_inputs[0])
            else:
                image_inputs_batch.append(image_inputs)

            if isinstance(video_inputs, list) and len(video_inputs) == 1:
                video_inputs_batch.append(video_inputs[0])
            else:
                video_inputs_batch.append(video_inputs)

        processor_kwargs = {
            "text": rendered_texts,
            "images": image_inputs_batch,
            "padding": True,
            "return_tensors": "pt",
        }
        if any(video_input is not None for video_input in video_inputs_batch):
            processor_kwargs["videos"] = video_inputs_batch

        batch_inputs = self.processor(**processor_kwargs)

        labels = batch_inputs["input_ids"].clone()
        labels[batch_inputs["attention_mask"] == 0] = -100
        batch_inputs["labels"] = labels

        if not self._debug_printed:
            first_text = rendered_texts[0] if rendered_texts else ""
            contains_vision_tokens = (
                "<|vision_start|>" in first_text
                or "<|image_pad|>" in first_text
                or "<|vision_end|>" in first_text
            )
            print(f"First batch contains Qwen vision tokens: {contains_vision_tokens}")
            print(f"input_ids shape: {tuple(batch_inputs['input_ids'].shape)}")
            if "pixel_values" in batch_inputs:
                print(f"pixel_values shape: {tuple(batch_inputs['pixel_values'].shape)}")
            if "image_grid_thw" in batch_inputs:
                print(f"image_grid_thw shape: {tuple(batch_inputs['image_grid_thw'].shape)}")
            self._debug_printed = True

        return batch_inputs

    def _resize_image(self, image):
        width, height = image.size
        longest_side = max(width, height)
        if longest_side <= self.max_image_size:
            return image.copy()

        scale = self.max_image_size / float(longest_side)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        return image.resize((new_width, new_height))


def bitsandbytes_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


def build_model_and_processor(config: TrainConfig):
    torch = _require_dependency("torch", "pip install torch torchvision")
    try:
        import torch._dynamo as torch_dynamo

        torch_dynamo.config.suppress_errors = True
    except Exception:
        pass

    _require_dependency("accelerate", "pip install accelerate")
    _require_dependency("peft", "pip install peft")
    _require_dependency("PIL", "pip install pillow")
    _require_dependency("qwen_vl_utils", "pip install qwen-vl-utils")

    transformers = _require_dependency(
        "transformers",
        "pip install transformers sentencepiece",
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    quantization_config = None
    qlora_enabled = False
    if config.use_qlora:
        if not bitsandbytes_available():
            raise RuntimeError(
                "QLoRA was requested but bitsandbytes is not installed. "
                "Install it with: pip install bitsandbytes"
            )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if _supports_bf16(torch) else torch.float16,
        )
        qlora_enabled = True

    torch_dtype = _resolve_torch_dtype(torch)
    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "attn_implementation": config.attn_implementation,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        if torch.cuda.is_available():
            model_kwargs["device_map"] = {"": torch.cuda.current_device()}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_name,
        **model_kwargs,
    )

    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if qlora_enabled:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor, torch, qlora_enabled, transformers.__version__


def _resolve_torch_dtype(torch_module):
    if torch_module.cuda.is_available():
        if _supports_bf16(torch_module):
            return torch_module.bfloat16
        return torch_module.float16
    return torch_module.float32


def _supports_bf16(torch_module) -> bool:
    return bool(
        torch_module.cuda.is_available()
        and hasattr(torch_module.cuda, "is_bf16_supported")
        and torch_module.cuda.is_bf16_supported()
    )


def train(config: TrainConfig) -> dict[str, Any]:
    datasets_module = _require_dependency("datasets", "pip install datasets")
    transformers = _require_dependency("transformers", "pip install transformers sentencepiece")
    image_module = _require_dependency("PIL.Image", "pip install pillow")

    from transformers import Trainer, TrainingArguments

    model, processor, torch, qlora_enabled, transformers_version = build_model_and_processor(config)

    gpu_info = print_gpu_info(torch)

    _ = datasets_module  # Keeps the dependency explicit for the training environment.

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    collator = QwenVisionDataCollator(
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

    training_arguments = TrainingArguments(
        **_build_training_arguments_kwargs(
            TrainingArguments,
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
            fp16=bool(torch.cuda.is_available() and not _supports_bf16(torch)),
            bf16=bool(_supports_bf16(torch)),
            gradient_checkpointing=bool(config.gradient_checkpointing),
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

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    training_log = {
        "config": asdict(config),
        "gpu_info": gpu_info,
        "transformers_version": transformers_version,
        "qlora_enabled": qlora_enabled,
        "train_metrics": _to_jsonable(train_result.metrics),
        "eval_metrics": _to_jsonable(eval_metrics),
        "log_history": _to_jsonable(trainer.state.log_history),
    }
    with (output_dir / "training_log.json").open("w", encoding="utf-8") as handle:
        json.dump(training_log, handle, indent=2)

    return training_log


def _build_training_arguments_kwargs(training_arguments_class, **base_kwargs) -> dict[str, Any]:
    signature = inspect.signature(training_arguments_class.__init__)
    supported_parameters = signature.parameters

    kwargs = dict(base_kwargs)
    if "evaluation_strategy" in supported_parameters:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in supported_parameters:
        kwargs["eval_strategy"] = "epoch"

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
    config = load_config(args.config, args)

    print("Starting experimental Qwen2.5-VL LoRA fine-tuning")
    print(f"Model: {config.model_name}")
    print(f"Train file: {config.train_file}")
    print(f"Validation file: {config.val_file}")
    print(f"Output dir: {config.output_dir}")
    print(f"QLoRA enabled: {config.use_qlora}")
    try:
        training_log = train(config)
    except RuntimeError as exc:
        error_text = str(exc)
        lowered = error_text.lower()
        if (
            "illegal memory access" in lowered
            or "cuda error" in lowered
            or "invalid program counter" in lowered
        ) and ("bitsandbytes" in lowered or config.use_qlora):
            print(
                "Detected a CUDA / bitsandbytes stability failure. "
                "On Windows RTX 4050 laptops, plain LoRA is the safer fallback. "
                "Retry with: --no-qlora"
            )
        raise

    print("Training complete")
    print(json.dumps(training_log["eval_metrics"], indent=2))


if __name__ == "__main__":
    main()
