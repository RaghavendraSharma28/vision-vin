from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


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
        description="Run inference with a fine-tuned SmolVLM2 model or adapter."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the fine-tuned model directory or LoRA adapter directory.",
    )
    parser.add_argument(
        "--model-name",
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        help="Base model name used when loading a LoRA adapter.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    torch = _require_dependency("torch", "pip install torch torchvision")
    pil_image = _require_dependency("PIL.Image", "pip install pillow")
    _require_dependency("transformers", "pip install transformers sentencepiece")

    from transformers import AutoModelForImageTextToText, AutoProcessor

    image_path = Path(args.image)
    model_path = Path(args.model)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        print(f"GPU name: {properties.name}")
        print(f"Total VRAM (GB): {round(properties.total_memory / (1024**3), 2)}")

    dtype = (
        torch.bfloat16
        if cuda_available and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        else (torch.float16 if cuda_available else torch.float32)
    )

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)

    if (model_path / "adapter_config.json").exists():
        _require_dependency("peft", "pip install peft")
        from peft import PeftModel

        base_model = AutoModelForImageTextToText.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(model_path))
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    if cuda_available:
        model = model.to("cuda")
    model.eval()

    with pil_image.open(image_path) as image:
        rgb_image = image.convert("RGB")

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
        images=[rgb_image],
        padding=True,
        return_tensors="pt",
    )
    if cuda_available:
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = generated[:, prompt_length:]
    raw_output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("Raw output:")
    print(raw_output)
    print("\nParsed JSON:")
    try:
        print(json.dumps(json.loads(raw_output), indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print(json.dumps({"raw_output": raw_output}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
