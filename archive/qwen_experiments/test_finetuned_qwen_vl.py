from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


def _require_dependency(module_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            f"Missing required package '{module_name}'. Install it with: {install_hint}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference with a base Qwen2.5-VL model plus a LoRA adapter."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to the fine-tuned LoRA adapter directory.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Base model name.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    torch = _require_dependency("torch", "pip install torch torchvision")
    _require_dependency("qwen_vl_utils", "pip install qwen-vl-utils")
    pil_image = _require_dependency("PIL.Image", "pip install pillow")
    transformers = _require_dependency(
        "transformers",
        "pip install transformers sentencepiece",
    )
    _require_dependency("peft", "pip install peft")

    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from mulkiya_vin.vlm import QWEN_MULKIYA_PROMPT, extract_json_object

    image_path = Path(args.image)
    adapter_path = Path(args.adapter)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        print(f"GPU name: {properties.name}")
        print(f"Total VRAM (GB): {round(properties.total_memory / (1024**3), 2)}")

    dtype = torch.bfloat16 if cuda_available and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported() else (torch.float16 if cuda_available else torch.float32)

    processor = AutoProcessor.from_pretrained(str(adapter_path), trust_remote_code=True)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    if cuda_available:
        model = model.to("cuda")
    model.eval()

    with pil_image.open(image_path) as image:
        rgb_image = image.convert("RGB")

    user_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": rgb_image},
                {"type": "text", "text": QWEN_MULKIYA_PROMPT},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(
        user_messages,
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

    parsed = extract_json_object(raw_output)
    print("Raw output:")
    print(raw_output)
    print("\nParsed JSON:")
    print(json.dumps(parsed or {"raw_output": raw_output}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
