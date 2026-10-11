from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import VehicleMetadata
from .vin import clean_chassis_no, is_valid_chassis_format

QWEN_MULKIYA_PROMPT = """
You are reading a UAE vehicle registration card, also called a Mulkiya.

Extract only the following fields from the image:
- chassis_no
- model_year
- origin
- vehicle_description

The chassis number may be labelled as:
- Chassis No
- Chassis No.
- Chassis Number

Rules:
- Return only valid JSON.
- Do not include explanations.
- If a field is not visible, use null.
- chassis_no should be exactly as shown, uppercase, no spaces.
- Do not guess missing characters.
- Pay special attention to OCR-confusable characters:
  5 vs S
  8 vs B
  0 vs O
  1 vs I
  2 vs Z
  3 vs S

Expected JSON format:
{
  "chassis_no": "...",
  "model_year": "...",
  "origin": "...",
  "vehicle_description": "..."
}
""".strip()

QWEN_MULKIYA_CROP_PROMPT = """
You are looking at a cropped row from a UAE Mulkiya.
Read the chassis number shown next to the label "Chassis No".
Return only JSON:
{
  "chassis_no": "...",
  "model_year": null,
  "origin": null,
  "vehicle_description": null
}
If the chassis number is not clearly visible, return null.
Do not guess.
""".strip()

QWEN_MULKIYA_RETRY_PROMPT = """
You are reading a UAE Mulkiya.
Focus only on the chassis number field.
Return only JSON:
{
  "chassis_no": "...",
  "model_year": null,
  "origin": null,
  "vehicle_description": null
}
If the chassis number is not clearly visible, return null.
Do not guess.
""".strip()


class VlmExtractor:
    def extract(self, image_path: str | Path) -> VehicleMetadata:
        raise NotImplementedError


class NullVlmExtractor(VlmExtractor):
    def extract(self, image_path: str | Path) -> VehicleMetadata:
        return VehicleMetadata()


def load_qwen_model(
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    device: str | None = "cpu",
):
    """Load Qwen2.5-VL once for a baseline run."""
    try:
        import torch
        import transformers
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Qwen dependencies are missing. Install them with: "
            "pip install transformers accelerate pillow qwen-vl-utils"
        ) from exc

    transformers_version = getattr(transformers, "__version__", "unknown")
    print(f"transformers version: {transformers_version}")

    if _version_tuple(transformers_version) < (4, 49):
        raise RuntimeError(
            "Qwen2.5-VL requires a newer transformers version. "
            f"Current version: {transformers_version}. "
            "Run: pip install --upgrade transformers accelerate"
        )

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Could not import Qwen2_5_VLForConditionalGeneration from transformers. "
            f"Current transformers version: {transformers_version}. "
            "Run: pip install --upgrade transformers accelerate"
        ) from exc 

    processor = AutoProcessor.from_pretrained(model_name)

    cuda_available = torch.cuda.is_available()
    device_mode = (device or "cpu").lower().strip()
    if device_mode not in {"cpu", "cuda", "auto"}:
        raise ValueError(f"Unsupported device mode: {device}")

    if device_mode == "auto":
        device_mode = "cuda" if cuda_available else "cpu"

    print(f"Qwen device mode: {device_mode}")

    if device_mode == "cuda":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
    else:
        if device_mode == "cpu" and not cuda_available:
            print("WARNING: CUDA not available. Qwen will run on CPU and may be extremely slow. Use --limit for testing.")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
        )
        model.to("cpu")

    model.eval()
    return model, processor


def extract_with_qwen(
    image_path: str,
    model,
    processor,
    max_new_tokens: int = 256,
    prompt: str | None = None,
) -> dict:
    """Run Qwen on one image and parse the requested JSON fields."""
    try:
        from PIL import Image
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError(
            "Qwen dependencies are missing. Install them with: "
            "pip install transformers accelerate pillow qwen-vl-utils"
        ) from exc

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt or QWEN_MULKIYA_PROMPT},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    if hasattr(model, "device"):
        inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    parsed = extract_json_object(raw_output)
    if parsed is None:
        return {
            "raw_output": raw_output,
            "chassis_no": None,
            "model_year": None,
            "origin": None,
            "vehicle_description": None,
            "parse_success": False,
        }

    return {
        "raw_output": raw_output,
        "chassis_no": parsed.get("chassis_no"),
        "model_year": parsed.get("model_year"),
        "origin": parsed.get("origin"),
        "vehicle_description": parsed.get("vehicle_description"),
        "parse_success": True,
    }


def extract_json_object(text: str) -> dict | None:
    """Extract the first JSON object from model text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def extract_chassis_candidate_from_qwen_result(result: dict) -> str | None:
    """Clean the Qwen chassis field for validation and correction."""
    chassis_no = result.get("chassis_no")
    if not chassis_no:
        return None

    cleaned = clean_chassis_no(chassis_no)
    if not cleaned:
        return None

    if is_valid_chassis_format(cleaned):
        return cleaned
    return cleaned


def _version_tuple(version_text: str) -> tuple[int, ...]:
    numbers = []
    for part in str(version_text).split("."):
        digits = ""
        for char in part:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        numbers.append(int(digits))
    return tuple(numbers)
