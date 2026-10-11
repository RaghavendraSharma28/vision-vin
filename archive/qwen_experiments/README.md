# Qwen VL Experiments — Archived

Archived 2026-06-30.

## Status: ABANDONED

**Reason:** `CUDA error: an illegal instruction was encountered` on Windows RTX 4050 (Ada Lovelace, sm_89)
during VLM image-token merging. Reproducible across multiple attempts and CUDA configurations.
Requires a Linux GPU environment to resolve.

## Files

| File | Purpose |
|---|---|
| `finetune_qwen_vl_lora.py` | LoRA / QLoRA training script for Qwen2.5-VL-3B |
| `test_finetuned_qwen_vl.py` | Single-image inference with a trained Qwen adapter |
| `finetune_qwen_vl_lora.yaml` | Training hyperparameters |

## Notes

- Qwen2.5-VL-3B is a capable model for structured VLM extraction.
- If running on Linux with an A100/H100, this path is worth revisiting.
- Dataset format (`finetune_train.jsonl`) is identical — these scripts can be
  reused without changes on a compatible machine.

## Replacement

BLIP (`Salesforce/blip-image-captioning-base`) on CPU was chosen as the stable
production fine-tuning path.
