# SmolVLM2 Experiments — Archived

Archived 2026-06-30.

## Status: ABANDONED

**Reason:** `CUDA error: an illegal instruction was encountered` on Windows RTX 4050 (Ada Lovelace, sm_89)
during forward pass — confirmed by `smoke_test_vlm_gpu.py`.  CPU training worked but was too slow
for practical iteration (500M params, fp32).

SmolVLM2-500M was chosen over Qwen2.5-VL-3B because it is smaller and CPU-trainable.
However, the pace of CPU training made it impractical compared to BLIP (224M params with LoRA).

## Files

| File | Purpose |
|---|---|
| `finetune_smolvlm2.py` | Full training script with LoRA, resume, GPU config flags |
| `eval_smolvlm2_mulkiya.py` | Batch evaluation against test JSONL |
| `test_smolvlm2_mulkiya.py` | Single-image inference |
| `finetune_smolvlm2.yaml` | Training hyperparameters |

## Notes

- `finetune_smolvlm2.py` contains robust `--device cpu/cuda`, `--resume-from-checkpoint`,
  `--attention-implementation eager/sdpa/flash_attention_2`, and GPU config flags.
  These patterns were later ported to `finetune_blip_mulkiya.py`.
- SmolVLM2 chat-template + image-token-expansion masking logic is correct and tested on CPU.
- Dataset format is identical — scripts can be reused on a Linux GPU.

## Replacement

BLIP with LoRA on CPU (`finetune_blip_mulkiya.py`) is the current stable path.
