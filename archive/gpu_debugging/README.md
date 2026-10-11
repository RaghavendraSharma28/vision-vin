# GPU Debugging Scripts — Archived

Archived 2026-06-30.

## Purpose

These scripts were written during investigation of CUDA illegal-instruction errors
on Windows RTX 4050 (Ada Lovelace, sm_89) when training SmolVLM2 and BLIP.

## Files

| File | Purpose |
|---|---|
| `gpu_diagnostics.py` | Print PyTorch/CUDA/cuDNN/package versions; run basic CUDA smoke tests; diagnose illegal-instruction root cause |
| `smoke_test_vlm_gpu.py` | Minimal SmolVLM2 forward/backward pass on GPU without Trainer — isolates model vs Trainer crashes |

## Findings

- `gpu_diagnostics.py`: All basic CUDA ops passed (tensor alloc, matmul, backward, fp16, bf16, SDPA).
- `smoke_test_vlm_gpu.py`: **FAILED** — SmolVLM2 forward pass crashed with illegal instruction during image-token merge regardless of attention implementation.
- BLIP smoke test (`--smoke-test` flag in `finetune_blip_mulkiya.py`): Passed on GPU, but full training crashed with illegal memory access in fp16 causal mask creation.

## Decision

CPU-only training adopted as stable path. GPU experiments discontinued on this machine.
If GPU training is needed, use Linux + cloud GPU.
