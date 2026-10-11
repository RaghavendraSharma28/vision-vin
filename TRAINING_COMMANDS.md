# Training Commands — UAE Mulkiya VIN Extraction

Current recommended workflow.  All commands run on Windows CPU without GPU.

---

## Step 1 — Audit dataset

```powershell
python -m mulkiya_vin.cli audit
```

Writes `outputs/audit_summary.csv`.  Review for missing images, invalid chassis
numbers, and duplicate labels before training.

---

## Step 2 — Prepare dataset

```powershell
python -m mulkiya_vin.cli prepare-dataset
```

Writes:
- `outputs/dataset/finetune_train.jsonl`
- `outputs/dataset/finetune_val.jsonl`
- `outputs/dataset/finetune_test.jsonl`

---

## Step 3 — RapidOCR baseline

```powershell
python -m mulkiya_vin.cli ocr-baseline --ocr-engine rapidocr
```

Establishes the production baseline before measuring BLIP improvement.
Baseline: 109/110 chassis (99.09%).

---

## Step 4 — BLIP smoke test (50 samples, confirm training loop works)

```powershell
python scripts\finetune_blip_mulkiya.py --config configs\finetune_blip_mulkiya.yaml --max-samples 50
```

CPU is the default — no `--device` flag needed.  Trains LoRA adapters only
(~0.26% of parameters).  Should complete in under 30 minutes on a laptop CPU.

---

## Step 5 — Full BLIP training

```powershell
python scripts\finetune_blip_mulkiya.py --config configs\finetune_blip_mulkiya.yaml
```

Saves checkpoints to `outputs/models/blip_mulkiya/checkpoint-N/` after each epoch.

### Resume training from the latest checkpoint

```powershell
python scripts\finetune_blip_mulkiya.py --config configs\finetune_blip_mulkiya.yaml --resume-from-checkpoint latest
```

### Resume from a specific checkpoint

```powershell
python scripts\finetune_blip_mulkiya.py --config configs\finetune_blip_mulkiya.yaml --resume-from-checkpoint outputs\models\blip_mulkiya\checkpoint-619
```

---

## Step 6 — Evaluate BLIP

```powershell
python scripts\eval_blip_mulkiya.py --model outputs\models\blip_mulkiya --test-file outputs\dataset\finetune_test.jsonl
```

Reports chassis accuracy vs. RapidOCR baseline.
Saves `outputs/eval/blip_eval_results.csv`, `blip_eval_summary.csv`, `blip_eval_failures.csv`.

### Zero-shot base model evaluation (no training required)

```powershell
python scripts\eval_blip_mulkiya.py --model-name Salesforce/blip-image-captioning-base --test-file outputs\dataset\finetune_test.jsonl
```

---

## Conservative hybrid pipeline (production)

```powershell
python -m mulkiya_vin.cli hybrid-baseline --mode conservative
```

Uses RapidOCR + validation/correction.  This is the production path.
BLIP is a research enhancement, not a replacement.

---

## Install training dependencies

```powershell
pip install -r requirements-train.txt
```

---

## Archived experiments

Previous GPU training experiments (Qwen2.5-VL, SmolVLM2) are in `archive/`.
They are preserved for reference if GPU training is attempted on Linux in future.
See `archive/*/README.md` for context.
