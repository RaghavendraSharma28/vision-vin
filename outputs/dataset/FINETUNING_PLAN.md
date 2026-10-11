# Fine-Tuning Plan

## Recommended Model
Start with `Qwen/Qwen2.5-VL-3B-Instruct` or another small vision-language model in the same size range.

## Why Fine-Tuning
Fine-tuning is being prepared so the project can learn Mulkiya-specific layout patterns and field wording from a larger internal dataset.

## Training Objective
The model should read a UAE Mulkiya image and return only valid JSON containing:
- `chassis_no`
- `model_year`
- `origin`
- `vehicle_description`

## Dataset Format
This dataset export includes:
- `train.csv`
- `val.csv`
- `test.csv`
- `finetune_train.jsonl`
- `finetune_val.jsonl`
- `finetune_test.jsonl`

Each JSONL row contains an image path and a user/assistant message pair.

## Train / Validation / Test Split
The dataset is split into train, validation, and test sets so the future fine-tuned model can be trained and evaluated consistently.

## Evaluation Metrics
Recommended evaluation metrics:
- exact chassis number match accuracy
- edit distance for chassis number prediction
- model year exact match
- origin exact match
- vehicle description exact match or normalized string match

## Expected Output Format
The fine-tuned model should return only valid JSON:

```json
{"chassis_no":"...","model_year":"...","origin":"...","vehicle_description":"..."}
```

## Production Warning
Fine-tuning should not replace the existing validation and correction logic.

The current RapidOCR pipeline remains the production baseline.

The first deployment target for a fine-tuned model should be fallback handling for failed or review-flagged cases, not direct replacement of the production OCR flow.
