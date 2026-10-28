# Fine-tuning Liquid Image Decision Models

Fine-tune Liquid decision models that take text and images as input, using Modal.

## Motivation

This started with a [tweet from Leonie at Liquid](https://x.com/helloiamleonie/status/2110994738811715943):

> how do you fine-tune a decision model on your own data?

I adapted the [text-classification recipe she shared](https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/sft_for_decision_models_with_trl.ipynb)
to fine-tune [d1-3B](https://huggingface.co/LiquidAI/d1-3B) on text and images with [Modal](https://modal.com).
While I was working on the blog post, Liquid [announced LFM3](https://x.com/liquidai/status/2115128329003303107),
including the newer [d1-2B](https://huggingface.co/LiquidAI/d1-2B).
So this repo has a standalone training script for each model.

**TODO:** Link the blog post once it's published.

## Setup

```bash
uv sync
uv run modal setup
```

## Sample runs

Both scripts default to [Mind2Web](https://huggingface.co/datasets/chrislevy/mind2web-decisions).
These commands train on a small sample: 16 training examples, 4 validation examples,
and 4 training steps. Use them to check that setup works.

```bash
# Original d1-3B
uv run modal run --detach src/decision_model/liquid_d1_3b_image_finetune.py

# Newer d1-2B
uv run modal run --detach src/decision_model/liquid_d1_2b_image_finetune.py
```

## Full runs

Train either model on the full Mind2Web training split for one pass through the data:

```bash
# Original d1-3B
uv run modal run --detach src/decision_model/liquid_d1_3b_image_finetune.py --mode full

# Newer d1-2B
uv run modal run --detach src/decision_model/liquid_d1_2b_image_finetune.py --mode full
```

Datasets and saved models are stored in the `liquid-d1-training` Modal Volume.

## Evaluation

Evaluate either base model on all Mind2Web test splits:

```bash
uv run modal run --detach src/decision_model/evaluate.py --model 3b
uv run modal run --detach src/decision_model/evaluate.py --model 2b
```

To evaluate a fine-tuned model, use the `run_id` printed at the end of training:

```bash
uv run modal run --detach src/decision_model/evaluate.py --run-id <run_id>
```

This loads `runs/<run_id>/merged/` from the Volume and automatically uses the
model and dataset version recorded in `training.json`. The `adapter/` folder
holds the LoRA weights; `merged/` contains the complete model for inference.
Evaluation compares the model's chosen option with each test label, prints
accuracy, and saves the report under `evaluations/` in the same Volume.
