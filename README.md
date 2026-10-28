# Fine-tuning Liquid decision models on Modal

## Motivation

This started with a [tweet from Leonie at Liquid](https://x.com/helloiamleonie/status/2110994738811715943):

> how do you fine-tune a decision model on your own data?

I followed the [TRL recipe she shared](https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/sft_for_decision_models_with_trl.ipynb)
to fine-tune [d1-3B](https://huggingface.co/LiquidAI/d1-3B) on [Modal](https://modal.com).
While I was working on the blog post, Liquid [announced LFM3](https://x.com/liquidai/status/2115128329003303107),
including the newer [d1-2B](https://huggingface.co/LiquidAI/d1-2B).
So this repo has a standalone training script for each model.

**TODO:** Link the blog post once it's published.

## Setup

```bash
uv sync
uv run modal setup
```

Create a [Modal secret](https://modal.com/docs/guide/secrets) named `huggingface-secret`
with `HF_TOKEN` set to a Hugging Face read token.

## Run

Both scripts default to [Mind2Web](https://huggingface.co/datasets/chrislevy/mind2web-decisions).
These commands run a quick four-step training check:

```bash
# Original d1-3B
uv run modal run --detach src/decision_model/liquid_d1_3b_image_finetune.py

# Newer d1-2B
uv run modal run --detach src/decision_model/liquid_d1_2b_image_finetune.py
```

Add `--mode full` to either command for one pass through the training data.
Datasets and saved models are stored in the `liquid-d1-training` Modal Volume.
