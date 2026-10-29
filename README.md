# Fine-tuning Liquid Image Decision Models

Fine-tune Liquid decision models that take text and images as input, using Modal.

This repo contains the code for [my personal blog post](https://drchrislevy.com/blog/liquid-image-decisions) on fine-tuning Liquid image decision models.

## Motivation

This started with a [tweet from Leonie at Liquid](https://x.com/helloiamleonie/status/2110994738811715943):

> how do you fine-tune a decision model on your own data?

I adapted the [text-classification recipe she shared](https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/sft_for_decision_models_with_trl.ipynb)
to fine-tune [d1-3B](https://huggingface.co/LiquidAI/d1-3B) on text and images with [Modal](https://modal.com).
While I was working on the blog post, Liquid [announced LFM3](https://x.com/liquidai/status/2115128329003303107),
including the newer [d1-2B](https://huggingface.co/LiquidAI/d1-2B).
So this repo has a standalone training script for each model.

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

## Results

Next-element selection accuracy across all 4,157 Mind2Web test examples. Each model
trained for one epoch on 4,574 examples using an RTX PRO 6000; training time excludes
loading and evaluation.

| Model | Before fine-tuning | After fine-tuning | Training time |
| --- | ---: | ---: | ---: |
| d1-3B | 44.21% | 91.34% | ~44 min |
| d1-2B | 51.12% | 95.93% | ~20 min |

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

## Serve a fine-tuned model

Copy the `run_id` printed after training into `MODEL_RUN_ID`, then deploy:

```bash
export MODEL_RUN_ID="your-training-run-id"
uv run modal deploy src/decision_model/inference.py
```

`inference.py` reads that environment variable to load `runs/<run_id>/merged/`
from the Volume.
Either model runs on one RTX PRO 6000 with up to 16 concurrent requests and scales
down after five idle minutes. The model processes one request at a time.

Save this example as `payload.json`, replacing `<base64-encoded screenshot>` with
your PNG image encoded as base64:

```json
{
  "state": {"task_and_past_actions": "Search for flights from Halifax to Toronto."},
  "questions": {
    "next_element": {
      "type": "choice",
      "instructions": "Which element should the user interact with next?",
      "criteria": {
        "flights": "Flights tab at the top of the page",
        "hotels": "Hotels tab beside the Flights tab",
        "help": "Help link in the footer"
      }
    }
  },
  "images": ["data:image/png;base64,<base64-encoded screenshot>"]
}
```

Use the URL printed by Modal to send it. The response contains the chosen option
and its probabilities:

```bash
URL="https://your-deployment-url"
curl --retry 60 --retry-delay 1 --retry-max-time 180 --fail-with-body \
  "$URL/v1/systemone" -H 'Content-Type: application/json' \
  --data-binary @payload.json
```

The retries allow the model to start after the endpoint has scaled down.
