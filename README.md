# Image decision model training

Fine-tune Liquid's image decision models with a standalone script for each model:

| Model | Architecture | Training script |
| --- | --- | --- |
| [d1-2B](https://huggingface.co/LiquidAI/d1-2B) | LFM3 | [`liquid_d1_2b_image_finetune.py`](src/decision_model/liquid_d1_2b_image_finetune.py) |
| [d1-3B](https://huggingface.co/LiquidAI/d1-3B) | LFM2.5-VL | [`liquid_d1_3b_image_finetune.py`](src/decision_model/liquid_d1_3b_image_finetune.py) |

Both scripts load the same Hugging Face dataset format, render Liquid's native
decision prompts, and use TRL for image batching and completion-only loss on the
answer code plus EOS. Each script works independently and supports both datasets.

d1-2B uses LFM3 with bidirectional global attention over each decision prompt. A small
`SFTTrainer` subclass applies the model's native attention mask while keeping
the answer hidden from the prompt. LoRA trains the language model, including
its attention gates. d1-3B uses the standard causal `SFTTrainer`. Both keep the
vision encoder frozen.

## Datasets

Both datasets have the same five columns:

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | --- |
| [Image segment decisions](https://huggingface.co/datasets/chrislevy/image-segment-decisions) | 9,969 | 997 | 3,000 |
| [Mind2Web](https://huggingface.co/datasets/chrislevy/mind2web-decisions) | 4,574 | 512 | task: 833; website: 634; domain: 2,690 |

```python
Features({
    "id": Value("string"),
    "images": List(Image(mode="RGB")),
    "state": Json(),
    "question": Json(),
    "label": Value("string"),
})
```

- `images`: ordered list of embedded PNGs, decoded to PIL images when loaded.
  Every current example has one image; the format supports multiple images.
- `state`: optional context as a JSON object. Image segment decisions use `{}`; Mind2Web
  includes the task, past actions, and original screenshot dimensions.
- `question`: `{"type": "choice", "instructions": "...", "criteria": {"option_key": "description", ...}}`.
  Candidate keys and descriptions can differ between examples.
- `label`: the correct key in that example's `question.criteria`.
- `id`: stable example identifier, also used to deterministically shuffle options.

The `Json` feature preserves variable object keys and their order. Use the pinned
`datasets==4.8.4` version; the Modal image already includes it. The trainer requires
`train` and `validation` splits. Test splits are reserved for evaluation.

Image segment decisions use images and segment descriptions, with the assigned
segment as the target. Mind2Web asks which candidate element to select; bounding
boxes use original screenshot coordinates, with those dimensions supplied in state.

The dataset cards include example payloads and streaming snippets.
Images are stored inside the dataset, so training needs
no backup ZIP, image URLs, or separate dataset upload to Modal.

## Setup

```bash
uv sync
uv run modal setup
```

Training uses your active Modal configuration. In your Modal workspace, create a Secret named
`huggingface-secret` with `HF_TOKEN` set to a Hugging Face token that can read the
datasets. Both scripts require this secret for public and private datasets.
See [Modal's Secrets guide](https://modal.com/docs/guide/secrets).

Both scripts create the `liquid-d1-training` Volume if needed and share its
dataset cache. Base models are cached by revision, and run names include the
model size to keep training outputs separate.

## Run

Mind2Web is the default dataset for both scripts. Choose a model and run a
four-step smoke test with 16 training and 4 validation records:

```bash
# d1-2B
uv run modal run --detach src/decision_model/liquid_d1_2b_image_finetune.py

# d1-3B
uv run modal run --detach src/decision_model/liquid_d1_3b_image_finetune.py
```

Both scripts accept the same options. Choose another dataset with `--dataset`:

```bash
uv run modal run --detach src/decision_model/liquid_d1_2b_image_finetune.py \
  --dataset chrislevy/image-segment-decisions
```

Use `--mode full` to train for one epoch. Optionally supply `--revision <commit>`
to select a specific dataset version. Each run resolves the requested revision
(default `main`) to a commit SHA and records it in `training.json`.

Each run saves `adapter/`, `merged/`, and `training.json` under
`liquid-d1-training:/runs/<run_id>/`. Both reports include the model and dataset
revisions and validation loss before and after training; this is not test-set
accuracy. The 2B report also includes training time and peak GPU memory.

Load a merged checkpoint with `AutoModel.from_pretrained(path, trust_remote_code=True)`
and call `model.system_one(state, {"decision": question}, media=images)` for d1-2B,
or `model.system_one(state, {"decision": question}, images=images)` for d1-3B.
For d1-2B, use a separate model instance for inference: initializing its native
inference engine fuses projections that LoRA needs separately during training.
Fine-tuned probabilities need their own calibration; the base model's calibration
is not copied into the merged checkpoint.
