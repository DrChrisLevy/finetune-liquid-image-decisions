# Image decision model training

Fine-tune Liquid d1-3B on image decisions with one dataset-independent script:
[`liquid_d1_image_finetune.py`](src/decision_model/liquid_d1_image_finetune.py).
It loads a Hugging Face dataset, renders Liquid's native decision prompts, and
uses TRL for image batching and completion-only loss on the answer code plus EOS.

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

Training uses your active Modal configuration. Create a Modal Secret named
`huggingface-secret` with `HF_TOKEN` set to a Hugging Face token that can read the
datasets. This secret is already configured in the current workspace.
See [Modal's Secrets guide](https://modal.com/docs/guide/secrets).

The script creates the `liquid-d1-training` Volume if needed and uses it for
caching datasets, the base model, and training outputs.

## Run

Run a four-step smoke test with 16 training and 4 validation records:

```bash
uv run modal run --detach src/decision_model/liquid_d1_image_finetune.py \
  --dataset chrislevy/image-segment-decisions --mode smoke

uv run modal run --detach src/decision_model/liquid_d1_image_finetune.py \
  --dataset chrislevy/mind2web-decisions --mode smoke
```

Use `--mode full` to train for one epoch. Optionally supply `--revision <commit>`
to select a specific dataset version. Each run resolves the requested revision
(default `main`) to a commit SHA and records it in `training.json`.

Each run saves `adapter/`, `merged/`, and `training.json` under
`liquid-d1-training:/runs/<run_id>/`. The report includes validation loss before
and after training; this is not test-set accuracy.
