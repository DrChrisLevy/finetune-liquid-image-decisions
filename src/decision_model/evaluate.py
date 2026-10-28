"""Evaluate a Liquid image decision model on every test split of a normalized dataset.

Base model: uv run modal run --detach src/decision_model/evaluate.py --model 3b
Fine-tuned: uv run modal run --detach src/decision_model/evaluate.py --run-id <run_id>
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import modal
from decision_model import liquid_d1_2b_image_finetune as d1_2b
from decision_model import liquid_d1_3b_image_finetune as d1_3b

app = modal.App("liquid-d1-image-evaluate")
ROOT, volume = d1_2b.ROOT, d1_2b.volume
MODELS = {"2b": d1_2b, "3b": d1_3b}


@app.function(
    image=d1_2b.image.add_local_python_source("decision_model"),
    gpu="RTX-PRO-6000",
    cpu=4,
    memory=24576,
    timeout=2 * 60 * 60,
    volumes={str(ROOT): volume},
)
def evaluate(model: str = "2b", run_id: str = "", dataset: str = d1_2b.DEFAULT_DATASET,
             revision: str = "main"):
    """Load base weights or a saved merged model and measure test accuracy."""
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModel, set_seed

    volume.reload()
    set_seed(42)
    if run_id:
        run = json.loads((ROOT / "runs" / run_id / "training.json").read_text())
        model = run["model"].rsplit("-", 1)[-1].lower()
        dataset, revision = run["dataset"], run["dataset_revision"]
        model_revision = run["model_revision"]
        checkpoint = ROOT / "runs" / run_id / "merged"
    else:
        if model not in MODELS:
            raise ValueError("Choose --model 2b or --model 3b.")
        model_revision = MODELS[model].REVISION
        checkpoint = Path(snapshot_download(
            MODELS[model].MODEL, revision=model_revision,
            local_dir=str(ROOT / "model" / model_revision),
        ))
    if model not in MODELS:
        raise ValueError(f"Unsupported model: {model}")
    data_revision = HfApi().dataset_info(dataset, revision=revision).sha
    data = load_dataset(dataset, revision=data_revision, cache_dir=str(ROOT / "hf-datasets"))
    splits = [name for name in data if name.startswith("test")]
    if not splits:
        raise ValueError("The dataset needs a test split (for example, test or test_task).")

    loaded = AutoModel.from_pretrained(
        checkpoint, trust_remote_code=True, dtype=torch.bfloat16,
    ).to("cuda").eval()
    scores = {}
    with torch.inference_mode():
        for split in splits:
            correct = 0
            for index, row in enumerate(data[split]):
                images = {"media" if model == "2b" else "images": row["images"]}
                result = loaded.system_one(row["state"], {"next_element": row["question"]}, **images)
                choice = result["answers"]["next_element"]["choice"]
                if choice not in row["question"]["criteria"]:
                    raise ValueError(f"Invalid prediction for {row['id']}: {choice}")
                correct += choice == row["label"]
                if (index + 1) % 100 == 0:
                    print(f"{split}: {index + 1}/{len(data[split])} evaluated", flush=True)
            scores[split] = {"correct": correct, "total": len(data[split])}
    total = sum(score["total"] for score in scores.values())
    if not total:
        raise ValueError("The test splits are empty.")
    correct = sum(score["correct"] for score in scores.values())
    report = {
        "model": MODELS[model].MODEL,
        "model_revision": model_revision,
        "run_id": run_id or None,
        "dataset": dataset,
        "dataset_revision": data_revision,
        "correct": correct,
        "total": total,
        "accuracy_percent": 100 * correct / total,
        "splits": scores,
    }
    name = f"d1-{model}-{'finetuned' if run_id else 'base'}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output = ROOT / "evaluations" / f"{name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report["output_path"] = str(output)
    output.write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


@app.local_entrypoint()
def main(model: str = "2b", run_id: str = "", dataset: str = d1_2b.DEFAULT_DATASET,
         revision: str = "main"):
    report = evaluate.remote(model, run_id, dataset, revision)
    print(json.dumps(report, indent=2))
