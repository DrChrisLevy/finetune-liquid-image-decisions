"""Fine-tune Liquid d1-2B on a normalized Hugging Face image decision dataset.

Run: uv run modal run --detach src/decision_model/liquid_d1_2b_image_finetune.py
Defaults to Mind2Web and a smoke test; use --dataset to choose another dataset
and --mode full for one epoch. Requires train and validation splits with
id, images (List[Image]), state (Json), question (Json), and label (string).

Uses d1-2B's native prompt/media/attention helpers, with Liquid's LFM3 LoRA setup:
https://huggingface.co/LiquidAI/d1-2B
https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/lfm3_vl_sft_with_trl.ipynb
"""

import importlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import modal

VOLUME_NAME = "liquid-d1-training"
ROOT = Path(f"/mnt/{VOLUME_NAME}")
DEFAULT_DATASET = "chrislevy/mind2web-decisions"
MODEL = "LiquidAI/d1-2B"
REVISION = "97c57047d5852d77e02dfac211f8768aff77a8b2"
app = modal.App("liquid-d1-2b-image-finetune")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.14.0",
        "torchvision==0.29.0",
        "transformers==5.19.0",
        "trl==1.15.0",
        "peft==0.21.2",
        "datasets==4.8.4",
        "accelerate==1.15.0",
        "huggingface-hub==1.31.0",
        "safetensors==0.8.0",
        "pillow==12.3.0",
        "flash-linear-attention==0.5.2",
        "einops",
    )
    .env({"HF_HUB_CACHE": str(ROOT / "hf-hub")})
)


def prepare_dataset(dataset, processor, prompt):
    """Render d1 decisions; TRL handles image batching and completion-only labels."""
    from datasets import Image, List

    tokenizer = processor.tokenizer
    media = importlib.import_module(f"{prompt.__package__}.media")
    image_marker = media.template_markup(processor, "image")

    def render(row):
        if (
            not row["images"] or not isinstance(row["state"], dict)
            or row["question"]["type"] != "choice"
            or row["label"] not in row["question"]["criteria"]
        ):
            raise ValueError(f"Invalid image decision: {row['id']}")
        names = list(row["question"]["criteria"])
        random.Random(f"42:{row['id']}").shuffle(names)
        question = {
            **row["question"],
            "criteria": {name: row["question"]["criteria"][name] for name in names},
        }
        code, _ = dict(zip(names, prompt.aliases(tokenizer, names)))[row["label"]]
        text = prompt.render(
            tokenizer,
            row["state"],
            prompt.as_question(question),
            bos=tokenizer.bos_token or "",
            lead=prompt.default_lead("lfm3_vl", tokenizer),
            media=image_marker * len(row["images"]),
        )
        return {"prompt": text, "completion": code + tokenizer.eos_token}

    # Render text without decoding or re-encoding the embedded images.
    return (
        dataset.cast_column("images", List(Image(mode="RGB", decode=False)))
        .map(
            render,
            remove_columns=[c for c in dataset.column_names if c != "images"],
            writer_batch_size=16,
        )
        .cast_column("images", List(Image(mode="RGB")))
    )


@app.function(
    image=image,
    gpu="RTX-PRO-6000",
    cpu=4,
    memory=16384,
    timeout=4 * 60 * 60,
    volumes={str(ROOT): volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(dataset: str = DEFAULT_DATASET, mode: str = "smoke", revision: str = "main"):
    """Fine-tune, evaluate, and save the adapter and native d1 checkpoint."""
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoProcessor, set_seed
    from trl import SFTConfig, SFTTrainer

    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'")
    set_seed(42)
    data_revision = HfApi().dataset_info(dataset, revision=revision).sha
    data = load_dataset(
        dataset, revision=data_revision, token=True, cache_dir=str(ROOT / "hf-datasets")
    )
    train_rows, val_rows = data["train"], data["validation"]
    if mode == "smoke":
        train_rows = train_rows.select(range(min(16, len(train_rows))))
        val_rows = val_rows.select(range(min(4, len(val_rows))))
    checkpoint = Path(
        snapshot_download(
            MODEL,
            revision=REVISION,
            local_dir=str(ROOT / "model" / REVISION),
        )
    )
    # Load the native d1 class, but don't call system_one on this training instance:
    # creating its inference engine fuses projections that LoRA needs separately.
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, dtype=torch.bfloat16)
    package = type(model).__module__.rsplit(".", 1)[0]
    prompt = importlib.import_module(f"{package}.prompt")
    state_attention = importlib.import_module(f"{package}.attention").state_attention
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.d1_codes = model.config.d1_codes
    processor.tokenizer.d1_prompt = getattr(model.config, "d1_prompt", 1)
    train_dataset = prepare_dataset(train_rows, processor, prompt)
    val_dataset = prepare_dataset(val_rows, processor, prompt)

    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "o_proj",
                "out_proj",
                "in_proj",
                "w1",
                "w2",
                "w3",
            ],
            exclude_modules=r".*vision_tower.*",
        ),
    ).to("cuda")
    model.config.use_cache = False
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run_id = f"d1-2b-{mode}-{dataset.split('/')[-1]}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    print(f"Training {MODEL}: {run_id}", flush=True)

    class DecisionTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # With one question, d1's set attention reads the whole prompt both ways.
            # Answers stay causal and invisible to the prompt, preventing label leakage.
            prompt_lengths = (inputs["labels"] != -100).int().argmax(dim=1)
            with state_attention(model, prompt_lengths):
                return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    trainer = DecisionTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(run_dir / "trainer"),
            num_train_epochs=1,
            max_steps=4 if mode == "smoke" else -1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            per_device_eval_batch_size=2,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_steps=0.05,
            logging_steps=1 if mode == "smoke" else 10,
            eval_strategy="no",
            save_strategy="no",
            bf16=True,
            report_to="none",
            completion_only_loss=True,
            max_length=None,
            seed=42,
            data_seed=42,
            dataloader_num_workers=0,
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
    )
    before = trainer.evaluate()["eval_loss"]
    training_result = trainer.train()
    after = trainer.evaluate()["eval_loss"]
    optimizer_steps = trainer.state.global_step
    del trainer
    model.save_pretrained(run_dir / "adapter")
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged_dir = run_dir / "merged"
    merged.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)
    for source in checkpoint.iterdir():
        if source.suffix == ".py" or source.name == "LICENSE":
            shutil.copy2(source, merged_dir / source.name)
    config_path = merged_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["auto_map"] = json.loads((checkpoint / "config.json").read_text())[
        "auto_map"
    ]
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    report = {
        "run_id": run_id,
        "mode": mode,
        "label_source": "dataset.label",
        "dataset": dataset,
        "dataset_revision": data_revision,
        "model": MODEL,
        "model_revision": REVISION,
        "validation_split": "validation",
        "optimizer_steps": optimizer_steps,
        "trainable_parameters": trainable_parameters,
        "train_runtime_seconds": training_result.metrics["train_runtime"],
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "validation_loss_before": before,
        "validation_loss_after": after,
        "output_dir": str(run_dir),
    }
    (run_dir / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    print(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main(dataset: str = DEFAULT_DATASET, mode: str = "smoke", revision: str = "main"):
    call = train.spawn(dataset, mode, revision)
    print(f"Training call: {call.object_id}")
    print(call.get())
