"""Image-aware decision fine-tuning with Liquid d1-3B on Modal.

Run: uv run modal run --detach src/decision_model/liquid_d1_image_finetune.py --mode smoke
Use --mode full for the 9,969/997 split and one epoch. This reproduces our
LoRA experiment, not Liquid's original d1 training. Training and validation
targets are the original Brand DNA segment assignments, joined by record ID.

Motivation 
https://x.com/helloiamleonie/status/2110994738811715943?s=20
https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/sft_for_decision_models_with_trl.ipynb
"""

import gzip
import hashlib
import importlib
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal

VOLUME_NAME = "liquid-d1-training"
ROOT = Path(f"/mnt/{VOLUME_NAME}")
REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"
app = modal.App("liquid-d1-image-finetune")
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
    .env({"HF_HUB_DISABLE_TELEMETRY": "1"})
)


def source_split(split, dataset_dir=ROOT / "dataset"):
    """Read a checksum-verified original Brand DNA benchmark split."""
    source = dataset_dir / "source-benchmark-v2"
    manifest = json.loads((source / "manifest.json").read_text())
    data = (source / f"{split}.jsonl.gz").read_bytes()
    if manifest["label_source"] != "brand_dna_assignment":
        raise ValueError("Expected Brand DNA ground-truth labels")
    if hashlib.sha256(data).hexdigest() != manifest["splits"][split]["sha256"]:
        raise ValueError(f"{split}: source checksum mismatch")
    rows = [json.loads(line) for line in gzip.decompress(data).splitlines()]
    if len(rows) != manifest["splits"][split]["records"]:
        raise ValueError(f"{split}: source record count mismatch")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"{split}: duplicate source IDs")
    return rows, manifest


def load_brand_dna_split(split, dataset_dir=ROOT / "dataset"):
    """Reuse prepared inputs, replacing every target with its source assignment."""
    source, _ = source_split(split, dataset_dir)
    by_id = {row["id"]: row for row in source}
    rows = json.loads((dataset_dir / "demo-data-20261023" / f"{split}.json").read_text())
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"{split}: empty split or duplicate prepared IDs")
    for row in rows:
        reference = by_id[row["id"]]  # Missing ground truth must fail, never fall back.
        if row["state"] != reference["state"] or row["question"] != reference["questions"]["segment"]:
            raise ValueError(f"{split}: inputs differ from source for {row['id']}")
        label = reference["expected"]["segment"]
        if label not in row["question"]["criteria"]:
            raise ValueError(f"{split}: invalid Brand DNA target for {row['id']}")
        row["label"] = label
    return rows


def prepare_dataset(rows, processor, prompt, data_dir):
    """Render d1 decisions; TRL handles image batching and completion-only labels."""
    from datasets import Dataset, Image, List

    tokenizer = processor.tokenizer
    examples = []
    for row in rows:
        names = list(row["question"]["criteria"])
        random.Random(f"42:{row['id']}").shuffle(names)
        question = {
            **row["question"],
            "criteria": {name: row["question"]["criteria"][name] for name in names},
        }
        code, _ = dict(zip(names, prompt.aliases(tokenizer, names)))[row["label"]]
        content = [{"type": "image"} for _ in row["images"]]
        content.append({"type": "text", "text": "\x00"})
        template = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=False,
        )
        head = "<|im_start|>user\n"
        marker = template[template.index(head) + len(head) : template.index("\x00")]
        text = prompt.render(
            tokenizer,
            row["state"],
            prompt.as_question(question),
            bos=tokenizer.bos_token or "",
            images=marker,
        )
        examples.append(
            {
                "prompt": text,
                "completion": code + tokenizer.eos_token,
                "images": [str(data_dir / path) for path in row["images"]],
            }
        )
    return Dataset.from_list(examples).cast_column("images", List(Image(mode="RGB")))


@app.function(
    image=image,
    gpu="RTX-PRO-6000",
    cpu=4,
    memory=16384,
    timeout=4 * 60 * 60,
    volumes={str(ROOT): volume},
)
def train(mode: str = "smoke"):
    """Fine-tune, evaluate, and save the adapter and native d1 checkpoint."""
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
    from trl import SFTConfig, SFTTrainer

    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'")
    set_seed(42)
    data_dir = ROOT / "dataset/demo-data-20261023"
    train_rows = load_brand_dna_split("train")
    val_rows = load_brand_dna_split("validation")
    if mode == "smoke":
        train_rows, val_rows = train_rows[:16], val_rows[:4]
    checkpoint = Path(
        snapshot_download(
            "LiquidAI/d1-3B",
            revision=REVISION,
            local_dir=str(ROOT / "model" / REVISION),
        )
    )
    sys.path.insert(0, str(checkpoint))
    prompt = importlib.import_module("prompt")
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    train_dataset = prepare_dataset(train_rows, processor, prompt, data_dir)
    val_dataset = prepare_dataset(val_rows, processor, prompt, data_dir)
    print("Example prompt:", train_dataset[0]["prompt"][:1000])

    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, dtype=torch.bfloat16
    )
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
    run_id = f"{mode}-brand-dna-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    trainer = SFTTrainer(
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
    trainer.train()
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
        "label_source": "brand_dna_assignment",
        "optimizer_steps": optimizer_steps,
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
def main(mode: str = "smoke"):
    call = train.spawn(mode)
    print(f"Training call: {call.object_id}")
