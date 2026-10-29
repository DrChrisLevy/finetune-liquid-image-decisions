# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets==5.1.0", "pillow==13.0.0"]
# ///
"""Download the demo's exact 100 test records from its pinned public dataset.

Run once before prepare.py: uv run download_samples.py
"""
import base64
import io
import json
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent


def main(output=None):
    output = Path(output) if output is not None else ROOT / "data"
    manifest = json.loads((ROOT / "sample_ids.json").read_text())
    records = manifest["records"]
    assert len(records) == len({row["id"] for row in records}) == 100
    splits = {row["split"] for row in records}
    assert splits == {"test_task", "test_website", "test_domain"}
    published = {"dataset": manifest["dataset"], "revision": manifest["revision"], "splits": {}}
    found = {}
    for split in sorted(splits):
        data = load_dataset(manifest["dataset"], revision=manifest["revision"], split=split)
        ids, labels = data["id"], data["label"]
        published["splits"][split] = dict(zip(ids, labels))
        wanted = {row["id"] for row in records if row["split"] == split}
        indices = [i for i, row_id in enumerate(ids) if row_id in wanted]
        assert len(indices) == len(wanted), f"Missing sample IDs in {split}"
        for row in data.select(indices):
            assert row["label"] in row["question"]["criteria"]
            images = []
            for image in row["images"]:
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="PNG")
                images.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
            found[row["id"]] = {
                "id": row["id"], "split": split, "expected": row["label"],
                "image_sizes": [list(image.size) for image in row["images"]],
                "payload": {"state": row["state"], "questions": {"next_element": row["question"]},
                            "images": images},
            }
        print(f"Loaded {len(indices)} examples from {split}.", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "source-samples.jsonl").write_text(
        "".join(json.dumps(found[row["id"]]) + "\n" for row in records)
    )
    (output / "published-test-labels.json").write_text(json.dumps(published) + "\n")
    print("Downloaded 100 test examples. Next: uv run prepare.py")


if __name__ == "__main__":
    main()
