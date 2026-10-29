# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow==13.0.0"]
# ///
"""Prepare cached test examples for the four-model demo; no model-based selection."""
import base64
import io
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data/source-samples.jsonl'
published = json.loads((ROOT / 'data/published-test-labels.json').read_text())
source_rows = [json.loads(line) for line in SOURCE.read_text().splitlines()]
assert len(source_rows) == len({row['id'] for row in source_rows}) == 100
for row in source_rows:
    assert row['split'] in {'test_task', 'test_website', 'test_domain'}
    assert published['splits'][row['split']][row['id']] == row['expected']
(ROOT / 'data/images').mkdir(parents=True, exist_ok=True)
examples = []
for index, row in enumerate(source_rows):
    payload = row['payload']
    state, question = payload['state'], payload['questions']['next_element']
    assert len(payload['images']) == 1, 'This demo prepares single-screenshot Mind2Web records.'
    raw = base64.b64decode(payload['images'][0].split(',', 1)[1])
    image = Image.open(io.BytesIO(raw)).convert('RGB')
    source_size = image.size
    ratio = min(1, 1568 / max(image.size))
    resized = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    canvas = (max(280, resized[0]), max(280, resized[1]))
    offset = ((canvas[0] - resized[0]) // 2, (canvas[1] - resized[1]) // 2)
    changed = canvas != image.size or resized != image.size
    name = f'{index}-0.png'
    path = ROOT / 'data/images' / name
    if changed:
        image = image.resize(resized, Image.Resampling.LANCZOS)
        prepared = Image.new('RGB', canvas, 'white')
        prepared.paste(image, offset)
        prepared.save(path)
        original_w, original_h = state['screenshot_size_pixels']
        sx, sy = resized[0] / original_w, resized[1] / original_h
        criteria = {}
        for key, description in question['criteria'].items():
            candidate = json.loads(description)
            x, y, w, h = candidate['bbox']
            candidate['bbox'] = [round(x * sx + offset[0], 4), round(y * sy + offset[1], 4),
                                 round(w * sx, 4), round(h * sy, 4)]
            criteria[key] = json.dumps(candidate)
        question = {**question, 'criteria': criteria}
        state = {**state, 'screenshot_size_pixels': list(canvas)}
    else:
        path.write_bytes(raw)
    assert row['expected'] in question['criteria']
    examples.append({'index': index, 'id': row['id'], 'split': row['split'], 'label': row['expected'],
                     'images': [name], 'image_sizes': [list(canvas)], 'source_size': list(source_size),
                     'prepared': changed, 'state': state, 'question': question})
(ROOT / 'data/examples.json').write_text(json.dumps(examples))
print(f'Prepared {len(examples)} test examples; {sum(r["prepared"] for r in examples)} resized/padded for shared API limits.')
