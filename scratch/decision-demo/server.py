# /// script
# requires-python = ">=3.11"
# dependencies = ["fastapi>=0.115", "uvicorn>=0.34", "httpx>=0.28", "python-dotenv>=1.0"]
# ///
"""Local API proxy. Run: uv run server.py. Keys never reach the browser."""
import asyncio
import base64
import copy
import json
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

import httpx
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
ROWS = json.loads((ROOT / 'data/examples.json').read_text())
LIQUID_URL = 'https://drchrislevy--liquid-d1-image-inference-inference.us-east.modal.direct'
LIQUID_3B_URL = 'https://drchrislevy--liquid-d1-3b-image-inference-inference.us-east.modal.direct'
Provider = Literal['liquid', 'liquid3b', 'openai', 'jev']


def settings():
    # Re-read the local file so keys can be added without a restart.
    return {**dotenv_values(ROOT.parents[1] / '.env'), **dotenv_values(ROOT / '.env'), **os.environ}


def provider_config():
    env = settings()
    return {
        'liquid': {'name': 'Liquid d1-2B', 'model': 'Fine-tuned · RTX PRO 6000', 'enabled': True},
        'liquid3b': {'name': 'Liquid d1-3B', 'model': 'Fine-tuned · RTX PRO 6000',
                     'enabled': bool(env.get('LIQUID_3B_URL') or LIQUID_3B_URL), 'key_name': 'LIQUID_3B_URL'},
        'openai': {'name': 'OpenAI Decisions', 'model': 'gpt-6-luna',
                   'enabled': bool(env.get('OPENAI_API_KEY')), 'key_name': 'OPENAI_API_KEY'},
        'jev': {'name': 'TypeSafe JEV', 'model': env.get('JEV_MODEL') or 'jev-1.14.0',
                'enabled': bool(env.get('TYPESAFE_API_KEY') or env.get('JEV_API_KEY')),
                'key_name': 'TYPESAFE_API_KEY'},
    }


def example(index):
    if not 0 <= index < len(ROWS):
        raise HTTPException(404, 'Unknown example.')
    return ROWS[index]


@lru_cache(maxsize=128)
def image_url(name):
    return 'data:image/png;base64,' + base64.b64encode((ROOT / 'data/images' / name).read_bytes()).decode()


def payload_for(provider, row):
    """Only allow model inputs through: no ID, split, label, or previous answers."""
    images = [image_url(name) for name in row['images']]
    state, question = copy.deepcopy(row['state']), copy.deepcopy(row['question'])
    if provider in ('liquid', 'liquid3b'):
        return {'state': state, 'questions': {'next_element': question}, 'images': images}
    if provider == 'openai':
        return {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': [
            {'type': 'input_json', 'data': state},
            *[{'type': 'input_image', 'image_url': image} for image in images],
        ]}], 'questions': [{'name': 'next_element', 'type': 'choice',
                            'instructions': question['instructions'],
                            'choices': [{'value': k, 'description': v} for k, v in question['criteria'].items()]}]}
    if provider == 'jev':
        # Stable opaque references make preview/download match the sent request.
        refs = [str(uuid.uuid5(uuid.NAMESPACE_URL, f'{row["id"]}/image/{i}')) for i in range(len(images))]
        return {'model': provider_config()['jev']['model'], 'state': {**state, 'images': refs},
                'questions': {'next_element': question}, 'assets': {
                    ref: {'type': 'image', 'format': 'png', 'data_base64': image.split(',', 1)[1]}
                    for ref, image in zip(refs, images)}}
    raise ValueError('Unknown provider')


def compact_images(value):
    if isinstance(value, dict):
        return {k: (f'<{len(v):,} base64 characters>' if k == 'data_base64' else compact_images(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [compact_images(v) for v in value]
    if isinstance(value, str) and value.startswith('data:image/'):
        return value.split(',', 1)[0] + f',<{len(value.split(",", 1)[1]):,} base64 characters>'
    return value


def normalize_answer(provider, body, criteria):
    if provider == 'openai':
        answer = next(a for a in body['answers'] if a.get('name') == 'next_element')
        if answer.get('type') == 'refusal':
            raise ValueError('The model declined this question.')
        probabilities = {p['value']: p['probability'] for p in answer['probabilities']}
    else:
        answer = body['answers']['next_element']
        probabilities = answer.get('probabilities', {})
    choice = answer['choice']
    if choice not in criteria:
        raise ValueError('The provider returned a choice outside the candidates.')
    if any(k not in criteria or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
           for k, v in probabilities.items()):
        raise ValueError('The provider returned an invalid probability distribution.')
    return choice, probabilities


@asynccontextmanager
async def lifespan(app):
    async with httpx.AsyncClient(timeout=90, limits=httpx.Limits(max_connections=16, max_keepalive_connections=16)) as client:
        app.state.client = client
        yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', 'testserver'])


@app.middleware('http')
async def local_origin(request, call_next):
    # No cross-origin API calls from unrelated websites on the same machine.
    origin = request.headers.get('origin')
    if origin and origin not in {'http://localhost:5178', 'http://127.0.0.1:5178',
                                 'http://localhost:8018', 'http://127.0.0.1:8018'}:
        return JSONResponse({'detail': 'Use the local demo page.'}, status_code=403)
    return await call_next(request)


@app.get('/api/config')
async def config():
    return {'providers': provider_config(), 'dataset': 'chrislevy/mind2web-decisions', 'count': len(ROWS)}


@app.get('/api/examples')
async def examples():
    return [{**r, 'images': ['/images/' + name for name in r['images']]} for r in ROWS]


@app.get('/api/payload/{index}/{provider}')
async def preview(index: int, provider: Provider, full: bool = False):
    payload = payload_for(provider, example(index))
    return payload if full else compact_images(payload)


@app.post('/api/warm')
async def warm(provider: Literal['liquid', 'liquid3b'] = 'liquid'):
    start = time.perf_counter()
    key, default = ('LIQUID_URL', LIQUID_URL) if provider == 'liquid' else ('LIQUID_3B_URL', LIQUID_3B_URL)
    url = (settings().get(key) or default).rstrip('/')
    if not url:
        raise HTTPException(409, f'Set {key} on the local backend.')
    while time.perf_counter() - start < 600:
        try:
            response = await app.state.client.get(url + '/health', timeout=20)
            if response.is_success:
                body = response.json()
                return {'status': 'ready', 'model': body.get('model'), 'run_id': body.get('run_id')}
            if response.status_code not in (429, 502, 503, 504):
                raise HTTPException(502, f'Liquid health check returned HTTP {response.status_code}.')
        except httpx.RequestError:
            pass
        await asyncio.sleep(2)
    raise HTTPException(504, 'Liquid has not started yet. Check the Modal dashboard and try again.')


class Predict(BaseModel):
    index: int


@app.post('/api/predict/{provider}')
async def predict(provider: Provider, request: Predict):
    row = example(request.index)
    if not provider_config()[provider]['enabled']:
        raise HTTPException(409, f'Set {provider_config()[provider]["key_name"]} on the local backend.')
    env = settings()
    headers = {}
    if provider in ('liquid', 'liquid3b'):
        key, default = ('LIQUID_URL', LIQUID_URL) if provider == 'liquid' else ('LIQUID_3B_URL', LIQUID_3B_URL)
        url = (env.get(key) or default).rstrip('/') + '/v1/systemone'
    elif provider == 'openai':
        url = 'https://api.openai.com/v1/decisions'
        headers['Authorization'] = 'Bearer ' + env['OPENAI_API_KEY']
    else:
        url = 'https://api.typesafe.ai/v1/systemone'
        headers = {'Authorization': 'Bearer ' + (env.get('TYPESAFE_API_KEY') or env['JEV_API_KEY']),
                   'TypeSafe-Danger-Zone-Images': 'true'}
    # Serialize/read images before timing; measure one complete service HTTP round trip.
    content = json.dumps(payload_for(provider, row)).encode()
    start = time.perf_counter()
    try:
        response = await app.state.client.post(url, content=content,
                                               headers={**headers, 'Content-Type': 'application/json'})
        latency = (time.perf_counter() - start) * 1000
        if not response.is_success:
            # Never echo upstream text: it can contain payloads or credentials.
            tips = {401: 'Check the API key.', 403: 'Check model access for this key.',
                    429: 'Rate limited; try again shortly.', 503: 'The GPU is starting; try again shortly.',
                    400: 'The provider rejected the input. Check its image/model limits.',
                    422: 'The provider rejected the request format.'}
            raise HTTPException(502, f'Provider HTTP {response.status_code}. {tips.get(response.status_code, "Try again.")}')
        body = response.json()
        choice, probabilities = normalize_answer(provider, body, row['question']['criteria'])
        return {'provider': provider, 'index': row['index'], 'choice': choice, 'correct': choice == row['label'],
                'expected': row['label'], 'latency_ms': round(latency, 2), 'probabilities': probabilities,
                'model': body.get('model') or provider_config()[provider]['name'], 'raw': body}
    except httpx.TimeoutException as e:
        raise HTTPException(504, 'Provider timed out. If Liquid is starting, wait a moment and try again.') from e
    except httpx.RequestError as e:
        raise HTTPException(502, 'Connection to the provider failed. Try again.') from e
    except (ValueError, KeyError, StopIteration, TypeError) as e:
        raise HTTPException(502, 'Provider returned an unexpected answer or a refusal.') from e


app.mount('/images', StaticFiles(directory=ROOT / 'data/images'), name='images')
if (ROOT / 'dist').exists():
    app.mount('/', StaticFiles(directory=ROOT / 'dist', html=True), name='frontend')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8018)
