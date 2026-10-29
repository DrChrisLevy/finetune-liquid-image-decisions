"""Serve a fine-tuned d1-2B or d1-3B on an RTX PRO 6000.

Deploy: MODEL_RUN_ID=<run_id> uv run modal deploy src/decision_model/inference.py
POST /v1/systemone with state, questions, and images (base64 image data URLs).
"""
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import threading

import modal
from decision_model.liquid_d1_2b_image_finetune import ROOT, image, volume

app = modal.App("liquid-d1-image-inference")
RUN_ID = os.environ.get("MODEL_RUN_ID", "")


@app.server(
    image=image.uv_pip_install("fastapi[standard]").add_local_python_source("decision_model"),
    env={"MODEL_RUN_ID": RUN_ID},
    gpu="RTX-PRO-6000",
    cpu=4,
    memory=24576,
    volumes={str(ROOT): volume},
    port=8000,
    max_concurrency=16,
    scaledown_window=300,
    startup_timeout=600,
    unauthenticated=True,
)
class Inference:
    @modal.enter()
    def start(self):
        import torch
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from PIL import Image
        from pydantic import BaseModel, Field
        from transformers import AutoModel

        if not RUN_ID:
            raise ValueError("Set MODEL_RUN_ID to a completed training run before deploying.")
        run = ROOT / "runs" / RUN_ID
        model_id = json.loads((run / "training.json").read_text())["model"]
        image_arg = {"LiquidAI/d1-2B": "media", "LiquidAI/d1-3B": "images"}[model_id]
        model = AutoModel.from_pretrained(
            run / "merged", trust_remote_code=True, dtype=torch.bfloat16,
        ).to("cuda").eval()
        self.image_pool = ThreadPoolExecutor(max_workers=4)
        # d1-2B installs temporary attention hooks: serialize calls to the shared model.
        self.model_pool = ThreadPoolExecutor(max_workers=1)
        api = FastAPI()

        class Payload(BaseModel):
            state: dict = Field(default_factory=dict)
            questions: dict
            images: list[str] = Field(min_length=1)

        def decode(values):
            images = []
            for value in values:
                header, _, encoded = value.partition(",")
                if not header.startswith("data:image/") or not header.endswith(";base64"):
                    raise ValueError("Use base64 image data URLs.")
                with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
                    images.append(image.convert("RGB"))
            return images

        def predict(payload, images):
            with torch.inference_mode():
                return model.system_one(payload.state, payload.questions, **{image_arg: images})

        @api.get("/health")
        async def health():
            return {"status": "ok", "model": model_id, "run_id": RUN_ID}

        @api.post("/v1/systemone")
        async def decide(payload: Payload):
            loop = asyncio.get_running_loop()
            try:
                images = await loop.run_in_executor(self.image_pool, decode, payload.images)
            except (ValueError, OSError) as error:
                raise HTTPException(400, "Invalid image: use a base64 image data URL.") from error
            return await loop.run_in_executor(self.model_pool, predict, payload, images)

        self.server = uvicorn.Server(uvicorn.Config(api, host="0.0.0.0", port=8000, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()

    @modal.exit()
    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=10)
        self.image_pool.shutdown(wait=False, cancel_futures=True)
        self.model_pool.shutdown(wait=False, cancel_futures=True)
