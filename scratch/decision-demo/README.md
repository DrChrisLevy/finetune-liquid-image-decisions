# Local decision demo

Local React/FastAPI demo comparing Liquid d1-2B, Liquid d1-3B, OpenAI Decisions, and TypeSafe JEV on the same 100 Mind2Web test examples. Backed up on the `scratch/decision-demo` branch.

From the repository root, prepare the demo once:

```bash
cd scratch/decision-demo
npm ci
uv run download_samples.py
uv run prepare.py
```

`sample_ids.json` fixes the dataset revision, example IDs, and order. The download uses Hugging Face's local cache; the first run downloads the dataset files. Images and generated data stay local.

Then use two terminals in this directory:

```bash
uv run server.py
```

```bash
npm run dev
```

Open **http://127.0.0.1:5178**. Add your keys to `.env` at the repository root (or in this scratch directory):

```dotenv
OPENAI_API_KEY=your-openai-key
TYPESAFE_API_KEY=your-typesafe-key
```

Keys are only read by Python. Both `.env` locations are ignored by Git. The backend rereads them on each request; configuration refreshes when you click **Compare models**. The scratch `.env` overrides the repository `.env`; shell environment variables take precedence over both. Optional `JEV_MODEL` defaults to `jev-1.14.0`; `LIQUID_URL` and `LIQUID_3B_URL` override the 2B and 3B deployments respectively.

The two Liquid models run on separate RTX PRO 6000 endpoints. They load the latest completed full Mind2Web checkpoints found in the `liquid-d1-training` Volume on October 29, 2026:

- d1-2B: `d1-2b-full-mind2web-decisions-20261029T005818Z`
- d1-3B: `d1-3b-full-mind2web-decisions-20261029T002341Z`

The endpoints are pinned to those runs. To use your own checkpoints, deploy them with the repository's inference script and set `LIQUID_URL` and `LIQUID_3B_URL` in `.env` to the URLs printed by Modal.

Both deployments use the unchanged repository inference script:

```bash
MODEL_RUN_ID=d1-2b-full-mind2web-decisions-20261029T005818Z uv run modal deploy ../../src/decision_model/inference.py
MODEL_RUN_ID=d1-3b-full-mind2web-decisions-20261029T002341Z uv run modal deploy --name liquid-d1-3b-image-inference ../../src/decision_model/inference.py
```

Run an initial comparison to warm the GPU, then **Reset session** before recording. Choose an example and click **Compare models**. Requests run concurrently; each card updates when its provider finishes. Select a predicted element to locate it on the screenshot. Inspect or download the exact provider payload. Labels only score responses; they are never sent to providers.

**Run 100** starts a fresh evaluation of the fixed sample, with four concurrent requests per example. Examples run sequentially. Each model's accuracy, correct/answered count, error count, and median latency update as responses arrive. Earlier comparisons do not affect this run. **Stop** finishes the current example and preserves the partial totals. Errors are shown separately and excluded from accuracy. Refreshing the page clears the session.

The sample contains 100 distinct held-out records: 34 from `test_task`, 33 from `test_website`, and 33 from `test_domain`, shuffled with seed 42. All IDs and labels were checked against the published dataset at revision `8d64ec24100a31724ea1d36baef9cfcdcb516b4d`, with no ID overlap with training or validation. The original 30 examples are included. Selection does not use model predictions. To meet JEV's image limits, screenshots are resized proportionally when necessary to a maximum side of 1568 pixels and padded to a minimum side of 280. Candidate coordinates are adjusted. All providers receive the same images, task, and candidates. The source dataset is untouched. This demo's accuracy is not directly comparable to the published full-test benchmark.

Latency measures one complete HTTP request from the local Python backend to the provider and back, including upload and network time. Image preparation happens before the timer. No automatic inference retries or response cache. Provider-side caching may apply. The Liquid health check loads the GPU but doesn't run a scored inference.

Live APIs: [OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions) (`gpt-6-luna`), [JEV images](https://docs.typesafe.ai/images) (image opt-in header), and the existing native Liquid Modal endpoint. The full 100-example UI run was verified on 2026-10-29: Liquid and JEV answered all 100; OpenAI answered 99 and refused example 68. That example remains in the sample. Expand failed requests in the run summary to inspect errors.

Offline checks:

```bash
uv run --with fastapi --with httpx --with python-dotenv --with pillow python test_demo.py
node --test test_evaluation.mjs
npm run build
```
