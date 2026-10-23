# vlmeter — Accuracy × Cost Meter for Multimodal LLMs

English | [简体中文](README.zh-CN.md)

Evaluate vision-language models on **your own images and labels**, side by side — API models (GPT-4o, DeepSeek, GLM-4V, Qwen-VL, Claude, Gemini) and locally served ones (vLLM, Ollama) — and get a comparison report of **accuracy and cost** to drive model selection.

```bash
vlmeter run --models deepseek-flash,gpt-4o,qwen2.5-vl-7b --data data/sample.jsonl --judge deepseek-v4-pro
```

**Features**

- One dataset, many models: OpenAI-compatible endpoints (GPT-4o / DeepSeek / GLM-4V / Qwen-VL / vLLM / sglang / LM Studio), Anthropic Claude, Google Gemini, local Ollama — adding a model is a few lines in `vlmeter/config/models.yaml`, no code changes
- Three task types: `classification` (option matching), `checklist` (**multiple yes/no scene checks per image** — e.g. "bicycle present + checkout counter clean" — answered in a single call), `vqa` (open-ended, with exact / contains / numeric / LLM-judge scoring)
- Resumable runs: results cached per question; reruns only fill in what's missing; failed requests retry automatically
- **Cost accounting**: per-model token pricing × actual usage, with a built-in accuracy-vs-cost tradeoff table (total / per-call / per-1000-calls)
- Reports in two formats: Markdown + a **self-contained HTML report** (single file, no external deps, shareable) with per-check accuracy heatmaps and expandable wrong-case details

## Common use cases

1. **Model selection before launch** — public leaderboard scores don't predict your domain. Evaluate 200–500 real business images with labels; the accuracy-vs-cost table answers questions like "93% at ¥6/1k calls vs 98% at ¥80/1k calls" directly.
2. **Site-inspection compliance checks (checklist's home turf)** — one store/factory photo, many yes/no checks ("bicycles inside parking lines", "checkout counter free of clutter", "fire exit unblocked"). The same prompt structure carries straight into production.
3. **Annotation & criteria iteration** — subjective checks ("clean", "tidy") fail on ambiguous standards, not model ability. Read the model's `reason` in wrong cases; if the model "has a point", tighten the `criteria` and rerun to quantify the improvement per check.
4. **Regression testing against silent API upgrades** — API models get upgraded without notice. Freeze your dataset (results are cached by content fingerprint) and rerun monthly or on vendor announcements; accuracy drops surface immediately.
5. **Self-hosted vs API break-even** — add a local vLLM model (API cost = 0) to the same run; if it scores 95% vs the API's 97%, the tradeoff table tells you at what call volume giving up 2 points pays for the GPU.
6. **Vendor acceptance testing** — a supplier claims "99% accuracy". Run your own labeled acceptance set, break it down per check/category, and challenge cases line by line from the wrong-case list.

## Install

```bash
pip install vlmeter                     # or from source: pip install -r requirements.txt
cp .env.example .env                    # fill in the API keys you have (auto-loaded by run)
```

## Quick start (10 minutes)

```bash
python3 scripts/make_sample_data.py   # generate a 15-item synthetic sample set (shapes/colors/counting/OCR)
vlmeter list-models                   # see registered models
vlmeter run --models mock --data data/sample.jsonl      # offline self-test, costs nothing
vlmeter run --models deepseek-flash --data data/sample.jsonl --judge deepseek-v4-pro
vlmeter report --data data/sample.jsonl                # regenerate reports anytime
```

Every run writes a Markdown report (`results/report_*.md`) and a self-contained HTML report (`results/report_*.html` — double-click to open, safe to send to anyone). Raw per-question results live in `results/<model>/*.jsonl`.

## Using your own data

### Scene checks (recommended): many checks per image

The typical need: give one image, answer multiple yes/no checks — "is there a bicycle?", "is the checkout counter clean?". One data row = one image + all its checks, answered in **one model call** (saves N-1 calls):

```json
{"id": "shop001", "image": "images/shop001.jpg", "task": "checklist",
 "checks": [
   {"name": "bicycle present", "answer": true},
   {"name": "checkout counter clean", "answer": false,
    "criteria": "no litter, no visible stains, no clutter on the counter; light dust does not count"},
   {"name": "no clutter in store", "answer": true}
 ]}
```

- `answer` is the human-labeled ground truth (true/false)
- `name` is a short label (used as the report column and result ID); for **subjective checks always add `criteria`** — the judging standard goes into the prompt only, aligning the model with your annotators. This is the single biggest factor in subjective-check accuracy
- The model returns `verdict` (yes/no) + `reason` (one-sentence visual evidence) per check. Reasons are stored and shown in the report's wrong-case section, so you can tell "model was wrong" from "the standard was ambiguous"
- The report breaks accuracy down **per check name** (e.g. "checkout counter clean: 71%"), making weak scenes obvious
- Parsing tolerates format variants (bare yes/no, booleans, arrays); unanswered or garbage answers count as wrong and are tagged `parse_fail`

Labels work in Chinese or English — the sample data and prompts use Chinese; English content is handled equally well.

### Single-question mode

One row per question, for datasets where each image gets a different question. Prepare a JSONL (or CSV with the same columns), image paths relative to the dataset file:

```json
{"id": "q001", "image": "images/001.jpg", "task": "classification",
 "question": "Which product model is shown?", "choices": ["A1", "B2", "C3"],
 "answer": "B2", "category": "model-id"}
{"id": "q002", "image": "images/002.jpg", "task": "vqa",
 "question": "Where is the damage: top-left, top-right, bottom-left or bottom-right?",
 "answer": "bottom-left", "match": "judge", "category": "defect-location"}
{"id": "q003", "image": "images/003.jpg", "task": "vqa",
 "question": "How many safety hazards are visible? Answer with a number only.",
 "answer": "3", "match": "numeric", "category": "counting"}
```

Field reference:

| Field | Required | Notes |
|---|---|---|
| `image` | ✅ | Path to the image; auto-compressed if > 4MB or longest side > 1568px |
| `task` |  | `classification` (needs `choices`) / `checklist` (needs `checks`) / `vqa` (default) |
| `question` | cls/vqa | The question text |
| `answer` | cls/vqa | Ground truth; for classification use the option text or its letter |
| `choices` |  | Options for classification |
| `checks` |  | Checklist items `[{name, answer, criteria?}]`; answer is true/false |
| `match` |  | VQA scoring strategy; default is judge-if-configured, else contains |
| `category` |  | Business category for per-category breakdown (checklists use check names automatically) |
| `tolerance` |  | Allowed error for numeric matching, default 0 |

**Choosing a VQA scoring strategy**

- `numeric`: answers are numbers (counts, readings); Chinese numerals ("三个") supported
- `exact`: unique unambiguous answers (IDs, yes/no), case/punctuation-insensitive
- `contains`: gold answer must appear in the prediction (loose; good for keyword answers)
- `judge`: LLM-judge semantic scoring (recommended for open-ended answers). The judge only sees question/gold/prediction **text** — never the image — so any text model can judge. Most robust for long or open answers.

## Adding models

Edit `vlmeter/config/models.yaml` and add an entry (flip `enabled` to `true`):

```yaml
  glm-4v:
    enabled: true
    provider: openai                 # any OpenAI-compatible endpoint
    model: glm-4v-plus
    base_url: https://open.bigmodel.cn/api/paas/v4/
    api_key_env: ZHIPUAI_API_KEY     # read from .env / environment
    concurrency: 4
```

Local models, two ways:

```bash
# Option 1: vLLM (recommended, OpenAI-compatible)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000
# Option 2: Ollama
ollama pull llava
```

Preset entries `qwen2.5-vl-7b` (vLLM) and `llava` (Ollama) are already in the yaml — just enable them.

> Note: if your machine sets `OPENAI_BASE_URL` to a proxy, the official-OpenAI entry must hard-code `base_url: https://api.openai.com/v1/` with its own key variable, so keys never get sent to the wrong server.

## Running & reports

```bash
vlmeter run --models a,b,c --data data/my.jsonl \
    --judge deepseek-v4-pro \         # optional judge model (any text model)
    --limit 20 \                      # pilot run on the first 20 items
    --fresh                           # ignore cache, rerun everything
vlmeter report --data data/my.jsonl   # regenerate reports anytime
```

- Interrupted or rate-limited? Run the same command again — only failed/missing items are retried
- Result files are keyed by dataset content hash; switching datasets starts fresh files automatically
- Reasoning models (e.g. DeepSeek) can exhaust `max_tokens` on thinking and return empty output — the runner retries automatically; if it happens often, raise `max_tokens` in the model config

## Reading accuracy and cost

Both key metrics land directly in the report for selection decisions:

**Accuracy** (drill-down, so a single number can't mislead you):
- Overview: overall / classification / checklist / VQA accuracy
- Per-category table: **one column per check** (e.g. "checkout counter clean: 71%") to locate weak scenes
- Wrong cases: gold vs prediction + the model's own `reason`, separating real errors from ambiguous standards

**Cost** (measured, not guessed):
- Set `price_input` / `price_output` (per 1M tokens) in `vlmeter/config/models.yaml` — ⚠️ bundled prices are examples, verify against provider pricing pages
- Input/output tokens are recorded from each API response; checklist multi-check images are billed as **one call** (auto-dedup)
- The cost table: total for this run, per-call cost, **per-1000-calls** (production extrapolation), average latency — side by side with accuracy
- Local models (vLLM/Ollama): set price 0 — API cost 0 (GPU electricity is out of scope)
- Judge calls are billed separately (evaluation infrastructure, not the model under test)

**A typical tradeoff**: model A at 98% accuracy / ¥30 per 1000 calls vs model B at 93% / ¥3 — if 93% is acceptable (e.g. human review as a fallback), B is 10× cheaper; if errors are expensive (compliance checks), A wins. The report provides the numbers; the decision is business.

**Counting rules**

- **Accuracy** = correct decision points / scored decision points (a checklist image expands into multiple decision points; request failures and unscored items are excluded from the denominator)
- **Parse failures** (model ignored the letter format and couldn't be mapped to an option) count as wrong; with a judge configured, free-form answers get mapped back to options, greatly reducing this penalty
- **Wrong cases** list gold vs prediction per item — read these before drawing conclusions

## Project layout

```
vlmeter/                 the package
  models/              model adapters (openai-compat / anthropic / gemini / ollama / mock)
  dataset.py           dataset loading & validation
  prompting.py         per-task prompt templates
  scoring.py           option parsing + exact/contains/numeric scoring
  judge.py             LLM judge (VQA semantics + free-form classification mapping)
  runner.py            async concurrency, retries, resume
  report.py            Markdown report
  report_html.py       self-contained HTML report
vlmeter/config/models.yaml     model registry
scripts/make_sample_data.py    synthetic sample data
tests/                 scoring unit tests + adapter contract tests
```

## Tests

```bash
python3 tests/test_scoring.py      # scoring logic
python3 tests/test_adapters.py     # adapter request shape (local mock server, no keys needed)
```

## Publishing a release to PyPI

One-time setup:

1. Register a [PyPI](https://pypi.org) account and create an API token under **Account Settings → API tokens** (account scope; switch to project scope once the name is claimed)
2. Put the token in `~/.pypirc`:

   ```ini
   [distutils]
   index-servers =
       pypi

   [pypi]
   username = __token__
   password = pypi-your-token
   ```

Each release, four steps (example: 0.2.3):

```bash
# 1. Bump the version (both places)
sed -i 's/__version__ = "0.2.2"/__version__ = "0.2.3"/' vlmeter/__init__.py
sed -i 's/version = "0.2.2"/version = "0.2.3"/' pyproject.toml

# 2. Regression + build + check (clear dist/ first to avoid re-uploading old files)
python3 tests/test_scoring.py && rm -rf dist && python3 -m build && python3 -m twine check dist/*

# 3. Upload (~30s of PyPI index lag after upload is normal)
python3 -m twine upload dist/*

# 4. Commit, tag, push
git add vlmeter/__init__.py pyproject.toml && git commit -m "chore: release v0.2.3" && git push
git tag v0.2.3 && git push origin v0.2.3
```

Notes:

- **A version can never be re-uploaded**: PyPI does not allow overwriting published files — any code change requires a version bump
- If isolated builds fail (restricted networks), use `python3 -m build --no-isolation` with system setuptools
- `.github/workflows/publish.yml` is an alternate CI channel (manual trigger) once PyPI Trusted Publishing is configured; local twine is the primary path
- Local twine reads the token from `~/.pypirc`; nothing secret ever gets pushed to git

## Known limitations

- The Ollama adapter is verified against a contract mock but not a real Ollama instance (no GPU on the dev machine); the vLLM path shares the OpenAI-compat adapter already verified with DeepSeek
- The judge itself can misjudge; for critical conclusions, spot-check the wrong cases in the report
- The dataset fingerprint is based on the JSONL/CSV content — replacing an image file in place does not trigger a rerun (use `--fresh`)
