# TARAG Local Runnable Version (GPU)

This repository implements the full pipeline and has been validated on GPU:

1. Knowledge base cleaning (local LLM)
2. Query parsing (disease/pest + time extraction)
3. Two-stage retrieval (BM25 over `disease` + embedding rerank over `stage`)
4. Answer generation (based on retrieved docs)
5. Answer verification (generated answer vs retrieved evidence)
6. Batch inference for query files (`ask-batch`)

## Project Layout

```text
agri/
├─ main.py
├─ tarag/
│  ├─ cleaner.py
│  ├─ io_utils.py
│  ├─ local_llm.py
│  ├─ pipeline.py
│  ├─ retriever.py
│  └─ schemas.py
├─ data/
│  ├─ query5.json
│  ├─ sample_raw_docs.json
│  └─ sample_clean_docs.json
├─ models/
│  └─ DeepSeek-R1-Distill-Qwen-1.5B
├─ embedding_models/
│  └─ bge-m3
└─ output/
```

## Environment Setup (Windows)

### 1) Use `tarag_env`

```powershell
conda activate tarag_env
```

### 2) Install dependencies

For this repo state, use:

```powershell
pip install -r requirements.windows.txt
```

Notes:
- `requirements.txt` contains machine-specific `@ file:///...` entries.
- `requirements.windows.txt` is the installable Windows-friendly version.

### 3) Install GPU PyTorch (required)

Make sure you install CUDA wheels (not `+cpu`):

```powershell
pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Check GPU runtime:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Expected output pattern:
- `2.6.0+cu124`
- `True`
- `12.4`

## Model Paths

Recommended explicit paths:

- LLM: `models/DeepSeek-R1-Distill-Qwen-1.5B`
- Embedding model: `embedding_models/bge-m3`

## Data Formats

### Cleaned KB format

```json
[
  {
    "disease": "Tobacco sooty mold",
    "stage": "July to August",
    "treatment": "Spray for aphid control"
  }
]
```

### Query file format (batch input)

```json
[
  {
    "golden_answer": "2359_2",
    "query": "In August, weather is hot. Is this when thrips outbreaks are more likely, and how should we control them?",
    "answer": "..."
  }
]
```

`query` is required and must be a non-empty string.

## Commands

### 1) Build/Clean KB

```powershell
python main.py build-kb `
  --input data/sample_raw_docs.json `
  --output data/clean_docs.json `
  --model-dir models/DeepSeek-R1-Distill-Qwen-1.5B
```

### 2) Single query inference

```powershell
python main.py ask `
  --question "How to control tobacco black shank 15 days after transplanting?" `
  --kb data/sample_clean_docs.json `
  --model-dir models/DeepSeek-R1-Distill-Qwen-1.5B `
  --embedding-model-dir embedding_models/bge-m3 `
  --bm25-top-k 100 `
  --rerank-top-k 10 `
  --generation-top-k 5
```

### 3) Batch inference (`ask-batch`)

Core args:
- `--query-file` (required)
- `--kb` (required)
- `--output` (required, one JSON array file)
- `--start-index` default `0`
- `--limit` optional (process to end if omitted)
- `--progress-every` default `10`

## Recommended Run Flow

### Step A: Run 1 record first

```powershell
python main.py ask-batch `
  --query-file data/query5.json `
  --kb data/sample_clean_docs.json `
  --model-dir models/DeepSeek-R1-Distill-Qwen-1.5B `
  --embedding-model-dir embedding_models/bge-m3 `
  --start-index 0 `
  --limit 1 `
  --output output/batch_query5_1.json `
  --progress-every 1
```

### Step B: Run 10 records

```powershell
python main.py ask-batch `
  --query-file data/query5.json `
  --kb data/sample_clean_docs.json `
  --model-dir models/DeepSeek-R1-Distill-Qwen-1.5B `
  --embedding-model-dir embedding_models/bge-m3 `
  --start-index 0 `
  --limit 10 `
  --output output/batch_query5_10.json `
  --progress-every 2
```

### Step C: Full run

Switch to your full query file (for example `data/query.json`) and remove `--limit`:

```powershell
python main.py ask-batch `
  --query-file data/query.json `
  --kb data/sample_clean_docs.json `
  --model-dir models/DeepSeek-R1-Distill-Qwen-1.5B `
  --embedding-model-dir embedding_models/bge-m3 `
  --output output/batch_query_full.json
```

## `ask-batch` Output Schema

Each item in the output array:

```json
{
  "index": 0,
  "query": "...",
  "golden_answer": "...",
  "reference_answer": "...",
  "result": {
    "question": "...",
    "parsed_query": {},
    "retrieved": [],
    "answer": "...",
    "verification": {}
  },
  "status": "ok",
  "error": "",
  "elapsed_seconds": 51.863
}
```

If `status=error`, processing still continues for remaining items.

## Troubleshooting

### `torch.load` security restriction error

Cause: old PyTorch version.  
Fix: upgrade to `torch>=2.6` with CUDA wheels.

### GPU is available but pipeline still uses CPU

Check:
- `torch.__version__` includes CUDA suffix (for example `+cu124`)
- `torch.cuda.is_available()` is `True`

### `data/query.json` not found

Start with `data/query5.json`, then switch path to your full dataset.
