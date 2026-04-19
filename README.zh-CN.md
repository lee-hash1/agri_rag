# TARAG 本地可运行版（GPU）

这个仓库实现了你要的完整链路，并已验证可在 GPU 上运行：

1. 知识库清洗（本地大模型）
2. 问题解析（提取病虫害 + 时间）
3. 两阶段检索（BM25 disease Top-K + stage embedding 重排）
4. 生成答案（基于检索文档）
5. 答案校验（生成答案 vs 检索证据）
6. 批量推理（`query` 文件单条/小批/全量）

## 目录结构

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

## 环境准备（Windows）

### 1) 使用 `tarag_env`

```powershell
conda activate tarag_env
```

### 2) 依赖安装

如果你是本仓库当前环境，建议直接使用：

```powershell
pip install -r requirements.windows.txt
```

说明：
- `requirements.txt` 里有部分本机路径依赖（`@ file:///...`），不适合直接复现安装。
- `requirements.windows.txt` 是可安装版本（已去除 Windows 不兼容项）。

### 3) GPU 版 PyTorch（必须）

确保是 CUDA 版（不是 `+cpu`）：

```powershell
pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

检查 GPU：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

期望输出类似：
- `2.6.0+cu124`
- `True`
- `12.4`

## 模型目录

本项目默认使用以下子目录（建议显式传参）：

- LLM：`models/DeepSeek-R1-Distill-Qwen-1.5B`
- Embedding：`embedding_models/bge-m3`

## 数据格式

### 知识库清洗输出格式

```json
[
  {
    "disease": "烟草煤污病",
    "stage": "7、8月份",
    "treatment": "喷药防治蚜虫"
  }
]
```

### Query 文件格式（批处理）

```json
[
  {
    "golden_answer": "2359_2",
    "query": "8月份天气热了，桑蓟马是不是更容易暴发，该怎么防治？",
    "answer": "..."
  }
]
```

其中 `query` 字段必须存在且为非空字符串。

## 命令说明

### 1) 清洗知识库

```powershell
python main.py build-kb `
  --input data/sample_raw_docs.json `
  --output data/clean_docs.json `
  --model-dir models/DeepSeek-R1-Distill-Qwen-1.5B
```

### 2) 单问题推理

```powershell
python main.py ask `
  --question "移栽15天后烟草低头黑病怎么防治？" `
  --kb data/sample_clean_docs.json `
  --model-dir models/DeepSeek-R1-Distill-Qwen-1.5B `
  --embedding-model-dir embedding_models/bge-m3 `
  --bm25-top-k 100 `
  --rerank-top-k 10 `
  --generation-top-k 5
```

### 3) 批量推理（`ask-batch`）

新增命令：`ask-batch`，用于 query 文件批处理。

核心参数：
- `--query-file`：query JSON 路径（必填）
- `--kb`：知识库 JSON 路径（必填）
- `--output`：输出 JSON 文件（必填，单个 JSON 数组）
- `--start-index`：起始索引，默认 `0`
- `--limit`：处理条数，不传则处理到末尾
- `--progress-every`：每 N 条打印一次进度，默认 `10`

## 跑通流程（推荐）

### Step A：先跑 1 条

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

### Step B：再跑 10 条

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

### Step C：全量跑

把 `--query-file` 改为你的完整文件（例如 `data/query.json`），去掉 `--limit`：

```powershell
python main.py ask-batch `
  --query-file data/query.json `
  --kb data/sample_clean_docs.json `
  --model-dir models/DeepSeek-R1-Distill-Qwen-1.5B `
  --embedding-model-dir embedding_models/bge-m3 `
  --output output/batch_query_full.json
```

## `ask-batch` 输出结构

每条记录结构如下：

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

说明：
- `status=error` 时不会中断全批次，错误会记录在 `error` 字段。

## 常见问题

### 1) 报错：`torch.load` 安全限制（CVE）

原因：`torch` 太低。  
解决：升级到 `torch>=2.6`，并使用 CUDA 版 wheel。

### 2) 明明有 GPU 但仍走 CPU

检查：
- `torch.__version__` 是否是 `+cu124`（或其他 CUDA 后缀）
- `torch.cuda.is_available()` 是否为 `True`

### 3) `query.json` 不存在

先用 `data/query5.json` 跑通，再替换为你自己的全量路径。
