from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def build_kb(args: argparse.Namespace) -> None:
    try:
        from tarag.cleaner import KnowledgeBaseCleaner
        from tarag.io_utils import load_raw_records, save_clean_docs
        from tarag.local_llm import LocalLLM
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. "
            "Please run `pip install -r requirements.txt` first."
        ) from exc

    llm = LocalLLM(
        model_dir=args.model_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    cleaner = KnowledgeBaseCleaner(
        llm=llm,
        batch_size=args.batch_size,
        passthrough_structured=args.passthrough_structured,
    )

    raw_records = load_raw_records(args.input)
    docs = cleaner.clean_records(raw_records)
    save_clean_docs(args.output, docs)

    print(f"Raw records: {len(raw_records)}")
    print(f"Clean docs : {len(docs)}")
    print(f"Saved to   : {Path(args.output).resolve()}")


def ask(args: argparse.Namespace) -> None:
    try:
        from tarag.io_utils import save_json
        from tarag.pipeline import TARAGPipeline
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. "
            "Please run `pip install -r requirements.txt` first."
        ) from exc

    pipeline = TARAGPipeline(
        kb_path=args.kb,
        model_dir=args.model_dir,
        embedding_model_dir=args.embedding_model_dir,
        bm25_top_k=args.bm25_top_k,
        rerank_top_k=args.rerank_top_k,
        generation_top_k=args.generation_top_k,
    )
    result = pipeline.run(args.question).to_dict()
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)

    if args.save:
        save_json(args.save, result)
        print(f"Saved to   : {Path(args.save).resolve()}")


def ask_batch(args: argparse.Namespace) -> None:
    try:
        from tarag.io_utils import load_query_records, save_json
        from tarag.pipeline import TARAGPipeline
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. "
            "Please run `pip install -r requirements.txt` first."
        ) from exc

    if args.start_index < 0:
        raise SystemExit("--start-index must be >= 0")
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be >= 1")

    query_records = load_query_records(args.query_file)
    total_records = len(query_records)
    if args.start_index >= total_records:
        raise SystemExit(
            f"--start-index ({args.start_index}) is out of range for {total_records} query records."
        )

    end_index = total_records
    if args.limit is not None:
        end_index = min(total_records, args.start_index + args.limit)

    selected_records = query_records[args.start_index:end_index]
    print(
        f"Loaded {total_records} query records; "
        f"processing range [{args.start_index}, {end_index}) "
        f"(count={len(selected_records)})."
    )

    pipeline = TARAGPipeline(
        kb_path=args.kb,
        model_dir=args.model_dir,
        embedding_model_dir=args.embedding_model_dir,
        bm25_top_k=args.bm25_top_k,
        rerank_top_k=args.rerank_top_k,
        generation_top_k=args.generation_top_k,
    )

    outputs: list[dict[str, object]] = []
    batch_start = time.perf_counter()
    for offset, record in enumerate(selected_records):
        global_index = args.start_index + offset
        question = record["query"]

        one_start = time.perf_counter()
        status = "ok"
        error = ""
        result_payload: dict[str, object] | None

        try:
            result_payload = pipeline.run(question).to_dict()
        except Exception as exc:  # noqa: BLE001
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            result_payload = None

        elapsed_seconds = round(time.perf_counter() - one_start, 6)
        outputs.append(
            {
                "index": global_index,
                "query": question,
                "golden_answer": record.get("golden_answer"),
                "reference_answer": record.get("answer"),
                "result": result_payload,
                "status": status,
                "error": error,
                "elapsed_seconds": elapsed_seconds,
            }
        )

        finished = offset + 1
        if finished % args.progress_every == 0 or finished == len(selected_records):
            print(
                f"[{finished}/{len(selected_records)}] "
                f"index={global_index} status={status} elapsed={elapsed_seconds:.3f}s"
            )

    total_elapsed = round(time.perf_counter() - batch_start, 3)
    save_json(args.output, outputs)
    print(f"Saved to   : {Path(args.output).resolve()}")
    print(f"Processed  : {len(outputs)} records in {total_elapsed}s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TARAG local framework CLI")
    subparsers = parser.add_subparsers(dest="command")

    kb_parser = subparsers.add_parser(
        "build-kb",
        help="Clean raw knowledge with local LLM into disease/stage/treatment docs.",
    )
    kb_parser.add_argument("--input", required=True, help="Raw input file (.json/.jsonl/.txt/.md)")
    kb_parser.add_argument("--output", required=True, help="Output JSON file for cleaned docs")
    kb_parser.add_argument("--model-dir", default="./models", help="Local LLM directory")
    kb_parser.add_argument("--batch-size", type=int, default=4, help="Batch size for LLM cleaning")
    kb_parser.add_argument("--max-new-tokens", type=int, default=1024, help="LLM max new tokens")
    kb_parser.add_argument("--temperature", type=float, default=0.2, help="LLM generation temperature")
    kb_parser.add_argument("--top-p", type=float, default=0.9, help="LLM top-p")
    kb_parser.add_argument(
        "--passthrough-structured",
        action="store_true",
        help="Skip LLM cleaning for already structured records.",
    )
    kb_parser.set_defaults(func=build_kb)

    ask_parser = subparsers.add_parser(
        "ask",
        help="Run TARAG pipeline: parse -> BM25 -> time embedding rerank -> generate -> verify.",
    )
    ask_parser.add_argument("--question", required=True, help="User question")
    ask_parser.add_argument("--kb", required=True, help="Clean knowledge base JSON path")
    ask_parser.add_argument("--model-dir", default="./models", help="Local LLM directory")
    ask_parser.add_argument(
        "--embedding-model-dir",
        default="./embedding_models",
        help="Local embedding model directory",
    )
    ask_parser.add_argument("--bm25-top-k", type=int, default=100, help="Top-k by BM25")
    ask_parser.add_argument("--rerank-top-k", type=int, default=10, help="Top-k after stage rerank")
    ask_parser.add_argument(
        "--generation-top-k",
        type=int,
        default=5,
        help="Top docs used by LLM for answer generation",
    )
    ask_parser.add_argument("--save", default="", help="Optional path to save JSON output")
    ask_parser.set_defaults(func=ask)

    ask_batch_parser = subparsers.add_parser(
        "ask-batch",
        help="Run TARAG pipeline for a query JSON file and save outputs as one JSON array.",
    )
    ask_batch_parser.add_argument("--query-file", required=True, help="Path to query JSON file")
    ask_batch_parser.add_argument("--kb", required=True, help="Clean knowledge base JSON path")
    ask_batch_parser.add_argument("--model-dir", default="./models", help="Local LLM directory")
    ask_batch_parser.add_argument(
        "--embedding-model-dir",
        default="./embedding_models",
        help="Local embedding model directory",
    )
    ask_batch_parser.add_argument("--start-index", type=int, default=0, help="Start index (inclusive)")
    ask_batch_parser.add_argument("--limit", type=int, default=None, help="Max records to process")
    ask_batch_parser.add_argument("--output", required=True, help="Output JSON file path")
    ask_batch_parser.add_argument("--bm25-top-k", type=int, default=100, help="Top-k by BM25")
    ask_batch_parser.add_argument("--rerank-top-k", type=int, default=10, help="Top-k after stage rerank")
    ask_batch_parser.add_argument(
        "--generation-top-k",
        type=int,
        default=5,
        help="Top docs used by LLM for answer generation",
    )
    ask_batch_parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N processed records",
    )
    ask_batch_parser.set_defaults(func=ask_batch)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
