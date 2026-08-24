#!/usr/bin/env python3
import argparse
import csv
import json
import time
from pathlib import Path

import faiss
import numpy as np


HOME = Path("/home/wen")
ROOT = HOME / "private-vector-search"
RESULTS_ROOT = ROOT / "results/plain_exact"

MODES = {
    "image": {
        "corpus": ROOT / "artifacts/image/embeddings/image_corpus_l2.npy",
        "query": None,
        "id_map": ROOT / "artifacts/image/embeddings/id_mapping.json",
        "query_source": "self_sample_from_corpus",
    },
    "text": {
        "corpus": ROOT / "artifacts/text/embeddings/text_corpus_l2.npy",
        "query": ROOT / "artifacts/text/embeddings/text_queries_l2.npy",
        "id_map": ROOT / "artifacts/text/embeddings/text_id_map.json",
        "query_source": "external_query",
    },
    "audio": {
        "corpus": ROOT / "artifacts/audio/embeddings/corpus/audio_corpus_l2.npy",
        "query": ROOT / "artifacts/audio/embeddings/query/audio_query_l2.npy",
        "id_map": ROOT / "artifacts/audio/embeddings/corpus/id_mapping.json",
        "query_source": "external_query",
    },
}


def normalize_rows(x):
    x = np.asarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def load_queries(config, corpus, query_index, num_queries):
    if config["query"] is None:
        end = min(query_index + num_queries, len(corpus))
        queries = np.asarray(corpus[query_index:end], dtype="float32").copy()
        query_ids = list(range(query_index, end))
        return normalize_rows(queries), query_ids

    all_queries = np.load(config["query"], mmap_mode="r")
    end = min(query_index + num_queries, len(all_queries))
    queries = np.asarray(all_queries[query_index:end], dtype="float32").copy()
    query_ids = list(range(query_index, end))
    return normalize_rows(queries), query_ids


def load_id_map(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def display_id(mode, id_map, row_id):
    item = id_map.get(str(int(row_id)))
    if isinstance(item, dict):
        if mode == "image":
            return item.get("external_id", str(row_id))
        if mode == "text":
            return item.get("title") or item.get("doc_id") or str(row_id)
        if mode == "audio":
            return item.get("cycle_id", str(row_id))
    if item is None:
        return str(row_id)
    return str(item)


def run_mode(mode, topk, num_queries, query_index, out_root):
    config = MODES[mode]
    corpus = np.load(config["corpus"], mmap_mode="r").astype("float32")
    queries, query_ids = load_queries(config, corpus, query_index, num_queries)
    id_map = load_id_map(config["id_map"])

    index = faiss.IndexFlatIP(corpus.shape[1])
    index.add(corpus)

    start = time.time()
    scores, ids = index.search(queries, topk)
    ms_per_query = (time.time() - start) * 1000 / len(queries)

    out_dir = out_root / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"exact_top{topk}_detail.csv"
    summary_path = out_dir / f"exact_top{topk}_summary.json"
    ids_path = out_dir / f"exact_top{topk}_indices.npy"
    scores_path = out_dir / f"exact_top{topk}_scores.npy"

    np.save(ids_path, ids.astype("int64"))
    np.save(scores_path, scores.astype("float32"))

    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "query_index",
                "rank",
                "row_id",
                "display_id",
                "score",
            ],
        )
        writer.writeheader()
        for qi, qid in enumerate(query_ids):
            for rank, (row_id, score) in enumerate(zip(ids[qi], scores[qi]), start=1):
                writer.writerow(
                    {
                        "mode": mode,
                        "query_index": qid,
                        "rank": rank,
                        "row_id": int(row_id),
                        "display_id": display_id(mode, id_map, row_id),
                        "score": float(score),
                    }
                )

    summary = {
        "mode": mode,
        "metric": "IndexFlatIP exact inner-product search over extracted embeddings",
        "corpus_path": str(config["corpus"]),
        "query_path": str(config["query"]) if config["query"] is not None else None,
        "query_source": config["query_source"],
        "corpus_vectors": int(corpus.shape[0]),
        "dimension": int(corpus.shape[1]),
        "num_queries": int(len(queries)),
        "query_index_start": int(query_index),
        "topk": int(topk),
        "ms_per_query": float(ms_per_query),
        "indices_path": str(ids_path),
        "scores_path": str(scores_path),
        "detail_path": str(detail_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run plaintext exact retrieval over extracted embeddings only."
    )
    parser.add_argument("--mode", choices=[*MODES.keys(), "all"], default="all")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()

    modes = list(MODES) if args.mode == "all" else [args.mode]
    summaries = [
        run_mode(mode, args.topk, args.num_queries, args.query_index, args.out)
        for mode in modes
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    combined_path = args.out / f"exact_top{args.topk}_summary_all.json"
    combined_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"combined summary -> {combined_path}")


if __name__ == "__main__":
    main()
