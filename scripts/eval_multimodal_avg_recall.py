#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np


ROOT = Path("/home/wen/private-vector-search")


CONFIGS = [
    {
        "mode": "image",
        "corpus_path": ROOT / "artifacts/image/embeddings/image_corpus_l2.npy",
        "query_path": None,
        "index_path": ROOT / "artifacts/image/index/image_ivfflat_nlist256.index",
        "nlist": 256,
        "nprobe": 16,
    },
    {
        "mode": "text",
        "corpus_path": ROOT / "artifacts/text/embeddings/text_corpus_l2.npy",
        "query_path": ROOT / "artifacts/text/embeddings/text_queries_l2.npy",
        "index_path": ROOT / "artifacts/text/index/text_ivfflat_nlist1024.index",
        "nlist": 1024,
        "nprobe": 64,
    },
    {
        "mode": "audio",
        "corpus_path": ROOT / "artifacts/audio/embeddings/corpus/audio_corpus_l2.npy",
        "query_path": ROOT / "artifacts/audio/embeddings/query/audio_query_l2.npy",
        "index_path": ROOT / "artifacts/audio/index/audio_icbhi_ivfflat.index",
        "nlist": 32,
        "nprobe": 8,
    },
]


def overlap_mean(exact_ids, cand_ids, exact_k):
    values = []
    for exact_row, cand_row in zip(exact_ids[:, :exact_k], cand_ids):
        exact_set = set(map(int, exact_row))
        cand_set = set(map(int, cand_row))
        values.append(len(exact_set & cand_set) / exact_k)
    return float(np.mean(values))


def order_same_rate(exact_ids, ivf_ids, k):
    values = []
    for exact_row, ivf_row in zip(exact_ids[:, :k], ivf_ids[:, :k]):
        values.append(bool(np.array_equal(exact_row, ivf_row)))
    return float(np.mean(values))


def load_queries(config, corpus, query_count, seed):
    query_path = config.get("query_path")
    rng = np.random.default_rng(seed)

    if query_path and Path(query_path).exists():
        queries = np.load(query_path, mmap_mode="r").astype("float32")
        if query_count and len(queries) > query_count:
            ids = rng.choice(len(queries), query_count, replace=False)
            queries = queries[ids]
        return queries, "external_query"

    count = min(query_count, len(corpus))
    ids = rng.choice(len(corpus), count, replace=False)
    return np.asarray(corpus[ids], dtype="float32"), "self_sample_from_corpus"


def eval_one(config, query_count, seed):
    mode = config["mode"]
    print(f"\n===== {mode} =====", flush=True)

    index = faiss.read_index(str(config["index_path"]))
    corpus = np.load(config["corpus_path"], mmap_mode="r").astype("float32")

    if index.ntotal < len(corpus):
        print(
            f"[WARN] truncate corpus from {len(corpus)} to index.ntotal={index.ntotal}",
            flush=True,
        )
        corpus = corpus[: index.ntotal]

    queries, query_source = load_queries(config, corpus, query_count, seed)
    dim = corpus.shape[1]

    exact = faiss.IndexFlatIP(dim)
    exact.add(corpus)

    t0 = time.time()
    _, exact_ids = exact.search(queries, 100)
    exact_ms = (time.time() - t0) * 1000 / len(queries)

    index.nprobe = config["nprobe"]
    t0 = time.time()
    _, ivf_ids = index.search(queries, 100)
    ivf_ms = (time.time() - t0) * 1000 / len(queries)

    result = {
        "mode": mode,
        "corpus_vectors": int(len(corpus)),
        "query_vectors": int(len(queries)),
        "dimension": int(dim),
        "query_source": query_source,
        "nlist": int(config["nlist"]),
        "nprobe": int(config["nprobe"]),
        "recall_at_20_ivf_top20": overlap_mean(exact_ids, ivf_ids[:, :20], 20),
        "cover_exact_top20_by_ivf_top100": overlap_mean(exact_ids, ivf_ids[:, :100], 20),
        "recall_at_100_ivf_top100": overlap_mean(exact_ids, ivf_ids[:, :100], 100),
        "top20_order_same_rate": order_same_rate(exact_ids, ivf_ids, 20),
        "exact_ms_per_query": float(exact_ms),
        "ivfflat_ms_per_query": float(ivf_ms),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/recall/avg_recall_summary_1000.json",
    )
    args = parser.parse_args()

    results = [eval_one(config, args.query_count, args.seed) for config in CONFIGS]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== saved =====")
    print(args.out)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
