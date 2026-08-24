#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import time
from pathlib import Path

import numpy as np
import faiss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--num-queries", type=int, default=2000)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.load(args.corpus).astype("float32")
    n, d = x.shape
    print("loaded corpus:", x.shape, x.dtype)

    rng = np.random.default_rng(args.seed)
    qids = rng.choice(n, size=min(args.num_queries, n), replace=False)
    q = x[qids].copy()

    # topk + 1 是因为库内查询会搜到自己，后面去掉 self
    index = faiss.IndexFlatIP(d)
    index.add(x)

    t0 = time.time()
    scores, indices = index.search(q, args.topk + 1)
    search_time = time.time() - t0

    clean_indices = []
    clean_scores = []

    for row_i, qid in enumerate(qids):
        inds = indices[row_i]
        scs = scores[row_i]

        keep_i = []
        keep_s = []
        for idx, score in zip(inds, scs):
            if idx == qid:
                continue
            keep_i.append(idx)
            keep_s.append(score)
            if len(keep_i) == args.topk:
                break

        clean_indices.append(keep_i)
        clean_scores.append(keep_s)

    clean_indices = np.array(clean_indices, dtype=np.int64)
    clean_scores = np.array(clean_scores, dtype=np.float32)

    np.save(out_dir / f"audio_exact_query_ids_{len(qids)}.npy", qids.astype(np.int64))
    np.save(out_dir / f"audio_exact_top{args.topk}_indices.npy", clean_indices)
    np.save(out_dir / f"audio_exact_top{args.topk}_scores.npy", clean_scores)

    metrics = {
        "num_vectors": int(n),
        "dim": int(d),
        "num_queries": int(len(qids)),
        "topk": int(args.topk),
        "search_time_sec": float(search_time),
        "avg_latency_ms_per_query": float(search_time / len(qids) * 1000),
        "qps": float(len(qids) / search_time),
    }

    with open(out_dir / "audio_exact_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
