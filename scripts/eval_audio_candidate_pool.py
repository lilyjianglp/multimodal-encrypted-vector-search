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
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--exact-dir", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nprobe", type=int, default=64)
    ap.add_argument("--candidate-topk", type=int, default=500)
    ap.add_argument("--exact-topk", type=int, default=500)
    return ap.parse_args()


def coverage(candidate, exact, k):
    vals = []
    for c, e in zip(candidate, exact):
        vals.append(len(set(c.tolist()) & set(e[:k].tolist())) / k)
    return float(np.mean(vals))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.load(args.corpus).astype("float32")
    n, d = x.shape
    print("loaded corpus:", x.shape)

    exact_dir = Path(args.exact_dir)
    qid_file = sorted(exact_dir.glob("audio_exact_query_ids_*.npy"))[0]
    qids = np.load(qid_file)
    q = x[qids].copy()

    exact_indices = np.load(exact_dir / f"audio_exact_top{args.exact_topk}_indices.npy")

    index = faiss.read_index(args.index)
    index.nprobe = args.nprobe

    t0 = time.time()
    scores, indices = index.search(q, args.candidate_topk + 1)
    search_time = time.time() - t0

    cand_indices = []
    cand_scores = []

    for i, qid in enumerate(qids):
        row_idx = []
        row_score = []
        for idx, score in zip(indices[i], scores[i]):
            if idx == qid:
                continue
            row_idx.append(idx)
            row_score.append(score)
            if len(row_idx) == args.candidate_topk:
                break
        cand_indices.append(row_idx)
        cand_scores.append(row_score)

    cand_indices = np.array(cand_indices, dtype=np.int64)
    cand_scores = np.array(cand_scores, dtype=np.float32)

    metrics = {
        "num_vectors": int(n),
        "dim": int(d),
        "num_queries": int(len(qids)),
        "nprobe": int(args.nprobe),
        "candidate_topk": int(args.candidate_topk),
        "search_time_sec": float(search_time),
        "avg_latency_ms_per_query": float(search_time / len(qids) * 1000),
        "qps": float(len(qids) / search_time),
        "exact_top10_coverage_by_candidate": coverage(cand_indices, exact_indices, 10),
        "exact_top20_coverage_by_candidate": coverage(cand_indices, exact_indices, 20),
        "exact_top50_coverage_by_candidate": coverage(cand_indices, exact_indices, 50),
        "exact_top100_coverage_by_candidate": coverage(cand_indices, exact_indices, 100),
        "exact_top500_coverage_by_candidate": coverage(cand_indices, exact_indices, 500),
        "ckks_reduction_ratio": float(n / args.candidate_topk),
    }

    np.save(out_dir / f"audio_candidate_top{args.candidate_topk}_indices.npy", cand_indices)
    np.save(out_dir / f"audio_candidate_top{args.candidate_topk}_scores.npy", cand_scores)

    out_json = out_dir / f"audio_candidate_pool_nprobe{args.nprobe}_top{args.candidate_topk}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
