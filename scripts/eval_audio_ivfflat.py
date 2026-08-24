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
    ap.add_argument("--exact-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--nprobe-list", type=str, default="16,32,64,128,256,512")
    ap.add_argument("--topk", type=int, default=100)
    return ap.parse_args()


def overlap_at_k(approx, exact, k):
    vals = []
    for a, e in zip(approx[:, :k], exact[:, :k]):
        vals.append(len(set(a.tolist()) & set(e.tolist())) / k)
    return float(np.mean(vals))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exact_dir = Path(args.exact_dir)

    x = np.load(args.corpus).astype("float32")
    n, d = x.shape
    print("loaded corpus:", x.shape)

    qid_files = sorted(exact_dir.glob("audio_exact_query_ids_*.npy"))
    if not qid_files:
        raise FileNotFoundError("No audio_exact_query_ids_*.npy found in exact-dir")

    qids = np.load(qid_files[0])
    q = x[qids].copy()
    exact_indices = np.load(exact_dir / f"audio_exact_top{args.topk}_indices.npy")

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, args.nlist, faiss.METRIC_INNER_PRODUCT)

    print("training IVFFlat...")
    t0 = time.time()
    index.train(x)
    train_time = time.time() - t0

    print("adding vectors...")
    t0 = time.time()
    index.add(x)
    add_time = time.time() - t0

    results = []

    for nprobe in [int(v) for v in args.nprobe_list.split(",")]:
        index.nprobe = nprobe

        t0 = time.time()
        scores, indices = index.search(q, args.topk + 1)
        search_time = time.time() - t0

        clean = []
        for row_i, qid in enumerate(qids):
            row = []
            for idx in indices[row_i]:
                if idx == qid:
                    continue
                row.append(idx)
                if len(row) == args.topk:
                    break
            clean.append(row)
        clean = np.array(clean, dtype=np.int64)

        item = {
            "nlist": int(args.nlist),
            "nprobe": int(nprobe),
            "num_vectors": int(n),
            "dim": int(d),
            "num_queries": int(len(qids)),
            "topk": int(args.topk),
            "train_time_sec": float(train_time),
            "add_time_sec": float(add_time),
            "search_time_sec": float(search_time),
            "avg_latency_ms_per_query": float(search_time / len(qids) * 1000),
            "qps": float(len(qids) / search_time),
            "overlap_at_10": overlap_at_k(clean, exact_indices, 10),
            "overlap_at_20": overlap_at_k(clean, exact_indices, 20),
            "overlap_at_50": overlap_at_k(clean, exact_indices, 50),
            "overlap_at_100": overlap_at_k(clean, exact_indices, 100),
        }

        results.append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2))

    with open(out_dir / f"audio_ivfflat_nlist{args.nlist}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    faiss.write_index(index, str(out_dir / f"audio_ivfflat_nlist{args.nlist}.index"))
    print("saved index:", out_dir / f"audio_ivfflat_nlist{args.nlist}.index")


if __name__ == "__main__":
    main()
