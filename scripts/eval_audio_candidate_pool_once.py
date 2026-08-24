#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from pathlib import Path

import numpy as np
import faiss

corpus_path = "data/audio/old_audio_backend/audio/audio_corpus_l2.npy"
exact_dir = Path("data/audio/eval_exact_top500")
index_path = "data/audio/eval_ivfflat_top500/audio_ivfflat_nlist4096.index"
out_dir = Path("data/audio/eval_candidate_pool_once")
out_dir.mkdir(parents=True, exist_ok=True)

nprobe = 64
max_candidate_topk = 500
exact_topk = 500

x = np.load(corpus_path).astype("float32")
n, d = x.shape
print("loaded corpus:", x.shape)

qid_file = sorted(exact_dir.glob("audio_exact_query_ids_*.npy"))[0]
qids = np.load(qid_file)
q = x[qids].copy()

exact = np.load(exact_dir / f"audio_exact_top{exact_topk}_indices.npy")

index = faiss.read_index(index_path)
index.nprobe = nprobe

t0 = time.time()
scores, indices = index.search(q, max_candidate_topk + 1)
search_time = time.time() - t0

cand = []
for i, qid in enumerate(qids):
    row = []
    for idx in indices[i]:
        if idx == qid:
            continue
        row.append(idx)
        if len(row) == max_candidate_topk:
            break
    cand.append(row)

cand = np.array(cand, dtype=np.int64)

def cov(candidate_k, exact_k):
    vals = []
    for c, e in zip(cand[:, :candidate_k], exact[:, :exact_k]):
        vals.append(len(set(c.tolist()) & set(e.tolist())) / exact_k)
    return float(np.mean(vals))

results = {
    "num_vectors": int(n),
    "dim": int(d),
    "num_queries": int(len(qids)),
    "nprobe": nprobe,
    "search_once_topk": max_candidate_topk,
    "search_time_sec": float(search_time),
    "avg_latency_ms_per_query": float(search_time / len(qids) * 1000),
    "qps": float(len(qids) / search_time),
    "candidate_results": {}
}

for candidate_k in [100, 200, 500]:
    results["candidate_results"][str(candidate_k)] = {
        "exact_top10_coverage": cov(candidate_k, 10),
        "exact_top20_coverage": cov(candidate_k, 20),
        "exact_top50_coverage": cov(candidate_k, 50),
        "exact_top100_coverage": cov(candidate_k, 100),
        "exact_top500_coverage": cov(candidate_k, 500),
        "ckks_reduction_ratio": float(n / candidate_k),
    }

np.save(out_dir / "audio_candidate_search_once_top500_indices.npy", cand)

with open(out_dir / "audio_candidate_pool_once_nprobe64_top500.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(json.dumps(results, ensure_ascii=False, indent=2))
