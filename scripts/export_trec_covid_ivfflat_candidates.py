import json
import time
from pathlib import Path

import faiss
import numpy as np


CORPUS_EMB = Path("/mnt/d/datasets/text/trec_covid/embeddings/text_corpus_l2.npy")
QUERY_EMB = Path("/mnt/d/datasets/text/trec_covid/embeddings/text_queries_l2.npy")

INDEX_DIR = Path("/mnt/d/datasets/text/trec_covid/index")
CAND_DIR = Path("/mnt/d/datasets/text/trec_covid/candidates")

INDEX_DIR.mkdir(parents=True, exist_ok=True)
CAND_DIR.mkdir(parents=True, exist_ok=True)

nlist = 1024
nprobe = 64
topk = 100


def main():
    corpus = np.load(CORPUS_EMB).astype("float32")
    queries = np.load(QUERY_EMB).astype("float32")

    print("corpus:", corpus.shape)
    print("queries:", queries.shape)

    dim = corpus.shape[1]

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    print(f"training IVFFlat: nlist={nlist}")
    t0 = time.time()
    index.train(corpus)
    train_time = time.time() - t0

    print("adding corpus")
    t0 = time.time()
    index.add(corpus)
    add_time = time.time() - t0

    index.nprobe = nprobe

    print(f"searching candidates: nprobe={nprobe}, topk={topk}")
    t0 = time.time()
    scores, indices = index.search(queries, topk)
    search_time = time.time() - t0

    index_path = INDEX_DIR / "text_ivfflat_nlist1024.index"
    faiss.write_index(index, str(index_path))

    # 导出 IVF centers，后面给 client_pick_clusters.py 用
    centers = np.zeros((nlist, dim), dtype="float32")
    for i in range(nlist):
        centers[i] = index.quantizer.reconstruct(i)

    centers_path = INDEX_DIR / "text_ivfflat_nlist1024_centers.npy"
    np.save(centers_path, centers)

    cand_indices_path = CAND_DIR / "text_ivfflat_nprobe64_top100_indices.npy"
    cand_scores_path = CAND_DIR / "text_ivfflat_nprobe64_top100_scores.npy"

    np.save(cand_indices_path, indices.astype("int64"))
    np.save(cand_scores_path, scores.astype("float32"))

    summary = {
        "index": "IVFFlat",
        "nlist": nlist,
        "nprobe": nprobe,
        "topk": topk,
        "num_corpus": int(corpus.shape[0]),
        "num_queries": int(queries.shape[0]),
        "dim": int(dim),
        "train_time_sec": float(train_time),
        "add_time_sec": float(add_time),
        "search_time_sec": float(search_time),
        "avg_latency_ms_per_query": float(search_time / len(queries) * 1000),
        "qps": float(len(queries) / search_time),
        "index_path": str(index_path),
        "centers_path": str(centers_path),
        "candidate_indices_path": str(cand_indices_path),
        "candidate_scores_path": str(cand_scores_path),
    }

    with open(CAND_DIR / "text_ivfflat_nprobe64_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
