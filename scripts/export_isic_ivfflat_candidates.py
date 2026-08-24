import json
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


META_PATH = Path("/mnt/d/datasets/isic2020/metadata_clean.csv")
EMB_PATH = Path("/mnt/d/datasets/isic2020/embeddings/image_corpus_l2.npy")

INDEX_DIR = Path("/mnt/d/datasets/isic2020/index")
CAND_DIR = Path("/mnt/d/datasets/isic2020/candidates")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
CAND_DIR.mkdir(parents=True, exist_ok=True)

nlist = 256
nprobe = 16
topk = 100


def main():
    df = pd.read_csv(META_PATH)
    x = np.load(EMB_PATH).astype("float32")

    train_mask = df["split"].values == "train"
    df_train = df[train_mask].reset_index(drop=True)
    x_train = x[train_mask]

    print("train embeddings:", x_train.shape)

    dim = x_train.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    print(f"training IVFFlat: nlist={nlist}")
    t0 = time.time()
    index.train(x_train)
    train_time = time.time() - t0

    print("adding vectors")
    t0 = time.time()
    index.add(x_train)
    add_time = time.time() - t0

    index.nprobe = nprobe

    print(f"searching candidates: nprobe={nprobe}, topk={topk}")
    t0 = time.time()
    scores, indices = index.search(x_train, topk + 1)
    search_time = time.time() - t0

    clean_indices = []
    clean_scores = []
    for i in range(len(x_train)):
        inds = indices[i]
        scs = scores[i]
        keep = inds != i
        inds = inds[keep][:topk]
        scs = scs[keep][:topk]

        if len(inds) < topk:
            pad = topk - len(inds)
            inds = np.pad(inds, (0, pad), constant_values=-1)
            scs = np.pad(scs, (0, pad), constant_values=-1e9)

        clean_indices.append(inds)
        clean_scores.append(scs)

    clean_indices = np.vstack(clean_indices).astype("int64")
    clean_scores = np.vstack(clean_scores).astype("float32")

    index_path = INDEX_DIR / "image_ivfflat_nlist256.index"
    faiss.write_index(index, str(index_path))

    np.save(CAND_DIR / "ivfflat_nprobe16_top100_indices.npy", clean_indices)
    np.save(CAND_DIR / "ivfflat_nprobe16_top100_scores.npy", clean_scores)

    summary = {
        "index": "IVFFlat",
        "nlist": nlist,
        "nprobe": nprobe,
        "topk": topk,
        "num_vectors": int(len(x_train)),
        "dim": int(dim),
        "train_time_sec": train_time,
        "add_time_sec": add_time,
        "search_time_sec": search_time,
        "avg_latency_ms_per_query": search_time / len(x_train) * 1000,
        "qps": len(x_train) / search_time,
        "index_path": str(index_path),
        "candidate_indices_path": str(CAND_DIR / "ivfflat_nprobe16_top100_indices.npy"),
        "candidate_scores_path": str(CAND_DIR / "ivfflat_nprobe16_top100_scores.npy"),
    }

    with open(CAND_DIR / "ivfflat_nprobe16_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
