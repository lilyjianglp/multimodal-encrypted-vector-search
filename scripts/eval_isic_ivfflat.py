import json
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


META_PATH = Path("/mnt/d/datasets/isic2020/metadata_clean.csv")
EMB_PATH = Path("/mnt/d/datasets/isic2020/embeddings/image_corpus_l2.npy")
EXACT_TOP20_PATH = Path("/mnt/d/datasets/isic2020/eval_exact/exact_top20_indices.npy")
OUT_DIR = Path("/mnt/d/datasets/isic2020/eval_ivfflat")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def metrics(query_labels, retrieved_labels, ks=(1, 5, 10, 20)):
    ans = {}
    for k in ks:
        same = retrieved_labels[:, :k] == query_labels[:, None]
        ans[f"precision@{k}"] = float(same.mean())
        ans[f"hit@{k}"] = float(same.any(axis=1).mean())
    return ans


def overlap(approx_indices, exact_indices, ks=(1, 5, 10, 20)):
    ans = {}
    for k in ks:
        vals = []
        for a, e in zip(approx_indices[:, :k], exact_indices[:, :k]):
            vals.append(len(set(a.tolist()) & set(e.tolist())) / k)
        ans[f"overlap@{k}"] = float(np.mean(vals))
    return ans


def main():
    df = pd.read_csv(META_PATH)
    x = np.load(EMB_PATH).astype("float32")

    train_mask = df["split"].values == "train"
    df_train = df[train_mask].reset_index(drop=True)
    x_train = x[train_mask]
    labels = df_train["label"].astype(str).values

    exact_top20 = np.load(EXACT_TOP20_PATH)

    print("train embeddings:", x_train.shape)
    print(df_train["label"].value_counts())

    dim = x_train.shape[1]
    nlist = 256

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    print(f"training IVFFlat: nlist={nlist}")
    t0 = time.time()
    index.train(x_train)
    print("train time:", time.time() - t0)

    print("adding vectors")
    t0 = time.time()
    index.add(x_train)
    print("add time:", time.time() - t0)
    print("ntotal:", index.ntotal)

    all_results = {}

    for nprobe in [4, 8, 16, 32, 64, 128, 256]:
        index.nprobe = nprobe

        print(f"\nsearching nprobe={nprobe}")
        t0 = time.time()
        scores, indices = index.search(x_train, 21)
        elapsed = time.time() - t0

        clean_indices = []
        for i in range(len(x_train)):
            inds = indices[i]
            keep = inds != i
            inds = inds[keep][:20]

            if len(inds) < 20:
                inds = np.pad(inds, (0, 20 - len(inds)), constant_values=-1)

            clean_indices.append(inds)

        clean_indices = np.vstack(clean_indices)
        safe_indices = np.where(clean_indices < 0, 0, clean_indices)
        retrieved_labels = labels[safe_indices]

        result = {
            "nprobe": nprobe,
            "nlist": nlist,
            "search_time_sec": float(elapsed),
            "avg_latency_ms_per_query": float(elapsed / len(x_train) * 1000),
            "qps": float(len(x_train) / elapsed),
            "overall": metrics(labels, retrieved_labels),
            "exact_overlap": overlap(clean_indices, exact_top20),
            "by_label": {},
        }

        for lab in sorted(set(labels)):
            mask = labels == lab
            result["by_label"][lab] = metrics(labels[mask], retrieved_labels[mask])
            result["by_label"][lab]["num_queries"] = int(mask.sum())

        all_results[f"nprobe_{nprobe}"] = result
        print(json.dumps(result, indent=2, ensure_ascii=False))

    with open(OUT_DIR / "ivfflat_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\nsaved to:", OUT_DIR)


if __name__ == "__main__":
    main()
