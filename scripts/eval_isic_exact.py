import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


META_PATH = Path("/mnt/d/datasets/isic2020/metadata_clean.csv")
EMB_PATH = Path("/mnt/d/datasets/isic2020/embeddings/image_corpus_l2.npy")
OUT_DIR = Path("/mnt/d/datasets/isic2020/eval_exact")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def topk_label_precision(query_labels, retrieved_labels, ks=(1, 5, 10, 20)):
    results = {}
    for k in ks:
        same = retrieved_labels[:, :k] == query_labels[:, None]
        results[f"precision@{k}"] = float(same.mean())
        results[f"hit@{k}"] = float(same.any(axis=1).mean())
    return results


def main():
    df = pd.read_csv(META_PATH)
    x = np.load(EMB_PATH).astype("float32")

    print("all embeddings:", x.shape)
    print("metadata:", df.shape)

    # ISIC test split 没有 label，所以这里只用 train 做有监督检索评价
    train_mask = df["split"].values == "train"
    df_train = df[train_mask].reset_index(drop=True)
    x_train = x[train_mask]

    labels = df_train["label"].astype(str).values

    print("train embeddings:", x_train.shape)
    print(df_train["label"].value_counts())

    # 已经 L2 normalize，所以内积就是 cosine similarity
    dim = x_train.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(x_train)

    # 查 top 21，因为第一个通常是自己，要去掉 self
    topk = 21
    scores, indices = index.search(x_train, topk)

    # 去掉 self-match
    clean_indices = []
    clean_scores = []
    for i in range(len(x_train)):
        inds = indices[i]
        scs = scores[i]
        keep = inds != i
        clean_indices.append(inds[keep][:20])
        clean_scores.append(scs[keep][:20])

    clean_indices = np.vstack(clean_indices)
    clean_scores = np.vstack(clean_scores)

    retrieved_labels = labels[clean_indices]

    overall = topk_label_precision(labels, retrieved_labels)

    result = {
        "num_train": int(len(df_train)),
        "dim": int(dim),
        "label_counts": df_train["label"].value_counts().to_dict(),
        "overall": overall,
        "by_label": {},
    }

    for lab in sorted(set(labels)):
        mask = labels == lab
        result["by_label"][lab] = topk_label_precision(labels[mask], retrieved_labels[mask])
        result["by_label"][lab]["num_queries"] = int(mask.sum())

    print(json.dumps(result, indent=2, ensure_ascii=False))

    with open(OUT_DIR / "exact_search_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    np.save(OUT_DIR / "exact_top20_indices.npy", clean_indices)
    np.save(OUT_DIR / "exact_top20_scores.npy", clean_scores)

    print("saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
