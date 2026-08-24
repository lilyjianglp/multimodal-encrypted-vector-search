import json
from pathlib import Path

import numpy as np
import pandas as pd


META_PATH = Path("/mnt/d/datasets/isic2020/metadata_clean.csv")
EMB_PATH = Path("/mnt/d/datasets/isic2020/embeddings/image_corpus_l2.npy")
CAND_PATH = Path("/mnt/d/datasets/isic2020/candidates/ivfflat_nprobe16_top100_indices.npy")
EXACT_TOP20_PATH = Path("/mnt/d/datasets/isic2020/eval_exact/exact_top20_indices.npy")
OUT_DIR = Path("/mnt/d/datasets/isic2020/eval_candidates")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def label_metrics(query_labels, retrieved_labels, ks=(1, 5, 10, 20)):
    ans = {}
    for k in ks:
        same = retrieved_labels[:, :k] == query_labels[:, None]
        ans[f"precision@{k}"] = float(same.mean())
        ans[f"hit@{k}"] = float(same.any(axis=1).mean())
    return ans


def overlap_at_k(a, b, ks=(1, 5, 10, 20)):
    ans = {}
    for k in ks:
        vals = []
        for x, y in zip(a[:, :k], b[:, :k]):
            vals.append(len(set(x.tolist()) & set(y.tolist())) / k)
        ans[f"overlap@{k}"] = float(np.mean(vals))
    return ans


def coverage_exact_top20_by_candidate100(cand100, exact20):
    vals = []
    for c, e in zip(cand100, exact20):
        vals.append(len(set(c.tolist()) & set(e.tolist())) / 20)
    return float(np.mean(vals))


def main():
    df = pd.read_csv(META_PATH)
    x = np.load(EMB_PATH).astype("float32")
    cand100 = np.load(CAND_PATH)
    exact20 = np.load(EXACT_TOP20_PATH)

    train_mask = df["split"].values == "train"
    df_train = df[train_mask].reset_index(drop=True)
    x_train = x[train_mask]
    labels = df_train["label"].astype(str).values

    print("x_train:", x_train.shape)
    print("cand100:", cand100.shape)
    print("exact20:", exact20.shape)

    # 在 Top-100 候选池内，用原始 512 维归一化向量重新计算明文内积并排序
    rerank_top20 = []
    rerank_scores = []

    for i in range(len(x_train)):
        cands = cand100[i]
        cands = cands[cands >= 0]

        q = x_train[i]
        cv = x_train[cands]

        scores = cv @ q
        order = np.argsort(-scores)

        top = cands[order][:20]
        sc = scores[order][:20]

        if len(top) < 20:
            pad = 20 - len(top)
            top = np.pad(top, (0, pad), constant_values=-1)
            sc = np.pad(sc, (0, pad), constant_values=-1e9)

        rerank_top20.append(top)
        rerank_scores.append(sc)

    rerank_top20 = np.vstack(rerank_top20).astype("int64")
    rerank_scores = np.vstack(rerank_scores).astype("float32")

    safe = np.where(rerank_top20 < 0, 0, rerank_top20)
    retrieved_labels = labels[safe]

    result = {
        "candidate_pool": "IVFFlat nlist=256 nprobe=16 top100",
        "num_queries": int(len(x_train)),
        "dim": int(x_train.shape[1]),
        "candidate_topk": 100,
        "exact_top20_coverage_by_candidate100": coverage_exact_top20_by_candidate100(cand100, exact20),
        "rerank_exact_overlap": overlap_at_k(rerank_top20, exact20),
        "rerank_overall": label_metrics(labels, retrieved_labels),
        "rerank_by_label": {},
    }

    for lab in sorted(set(labels)):
        mask = labels == lab
        result["rerank_by_label"][lab] = label_metrics(labels[mask], retrieved_labels[mask])
        result["rerank_by_label"][lab]["num_queries"] = int(mask.sum())

    print(json.dumps(result, indent=2, ensure_ascii=False))

    with open(OUT_DIR / "candidate_pool_rerank_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    np.save(OUT_DIR / "candidate_rerank_top20_indices.npy", rerank_top20)
    np.save(OUT_DIR / "candidate_rerank_top20_scores.npy", rerank_scores)

    print("saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
