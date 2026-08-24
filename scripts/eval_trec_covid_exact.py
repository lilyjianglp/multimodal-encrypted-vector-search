import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


CORPUS_EMB = Path("/mnt/d/datasets/text/trec_covid/embeddings/text_corpus_l2.npy")
QUERY_EMB = Path("/mnt/d/datasets/text/trec_covid/embeddings/text_queries_l2.npy")
QRELS = Path("/mnt/d/datasets/text/trec_covid/qrels_clean.csv")
OUT_DIR = Path("/mnt/d/datasets/text/trec_covid/eval_exact")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_qrels(qrels_df):
    rel = {}
    for row in qrels_df.itertuples(index=False):
        qid = int(row.query_local_id)
        did = int(row.doc_local_id)
        score = float(row.score)
        if score > 0:
            rel.setdefault(qid, {})[did] = score
    return rel


def dcg_at_k(rels, k):
    rels = np.asarray(rels[:k], dtype="float32")
    if len(rels) == 0:
        return 0.0
    gains = (2 ** rels - 1) / np.log2(np.arange(2, len(rels) + 2))
    return float(gains.sum())


def eval_retrieval(indices, qrels, ks=(10, 20, 100, 1000)):
    results = {}

    for k in ks:
        recalls = []
        precisions = []
        ndcgs = []
        mrrs = []

        for qid in range(indices.shape[0]):
            retrieved = indices[qid, :k].tolist()
            rel_dict = qrels.get(qid, {})
            rel_set = set(rel_dict.keys())

            if not rel_set:
                continue

            hits = [1 if did in rel_set else 0 for did in retrieved]
            num_hit = sum(hits)

            recalls.append(num_hit / len(rel_set))
            precisions.append(num_hit / k)

            # MRR
            rr = 0.0
            for rank, h in enumerate(hits, start=1):
                if h:
                    rr = 1.0 / rank
                    break
            mrrs.append(rr)

            # nDCG
            retrieved_rels = [rel_dict.get(did, 0.0) for did in retrieved]
            ideal_rels = sorted(rel_dict.values(), reverse=True)
            dcg = dcg_at_k(retrieved_rels, k)
            idcg = dcg_at_k(ideal_rels, k)
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        results[f"recall@{k}"] = float(np.mean(recalls))
        results[f"precision@{k}"] = float(np.mean(precisions))
        results[f"ndcg@{k}"] = float(np.mean(ndcgs))
        results[f"mrr@{k}"] = float(np.mean(mrrs))

    return results


def main():
    corpus = np.load(CORPUS_EMB).astype("float32")
    queries = np.load(QUERY_EMB).astype("float32")
    qrels_df = pd.read_csv(QRELS)

    print("corpus:", corpus.shape)
    print("queries:", queries.shape)
    print("qrels:", qrels_df.shape)

    qrels = build_qrels(qrels_df)
    print("num qrels queries:", len(qrels))

    dim = corpus.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(corpus)

    topk = 1000
    print(f"searching exact top{topk}")
    scores, indices = index.search(queries, topk)

    metrics = eval_retrieval(indices, qrels, ks=(10, 20, 100, 1000))

    result = {
        "method": "Exact IndexFlatIP",
        "num_corpus": int(corpus.shape[0]),
        "num_queries": int(queries.shape[0]),
        "dim": int(dim),
        "topk": topk,
        "metrics": metrics,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    with open(OUT_DIR / "text_exact_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    np.save(OUT_DIR / "text_exact_top1000_indices.npy", indices)
    np.save(OUT_DIR / "text_exact_top1000_scores.npy", scores)

    print("saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
