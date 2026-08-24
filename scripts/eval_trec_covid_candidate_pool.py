import json
from pathlib import Path

import numpy as np
import pandas as pd


CORPUS_EMB = Path("/mnt/d/datasets/text/trec_covid/embeddings/text_corpus_l2.npy")
QUERY_EMB = Path("/mnt/d/datasets/text/trec_covid/embeddings/text_queries_l2.npy")
QRELS = Path("/mnt/d/datasets/text/trec_covid/qrels_clean.csv")

CAND_PATH = Path("/mnt/d/datasets/text/trec_covid/candidates/text_ivfflat_nprobe64_top100_indices.npy")
EXACT_TOP1000 = Path("/mnt/d/datasets/text/trec_covid/eval_exact/text_exact_top1000_indices.npy")

OUT_DIR = Path("/mnt/d/datasets/text/trec_covid/eval_candidates")
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


def eval_retrieval(indices, qrels, ks=(10, 20, 100)):
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

            rr = 0.0
            for rank, h in enumerate(hits, start=1):
                if h:
                    rr = 1.0 / rank
                    break
            mrrs.append(rr)

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


def overlap(a, b, ks=(10, 20, 100)):
    ans = {}
    for k in ks:
        vals = []
        for x, y in zip(a[:, :k], b[:, :k]):
            vals.append(len(set(x.tolist()) & set(y.tolist())) / k)
        ans[f"overlap@{k}"] = float(np.mean(vals))
    return ans


def coverage_exact_top100_by_candidate100(cand100, exact1000):
    exact100 = exact1000[:, :100]
    vals = []
    for c, e in zip(cand100, exact100):
        vals.append(len(set(c.tolist()) & set(e.tolist())) / 100)
    return float(np.mean(vals))


def main():
    corpus = np.load(CORPUS_EMB).astype("float32")
    queries = np.load(QUERY_EMB).astype("float32")
    cand100 = np.load(CAND_PATH)
    exact1000 = np.load(EXACT_TOP1000)
    qrels_df = pd.read_csv(QRELS)
    qrels = build_qrels(qrels_df)

    print("corpus:", corpus.shape)
    print("queries:", queries.shape)
    print("cand100:", cand100.shape)
    print("exact1000:", exact1000.shape)

    rerank_top100 = []
    rerank_scores = []

    for i in range(len(queries)):
        cands = cand100[i]
        q = queries[i]
        cv = corpus[cands]

        scores = cv @ q
        order = np.argsort(-scores)

        rerank_top100.append(cands[order])
        rerank_scores.append(scores[order])

    rerank_top100 = np.vstack(rerank_top100).astype("int64")
    rerank_scores = np.vstack(rerank_scores).astype("float32")

    result = {
        "candidate_pool": "IVFFlat nlist=1024 nprobe=64 top100",
        "num_queries": int(len(queries)),
        "num_corpus": int(len(corpus)),
        "dim": int(corpus.shape[1]),
        "candidate_topk": 100,
        "exact_top100_coverage_by_candidate100": coverage_exact_top100_by_candidate100(cand100, exact1000),
        "rerank_exact_overlap": overlap(rerank_top100, exact1000),
        "rerank_metrics": eval_retrieval(rerank_top100, qrels),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    with open(OUT_DIR / "text_candidate_pool_rerank_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    np.save(OUT_DIR / "text_candidate_rerank_top100_indices.npy", rerank_top100)
    np.save(OUT_DIR / "text_candidate_rerank_top100_scores.npy", rerank_scores)

    print("saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
