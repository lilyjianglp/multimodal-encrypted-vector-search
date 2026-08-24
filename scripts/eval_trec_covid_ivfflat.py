import json
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


CORPUS_EMB = Path("/mnt/d/datasets/text/trec_covid/embeddings/text_corpus_l2.npy")
QUERY_EMB = Path("/mnt/d/datasets/text/trec_covid/embeddings/text_queries_l2.npy")
QRELS = Path("/mnt/d/datasets/text/trec_covid/qrels_clean.csv")
EXACT_TOP1000 = Path("/mnt/d/datasets/text/trec_covid/eval_exact/text_exact_top1000_indices.npy")
OUT_DIR = Path("/mnt/d/datasets/text/trec_covid/eval_ivfflat")
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
        recalls, precisions, ndcgs, mrrs = [], [], [], []

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


def overlap(indices, exact, ks=(10, 20, 100, 1000)):
    ans = {}
    for k in ks:
        vals = []
        for a, e in zip(indices[:, :k], exact[:, :k]):
            vals.append(len(set(a.tolist()) & set(e.tolist())) / k)
        ans[f"overlap@{k}"] = float(np.mean(vals))
    return ans


def main():
    corpus = np.load(CORPUS_EMB).astype("float32")
    queries = np.load(QUERY_EMB).astype("float32")
    qrels_df = pd.read_csv(QRELS)
    exact = np.load(EXACT_TOP1000)

    qrels = build_qrels(qrels_df)

    print("corpus:", corpus.shape)
    print("queries:", queries.shape)

    dim = corpus.shape[1]
    nlist = 1024

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    print(f"training IVFFlat nlist={nlist}")
    t0 = time.time()
    index.train(corpus)
    train_time = time.time() - t0
    print("train time:", train_time)

    print("adding corpus")
    t0 = time.time()
    index.add(corpus)
    add_time = time.time() - t0
    print("add time:", add_time)

    all_results = {}

    for nprobe in [4, 8, 16, 32, 64, 128, 256]:
        index.nprobe = nprobe

        print(f"\nsearching nprobe={nprobe}")
        t0 = time.time()
        scores, indices = index.search(queries, 1000)
        search_time = time.time() - t0

        result = {
            "method": "IVFFlat",
            "nlist": nlist,
            "nprobe": nprobe,
            "num_corpus": int(corpus.shape[0]),
            "num_queries": int(queries.shape[0]),
            "dim": int(dim),
            "train_time_sec": float(train_time),
            "add_time_sec": float(add_time),
            "search_time_sec": float(search_time),
            "avg_latency_ms_per_query": float(search_time / len(queries) * 1000),
            "qps": float(len(queries) / search_time),
            "metrics": eval_retrieval(indices, qrels),
            "exact_overlap": overlap(indices, exact),
        }

        all_results[f"nprobe_{nprobe}"] = result
        print(json.dumps(result, indent=2, ensure_ascii=False))

        np.save(OUT_DIR / f"text_ivfflat_nprobe{nprobe}_top1000_indices.npy", indices)
        np.save(OUT_DIR / f"text_ivfflat_nprobe{nprobe}_top1000_scores.npy", scores)

    with open(OUT_DIR / "text_ivfflat_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
