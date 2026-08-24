#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import faiss
import numpy as np


HOME = Path("/home/wen")
ROOT = HOME / "private-vector-search"
CKKS_ROOT = HOME / "Desktop/backend/ckks"
BACKEND_ROOT = ROOT / "backend_data"
RESULTS_ROOT = ROOT / "results/final_ckks_recall_batched"

MODES = {
    "image": {
        "corpus": ROOT / "artifacts/image/embeddings/image_corpus_l2.npy",
        "query": None,
        "query_source": "self_sample_from_corpus",
        "index": ROOT / "artifacts/image/index/image_ivfflat_nlist256.index",
        "id_map": ROOT / "artifacts/image/embeddings/id_mapping.json",
        "centers": CKKS_ROOT / "image/centers.npy",
        "d0512": CKKS_ROOT / "image/D0512",
        "data_dir": BACKEND_ROOT / "data_image",
        "gateway": "127.0.0.1:50052",
        "nprobe": 16,
        "top_t": 16,
        "l": 32,
    },
    "text": {
        "corpus": ROOT / "artifacts/text/embeddings/text_corpus_l2.npy",
        "query": ROOT / "artifacts/text/embeddings/text_queries_l2.npy",
        "query_source": "external_query",
        "index": ROOT / "artifacts/text/index/text_ivfflat_nlist1024.index",
        "id_map": ROOT / "artifacts/text/embeddings/text_id_map.json",
        "centers": CKKS_ROOT / "text/centers.npy",
        "d0512": CKKS_ROOT / "text/D0512",
        "data_dir": BACKEND_ROOT / "data_text",
        "gateway": "127.0.0.1:50058",
        "nprobe": 64,
        "top_t": 32,
        "l": 32,
    },
    "audio": {
        "corpus": ROOT / "artifacts/audio/embeddings/corpus/audio_corpus_l2.npy",
        "query": ROOT / "artifacts/audio/embeddings/query/audio_query_l2.npy",
        "query_source": "external_query",
        "index": ROOT / "artifacts/audio/index/audio_icbhi_ivfflat.index",
        "id_map": ROOT / "artifacts/audio/embeddings/corpus/id_mapping.json",
        "centers": CKKS_ROOT / "audio/centers.npy",
        "d0512": CKKS_ROOT / "audio/D0512",
        "data_dir": BACKEND_ROOT / "data_audio",
        "gateway": "127.0.0.1:50057",
        "nprobe": 8,
        "top_t": 8,
        "l": 32,
    },
}

SLOTS = [i * 512 for i in range(8)]


def run(command, env=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=CKKS_ROOT, env=env, check=True)


def clear_directory(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def ensure_diag_link(data_dir, target):
    link = data_dir / "diag_blocks"
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    link.symlink_to(target)


def load_query(path, query_index, corpus=None):
    if path is None:
        if corpus is None:
            raise ValueError("corpus is required when query path is None")
        if query_index >= len(corpus):
            raise IndexError(f"query_index={query_index} out of corpus range {len(corpus)}")
        query = np.asarray(corpus[query_index], dtype="float32")
        query = query.copy()
        query /= max(float(np.linalg.norm(query)), 1e-12)
        return query

    query = np.load(path, mmap_mode="r").astype("float32")
    if query.ndim == 2:
        query = np.asarray(query[query_index], dtype="float32")
    if query.shape != (512,):
        raise ValueError(f"Expected query shape (512,), got {query.shape}")
    query = query.copy()
    query /= max(float(np.linalg.norm(query)), 1e-12)
    return query


def wait_for_raw(raw_dir, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        files = sorted(raw_dir.glob("scores_*.bin"))
        if len(files) >= 4:
            return
        time.sleep(1)
    raise TimeoutError(f"Expected 4 RAW score files in {raw_dir}")


def recover_scores(raw_dir):
    parts = [
        np.loadtxt(path, delimiter=",", skiprows=1, usecols=1)
        for path in sorted(raw_dir.glob("scores_*.csv"))
    ]
    if len(parts) != 4:
        raise RuntimeError(f"Expected 4 decrypted CSV files, got {len(parts)}")
    return np.sum(parts, axis=0) * 512


def load_metadata(id_map, row_id):
    if not id_map.exists():
        return {}
    data = json.loads(id_map.read_text(encoding="utf-8"))
    item = data.get(str(int(row_id)))
    if isinstance(item, dict):
        return item
    if item is None:
        return {}
    return {"external_id": item}


def display_id(mode, metadata, row_id):
    if mode == "image":
        return metadata.get("external_id", str(row_id))
    if mode == "text":
        return metadata.get("title") or metadata.get("doc_id") or str(row_id)
    if mode == "audio":
        return metadata.get("cycle_id", str(row_id))
    return str(row_id)


def score_batch(mode, config, query, candidate_ids, batch_id, timeout, skip_eval_keys):
    layout = [0] * 4096
    for slot, row_id in zip(SLOTS, candidate_ids):
        layout[slot] = int(row_id)

    layout_path = Path(f"/tmp/{mode}_batched_layout_b{batch_id}.json")
    query_path = Path(f"/tmp/{mode}_batched_query_b{batch_id}.npy")
    clusters_path = Path(f"/tmp/{mode}_batched_clusters_b{batch_id}.txt")
    layout_path.write_text(json.dumps(layout))
    np.save(query_path, query)

    clear_directory(config["d0512"])
    run(
        [
            str(CKKS_ROOT / "make_dia1_768"),
            "--context",
            str(CKKS_ROOT / "context.seal"),
            "--ids",
            str(layout_path),
            "--dim",
            "512",
            "--outdir",
            str(config["d0512"]),
            "--mode",
            "from-npy",
            "--npy",
            str(config["corpus"]),
        ]
    )
    ensure_diag_link(config["data_dir"], config["d0512"])
    run(
        [
            str(CKKS_ROOT / "ckks_make_ctq_from_npy"),
            "--context",
            str(CKKS_ROOT / "context.seal"),
            "--pk",
            str(CKKS_ROOT / "pk.bin"),
            "--npy",
            str(query_path),
            "--dim",
            "512",
            "--out",
            str(CKKS_ROOT / "ct_q.bin"),
        ]
    )
    run(
        [
            "python3",
            str(CKKS_ROOT / "client_pick_clusters.py"),
            "--q",
            str(query_path),
            "--centers",
            str(config["centers"]),
            "--topT",
            str(config["top_t"]),
            "--L",
            str(config["l"]),
            "--session",
            f"batched-final-{mode}-{batch_id}",
            "--metric",
            "cos",
            "--out",
            str(clusters_path),
        ]
    )
    if not skip_eval_keys:
        run(["python3", str(CKKS_ROOT / "upload_evalkeys_v1.py")])

    raw_dir = CKKS_ROOT / "scores_raw_raw"
    clear_directory(raw_dir)
    env = os.environ.copy()
    env["CLUSTERS"] = clusters_path.read_text().strip()
    env["GATEWAY_ADDR"] = config["gateway"]
    run(["python3", str(CKKS_ROOT / "search_real_ct.py"), "--mode", mode], env=env)
    wait_for_raw(raw_dir, timeout)
    run(
        [
            str(CKKS_ROOT / "ckks_decrypt_dump"),
            "--context",
            str(CKKS_ROOT / "context.seal"),
            "--sk",
            str(CKKS_ROOT / "sk.bin"),
            "--scores_dir",
            str(raw_dir),
            "--dim",
            "512",
        ]
    )

    all_scores = recover_scores(raw_dir)
    return [
        {"batch_id": batch_id, "slot": slot, "row_id": int(row_id), "ckks_score": float(all_scores[slot])}
        for slot, row_id in zip(SLOTS, candidate_ids)
    ]


def evaluate(mode, config, query_index, final_k, rerank_k, candidate_k, timeout, skip_eval_keys):
    index = faiss.read_index(str(config["index"]))
    corpus = np.load(config["corpus"], mmap_mode="r").astype("float32")
    if index.ntotal < len(corpus):
        corpus = corpus[: index.ntotal]

    query = load_query(config["query"], query_index, corpus=corpus)
    query2d = query.reshape(1, -1)

    exact = faiss.IndexFlatIP(corpus.shape[1])
    exact.add(corpus)
    _, exact_ids = exact.search(query2d, max(final_k, candidate_k))

    index.nprobe = config["nprobe"]
    _, ivf_ids = index.search(query2d, candidate_k)
    if rerank_k > candidate_k:
        raise ValueError(f"rerank_k={rerank_k} cannot exceed candidate_k={candidate_k}")
    candidate_ids = [int(x) for x in ivf_ids[0][:rerank_k]]

    scored = []
    for batch_id, offset in enumerate(range(0, rerank_k, len(SLOTS))):
        batch_ids = candidate_ids[offset : offset + len(SLOTS)]
        scored.extend(score_batch(mode, config, query, batch_ids, batch_id, timeout, skip_eval_keys))

    records = []
    by_id = {row["row_id"]: row for row in scored}
    for plain_rank, row_id in enumerate(candidate_ids, start=1):
        row = by_id[row_id]
        plain = float(corpus[row_id] @ query)
        metadata = load_metadata(config["id_map"], row_id)
        records.append(
            {
                "query_index": query_index,
                "candidate_rank": plain_rank,
                "batch_id": row["batch_id"],
                "slot": row["slot"],
                "row_id": row_id,
                "display_id": display_id(mode, metadata, row_id),
                "plain_score": plain,
                "ckks_score": row["ckks_score"],
                "abs_error": abs(plain - row["ckks_score"]),
                "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            }
        )

    ckks_final = [
        row["row_id"]
        for row in sorted(records, key=lambda x: x["ckks_score"], reverse=True)[:final_k]
    ]
    plain_final = [
        row["row_id"]
        for row in sorted(records, key=lambda x: x["plain_score"], reverse=True)[:final_k]
    ]
    exact_topk = [int(x) for x in exact_ids[0][:final_k]]
    recall = len(set(exact_topk) & set(ckks_final)) / final_k
    rerank_candidate_recall = len(set(exact_topk) & set(candidate_ids)) / final_k
    ranking_consistent = ckks_final == plain_final

    return {
        "query_index": query_index,
        "exact_topk": exact_topk,
        "ivf_candidates": [int(x) for x in ivf_ids[0]],
        "scored_candidate_ids": candidate_ids,
        "plain_final_topk_from_scored": plain_final,
        "ckks_final_topk": ckks_final,
        "recall_at_k": recall,
        "rerank_candidate_recall_at_k": rerank_candidate_recall,
        "ranking_consistent": ranking_consistent,
        "max_abs_error": max(row["abs_error"] for row in records),
        "mean_abs_error": float(np.mean([row["abs_error"] for row in records])),
        "details": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--num-queries", type=int, default=1)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--final-k", type=int)
    parser.add_argument("--rerank-k", type=int)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--top-t", type=int, help="Number of real clusters selected before padding with fake clusters.")
    parser.add_argument("--l", type=int, help="Total cluster count sent to Gateway, including fake clusters.")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--skip-eval-keys", action="store_true")
    args = parser.parse_args()

    final_k = args.final_k if args.final_k is not None else args.k
    rerank_k = args.rerank_k if args.rerank_k is not None else args.k
    if final_k > rerank_k:
        raise ValueError(f"final_k={final_k} cannot exceed rerank_k={rerank_k}")
    if rerank_k > args.candidate_k:
        raise ValueError(f"rerank_k={rerank_k} cannot exceed candidate_k={args.candidate_k}")

    if rerank_k % len(SLOTS) != 0:
        print(f"[INFO] rerank_k={rerank_k} will use a partially filled final batch", flush=True)

    config = dict(MODES[args.mode])
    if args.top_t is not None:
        config["top_t"] = args.top_t
    if args.l is not None:
        config["l"] = args.l
    if config["top_t"] > config["l"]:
        raise ValueError(f"top_t={config['top_t']} cannot exceed l={config['l']}")
    out_dir = RESULTS_ROOT / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [
        evaluate(
            args.mode,
            config,
            args.query_index + offset,
            final_k,
            rerank_k,
            args.candidate_k,
            args.timeout,
            args.skip_eval_keys,
        )
        for offset in range(args.num_queries)
    ]

    detail_path = out_dir / (
        f"detail_final{final_k}_rerank{rerank_k}_topT{config['top_t']}_L{config['l']}.csv"
    )
    fields = [
        "query_index",
        "candidate_rank",
        "batch_id",
        "slot",
        "row_id",
        "display_id",
        "plain_score",
        "ckks_score",
        "abs_error",
        "metadata",
    ]
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerows(result["details"])

    summary = {
        "mode": args.mode,
        "k": final_k,
        "final_k": final_k,
        "rerank_k": rerank_k,
        "candidate_k": args.candidate_k,
        "top_t": config["top_t"],
        "l": config["l"],
        "fake_cluster_count": config["l"] - config["top_t"],
        "real_cluster_ratio": config["top_t"] / config["l"],
        "num_queries": args.num_queries,
        "query_source": config.get("query_source", "unknown"),
        "batch_size": len(SLOTS),
        "mean_final_recall_at_k": float(np.mean([r["recall_at_k"] for r in results])),
        "mean_rerank_candidate_recall_at_k": float(
            np.mean([r["rerank_candidate_recall_at_k"] for r in results])
        ),
        "mean_max_abs_error": float(np.mean([r["max_abs_error"] for r in results])),
        "mean_abs_error": float(np.mean([d["abs_error"] for r in results for d in r["details"]])),
        "ranking_consistent_rate": float(np.mean([r["ranking_consistent"] for r in results])),
        "method": "batched CKKS scoring; 8 verified effective candidate slots per batch",
        "queries": [
            {
                "query_index": r["query_index"],
                "exact_topk": r["exact_topk"],
                "scored_candidate_ids": r["scored_candidate_ids"],
                "plain_final_topk_from_scored": r["plain_final_topk_from_scored"],
                "ckks_final_topk": r["ckks_final_topk"],
                "recall_at_k": r["recall_at_k"],
                "rerank_candidate_recall_at_k": r["rerank_candidate_recall_at_k"],
                "ranking_consistent": r["ranking_consistent"],
                "max_abs_error": r["max_abs_error"],
                "mean_abs_error": r["mean_abs_error"],
            }
            for r in results
        ],
    }
    summary_path = out_dir / (
        f"summary_final{final_k}_rerank{rerank_k}_topT{config['top_t']}_L{config['l']}.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"summary -> {summary_path}")
    print(f"detail  -> {detail_path}")


if __name__ == "__main__":
    main()
