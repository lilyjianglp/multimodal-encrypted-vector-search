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
RESULTS_ROOT = ROOT / "results/final_ckks_recall"

MODES = {
    "image": {
        "corpus": ROOT / "artifacts/image/embeddings/image_corpus_l2.npy",
        "query": Path("/tmp/query_image.npy"),
        "index": ROOT / "artifacts/image/index/image_ivfflat_nlist256.index",
        "id_map": ROOT / "artifacts/image/embeddings/id_mapping.json",
        "centers": CKKS_ROOT / "image/centers.npy",
        "d0512": CKKS_ROOT / "image/D0512",
        "data_dir": BACKEND_ROOT / "data_image",
        "gateway": "127.0.0.1:50052",
        "nlist": 256,
        "nprobe": 16,
        "top_t": 16,
        "l": 32,
    },
    "text": {
        "corpus": ROOT / "artifacts/text/embeddings/text_corpus_l2.npy",
        "query": ROOT / "artifacts/text/embeddings/text_queries_l2.npy",
        "index": ROOT / "artifacts/text/index/text_ivfflat_nlist1024.index",
        "id_map": ROOT / "artifacts/text/embeddings/text_id_map.json",
        "centers": CKKS_ROOT / "text/centers.npy",
        "d0512": CKKS_ROOT / "text/D0512",
        "data_dir": BACKEND_ROOT / "data_text",
        "gateway": "127.0.0.1:50058",
        "nlist": 1024,
        "nprobe": 64,
        "top_t": 32,
        "l": 32,
    },
    "audio": {
        "corpus": ROOT / "artifacts/audio/embeddings/corpus/audio_corpus_l2.npy",
        "query": ROOT / "artifacts/audio/embeddings/query/audio_query_l2.npy",
        "index": ROOT / "artifacts/audio/index/audio_icbhi_ivfflat.index",
        "id_map": ROOT / "artifacts/audio/embeddings/corpus/id_mapping.json",
        "centers": CKKS_ROOT / "audio/centers.npy",
        "d0512": CKKS_ROOT / "audio/D0512",
        "data_dir": BACKEND_ROOT / "data_audio",
        "gateway": "127.0.0.1:50057",
        "nlist": 32,
        "nprobe": 8,
        "top_t": 8,
        "l": 32,
    },
}

VERIFIED_EFFECTIVE_SLOTS = [i * 512 for i in range(8)]


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


def load_query(path, query_index):
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
            return files
        time.sleep(1)
    raise TimeoutError(f"Expected 4 RAW score files in {raw_dir}")


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


def recover_ckks_scores(raw_dir):
    parts = [
        np.loadtxt(path, delimiter=",", skiprows=1, usecols=1)
        for path in sorted(raw_dir.glob("scores_*.csv"))
    ]
    if len(parts) != 4:
        raise RuntimeError(f"Expected 4 decrypted CSV files, got {len(parts)}")
    return np.sum(parts, axis=0) * 512


def evaluate_query(mode, config, query, query_index, k, candidate_k, timeout, skip_eval_keys):
    index = faiss.read_index(str(config["index"]))
    corpus = np.load(config["corpus"], mmap_mode="r").astype("float32")
    if index.ntotal < len(corpus):
        corpus = corpus[: index.ntotal]

    query2d = query.reshape(1, -1)
    exact = faiss.IndexFlatIP(corpus.shape[1])
    exact.add(corpus)
    _, exact_ids = exact.search(query2d, max(k, candidate_k))

    index.nprobe = config["nprobe"]
    _, ivf_ids = index.search(query2d, candidate_k)
    ivf_candidates = [int(x) for x in ivf_ids[0]]

    scored_candidate_ids = ivf_candidates[:k]
    layout = [0] * 4096
    for slot, row_id in zip(VERIFIED_EFFECTIVE_SLOTS[:k], scored_candidate_ids):
        layout[slot] = int(row_id)

    layout_path = Path(f"/tmp/{mode}_final_ckks_layout_q{query_index}.json")
    query_path = Path(f"/tmp/{mode}_final_ckks_query_q{query_index}.npy")
    clusters_path = Path(f"/tmp/{mode}_final_ckks_clusters_q{query_index}.txt")
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
            f"final-ckks-{mode}-{query_index}",
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

    ckks_all_scores = recover_ckks_scores(raw_dir)
    records = []
    for rank_in_layout, (slot, row_id) in enumerate(
        zip(VERIFIED_EFFECTIVE_SLOTS[:k], scored_candidate_ids), start=1
    ):
        plain = float(corpus[row_id] @ query)
        ckks = float(ckks_all_scores[slot])
        metadata = load_metadata(config["id_map"], row_id)
        records.append(
            {
                "query_index": query_index,
                "layout_rank": rank_in_layout,
                "slot": slot,
                "row_id": int(row_id),
                "display_id": display_id(mode, metadata, row_id),
                "plain_score": plain,
                "ckks_score": ckks,
                "abs_error": abs(plain - ckks),
                "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            }
        )

    ckks_order = sorted(records, key=lambda item: item["ckks_score"], reverse=True)
    ckks_final_topk = [row["row_id"] for row in ckks_order[:k]]
    exact_topk = [int(x) for x in exact_ids[0][:k]]
    recall = len(set(exact_topk) & set(ckks_final_topk)) / k
    ranking_consistent = [row["row_id"] for row in records] == ckks_final_topk

    return {
        "query_index": query_index,
        "exact_topk": exact_topk,
        "ivf_candidates": ivf_candidates,
        "scored_candidate_ids": scored_candidate_ids,
        "ckks_final_topk": ckks_final_topk,
        "recall_at_k": recall,
        "ranking_consistent": ranking_consistent,
        "max_abs_error": max(row["abs_error"] for row in records),
        "mean_abs_error": float(np.mean([row["abs_error"] for row in records])),
        "details": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--num-queries", type=int, default=1)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--skip-eval-keys", action="store_true")
    args = parser.parse_args()

    if args.k > len(VERIFIED_EFFECTIVE_SLOTS):
        raise ValueError(
            "Current verified CKKS layout supports only 8 effective candidate slots. "
            "Use --k 8 unless the layout is extended and revalidated."
        )

    config = MODES[args.mode]
    out_dir = RESULTS_ROOT / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for offset in range(args.num_queries):
        query_index = args.query_index + offset
        query = load_query(config["query"], query_index)
        result = evaluate_query(
            args.mode,
            config,
            query,
            query_index,
            args.k,
            args.candidate_k,
            args.timeout,
            args.skip_eval_keys,
        )
        results.append(result)

    detail_path = out_dir / "detail.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "query_index",
            "layout_rank",
            "slot",
            "row_id",
            "display_id",
            "plain_score",
            "ckks_score",
            "abs_error",
            "metadata",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerows(result["details"])

    summary = {
        "mode": args.mode,
        "k": args.k,
        "candidate_k": args.candidate_k,
        "num_queries": args.num_queries,
        "mean_final_recall_at_k": float(np.mean([r["recall_at_k"] for r in results])),
        "mean_max_abs_error": float(np.mean([r["max_abs_error"] for r in results])),
        "mean_abs_error": float(
            np.mean([detail["abs_error"] for r in results for detail in r["details"]])
        ),
        "ranking_consistent_rate": float(
            np.mean([bool(r["ranking_consistent"]) for r in results])
        ),
        "limitation": "current CKKS layout supports 8 verified effective candidate slots",
        "queries": [
            {
                "query_index": r["query_index"],
                "exact_topk": r["exact_topk"],
                "ivf_candidates": r["ivf_candidates"],
                "scored_candidate_ids": r["scored_candidate_ids"],
                "ckks_final_topk": r["ckks_final_topk"],
                "recall_at_k": r["recall_at_k"],
                "ranking_consistent": r["ranking_consistent"],
                "max_abs_error": r["max_abs_error"],
                "mean_abs_error": r["mean_abs_error"],
            }
            for r in results
        ],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"summary -> {summary_path}")
    print(f"detail  -> {detail_path}")


if __name__ == "__main__":
    main()
