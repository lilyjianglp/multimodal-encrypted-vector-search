#!/usr/bin/env python3
"""Prepare a one-query SIFT1M bundle for the real network service chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import subprocess
import time
from pathlib import Path

import faiss
import h5py
import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/sift1m/sift-128-euclidean.hdf5"))
    parser.add_argument("--out", type=Path, default=Path("results/sift1m/network_e2e"))
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--top-t", type=int, default=16)
    parser.add_argument("--total-l", type=int, default=32)
    parser.add_argument("--top-r", type=int, default=1536)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--niter", type=int, default=25)
    parser.add_argument("--nredo", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--session", default="sift-network-q0")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--skip-diag", action="store_true")
    parser.add_argument("--full-bsgs", action="store_true")
    return parser.parse_args()


def session_rng(session: str) -> random.Random:
    digest = hashlib.sha256(session.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "little", signed=False))


def augment_query(query: np.ndarray) -> np.ndarray:
    out = np.zeros(512, dtype=np.float32)
    out[: query.size] = query / np.float32(255.0)
    out[query.size] = 1.0
    return out


def augment_vectors(vectors: np.ndarray) -> np.ndarray:
    scaled = vectors.astype(np.float32, copy=False) / np.float32(255.0)
    out = np.zeros((len(vectors), 512), dtype=np.float32)
    out[:, : vectors.shape[1]] = 2.0 * scaled
    out[:, vectors.shape[1]] = -np.sum(scaled * scaled, axis=1)
    return out


def augment_centers(centers: np.ndarray) -> np.ndarray:
    return augment_vectors(centers)


def get_list_ids(index: faiss.IndexIVF, cluster_id: int) -> np.ndarray:
    size = index.invlists.list_size(cluster_id)
    if size == 0:
        return np.empty(0, dtype=np.int64)
    return faiss.rev_swig_ptr(index.invlists.get_ids(cluster_id), size).copy()


def write_index_bundle(
    data_dir: Path,
    centers_512: np.ndarray,
    mixed_clusters: list[int],
    postings: dict[int, np.ndarray],
    corpus_512: np.ndarray,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=False)
    np.asarray(centers_512, dtype="<f4", order="C").tofile(data_dir / "centers.snap")
    np.zeros(len(corpus_512), dtype=np.uint8).tofile(data_dir / "pq_codes.snap")
    np.zeros((1, 1, 512), dtype="<f4").tofile(data_dir / "pq_codebook.snap")

    database = sqlite3.connect(data_dir / "index.db")
    try:
        database.executescript(
            """
            CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);
            CREATE TABLE postings (
              cluster_id INTEGER,
              cand_id INTEGER,
              approx_score REAL,
              PRIMARY KEY(cluster_id, cand_id)
            );
            CREATE INDEX idx_postings_cluster_score
              ON postings(cluster_id, approx_score DESC);
            """
        )
        meta = {
            "dim": "512",
            "K": str(len(centers_512)),
            "M": "1",
            "Ks": "1",
            "N": str(len(corpus_512)),
            "centers_sha": hashlib.sha256(centers_512.tobytes()).hexdigest(),
        }
        database.executemany("INSERT INTO meta(k,v) VALUES(?,?)", meta.items())
        for cluster_id in mixed_clusters:
            local_ids = postings[cluster_id]
            rows = [
                (cluster_id, int(local_id), float(len(local_ids) - rank))
                for rank, local_id in enumerate(local_ids)
            ]
            database.executemany(
                "INSERT INTO postings(cluster_id,cand_id,approx_score) VALUES(?,?,?)",
                rows,
            )
        database.commit()
    finally:
        database.close()


def generate_diagonals(
    generator: Path,
    context: Path,
    corpus_path: Path,
    diag_dir: Path,
    candidate_count: int,
    full_bsgs: bool,
) -> None:
    diag_dir.mkdir(parents=True, exist_ok=False)
    generation_dir = diag_dir / "generation"
    generation_dir.mkdir()
    combined: list[dict] = []
    pack_slots = 4096
    pack_count = (candidate_count + pack_slots - 1) // pack_slots

    for pack_index in range(pack_count):
        begin = pack_index * pack_slots
        ids = list(range(begin, min(begin + pack_slots, candidate_count)))
        ids.extend([candidate_count] * (pack_slots - len(ids)))
        ids_path = generation_dir / f"pack-{pack_index:06d}-ids.json"
        ids_path.write_text(json.dumps(ids), encoding="utf-8")
        pack_dir = generation_dir / f"pack-{pack_index:06d}"
        started = time.perf_counter()
        command = [
                str(generator),
                "--context", str(context),
                "--ids", str(ids_path),
                "--dim", "512",
                "--outdir", str(pack_dir),
                "--mode", "from-npy",
                "--npy", str(corpus_path),
            ]
        if full_bsgs:
            command.extend(["--full-bsgs", "--bsgs-baby", "32"])
        subprocess.run(
            command,
            check=True,
        )
        metadata = json.loads((pack_dir / "diag_blocks.json").read_text())
        for dimension_block, block in enumerate(metadata["blocks"]):
            source = Path(block["mmap_path"])
            name = f"pack-{pack_index:06d}-blk-{dimension_block:06d}.dia"
            destination = diag_dir / name
            source.replace(destination)
            block["block_id"] = name.removesuffix(".dia")
            block["mmap_path"] = name
            combined.append(block)
        elapsed = time.perf_counter() - started
        print(f"generated diagonal pack {pack_index + 1}/{pack_count} in {elapsed:.2f}s", flush=True)

    (diag_dir / "diag_blocks.json").write_text(
        json.dumps({"blocks": combined}, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = arguments()
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    if not (0 < args.top_t <= args.total_l <= args.nlist):
        raise ValueError("require 0 < top-t <= total-l <= nlist")
    faiss.omp_set_num_threads(args.threads)

    with h5py.File(args.data, "r") as dataset:
        database = np.asarray(dataset["train"], dtype=np.float32, order="C")
        query = np.asarray(dataset["test"][0], dtype=np.float32)
        ground_truth = np.asarray(dataset["neighbors"][0, :10], dtype=np.int64)

    rng_np = np.random.default_rng(args.seed)
    train_ids = rng_np.choice(len(database), size=args.train_size, replace=False)
    quantizer = faiss.IndexFlatL2(database.shape[1])
    index = faiss.IndexIVFFlat(quantizer, database.shape[1], args.nlist, faiss.METRIC_L2)
    index.cp.niter = args.niter
    index.cp.nredo = args.nredo
    index.cp.seed = args.seed
    started = time.perf_counter()
    index.train(np.ascontiguousarray(database[train_ids]))
    index.add(database)
    print(f"trained and populated IVFFlat in {time.perf_counter() - started:.2f}s", flush=True)

    centers = np.empty((args.nlist, database.shape[1]), dtype=np.float32)
    index.quantizer.reconstruct_n(0, args.nlist, centers)
    query_512 = augment_query(query)
    centers_512 = augment_centers(centers)
    center_scores = centers_512 @ query_512
    true_clusters = np.argsort(-center_scores)[: args.top_t].astype(int).tolist()

    rng = session_rng(args.session)
    true_set = set(true_clusters)
    decoy_pool = [cluster for cluster in range(args.nlist) if cluster not in true_set]
    decoys = rng.sample(decoy_pool, args.total_l - args.top_t)
    mixed_clusters = true_clusters + decoys
    rng.shuffle(mixed_clusters)

    candidate_count = args.total_l * args.top_r
    corpus_512 = np.empty((candidate_count + 1, 512), dtype=np.float32)
    local_to_original = np.full(candidate_count + 1, -1, dtype=np.int64)
    # Dedicated padding row; q[128] is one, so its score is safely very small.
    corpus_512[-1] = 0.0
    corpus_512[-1, database.shape[1]] = -1_000_000.0

    local_postings: dict[int, np.ndarray] = {}
    cursor = 0
    for cluster_id in mixed_clusters:
        member_ids = get_list_ids(index, cluster_id)
        if len(member_ids):
            delta = database[member_ids] - centers[cluster_id]
            order = np.argsort(np.einsum("ij,ij->i", delta, delta), kind="stable")
            selected = member_ids[order[: args.top_r]]
        else:
            selected = member_ids
        take = len(selected)
        if take:
            corpus_512[cursor : cursor + take] = augment_vectors(database[selected])
            local_to_original[cursor : cursor + take] = selected
        if take < args.top_r:
            corpus_512[cursor + take : cursor + args.top_r] = corpus_512[-1]
        local_ids = np.arange(cursor, cursor + args.top_r, dtype=np.int64)
        local_postings[cluster_id] = local_ids
        cursor += args.top_r
    assert cursor == candidate_count

    args.out.mkdir(parents=True, exist_ok=False)
    np.save(args.out / "query.npy", query_512)
    np.save(args.out / "centers.npy", centers_512)
    np.save(args.out / "corpus.npy", corpus_512)
    np.save(args.out / "local_to_original.npy", local_to_original)
    (args.out / "clusters.txt").write_text(
        ",".join(map(str, mixed_clusters)), encoding="utf-8"
    )
    write_index_bundle(
        args.out / "index_data", centers_512, mixed_clusters, local_postings, corpus_512
    )

    candidate_original = local_to_original[:candidate_count]
    valid_candidates = set(candidate_original[candidate_original >= 0].tolist())
    candidate_recall = len(valid_candidates.intersection(ground_truth.tolist())) / 10.0
    metadata = {
        "dataset": "SIFT1M",
        "query_index": 0,
        "nlist": args.nlist,
        "top_t": args.top_t,
        "total_l": args.total_l,
        "top_r": args.top_r,
        "session": args.session,
        "true_clusters": true_clusters,
        "decoy_clusters": decoys,
        "mixed_clusters": mixed_clusters,
        "candidate_count": candidate_count,
        "real_candidate_count": int(np.count_nonzero(candidate_original >= 0)),
        "candidate_recall_at_10": candidate_recall,
        "ground_truth_top10": ground_truth.tolist(),
        "transform": "q'=[q/255,1], x'=[2x/255,-||x/255||^2]",
    }
    (args.out / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)

    if not args.skip_diag:
        generate_diagonals(
            args.generator,
            args.context,
            args.out / "corpus.npy",
            args.out / "index_data/diag_blocks",
            candidate_count,
            args.full_bsgs,
        )


if __name__ == "__main__":
    main()
