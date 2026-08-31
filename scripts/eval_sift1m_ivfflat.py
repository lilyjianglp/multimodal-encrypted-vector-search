#!/usr/bin/env python3
"""Benchmark Faiss IVFFlat Recall@10 on ANN-Benchmarks SIFT1M."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import struct
import time
from pathlib import Path

import faiss
import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/sift1m/sift-128-euclidean.hdf5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sift1m/ivfflat_metrics.json"),
    )
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument(
        "--nprobe",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--niter", type=int, default=25)
    parser.add_argument("--nredo", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--exact-query-count",
        type=int,
        default=100,
        help="Run exact Flat timing/ground-truth validation on this many queries; 0 disables it.",
    )
    parser.add_argument(
        "--ckks-fixture",
        type=Path,
        help="Optionally export a compact SIFT fixture for the local CKKS reranker.",
    )
    parser.add_argument("--ckks-query-count", type=int, default=100)
    parser.add_argument("--ckks-nprobe", type=int, default=64)
    parser.add_argument("--ckks-rerank-k", type=int, default=16)
    parser.add_argument(
        "--ckks-fixed-top-r",
        type=int,
        default=0,
        help="Export query-independent centroid Top-R postings instead of query-ranked IVF candidates.",
    )
    parser.add_argument(
        "--cluster-top-r",
        type=int,
        nargs="+",
        default=[],
        help=(
            "Also measure candidate recall when each probed cluster exposes only "
            "its R vectors nearest to the centroid, matching the index service."
        ),
    )
    return parser.parse_args()


def recall_at(approx: np.ndarray, exact: np.ndarray, k: int) -> float:
    approx_k = approx[:, :k]
    exact_k = exact[:, :k]
    hits = np.any(approx_k[:, :, None] == exact_k[:, None, :], axis=2)
    return float(hits.sum() / (len(approx_k) * k))


def timed_search(index: faiss.Index, queries: np.ndarray, k: int, repeats: int):
    index.search(queries[: min(100, len(queries))], k)
    timings = []
    distances = indices = None
    for _ in range(repeats):
        started = time.perf_counter()
        distances, indices = index.search(queries, k)
        timings.append(time.perf_counter() - started)
    assert distances is not None and indices is not None
    return distances, indices, timings


def export_ckks_fixture(
    path: Path,
    database: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    index: faiss.IndexIVF,
    query_count: int,
    nprobe: int,
    rerank_k: int,
) -> None:
    """Export L2 ranking as a 512-dimensional augmented inner product fixture."""
    query_count = min(query_count, len(queries))
    if query_count < 1:
        raise ValueError("ckks-query-count must be positive")
    if rerank_k < ground_truth.shape[1]:
        raise ValueError("ckks-rerank-k must cover ground-truth Top-K")

    index.nprobe = nprobe
    _, candidate_ids = index.search(queries[:query_count], rerank_k)
    if np.any(candidate_ids < 0):
        raise RuntimeError("IVFFlat returned an invalid candidate ID")

    # Scaling both q and x by the same positive constant preserves L2 ranking.
    # q'=[q,1], x'=[2x,-||x||^2] makes q' dot x' equal to
    # 2*q dot x - ||x||^2, which differs from -||q-x||^2 by query-only ||q||^2.
    value_scale = np.float32(255.0)
    source_queries = queries[:query_count] / value_scale
    source_candidates = database[candidate_ids] / value_scale

    padded_queries = np.zeros((query_count, 512), dtype=np.float32)
    padded_queries[:, : source_queries.shape[1]] = source_queries
    padded_queries[:, source_queries.shape[1]] = 1.0

    padded_candidates = np.zeros(
        (query_count, rerank_k, 512), dtype=np.float32
    )
    padded_candidates[:, :, : source_candidates.shape[2]] = 2.0 * source_candidates
    padded_candidates[:, :, source_candidates.shape[2]] = -np.sum(
        source_candidates * source_candidates, axis=2
    )

    exact_ids = np.asarray(
        ground_truth[:query_count], dtype="<i8", order="C"
    )
    candidate_ids = np.asarray(candidate_ids, dtype="<i8", order="C")
    padded_queries = np.asarray(padded_queries, dtype="<f4", order="C")
    padded_candidates = np.asarray(padded_candidates, dtype="<f4", order="C")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(
            struct.pack(
                "<4sIIII",
                b"SCK1",
                query_count,
                rerank_k,
                512,
                exact_ids.shape[1],
            )
        )
        candidate_ids.tofile(output)
        exact_ids.tofile(output)
        padded_queries.tofile(output)
        padded_candidates.tofile(output)
    print(
        f"exported CKKS fixture to {path} "
        f"(queries={query_count}, candidates={rerank_k}, nprobe={nprobe})",
        flush=True,
    )


def export_fixed_ckks_fixture(
    path: Path,
    database: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    index: faiss.IndexIVF,
    posting_ids: list[np.ndarray],
    query_count: int,
    nprobe: int,
    top_r: int,
) -> None:
    """Export the privacy-compatible, query-independent fixed Top-R candidate path."""
    query_count = min(query_count, len(queries))
    candidate_count = nprobe * top_r
    _, selected_clusters = index.quantizer.search(queries[:query_count], nprobe)
    candidate_ids = np.full((query_count, candidate_count), -1, dtype=np.int64)
    for query_index, cluster_ids in enumerate(selected_clusters):
        for cluster_offset, cluster_id in enumerate(cluster_ids):
            ids = posting_ids[int(cluster_id)][:top_r]
            begin = cluster_offset * top_r
            candidate_ids[query_index, begin : begin + len(ids)] = ids

    value_scale = np.float32(255.0)
    source_queries = queries[:query_count] / value_scale
    padded_queries = np.zeros((query_count, 512), dtype=np.float32)
    padded_queries[:, : source_queries.shape[1]] = source_queries
    padded_queries[:, source_queries.shape[1]] = 1.0

    padded_candidates = np.zeros(
        (query_count, candidate_count, 512), dtype=np.float32
    )
    for query_index in range(query_count):
        valid = candidate_ids[query_index] >= 0
        source = database[candidate_ids[query_index, valid]] / value_scale
        padded_candidates[query_index, valid, : source.shape[1]] = 2.0 * source
        padded_candidates[query_index, valid, source.shape[1]] = -np.sum(
            source * source, axis=1
        )

    exact_ids = np.asarray(ground_truth[:query_count], dtype="<i8", order="C")
    candidate_ids = np.asarray(candidate_ids, dtype="<i8", order="C")
    padded_queries = np.asarray(padded_queries, dtype="<f4", order="C")
    padded_candidates = np.asarray(padded_candidates, dtype="<f4", order="C")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(
            struct.pack(
                "<4sIIII",
                b"SCK1",
                query_count,
                candidate_count,
                512,
                exact_ids.shape[1],
            )
        )
        candidate_ids.tofile(output)
        exact_ids.tofile(output)
        padded_queries.tofile(output)
        padded_candidates.tofile(output)
    print(
        f"exported fixed-posting CKKS fixture to {path} "
        f"(queries={query_count}, candidates={candidate_count}, "
        f"nprobe={nprobe}, top_r={top_r})",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    faiss.omp_set_num_threads(args.threads)

    with h5py.File(args.data, "r") as dataset:
        distance = dataset.attrs.get("distance", "")
        if distance != "euclidean":
            raise ValueError(f"Expected euclidean data, got {distance!r}")
        database = np.asarray(dataset["train"], dtype=np.float32, order="C")
        queries = np.asarray(dataset["test"], dtype=np.float32, order="C")
        ground_truth = np.asarray(dataset["neighbors"][:, : args.topk], dtype=np.int64)

    n, dimension = database.shape
    if args.train_size > n:
        raise ValueError(f"train-size {args.train_size} exceeds database size {n}")
    if any(value < 1 or value > args.nlist for value in args.nprobe):
        raise ValueError("Every nprobe must be in [1, nlist]")

    rng = np.random.default_rng(args.seed)
    train_ids = rng.choice(n, size=args.train_size, replace=False)
    training = np.ascontiguousarray(database[train_ids])

    print(
        f"SIFT1M database={database.shape}, queries={queries.shape}, "
        f"nlist={args.nlist}, train={training.shape}, threads={args.threads}",
        flush=True,
    )

    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, args.nlist, faiss.METRIC_L2)
    index.cp.niter = args.niter
    index.cp.nredo = args.nredo
    index.cp.seed = args.seed

    started = time.perf_counter()
    index.train(training)
    train_seconds = time.perf_counter() - started
    print(f"trained in {train_seconds:.3f}s", flush=True)

    started = time.perf_counter()
    index.add(database)
    add_seconds = time.perf_counter() - started
    print(f"added {index.ntotal} vectors in {add_seconds:.3f}s", flush=True)

    list_sizes = np.asarray(
        [index.invlists.list_size(i) for i in range(args.nlist)], dtype=np.int64
    )
    cluster_stats = {
        "min": int(list_sizes.min()),
        "max": int(list_sizes.max()),
        "mean": float(list_sizes.mean()),
        "std": float(list_sizes.std()),
        "empty": int(np.count_nonzero(list_sizes == 0)),
        "imbalance_max_over_mean": float(list_sizes.max() / list_sizes.mean()),
    }

    # The production index service orders each posting list by a query-independent
    # score: 1 - 0.5 * squared_distance(vector, assigned_centroid).  Record the
    # corresponding rank for every database vector so that we can evaluate its
    # fixed Top-R truncation without computing all query-to-candidate distances.
    posting_rank = None
    assigned_cluster = None
    posting_ids = None
    if args.cluster_top_r:
        if any(value < 1 for value in args.cluster_top_r):
            raise ValueError("Every cluster-top-r must be positive")
        assigned_distance, assigned_cluster_2d = index.quantizer.search(database, 1)
        assigned_cluster = assigned_cluster_2d[:, 0]
        assigned_distance = assigned_distance[:, 0]
        posting_rank = np.empty(n, dtype=np.int32)
        posting_ids = []
        for cluster_id in range(args.nlist):
            member_ids = np.flatnonzero(assigned_cluster == cluster_id)
            order = np.argsort(assigned_distance[member_ids], kind="stable")
            ordered_ids = member_ids[order]
            posting_ids.append(ordered_ids)
            posting_rank[ordered_ids] = np.arange(len(member_ids), dtype=np.int32)

    exact_result = None
    exact_count = min(args.exact_query_count, len(queries))
    if exact_count:
        exact = faiss.IndexFlatL2(dimension)
        exact.add(database)
        _, exact_ids, exact_timings = timed_search(
            exact, queries[:exact_count], args.topk, 1
        )
        exact_result = {
            "query_count": exact_count,
            "recall_at_1": recall_at(exact_ids, ground_truth[:exact_count], 1),
            "recall_at_10": recall_at(
                exact_ids, ground_truth[:exact_count], args.topk
            ),
            "total_seconds": exact_timings[0],
            "latency_ms_per_query": exact_timings[0] * 1000 / exact_count,
            "qps": exact_count / exact_timings[0],
        }
        print(f"exact validation: {exact_result}", flush=True)

    results = []
    for nprobe in args.nprobe:
        index.nprobe = nprobe
        _, approximate, timings = timed_search(
            index, queries, args.topk, args.repeats
        )

        _, coarse_ids = index.quantizer.search(queries, nprobe)
        scanned = list_sizes[coarse_ids].sum(axis=1)
        median_seconds = statistics.median(timings)
        row = {
            "nprobe": nprobe,
            "recall_at_1": recall_at(approximate, ground_truth, 1),
            "recall_at_10": recall_at(approximate, ground_truth, args.topk),
            "median_total_seconds": median_seconds,
            "latency_ms_per_query": median_seconds * 1000 / len(queries),
            "qps": len(queries) / median_seconds,
            "timings_seconds": timings,
            "mean_scanned_candidates": float(scanned.mean()),
            "p95_scanned_candidates": float(np.percentile(scanned, 95)),
            "scanned_database_fraction": float(scanned.mean() / n),
        }
        if posting_rank is not None and assigned_cluster is not None:
            exact_cluster = assigned_cluster[ground_truth]
            exact_posting_rank = posting_rank[ground_truth]
            selected_cluster = np.any(
                exact_cluster[:, :, None] == coarse_ids[:, None, :], axis=2
            )
            row["fixed_cluster_top_r"] = {}
            for top_r in args.cluster_top_r:
                hits = selected_cluster & (exact_posting_rank < top_r)
                row["fixed_cluster_top_r"][str(top_r)] = {
                    "candidate_recall_at_10": float(hits.mean()),
                    "max_candidates": int(nprobe * top_r),
                }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    report = {
        "benchmark": "ANN-Benchmarks SIFT1M IVFFlat",
        "data_path": str(args.data.resolve()),
        "database_shape": list(database.shape),
        "query_shape": list(queries.shape),
        "metric": "squared L2",
        "ground_truth_k": args.topk,
        "parameters": {
            "nlist": args.nlist,
            "train_size": args.train_size,
            "niter": args.niter,
            "nredo": args.nredo,
            "seed": args.seed,
            "threads": args.threads,
            "repeats": args.repeats,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
        },
        "train_seconds": train_seconds,
        "add_seconds": add_seconds,
        "cluster_size": cluster_stats,
        "exact_flat": exact_result,
        "ivfflat": results,
    }
    if args.ckks_fixture:
        if args.ckks_fixed_top_r:
            if posting_ids is None:
                raise ValueError("--ckks-fixed-top-r requires --cluster-top-r")
            export_fixed_ckks_fixture(
                args.ckks_fixture,
                database,
                queries,
                ground_truth,
                index,
                posting_ids,
                args.ckks_query_count,
                args.ckks_nprobe,
                args.ckks_fixed_top_r,
            )
        else:
            export_ckks_fixture(
                args.ckks_fixture,
                database,
                queries,
                ground_truth,
                index,
                args.ckks_query_count,
                args.ckks_nprobe,
                args.ckks_rerank_k,
            )
        report["ckks_fixture"] = {
            "path": str(args.ckks_fixture.resolve()),
            "query_count": min(args.ckks_query_count, len(queries)),
            "nprobe": args.ckks_nprobe,
            "rerank_k": args.ckks_rerank_k,
            "padded_dimension": 512,
            "active_dimension": 129,
            "transform": "q'=[q/255,1], x'=[2x/255,-||x/255||^2]",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"saved report to {args.output}", flush=True)


if __name__ == "__main__":
    main()
