#!/usr/bin/env python3
"""Evaluate decrypted network E2E score blocks against SIFT1M ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("results/sift1m/network_e2e"))
    parser.add_argument("--request-pb", type=Path, default=Path("/tmp/req.pb"))
    parser.add_argument("--gateway-ms", type=int, required=True)
    parser.add_argument("--client-ms", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/sift1m/network_e2e_metrics.json"))
    args = parser.parse_args()

    metadata = json.loads((args.bundle / "metadata.json").read_text())
    candidate_count = int(metadata["candidate_count"])
    pack_slots = 4096
    dimension_blocks = 4
    pack_count = (candidate_count + pack_slots - 1) // pack_slots

    score_parts = []
    for ciphertext_index in range(pack_count * dimension_blocks):
        csv_path = args.bundle / "scores_out" / f"scores_{ciphertext_index:02d}.csv"
        score_parts.append(
            np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=1, dtype=np.float64)
        )
    encrypted_scores = np.concatenate(
        [
            np.sum(score_parts[begin : begin + dimension_blocks], axis=0)
            for begin in range(0, len(score_parts), dimension_blocks)
        ]
    )[:candidate_count]

    query = np.load(args.bundle / "query.npy").astype(np.float64)
    corpus = np.load(args.bundle / "corpus.npy", mmap_mode="r")[:candidate_count]
    plaintext_scores = np.asarray(corpus, dtype=np.float64) @ query
    local_to_original = np.load(args.bundle / "local_to_original.npy")[:candidate_count]
    valid = local_to_original >= 0

    valid_local_ids = np.flatnonzero(valid)
    plain_order = valid_local_ids[np.argsort(-plaintext_scores[valid], kind="stable")]
    encrypted_order = valid_local_ids[np.argsort(-encrypted_scores[valid], kind="stable")]
    plain_top10 = local_to_original[plain_order[:10]].astype(int).tolist()
    encrypted_top10 = local_to_original[encrypted_order[:10]].astype(int).tolist()
    official_top10 = [int(value) for value in metadata["ground_truth_top10"]]
    official_set = set(official_top10)

    absolute_error = np.abs(encrypted_scores - plaintext_scores)
    response_files = sorted((args.bundle / "scores_out").glob("scores_*.bin"))
    response_bytes = sum(path.stat().st_size for path in response_files)
    request_bytes = args.request_pb.stat().st_size
    eval_key_bytes = sum(
        (args.bundle.parent / "network_keys" / name).stat().st_size
        for name in ("galois.bin", "relin.bin")
    )

    first_relevant_rank = next(
        (rank for rank, item in enumerate(encrypted_top10, start=1) if item in official_set),
        None,
    )
    result = {
        "benchmark": "SIFT1M real Gateway/Index/HECompute/client E2E",
        "query_count": 1,
        "nlist": metadata["nlist"],
        "true_clusters": metadata["top_t"],
        "decoy_clusters": metadata["total_l"] - metadata["top_t"],
        "top_r_per_cluster": metadata["top_r"],
        "candidate_slots": candidate_count,
        "real_candidates": int(np.count_nonzero(valid)),
        "candidate_recall_at_10": metadata["candidate_recall_at_10"],
        "encrypted_recall_at_10": len(set(encrypted_top10) & official_set) / 10.0,
        "plaintext_recall_at_10": len(set(plain_top10) & official_set) / 10.0,
        "encrypted_plaintext_top10_set_consistency": len(set(encrypted_top10) & set(plain_top10)) / 10.0,
        "encrypted_plaintext_top10_order_equal": encrypted_top10 == plain_top10,
        "mrr_at_10_single_query": 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
        "official_top10": official_top10,
        "plaintext_top10": plain_top10,
        "encrypted_top10": encrypted_top10,
        "score_error_valid_candidates": {
            "mean_absolute": float(absolute_error[valid].mean()),
            "p95_absolute": float(np.percentile(absolute_error[valid], 95)),
            "max_absolute": float(absolute_error[valid].max()),
        },
        "score_error_all_slots": {
            "mean_absolute": float(absolute_error.mean()),
            "max_absolute": float(absolute_error.max()),
        },
        "latency_ms": {
            "gateway_reported": args.gateway_ms,
            "client_wall": args.client_ms,
        },
        "communication_payload_bytes": {
            "query_request": request_bytes,
            "encrypted_score_response": response_bytes,
            "query_total": request_bytes + response_bytes,
            "one_time_evaluation_keys": eval_key_bytes,
        },
        "ciphertext_counts": {"query": 1, "response": len(response_files)},
        "privacy_note": (
            "Gateway and Index observe the 32 numeric mixed cluster IDs but not explicit "
            "true/decoy labels; the client decrypts all 49,152 candidate scores."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
