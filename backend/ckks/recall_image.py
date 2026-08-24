#!/usr/bin/env python3
import json
import numpy as np
import argparse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--idmap", required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--group", type=int, default=12,
                    help="augmentation group size, default=12")
    return ap.parse_args()

def main():
    args = parse_args()

    print("========== Image Recall Evaluation (dedupe 12×) ==========")
    print(f"[+] Loading q: {args.q}")
    q = np.load(args.q)

    print(f"[+] Loading db: {args.db}")
    db = np.load(args.db)

    print(f"[+] Loading id_map: {args.idmap}")
    with open(args.idmap, "r") as f:
        id_map = json.load(f)  # dict: index -> cid

    # normalize
    q_norm = q / np.linalg.norm(q)
    db_norm = db / np.linalg.norm(db, axis=1, keepdims=True)

    print("[+] Computing cosine similarity…")
    sims = db_norm @ q_norm

    # ========== 去重，根据 group size 聚类 ==========
    group = args.group
    G = db.shape[0] // group

    grouped_scores = {}
    for g in range(G):
        start = g * group
        end = start + group
        block = sims[start:end]
        best_score = float(block.max())
        grouped_scores[g] = best_score

    # sort groups
    sorted_groups = sorted(grouped_scores.items(),
                           key=lambda x: x[1],
                           reverse=True)

    print("\n========= Top-{} Groups (after 12× dedupe) =========".format(args.topk))
    for rank, (g, score) in enumerate(sorted_groups[:args.topk], 1):
        orig_index = g * group
        cid = id_map[str(orig_index)]
        print(f"{rank}. group={g:<6} cid={cid:<20} score={score:.4f}")

    print("\n[OK] Done.")

if __name__ == "__main__":
    main()

