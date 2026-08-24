#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse

def load_vecs(path):
    print(f"[+] Loading features: {path}")
    return np.load(path).astype(np.float32)   # shape = (N, 512)

def load_query(path):
    return np.load(path).astype(np.float32)   # shape = (512,)

def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="数据库 features.npy")
    ap.add_argument("--q", required=True, help="查询向量 q.npy")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    # 1) 载入库向量
    V = load_vecs(args.db)      # (N, 512)
    N = V.shape[0]
    print(f"[+] DB size: {N}")

    # 2) 载入查询向量
    q = load_query(args.q)      # (512,)

    # 3) 归一化
    Vn = normalize(V)
    qn = q / (np.linalg.norm(q) + 1e-9)

    # 4) 计算余弦相似度
    scores = Vn @ qn            # (N,)

    # 5) 取 Top-K
    idx = np.argpartition(scores, -args.topk)[-args.topk:]
    idx = idx[np.argsort(-scores[idx])]

    print("\n========= Plaintext Top-{} =========".format(args.topk))
    for i, cid in enumerate(idx, 1):
        print(f"{i:2d}. cid={cid} score={scores[cid]:.4f}")

    # 保存
    out = {
        "top_ids": idx.tolist(),
        "scores": [float(scores[c]) for c in idx]
    }
    import json
    with open("plaintext_topk.json", "w") as f:
        json.dump(out, f, indent=2)

    print("[OK] saved → plaintext_topk.json")

if __name__ == "__main__":
    main()
