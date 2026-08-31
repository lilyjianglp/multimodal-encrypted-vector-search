#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, numpy as np, hashlib, random
from pathlib import Path

def l2norm(x: np.ndarray):
    n = np.linalg.norm(x) + 1e-12
    return x / n

def prng_from_session(s: str):
    h = hashlib.sha256(s.encode('utf-8')).digest()
    seed = int.from_bytes(h[:8], 'little', signed=False)
    rng = random.Random(seed)
    return rng

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="查询向量 .npy（可 256/512/768/1024/2048 任意维，已做映射到512后用本脚本；或者 centers 也是对应维度）")
    ap.add_argument("--centers", required=True, help="簇中心 .npy (K x D)，明文放客户端")
    ap.add_argument("--topT", type=int, default=4, help="真簇数量 T")
    ap.add_argument("--L", type=int, default=8, help="总簇数 L (T <= L)")
    ap.add_argument("--session", default="sess-0", help="会话id，用于假簇可复现的伪随机")
    ap.add_argument("--metric", choices=["ip","cos"], default="ip", help="打分：ip=点积, cos=余弦")
    ap.add_argument("--out", default="/tmp/clusters.txt", help="输出簇列表（逗号分隔）")
    args = ap.parse_args()

    q = np.load(args.q).astype(np.float32)
    C = np.load(args.centers).astype(np.float32)   # (K,D)
    if args.metric == "cos":
        q = l2norm(q); C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)

    # 内积打分
    scores = C @ q
    top_idx = np.argsort(-scores)[:args.topT]      # 真簇索引（数字）
    true_clusters = [str(int(i)) for i in top_idx.tolist()]

    # 伪随机补充合法的数字簇 ID。旧实现使用 fake_<id>，Gateway 可通过
    # is_digit_only 直接识别并跳过，因而不能混淆真实簇。数字诱饵会像
    # 真簇一样进入候选读取，使 Gateway/Index 只看到混合后的簇集合。
    need_fake = max(0, args.L - len(true_clusters))
    rng = prng_from_session(args.session)
    true_set = {int(i) for i in top_idx.tolist()}
    decoy_pool = [i for i in range(len(C)) if i not in true_set]
    if need_fake > len(decoy_pool):
        raise ValueError(f"L={args.L} exceeds available unique clusters K={len(C)}")
    fake_clusters = [str(i) for i in rng.sample(decoy_pool, need_fake)]

    # 洗牌：混淆真/假顺序
    clusters = true_clusters + fake_clusters
    rng.shuffle(clusters)

    Path(args.out).write_text(",".join(clusters), encoding="utf-8")
    print(",".join(clusters))

if __name__ == "__main__":
    main()
