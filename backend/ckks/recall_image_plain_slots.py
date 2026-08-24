#!/usr/bin/env python3
import numpy as np
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    q = np.load(args.q)
    db = np.load(args.db)

    # 计算余弦相似度
    qn = q / np.linalg.norm(q)
    dbn = db / np.linalg.norm(db, axis=1, keepdims=True)
    sims = dbn @ qn

    # 直接取 slot topk（不分组）
    idx = np.argsort(-sims)[:args.topk]

    print("===== Plain Top-K (slot level) =====")
    for r, i in enumerate(idx, 1):
        print(f"{r}. slot={i}  score={sims[i]:.4f}")

    # 保存到文件用于密态对比
    with open("gt_plain_slots.txt","w") as f:
        for i in idx:
            f.write(str(i)+"\n")
    print("[OK] Saved slot GT → gt_plain_slots.txt")

if __name__ == "__main__":
    main()
