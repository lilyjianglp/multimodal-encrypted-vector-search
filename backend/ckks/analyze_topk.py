#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import json
import argparse
from pathlib import Path
from collections import defaultdict
import time

# ==========================================================
# 读取 scores_*.csv
# ==========================================================
def load_blocks(scores_dir, pack_slots):
    files = sorted(Path(scores_dir).glob("scores_*.csv"))
    blocks = []
    for f in files:
        vals = []
        with open(f) as fh:
            next(fh)  # skip header
            for line in fh:
                slot, sc = line.strip().split(",")
                vals.append(float(sc))
        if len(vals) != pack_slots:
            raise ValueError(f"{f} 长度错误，期待 {pack_slots}")
        blocks.append(vals)
    return np.array(blocks)

# ==========================================================
# 音频 original_id 映射
# ==========================================================
def load_id_map(json_path):
    if not json_path or not Path(json_path).exists():
        return {}
    try:
        j = json.load(open(json_path))
    except:
        return {}
    mapping = {}
    if isinstance(j, dict):
        for k, v in j.items():
            try:
                cid = int(k)
            except:
                continue
            mapping[cid] = v
    return mapping

# ==========================================================
# 保存 JSON - 原有格式（仅ID）
# ==========================================================
def save_json(path, top_ids):
    obj = {"topk": top_ids}
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    print(f"[OK] 仅ID结果保存 → {path}")

# ==========================================================
# 保存 JSON - 带分数的版本
# ==========================================================
def save_json_with_scores(path, top_items_with_scores, mode):
    obj = {
        "topk": [
            {
                "id": item_id,
                "score": float(score),
                "rank": rank
            }
            for rank, (item_id, score) in enumerate(top_items_with_scores, 1)
        ],
        "mode": mode,
        "total_count": len(top_items_with_scores),
        "timestamp": time.time()
    }

    with open(path, "w", encoding='utf-8') as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    print(f"[OK] 带分数结果保存 → {path}")


# ==========================================================
# 主函数
# ==========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--slot_ids", required=True)
    ap.add_argument("--pack_slots", type=int, default=4096)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--id_map_json", default="")
    ap.add_argument("--img_group", type=int, default=12)
    ap.add_argument("--mode",
                    default="image",
                    choices=["image", "audio", "text", "raw", "recall"])
    ap.add_argument("--save_json")
    ap.add_argument("--save_scores_json")
    args = ap.parse_args()

    if not args.save_json:
        args.save_json = str(Path(args.scores_dir) / "topk.json")
    if not args.save_scores_json:
        args.save_scores_json = str(Path(args.scores_dir) / f"{Path(args.save_json).stem}_with_scores.json")

    print(f"======== analyze_topk（mode={args.mode}） ========")

    # ============= load scores =============
    blocks = load_blocks(args.scores_dir, args.pack_slots)
    scores_all = np.concatenate(blocks, axis=0)
    print(f"[+] Loaded {blocks.shape[0]} blocks × {args.pack_slots} slots = {len(scores_all)} scores")

    # ============= load slot_ids ============
    raw_ids = json.load(open(args.slot_ids))
    slot_ids = []
    for x in raw_ids:
        if isinstance(x, str) and not x.isdigit():
            slot_ids.append(x)
        else:
            try:
                slot_ids.append(int(x))
            except:
                slot_ids.append(x)

    print(f"[+] slot_ids loaded: {len(slot_ids)} entries")
    audio_map = load_id_map(args.id_map_json)

    # ======================================================
    # 保持 raw、recall、audio 全部原样
    # ======================================================
    if args.mode == "raw":
        slot_scores = [(cid, sc) for cid, sc in zip(slot_ids, scores_all)]
        slot_scores.sort(key=lambda x: -abs(x[1]))
        top = slot_scores[:args.topk]
        save_json(args.save_json, [cid for cid,_ in top])
        save_json_with_scores(args.save_scores_json, [(cid, abs(sc)) for cid, sc in top], args.mode)
        return

    if args.mode == "recall":
        cid_scores = {}
        for cid, sc in zip(slot_ids, scores_all):
            cid_scores[cid] = max(cid_scores.get(cid, -1e18), sc)
        items = sorted(cid_scores.items(), key=lambda x: -abs(x[1]))
        top = items[:args.topk]
        save_json(args.save_json, [cid for cid,_ in top])
        save_json_with_scores(args.save_scores_json, [(cid, abs(sc)) for cid, sc in top], args.mode)
        return

    if args.mode == "audio":
        best = defaultdict(lambda: -1e18)
        for cid, sc in zip(slot_ids, scores_all):
            if cid in audio_map:
                orig = audio_map[cid]
                best[orig] = max(best[orig], sc)
        items = sorted(best.items(), key=lambda x: -abs(x[1]))
        top = items[:args.topk]
        maxv = max(abs(sc) for _, sc in top) or 1
        save_json(args.save_json, [orig for orig,_ in top])
        save_json_with_scores(args.save_scores_json, [(orig, abs(sc)/maxv) for orig, sc in top], args.mode)
        return

    # ======================================================
    # 修复后的 TEXT（替换旧逻辑）
    # ======================================================
    if args.mode == "text":
        print("[*] 文本模式（修复版本：对齐 + 归一化）")

        text_ids = raw_ids
        N = len(text_ids)

        results = []
        for idx, sc in enumerate(scores_all):
            pmc = text_ids[idx % N]
            results.append((pmc, abs(sc)))

        maxv = max(sc for _, sc in results) or 1
        results = [(pmc, sc / maxv) for pmc, sc in results]

        results.sort(key=lambda x: -x[1])
        top = results[:args.topk]

        save_json(args.save_json, [pmc for pmc,_ in top])
        save_json_with_scores(args.save_scores_json, top, args.mode)
        return

    # ======================================================
    # 修复后的 IMAGE（替换旧逻辑）
    # ======================================================
    print("[*] 图像模式（修复版本：增强聚合）")

    group_best = {}
    for cid, sc in zip(slot_ids, scores_all):
        group_id = cid // args.img_group
        if group_id not in group_best or sc > group_best[group_id][0]:
            group_best[group_id] = (sc, cid)

    items = [(cid, sc) for sc, cid in group_best.values()]
    items.sort(key=lambda x: -abs(x[1]))
    top = items[:args.topk]

    maxv = max(abs(sc) for _, sc in top) or 1

    save_json(args.save_json, [cid for cid,_ in top])
    save_json_with_scores(
        args.save_scores_json,
        [(f"img_{cid:06d}", abs(sc)/maxv) for cid, sc in top],
        args.mode
    )


if __name__ == "__main__":
    main()

