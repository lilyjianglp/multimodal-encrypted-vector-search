#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse


def load_top_list(path):
    """自动适配 {"topk":[...]} 或 {"top_ids":[...]} 或纯 list"""
    obj = json.load(open(path))

    if isinstance(obj, list):
        return obj

    if "topk" in obj:
        return obj["topk"]

    if "top_ids" in obj:
        return obj["top_ids"]

    raise ValueError(f"{path} 中未找到 topk 或 top_ids 字段")


def to_groups(cid_list, img_group=12):
    """cid → group_id（例如：0..11 映射到 group 0）"""
    return [cid // img_group for cid in cid_list]


def compute_metrics(gt_groups, pred_groups):
    set_gt = set(gt_groups)
    set_pred = set(pred_groups)

    inter = set_gt & set_pred
    union = set_gt | set_pred

    recall = len(inter) / len(set_gt) if len(set_gt) else 0.0
    precision = len(inter) / len(set_pred) if len(set_pred) else 0.0
    jaccard = len(inter) / len(union) if len(union) else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "jaccard": jaccard,
        "intersection": sorted(list(inter)),
        "gt_size": len(set_gt),
        "pred_size": len(set_pred),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt_json", help="plaintext_topk.json")
    ap.add_argument("pred_json", help="he_topk.json")
    ap.add_argument("--img_group", type=int, default=12,
                    help="增强倍数 (默认=12)")
    args = ap.parse_args()

    print(f"[+] 使用增强分组 img_group = {args.img_group}")

    # 加载增强 ID
    gt = load_top_list(args.gt_json)
    pred = load_top_list(args.pred_json)

    # 映射为 group_id（原图级别）
    gt_groups = to_groups(gt, args.img_group)
    pred_groups = to_groups(pred, args.img_group)

    K = len(gt)
    print(f"[+] GT size = {len(gt)}, Pred size = {len(pred)}")
    print(f"[+] 计算 Group-Level Recall@{K} ...")

    m = compute_metrics(gt_groups, pred_groups)

    print("\n========= 原始样本级别（Group-Level）结果 =========")
    print(f"Recall@{K}    = {m['recall']:.4f}")
    print(f"Precision@{K} = {m['precision']:.4f}")
    print(f"Jaccard       = {m['jaccard']:.4f}")
    print(f"重叠 group 数  = {len(m['intersection'])}")
    print(f"重叠 group 列表 = {m['intersection']}")


if __name__ == "__main__":
    main()
