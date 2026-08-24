#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse


def load_top_list(path):
    """读取 JSON 并自动适配 top-k 列表字段名称"""
    obj = json.load(open(path))

    # 优先级：topk → top_ids → list 本体
    if isinstance(obj, list):
        return obj

    if "topk" in obj:
        return obj["topk"]

    if "top_ids" in obj:
        return obj["top_ids"]

    # 未找到可识别字段
    raise ValueError(f"{path} 中未找到 topk / top_ids 字段")


def compute_metrics(gt, pred):
    set_gt = set(gt)
    set_pred = set(pred)

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
    args = ap.parse_args()

    gt = load_top_list(args.gt_json)
    pred = load_top_list(args.pred_json)

    K = len(gt)
    print(f"[+] GT size = {len(gt)}, Pred size = {len(pred)}")
    print(f"[+] 计算 Recall@{K} ...")

    m = compute_metrics(gt, pred)

    print("\n========= 结果 =========")
    print(f"Recall@{K}    = {m['recall']:.4f}")
    print(f"Precision@{K} = {m['precision']:.4f}")
    print(f"Jaccard       = {m['jaccard']:.4f}")
    print(f"重叠数量       = {len(m['intersection'])}")
    print(f"重叠增强ID列表 = {m['intersection']}")


if __name__ == "__main__":
    main()

