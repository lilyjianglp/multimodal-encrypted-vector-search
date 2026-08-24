#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import pandas as pd

# ====== 修改为你的路径 ======
TOPK_JSON = "/home/wen/Desktop/backend/ckks/scores_raw_raw/topk.json"
TEXT_DIR = "/media/wen/F500/text_embeddings"
CASES_PARQUET = os.path.join(TEXT_DIR, "cases.parquet")

def normalize_pmc_id(c):
    """ pmc=PMC4036149_01 → PMC4036149_01 """
    if c.startswith("pmc="):
        return c.split("=", 1)[1]
    return c

def extract_text_from_case(case):
    """根据 case 字段自动提取文本内容"""
    parts = []

    if "case_summary" in case and isinstance(case["case_summary"], list):
        parts.append("Case Summary:\n" + "\n".join(case["case_summary"]))

    for key in ["finding", "impression", "history", "background", "text"]:
        if key in case and isinstance(case[key], str) and len(case[key]) > 0:
            parts.append(f"{key.capitalize()}:\n{case[key]}")

    if not parts:
        parts.append(str(case))

    return "\n\n".join(parts)

def main():
    print("========= Top-K 文本检索 =========\n")

    # ---- 读取 topk.json ----
    with open(TOPK_JSON, "r", encoding="utf8") as f:
        data = json.load(f)

    top_list = data.get("topk", [])

    print("[INFO] 正在加载 cases.parquet ...")
    df = pd.read_parquet(CASES_PARQUET)
    print("[OK] 加载完成\n")

    # ---- 遍历 Top-K ----
    for idx, item in enumerate(top_list, 1):
        cid_full = normalize_pmc_id(item)           # PMC5015624_01
        article_id = cid_full.split("_")[0]         # PMC5015624

        print(f"\n[{idx}] cid={cid_full}")

        # 找到对应 article
        rows = df[df["article_id"] == article_id]
        if len(rows) == 0:
            print("   ✗ 未找到 article_id\n")
            continue

        cases = rows.iloc[0]["cases"]  # list of dict

        # 在 cases 中查找具体段落
        target = None
        for c in cases:
            if c.get("case_id") == cid_full:
                target = c
                break

        if target is None:
            print("   ✗ article 找到，但没有该 case_id\n")
            continue

        # 提取文本内容
        text = extract_text_from_case(target)

        print("-----------------------------------------------------")
        print(text)
        print("-----------------------------------------------------")

        input("\n按回车查看下一条文本...\n")

    print("\n展示完成。")


if __name__ == "__main__":
    main()

