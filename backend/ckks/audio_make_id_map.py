#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

# ========== 路径（你可以改） ==========
IDS_PATH = Path("/home/wen/Desktop/backend/ckks/audio/ids.txt")
ORIG_PATH = Path("/home/wen/Desktop/backend/ckks/audio/original_id")
OUT_PATH = Path("/home/wen/Desktop/backend/ckks/audio/audio_id_map.json")

# ======================================

print(f"[+] 使用 ids.txt: {IDS_PATH}")
print(f"[+] 使用 original_id: {ORIG_PATH}")

# ==== 1) 读取 ids.txt（两列版）====
ids = []
with open(IDS_PATH) as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 1:
            # 单列：audio_000001
            ids.append(parts[0])
        else:
            # 双列：1 0   或   327297  e00610
            ids.append(parts[1])

N_ids = len(ids)
print(f"[+] ids.txt 解析成功，共 {N_ids} 行（使用右侧列作为 candidate_id）")

# ==== 2) 读取 original_id（单列）====
orig = []
with open(ORIG_PATH) as f:
    for line in f:
        line=line.strip()
        if line:
            orig.append(line)

N_orig = len(orig)
print(f"[+] original_id 解析成功，共 {N_orig} 行")

if N_ids != N_orig:
    raise ValueError(f"长度不一致: ids.txt={N_ids}, original_id={N_orig}")

# ==== 3) 生成 id mapping ====
mapping = {}
for cid_str, ori in zip(ids, orig):
    try:
        cid = int(cid_str)
    except:
        # 如果是 "audio_000123" 格式，尝试自动提取编号
        cid = int(''.join(filter(str.isdigit, cid_str)))
    mapping[cid] = ori

# ==== 4) 保存 JSON ====
with open(OUT_PATH, "w") as f:
    json.dump(mapping, f, indent=2, ensure_ascii=False)

print(f"[+] audio_id_map.json 生成成功：{OUT_PATH}")
print(f"[OK] 共映射 {len(mapping)} 个 candidate_id → original_id")
