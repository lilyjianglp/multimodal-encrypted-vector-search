#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os

# 输入文件（你的 image embedding 的 id_mapping.json）
IN = "/media/wen/F500/image_embeddings/id_mapping.json"

# 输出文件：生成到同目录下
OUT = "/media/wen/F500/image_embeddings/slot_ids.json"

def main():
    if not os.path.exists(IN):
        print(f"[Error] 找不到 id_mapping.json: {IN}")
        return
    
    with open(IN, "r") as f:
        mp = json.load(f)

    # mp: { "0":"img_000001", "1":"img_000002", ... }
    N = len(mp)
    slot_ids = [""] * N

    for k, v in mp.items():
        idx = int(k)
        slot_ids[idx] = v

    with open(OUT, "w") as f:
        json.dump(slot_ids, f, indent=2)

    print(f"[OK] 已生成 slot_ids.json：{OUT}（共 {N} 条）")

if __name__ == "__main__":
    main()

