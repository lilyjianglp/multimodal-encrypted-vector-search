#!/usr/bin/env python3
import json

IN = "/home/wen/Desktop/backend/ckks/image/id_mapping.json"
OUT = "/home/wen/Desktop/backend/ckks/image/slot_ids.json"

with open(IN, "r") as f:
    mp = json.load(f)

# mp 是：{ "0":"img_000000", "1":"img_000001", ... }
N = len(mp)
slot_ids = list(range(N))   # 数字 ID 数组

with open(OUT, "w") as f:
    json.dump(slot_ids, f)

print(f"[OK] Wrote {OUT} with {N} numeric slot IDs.")
