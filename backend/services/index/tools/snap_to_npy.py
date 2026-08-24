#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import os
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snap", help="centers.snap")
    ap.add_argument("out", help="centers.npy")
    ap.add_argument("--k", type=int, required=True, help="K (clusters)")
    ap.add_argument("--d", type=int, required=True, help="dimension (e.g., 512)")
    args = ap.parse_args()

    snap = args.snap
    out = args.out

    expected = args.k * args.d
    filesize = os.path.getsize(snap)

    if filesize != expected * 4:
        print(f"[ERROR] file size mismatch!")
        print(f"  file bytes = {filesize}")
        print(f"  expected   = {expected*4}")
        sys.exit(1)

    data = np.fromfile(snap, dtype=np.float32)
    data = data.reshape(args.k, args.d)

    np.save(out, data)
    print(f"[OK] wrote {out} with shape {data.shape}")

if __name__ == "__main__":
    main()
