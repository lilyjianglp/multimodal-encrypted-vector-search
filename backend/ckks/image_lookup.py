#!/usr/bin/env python3
import os
import json
import argparse
from PIL import Image
import matplotlib.pyplot as plt


def find_image(root, img_id):
    """在 root 目录中寻找 img_xxxxx 的图片文件"""
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    for ext in exts:
        fpath = os.path.join(root, img_id + ext)
        if os.path.exists(fpath):
            return fpath
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", required=True, help="topk.json 文件")
    parser.add_argument("--root", required=True, help="图片根目录")
    args = parser.parse_args()

    # ---- 读取 topk.json ----
    with open(args.topk, "r") as f:
        data = json.load(f)

    # data = {"topk": [int, int, ...]}
    if isinstance(data, dict) and "topk" in data:
        items = data["topk"]
    else:
        print("[ERROR] topk.json 格式不正确，必须包含 topk 字段")
        return

    print("\n========= Top-K 检索图片 =========\n")

    img_paths = []

    for rank, item in enumerate(items, start=1):
        # item 是整数，例如 94824 → "img_094824"
        if isinstance(item, int):
            img_id = f"img_{item:06d}"
        elif isinstance(item, str) and item.isdigit():
            img_id = f"img_{int(item):06d}"
        else:
            print(f"[{rank}] 无法识别的 ID 格式: {item}")
            continue

        fpath = find_image(args.root, img_id)

        if fpath:
            print(f"[{rank}] {img_id} -> {fpath}")
            img_paths.append((rank, img_id, fpath))
        else:
            print(f"[{rank}] {img_id} -> 未找到文件")

    print("\n开始逐张弹出图片...\n")

    # ==== 单张展示 ====
    for rank, img_id, fpath in img_paths:
        img = Image.open(fpath)

        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{rank}. {img_id}")

        plt.show()  # 阻塞模式，一张张看


if __name__ == "__main__":
    main()

