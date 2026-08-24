#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embed_image.py
输入图片 → ResNet50(2048) → PCA(2048→512) → vec512.npy
完全使用 offline 权重，与数据库中的 embedding 100% 一致
"""

import argparse
import numpy as np
import pickle
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


# -------------------------------
# 加载 PCA（从 pkl 文件）
# -------------------------------
def load_pca(pkl_path):
    with open(pkl_path, "rb") as f:
        pca = pickle.load(f)

    mean = pca.mean_.astype(np.float32)
    comp = pca.components_.astype(np.float32)

    print(f"[OK] Loaded PCA: mean {mean.shape}, components {comp.shape}")
    return mean, comp


# -------------------------------
# 加载本地 ResNet50 权重
# -------------------------------
def load_resnet50_local(weight_path, device):
    print(f"[INFO] Loading local ResNet50 weight: {weight_path}")

    state = torch.load(weight_path, map_location=device)

    # 建模（不下载任何权重）
    model = models.resnet50(weights=None)
    model.load_state_dict(state)

    # 去掉分类层
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return model, preprocess


# -------------------------------
# 图片 → 2048 向量
# -------------------------------
def img_to_2048(model, preprocess, device, path):
    img = Image.open(path).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(img_t).view(2048).cpu().numpy().astype("float32")

    return feat


# -------------------------------
# 主函数
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="输入图像路径")
    ap.add_argument("--pca", required=True, help="image_pca_2048_to_512.pkl 路径")
    ap.add_argument("--weight", required=True, help="ResNet50 本地权重路径")
    ap.add_argument("--out", default="vec512.npy")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 1) Load PCA
    mean, comp = load_pca(args.pca)

    # 2) Load ResNet50 local
    model, preprocess = load_resnet50_local(args.weight, device)

    # 3) 图像 → 2048
    print("[INFO] Extracting 2048-d feature ...")
    v2048 = img_to_2048(model, preprocess, device, args.image)

    # 4) PCA 2048 → 512
    v512 = np.dot(v2048 - mean, comp.T).astype(np.float32)

    # 5) 保存
    np.save(args.out, v512)
    print(f"[OK] Saved 512-d vector to: {args.out}")
    print("shape:", v512.shape)


if __name__ == "__main__":
    main()

