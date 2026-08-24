#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 ~/pics40 读取图片，通过 open-clip ViT-B-32 生成 512 维 embedding。
若原始图片不足 1 万，做数据增强（随机裁剪/翻转/色彩抖动）补足到 N=10000。
输出：
  ~/bigmix/D0512/img_512.npy      (Ni,512) float32，已 L2 归一
  ~/bigmix/D0512/corpus.f32       同上 row-major .f32
  ~/bigmix/D0512/ids.txt          0..Ni-1
  ~/Desktop/backend/ckks/id_map.csv   id 到原始图片路径映射（方便 show_topk_images）
"""
import os, glob, csv, math, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torch
import open_clip

SRC = Path.home() / "pics40"         # 你的图片目录（已存在）
OUT = Path.home() / "bigmix/D0512"   # 统一 512 维数据目录
OUT.mkdir(parents=True, exist_ok=True)
TARGET_N = 2000

# 1) 列出图片
paths = []
for ext in ("*.jpg","*.jpeg","*.png","*.webp","*.bmp"):
    paths += glob.glob(str(SRC / ext))
paths = sorted(paths)
assert len(paths) > 0, f"no images found in {SRC}"

# 2) 模型（ViT-B-32 -> 512 维）
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k",  # 公开权重，输出 512 维
    device=device
)
model.eval()

def pil_augment(img: Image.Image):
    # 轻量增强：随机水平翻转 + 轻度色彩/对比度/亮度抖动 + 随机resize-crop
    img = img.convert("RGB")
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    # 色彩抖动
    img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
    return img

def encode_batch(img_list):
    with torch.no_grad():
        ims = torch.stack([preprocess(im) for im in img_list]).to(device)
        feats = model.encode_image(ims)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()

# 3) 生成 N 条向量（原图 + 增强复制）
all_paths = []
while len(all_paths) < TARGET_N:
    for p in paths:
        all_paths.append(p)
        if len(all_paths) >= TARGET_N:
            break

# 4) 编码
B = 64
embeds = np.zeros((len(all_paths), 512), dtype=np.float32)
for s in range(0, len(all_paths), B):
    batch = []
    for q in all_paths[s:s+B]:
        im = Image.open(q)
        if q not in paths:   # 这里我们不区分，统一做轻增广
            im = pil_augment(im)
        else:
            # 原图也做轻增广以丰富分布（可关）
            im = pil_augment(im)
        batch.append(im)
    arr = encode_batch(batch)
    embeds[s:s+len(batch)] = arr
    print(f"encoded {s+len(batch)}/{len(all_paths)}")

# 5) 再次 L2 归一，落盘
embeds /= (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-12)
np.save(OUT/"img_512.npy", embeds)
# .f32
m = np.memmap(OUT/"corpus.f32", mode="w+", dtype=np.float32, shape=embeds.shape)
m[:] = embeds[:]; del m
# ids & id_map
with open(OUT/"ids.txt","w") as f:
    for i in range(embeds.shape[0]): f.write(str(i)+"\n")
with open(Path.home()/ "Desktop/backend/ckks/id_map.csv", "w", newline="") as f:
    w=csv.writer(f); w.writerow(["id","path"])
    for i,p in enumerate(all_paths): w.writerow([i,p])

print("Done ->", OUT/"img_512.npy", embeds.shape)
print("id_map.csv ->", Path.home()/ "Desktop/backend/ckks/id_map.csv")
