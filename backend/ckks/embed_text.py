#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
import pickle
import re
from transformers import BertTokenizer, BertModel


# =========================================
# 句子分割器：对医学病例效果最好的版本（无依赖）
# =========================================
def split_into_sentences(text):
    text = text.replace("\n", " ").strip()

    # 用正则按句号、问号、感叹号切分
    sents = re.split(r'(?<=[.!?])\s+', text)

    # 清理空句子
    sents = [s.strip() for s in sents if len(s.strip()) > 0]

    # 如果分句后太少，直接按500字符切一下防止超长
    if len(sents) <= 1 and len(text) > 400:
        sents = [text[i:i+300] for i in range(0, len(text), 300)]

    return sents


# =========================================
# 单句 embedding
# =========================================
def encode_single(text, tokenizer, model, device):
    with torch.no_grad():
        enc = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=256,           # 防止截断过长句子
            return_tensors="pt"
        ).to(device)

        out = model(**enc)
        last = out.last_hidden_state       # [1, seq, 768]
        mask = enc["attention_mask"].unsqueeze(-1)  # [1, seq, 1]

        summed = (last * mask).sum(dim=1)          # sum pooling
        count = mask.sum(dim=1).clamp(min=1)       # number of tokens
        return (summed / count)[0].cpu().numpy()   # → 768维


# =========================================
# 多句 mean pooling（核心优化！）
# =========================================
def encode_text_with_mean_pooling(text, tokenizer, model, device):
    sentences = split_into_sentences(text)

    feats = []
    for sent in sentences:
        feat = encode_single(sent, tokenizer, model, device)
        feats.append(feat)

    # mean pooling 所有句子向量
    feats = np.vstack(feats)              # [num_sent, 768]
    mean_feat = feats.mean(axis=0)        # [768]

    return mean_feat


# =========================================
# 主逻辑
# =========================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("output_npy", help="output npy file")
    ap.add_argument("--text", help="raw text input")
    ap.add_argument("--file", help="text file input")
    ap.add_argument("--pca", required=True)
    args = ap.parse_args()

    # ------- 读取文本 -------
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        raise ValueError("Must provide --text or --file")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_PATH = "/home/wen/Desktop/backend/ckks/text/Bio_ClinicalBERT/"
    print(f"[INFO] Using offline Bio_ClinicalBERT: {MODEL_PATH}")

    tokenizer = BertTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    model = BertModel.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    ).to(device).eval()

    # =========================================
    # ★★ 多句 mean pooling embedding（增强版）
    # =========================================
    feat768 = encode_text_with_mean_pooling(text, tokenizer, model, device)

    # ------- PCA -------
    with open(args.pca, "rb") as f:
        pca = pickle.load(f)

    vec512 = pca.transform([feat768])[0].astype("float32")

    # ------- L2 normalize -------
    vec512 = vec512 / (np.linalg.norm(vec512) + 1e-12)

    np.save(args.output_npy, vec512)
    print("saved:", args.output_npy, "shape=", vec512.shape)

