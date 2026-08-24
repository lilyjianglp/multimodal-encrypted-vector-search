#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
img_to_vec.py
将图片编码为 512 维向量（CLIP ViT-B/32），默认做 L2 归一化并保存为 .npy

改动要点：
- 默认使用 open_clip + ViT-B-32 + pretrained="openai"（与离线构库一致）
- 兼容 --openclip-pretrained 为「本地 .pt 路径」或「模型标识字符串」（如 openai）
- 自动处理 EXIF 方向
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_clip_model(provider: str,
                    model_name: str,
                    device: str,
                    model_path: str = None,
                    openclip_pretrained: str = None):
    """
    返回 (model, preprocess, dim)
    provider: "clip" | "open_clip"
    """
    provider = provider.lower()

    if provider == "clip":
        # openai/clip
        try:
            import clip  # pip install git+https://github.com/openai/CLIP.git
        except Exception as e:
            raise RuntimeError(
                f"无法导入 clip：{e}\n"
                "请先安装：pip install git+https://github.com/openai/CLIP.git"
            )

        # 可选：优先用本地权重（复制或硬链接到 clip 缓存）
        if model_path:
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"--model-path 文件不存在：{model_path}")
            os.environ.setdefault("XDG_CACHE_HOME", str(Path.home() / ".cache"))
            cache_target = Path(os.environ["XDG_CACHE_HOME"]) / "clip" / "ViT-B-32.pt"
            cache_target.parent.mkdir(parents=True, exist_ok=True)
            try:
                if Path(model_path).resolve() != cache_target.resolve():
                    try:
                        if cache_target.exists():
                            cache_target.unlink()
                        os.link(model_path, cache_target)
                    except Exception:
                        import shutil
                        shutil.copyfile(model_path, cache_target)
                print(f"[clip] 离线权重已就绪：{cache_target}")
            except Exception as e:
                print(f"[clip] 准备离线权重到缓存失败（不影响继续）：{e}")

        model, preprocess = clip.load(model_name, device=device, jit=False)
        model.eval()
        return model, preprocess, 512

    elif provider == "open_clip":
        # open_clip
        try:
            import open_clip  # pip install open_clip_torch
            import torch
        except Exception as e:
            raise RuntimeError(
                f"无法导入 open_clip：{e}\n"
                "请先安装：pip install open_clip_torch"
            )

        # 如果传了 --openclip-pretrained：
        #  - 若是本地文件路径：作为权重文件使用
        #  - 若是字符串（如 "openai"/"laion2b_s34b_b79k"）：作为模型标识使用
        if openclip_pretrained:
            if os.path.isfile(openclip_pretrained):
                pretrained_arg = openclip_pretrained  # 本地 .pt
            else:
                pretrained_arg = openclip_pretrained  # 字符串标识
        else:
            pretrained_arg = "openai"  # 与离线构库一致

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_arg, device=device
        )
        model.eval()
        return model, preprocess, 512

    else:
        raise ValueError(f"未知 provider：{provider}（应为 clip 或 open_clip）")


def encode_image_to_vec(model, preprocess, device, img_path: str, provider: str):
    provider = provider.lower()

    # 读取并矫正 EXIF 方向
    img = Image.open(img_path).convert("RGB")
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # 统一预处理
    img_t = preprocess(img).unsqueeze(0)

    # 将张量挪到设备
    try:
        import torch
    except Exception as e:
        raise RuntimeError(f"需要 torch 支持：{e}")
    img_t = img_t.to(device)

    with torch.no_grad():
        # 两个 provider 都是 encode_image
        vec = model.encode_image(img_t).float()

    vec = vec[0].detach().cpu().numpy()  # (512,)
    return vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_image", help="输入图片路径")
    ap.add_argument("out_npy", help="输出 .npy 路径")
    ap.add_argument("--provider", default="open_clip", choices=["clip", "open_clip"],
                    help="模型提供方，默认 open_clip（与构库一致）")
    ap.add_argument("--model", default="ViT-B-32", help="模型名（默认 ViT-B-32）")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="推理设备（默认 cpu）")
    ap.add_argument("--model-path", default=None,
                    help="【clip专用】本地权重 pt 路径（离线）")
    ap.add_argument("--openclip-pretrained", default="openai",
                    help="【open_clip专用】预训练标识或本地 .pt 路径（默认 openai）")
    ap.add_argument("--no-normalize", action="store_true",
                    help="不做 L2 归一化（默认归一化）")
    args = ap.parse_args()

    img_path = Path(args.input_image)
    out_path = Path(args.out_npy)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not img_path.exists():
        print(f"输入图片不存在：{img_path}")
        sys.exit(1)

    # 加载模型
    model, preprocess, dim = load_clip_model(
        provider=args.provider,
        model_name=args.model,
        device=args.device,
        model_path=args.model_path,
        openclip_pretrained=args.openclip_pretrained
    )

    # 编码
    vec = encode_image_to_vec(model, preprocess, args.device, str(img_path), provider=args.provider)

    # 归一化（与检索/构库一致）
    if not args.no_normalize:
        n = np.linalg.norm(vec) + 1e-12
        vec = vec / n

    np.save(out_path, vec.astype("float32"))
    print(f"saved: {out_path}  shape={vec.shape}  norm={np.linalg.norm(vec):.6f}  "
          f"provider={args.provider}, model={args.model}, pretrained={args.openclip_pretrained}")

if __name__ == "__main__":
    main()

