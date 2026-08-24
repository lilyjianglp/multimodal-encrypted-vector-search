#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import librosa

# ==========================
# LogMelExtractor
# ==========================
class LogMelExtractor(nn.Module):
    def __init__(self, sample_rate=16000, window_size=1024, hop_size=320,
                 mel_bins=64, fmin=50, fmax=14000):
        super().__init__()
        mel_fb = librosa.filters.mel(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax
        )
        self.mel_fb = torch.tensor(mel_fb, dtype=torch.float32)
        self.window_size = window_size
        self.hop_size = hop_size

    def forward(self, wav):
        wav = torch.tensor(wav, dtype=torch.float32)

        stft = torch.stft(
            wav,
            n_fft=self.window_size,
            hop_length=self.hop_size,
            win_length=self.window_size,
            return_complex=True
        )
        spec = stft.abs() ** 2
        mel = torch.matmul(self.mel_fb, spec)
        return torch.log(mel + 1e-8)

# ==========================
# CNN14
# ==========================
class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.c1 = nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch_out)
        self.c2 = nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch_out)

        nn.init.xavier_uniform_(self.c1.weight)
        nn.init.xavier_uniform_(self.c2.weight)

    def forward(self, x):
        x = F.relu_(self.b1(self.c1(x)))
        x = F.relu_(self.b2(self.c2(x)))
        return F.avg_pool2d(x, 2)

class Cnn14(nn.Module):
    def __init__(self):
        super().__init__()
        self.b0  = nn.BatchNorm2d(1)
        self.c1  = ConvBlock(1,64)
        self.c2  = ConvBlock(64,128)
        self.c3  = ConvBlock(128,256)
        self.c4  = ConvBlock(256,512)
        self.c5  = ConvBlock(512,1024)
        self.c6  = ConvBlock(1024,2048)
        self.fc1 = nn.Linear(2048,2048)
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        x = self.b0(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        x = torch.mean(x, dim=[2,3])
        return F.relu_(self.fc1(x))

# ==========================
# 单条音频 → 2048维
# ==========================
def embed_2048(path, mel, model, device):
    wav, sr = sf.read(path)
    if sr != 16000:
        return np.zeros(2048, dtype=np.float32)

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    mel_x = mel(wav).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(mel_x)[0].cpu().numpy().astype(np.float32)

    return feat


# ==========================
# 主程序：2048 → 512 PCA
# ==========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="输入音频文件")
    ap.add_argument("output_npy", help="输出 vec512.npy")
    ap.add_argument("--weight", required=True, help="Cnn14 权重路径")
    ap.add_argument("--pca_mean", required=True, help="audio_pca_mean.npy")
    ap.add_argument("--pca_comp", required=True, help="audio_pca_components.npy")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    mel = LogMelExtractor()
    model = Cnn14().to(device)
    state = torch.load(args.weight, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()

    # 提取 2048 维
    v2048 = embed_2048(args.audio, mel, model, device)

    # 加载 PCA (mean + components)
    mean = np.load(args.pca_mean).astype(np.float32)
    comp = np.load(args.pca_comp).astype(np.float32)

    # 2048 → 512
    v_center = v2048 - mean
    v512 = np.dot(v_center, comp.T).astype(np.float32)

    # L2 归一化
    v512 = v512 / (np.linalg.norm(v512) + 1e-12)

    # 输出
    np.save(args.output_npy, v512)
    print("Saved:", args.output_npy, "shape=", v512.shape)

