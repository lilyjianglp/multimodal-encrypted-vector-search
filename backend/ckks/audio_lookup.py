#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import subprocess

# ====== 修改为你的路径 ======
TOPK_JSON = "/home/wen/Desktop/backend/ckks/scores_raw_raw/topk.json"
AUDIO_DIR = "/media/wen/F500/audio_raw"

def play_audio(path):
    """自动调用 Linux 默认播放器播放 .wav 文件"""
    print(f"[PLAY] 播放音频文件: {path}")
    try:
        subprocess.Popen(["xdg-open", path])
        return
    except:
        pass

    try:
        subprocess.Popen(["ffplay", "-nodisp", "-autoexit", path])
        return
    except:
        pass

    try:
        subprocess.Popen(["aplay", path])
        return
    except:
        print("⚠ 无法自动播放音频，请手动检查播放器是否安装")


def main():
    print("========= Top-K 音频检索 =========\n")

    if not os.path.exists(TOPK_JSON):
        print("[ERROR] topk.json 不存在！请检查路径。")
        return

    with open(TOPK_JSON, "r", encoding="utf8") as f:
        data = json.load(f)

    # ---- 重点修改：当前 topk.json 结构 ----
    if isinstance(data, dict) and "topk" in data:
        top_list = data["topk"]
    else:
        print("[ERROR] topk.json 格式不正确，应包含 topk 字段。")
        return

    # ---- top_list 是字符串列表 ["id1","id2",...] ----
    for idx, item in enumerate(top_list, 1):
        cid = item            # item 是字符串
        original_id = item    # 对音频来说，original_id 就是 cid

        print(f"[{idx}] audio_id={cid}")

        wav_path = os.path.join(AUDIO_DIR, original_id + ".wav")
        print(f"   路径: {wav_path}")

        if os.path.exists(wav_path):
            print("   ✓ 找到音频文件，准备播放...")

            play_audio(wav_path)
            input("\n按回车播放下一条音频...\n")
        else:
            print("   ✗ 文件不存在，请检查路径或挂载点\n")

        print()

    print("展示完成。")


if __name__ == "__main__":
    main()