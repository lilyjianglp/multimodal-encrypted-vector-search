#!/usr/bin/env python3
import numpy as np
import json
from pathlib import Path

def debug_search():
    print("=== 搜索流程诊断 ===")
    
    # 1. 检查查询向量
    q_path = Path("/tmp/q.npy")
    if q_path.exists():
        q = np.load(q_path)
        print(f"1. 查询向量: shape={q.shape}, norm={np.linalg.norm(q):.6f}")
        print(f"   数值范围: [{q.min():.6f}, {q.max():.6f}]")
    else:
        print("1. ❌ 查询向量文件不存在")
    
    # 2. 检查簇选择
    clusters_path = Path("/tmp/clusters.txt")
    if clusters_path.exists():
        clusters = clusters_path.read_text().strip()
        print(f"2. 选择的簇: {clusters}")
    else:
        print("2. ❌ 簇选择文件不存在")
    
    # 3. 检查解密结果
    scores_dir = Path("/home/wen/Desktop/backend/ckks/scores_raw_raw")
    if scores_dir.exists():
        csv_files = list(scores_dir.glob("*.csv"))
        print(f"3. 找到 {len(csv_files)} 个分数文件")
        for csv_file in csv_files[:2]:  # 检查前两个文件
            try:
                # 简单检查CSV内容
                with open(csv_file, 'r') as f:
                    lines = f.readlines()[:5]
                    print(f"   {csv_file.name} 前几行:")
                    for line in lines:
                        print(f"     {line.strip()}")
            except:
                pass

if __name__ == "__main__":
    debug_search()
