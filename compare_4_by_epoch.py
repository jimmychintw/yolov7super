#!/usr/bin/env python3
"""
比較四組訓練結果，按 epoch 列出 mAP@0.5
"""

import numpy as np

def parse_results(filepath):
    """解析 results.txt 檔案，提取 mAP@0.5"""
    epochs = []
    map50 = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 11:
                try:
                    epoch_str = parts[0]
                    epoch = int(epoch_str.split('/')[0])
                    mAP = float(parts[10])
                    epochs.append(epoch)
                    map50.append(mAP)
                except (ValueError, IndexError):
                    continue
    return np.array(epochs), np.array(map50)

# 載入資料
e1, m1 = parse_results('/tmp/1b1h_500ep.txt')
e2, m2 = parse_results('/tmp/1b4h_standard_100ep.txt')
e3, m3 = parse_results('/tmp/1b4h_geometry_200ep.txt')
e4, m4 = parse_results('/tmp/1b4h_hybrid_balanced.txt')

# 建立對照表
print("=" * 100)
print(f"{'Epoch':<8} {'1B1H':<12} {'Standard':<12} {'Geometry':<12} {'Hybrid':<12} {'Best':<15}")
print("=" * 100)

# 關鍵 epochs
key_epochs = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 499]

for ep in key_epochs:
    v1 = m1[ep] if ep < len(m1) else None
    v2 = m2[ep] if ep < len(m2) else None
    v3 = m3[ep] if ep < len(m3) else None
    v4 = m4[ep] if ep < len(m4) else None

    values = []
    labels = []
    if v1 is not None:
        values.append(v1)
        labels.append('1B1H')
    if v2 is not None:
        values.append(v2)
        labels.append('Standard')
    if v3 is not None:
        values.append(v3)
        labels.append('Geometry')
    if v4 is not None:
        values.append(v4)
        labels.append('Hybrid')

    if values:
        best_idx = np.argmax(values)
        best_label = labels[best_idx]
        best_val = values[best_idx]
    else:
        best_label = '-'

    s1 = f"{v1:.4f}" if v1 is not None else "-"
    s2 = f"{v2:.4f}" if v2 is not None else "-"
    s3 = f"{v3:.4f}" if v3 is not None else "-"
    s4 = f"{v4:.4f}" if v4 is not None else "-"

    print(f"{ep:<8} {s1:<12} {s2:<12} {s3:<12} {s4:<12} {best_label:<15}")

print("=" * 100)

# 最終結果
print("\n" + "=" * 100)
print("最終結果 (Best mAP@0.5)")
print("=" * 100)
print(f"1B1H 500ep:        {np.max(m1):.4f} @ epoch {np.argmax(m1)}")
print(f"Standard 100ep:    {np.max(m2):.4f} @ epoch {np.argmax(m2)}")
print(f"Geometry 200ep:    {np.max(m3):.4f} @ epoch {np.argmax(m3)}")
print(f"Hybrid Balanced:   {np.max(m4):.4f} @ epoch {np.argmax(m4)} (訓練中, {len(m4)} epochs)")
print("=" * 100)
