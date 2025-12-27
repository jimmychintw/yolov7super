#!/usr/bin/env python3
"""
比較四組訓練結果的 mAP@0.5 曲線
"""

import matplotlib.pyplot as plt
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
                    epoch_str = parts[0]  # e.g., "0/499"
                    epoch = int(epoch_str.split('/')[0])
                    mAP = float(parts[10])  # mAP@0.5 is column 11 (index 10)
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

# 找出每組的最佳 mAP 和對應 epoch
best1_idx = np.argmax(m1)
best2_idx = np.argmax(m2)
best3_idx = np.argmax(m3)
best4_idx = np.argmax(m4)

# 設定圖表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# 上圖：完整範圍 (0-500 epochs)
ax1.plot(e1, m1, 'b-', linewidth=1.5, label=f'1B1H 500ep (Best: {m1[best1_idx]:.4f} @ ep{e1[best1_idx]})')
ax1.plot(e2, m2, 'g-', linewidth=1.5, label=f'1B4H Standard 100ep (Best: {m2[best2_idx]:.4f} @ ep{e2[best2_idx]})')
ax1.plot(e3, m3, 'r-', linewidth=1.5, label=f'1B4H Geometry 200ep (Best: {m3[best3_idx]:.4f} @ ep{e3[best3_idx]})')
ax1.plot(e4, m4, 'm-', linewidth=1.5, label=f'1B4H Hybrid Balanced (Best: {m4[best4_idx]:.4f} @ ep{e4[best4_idx]})')

# 標記最佳點
ax1.plot(e1[best1_idx], m1[best1_idx], 'b*', markersize=15)
ax1.plot(e2[best2_idx], m2[best2_idx], 'g*', markersize=15)
ax1.plot(e3[best3_idx], m3[best3_idx], 'r*', markersize=15)
ax1.plot(e4[best4_idx], m4[best4_idx], 'm*', markersize=15)

ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('mAP@0.5', fontsize=12)
ax1.set_title('Training Curves Comparison (Full Range: 0-500 Epochs)', fontsize=14)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 500)
ax1.set_ylim(0, 0.5)

# 下圖：前 100 epochs 公平比較
ax2.plot(e1[:100] if len(e1) >= 100 else e1, m1[:100] if len(m1) >= 100 else m1,
         'b-', linewidth=1.5, label='1B1H 500ep')
ax2.plot(e2, m2, 'g-', linewidth=1.5, label='1B4H Standard 100ep')
ax2.plot(e3[:100] if len(e3) >= 100 else e3, m3[:100] if len(m3) >= 100 else m3,
         'r-', linewidth=1.5, label='1B4H Geometry 200ep')
ax2.plot(e4[:100] if len(e4) >= 100 else e4, m4[:100] if len(m4) >= 100 else m4,
         'm-', linewidth=1.5, label='1B4H Hybrid Balanced')

# 在 epoch 100 標註數值
ep100_values = []
for name, epochs, maps, color in [
    ('1B1H', e1, m1, 'b'),
    ('Standard', e2, m2, 'g'),
    ('Geometry', e3, m3, 'r'),
    ('Hybrid', e4, m4, 'm')
]:
    if len(epochs) >= 100:
        idx = 99  # epoch 99 (0-indexed)
        val = maps[idx]
        ep100_values.append((name, val, color))
    elif len(epochs) > 0:
        val = maps[-1]
        ep100_values.append((name, val, color))

# 標註 epoch 100 的值
y_offset = 0.01
for i, (name, val, color) in enumerate(ep100_values):
    ax2.axhline(y=val, color=color, linestyle='--', alpha=0.3)
    ax2.annotate(f'{name}: {val:.4f}',
                xy=(100, val),
                xytext=(102, val + y_offset * (i - 1.5)),
                fontsize=9, color=color)

ax2.axvline(x=100, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('mAP@0.5', fontsize=12)
ax2.set_title('Training Curves Comparison (Epoch 0-100, Fair Comparison)', fontsize=14)
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 110)
ax2.set_ylim(0, 0.5)

plt.tight_layout()
plt.savefig('/Users/jimmy/Projects/Yolov7fast/training_4_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("圖表已儲存至 training_4_comparison.png")

# 印出統計
print("\n=== 訓練結果統計 ===")
print(f"{'訓練名稱':<30} {'Epochs':<10} {'Best mAP@0.5':<15} {'Best Epoch':<12}")
print("-" * 70)
print(f"{'1B1H 500ep':<30} {len(e1):<10} {m1[best1_idx]:<15.4f} {e1[best1_idx]:<12}")
print(f"{'1B4H Standard 100ep':<30} {len(e2):<10} {m2[best2_idx]:<15.4f} {e2[best2_idx]:<12}")
print(f"{'1B4H Geometry 200ep':<30} {len(e3):<10} {m3[best3_idx]:<15.4f} {e3[best3_idx]:<12}")
print(f"{'1B4H Hybrid Balanced':<30} {len(e4):<10} {m4[best4_idx]:<15.4f} {e4[best4_idx]:<12}")

print("\n=== Epoch 100 公平比較 ===")
for name, val, _ in ep100_values:
    print(f"{name}: {val:.4f}")
