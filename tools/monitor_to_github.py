#!/usr/bin/env python3
"""
訓練監控腳本 - 自動產生圖表並推送到 GitHub
手機可直接查看: https://raw.githubusercontent.com/jimmychintw/Yolov7fast/main/monitor/training_status.png
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
import os
import subprocess
import time
from datetime import datetime

# 設定
RESULTS_FILE = '/workspace/Yolov7fast/runs/train/1b4h_stage1_neck_tune_200ep/results.txt'
OUTPUT_DIR = '/workspace/Yolov7fast/monitor'
OUTPUT_IMAGE = f'{OUTPUT_DIR}/training_status.png'
UPDATE_INTERVAL = 300  # 5分鐘

# 中文字型 (server 可能沒有，用英文)
plt.rcParams['font.family'] = 'DejaVu Sans'

def parse_results(filepath):
    """解析 results.txt"""
    data = {
        'epoch': [], 'box': [], 'obj': [], 'cls': [], 'total': [],
        'precision': [], 'recall': [], 'mAP50': [], 'mAP50_95': []
    }

    if not os.path.exists(filepath):
        return data

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 14:
                try:
                    epoch_str = parts[0].split('/')[0]
                    data['epoch'].append(int(epoch_str))
                    data['box'].append(float(parts[2]))
                    data['obj'].append(float(parts[3]))
                    data['cls'].append(float(parts[4]))
                    data['total'].append(float(parts[5]))
                    data['precision'].append(float(parts[8]))
                    data['recall'].append(float(parts[9]))
                    data['mAP50'].append(float(parts[10]))
                    data['mAP50_95'].append(float(parts[11]))
                except:
                    continue

    return {k: np.array(v) for k, v in data.items()}

def generate_chart(data):
    """產生監控圖表"""
    if len(data['epoch']) == 0:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    current_epoch = data['epoch'][-1]
    total_epochs = 200
    current_mAP = data['mAP50'][-1]
    best_mAP = max(data['mAP50'])
    best_epoch = data['epoch'][np.argmax(data['mAP50'])]

    fig.suptitle(f"Stage1.1 Training Monitor | Epoch {current_epoch}/{total_epochs} | "
                 f"mAP: {current_mAP:.4f} | Best: {best_mAP:.4f}@ep{best_epoch}\n"
                 f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 fontsize=12, fontweight='bold')

    # 1. mAP@0.5
    ax1 = axes[0, 0]
    ax1.plot(data['epoch'], data['mAP50'], 'b-', linewidth=2)
    ax1.axhline(y=0.4268, color='green', linestyle='--', alpha=0.7, label='Stage1 final (0.4268)')
    ax1.axhline(y=0.4303, color='red', linestyle='--', alpha=0.7, label='Str A final (0.4303)')
    ax1.scatter([current_epoch], [current_mAP], color='blue', s=100, zorder=5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('mAP@0.5')
    ax1.set_title(f'mAP@0.5: {current_mAP:.4f}')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, total_epochs)

    # 2. Obj Loss
    ax2 = axes[0, 1]
    ax2.plot(data['epoch'], data['obj'], 'r-', linewidth=2)
    ax2.scatter([current_epoch], [data['obj'][-1]], color='red', s=100, zorder=5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Obj Loss')
    ax2.set_title(f'Obj Loss: {data["obj"][-1]:.4f}')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, total_epochs)

    # 3. Box Loss
    ax3 = axes[1, 0]
    ax3.plot(data['epoch'], data['box'], 'g-', linewidth=2)
    ax3.scatter([current_epoch], [data['box'][-1]], color='green', s=100, zorder=5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Box Loss')
    ax3.set_title(f'Box Loss: {data["box"][-1]:.4f}')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, total_epochs)

    # 4. Progress Bar
    ax4 = axes[1, 1]
    ax4.axis('off')

    progress = current_epoch / total_epochs

    # Progress bar
    ax4.barh([0], [progress], color='blue', height=0.3)
    ax4.barh([0], [1-progress], left=[progress], color='lightgray', height=0.3)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(-1, 1)

    # Stats text
    stats_text = f"""
    Progress: {current_epoch}/{total_epochs} ({progress*100:.1f}%)

    Current mAP@0.5: {current_mAP:.4f}
    Best mAP@0.5: {best_mAP:.4f} (epoch {best_epoch})

    Stage1 Final: 0.4268
    Strategy A Final: 0.4303

    Gap to Stage1: {current_mAP - 0.4268:+.4f}
    Gap to Str A:  {current_mAP - 0.4303:+.4f}
    """
    ax4.text(0.5, 0, stats_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='center', horizontalalignment='center',
             family='monospace')

    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_IMAGE, dpi=120, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    return True

def git_push():
    """Push to GitHub"""
    os.chdir('/workspace/Yolov7fast')

    cmds = [
        ['git', 'add', 'monitor/training_status.png'],
        ['git', 'commit', '-m', f'Update training status {datetime.now().strftime("%H:%M")}'],
        ['git', 'push']
    ]

    for cmd in cmds:
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            if 'nothing to commit' in str(e.stderr):
                continue
            print(f"Git error: {e}")

def main():
    print(f"Training Monitor Started")
    print(f"Output: {OUTPUT_IMAGE}")
    print(f"GitHub URL: https://raw.githubusercontent.com/jimmychintw/Yolov7fast/main/monitor/training_status.png")
    print(f"Update interval: {UPDATE_INTERVAL}s")
    print("-" * 50)

    while True:
        try:
            data = parse_results(RESULTS_FILE)

            if len(data['epoch']) > 0:
                if generate_chart(data):
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Epoch {data['epoch'][-1]}, mAP: {data['mAP50'][-1]:.4f}")
                    git_push()
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for data...")

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(UPDATE_INTERVAL)

if __name__ == '__main__':
    main()
