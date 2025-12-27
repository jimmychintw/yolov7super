#!/usr/bin/env python3
"""
YOLOv7-Tiny Model Structure Inspector
=====================================
用於盤點模型結構，輸出兩階段解凍（head/neck/backbone）所需資訊。

輸出：
- detect_info.json: Detect 層定位資訊
- modules_full.txt: 完整模組樹
- modules_window.csv: Detect 前後 window
- param_prefix_stats.csv: 參數前綴統計
- bn_modules.csv: BatchNorm 模組列表

Usage:
    python tools/inspect_tiny_model.py \
        --cfg cfg/training/yolov7-tiny.yaml \
        --weights runs/train/.../best.pt \
        --device 0 --img 320
"""

import argparse
import json
import csv
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# 加入專案根目錄到 path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn


def get_git_info():
    """取得 git commit 資訊"""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        return commit
    except:
        return "N/A"


def count_parameters(module):
    """計算模組的參數量"""
    return sum(p.numel() for p in module.parameters())


def get_detect_classes():
    """動態 import Detect 相關類別"""
    from models.yolo import Detect, IDetect
    detect_classes = [Detect, IDetect]

    # 嘗試 import MultiHeadDetect
    try:
        from models.multihead import MultiHeadDetect
        detect_classes.append(MultiHeadDetect)
    except ImportError:
        pass

    return tuple(detect_classes)


def find_detect_info(model):
    """
    找出 Detect 層的完整資訊

    Returns:
        dict: {
            'name': 完整 module name,
            'type': class 名稱,
            'index': 若為 list-like 結構的 index,
            'module': module 物件參考
        }
    """
    detect_classes = get_detect_classes()
    detect_info = None

    for name, module in model.named_modules():
        if isinstance(module, detect_classes):
            # 嘗試解析 index
            index = None
            parts = name.split('.')
            for part in parts:
                if part.isdigit():
                    index = int(part)

            detect_info = {
                'name': name,
                'type': type(module).__name__,
                'index': index,
                'num_params': count_parameters(module),
                'module': module
            }
            break

    return detect_info


def get_all_modules_info(model):
    """
    取得所有 named_modules 資訊

    Returns:
        list of dict
    """
    modules_info = []
    for name, module in model.named_modules():
        if name == '':
            name = '(root)'

        info = {
            'name': name,
            'type': type(module).__name__,
            'num_params': count_parameters(module),
            'is_leaf': len(list(module.children())) == 0
        }
        modules_info.append(info)

    return modules_info


def get_top_level_modules(model):
    """取得 top-level module 名稱列表"""
    top_level = []

    # 檢查 model.model 結構
    if hasattr(model, 'model') and isinstance(model.model, nn.Sequential):
        for i, m in enumerate(model.model):
            top_level.append({
                'index': i,
                'name': f'model.{i}',
                'type': type(m).__name__,
                'num_params': count_parameters(m)
            })
    elif isinstance(model, nn.Sequential):
        for i, m in enumerate(model):
            top_level.append({
                'index': i,
                'name': f'{i}',
                'type': type(m).__name__,
                'num_params': count_parameters(m)
            })
    else:
        # 直接列舉 named_children
        for name, m in model.named_children():
            top_level.append({
                'index': name,
                'name': name,
                'type': type(m).__name__,
                'num_params': count_parameters(m)
            })

    return top_level


def get_modules_window(modules_info, detect_name, before=30, after=5):
    """
    取得 Detect 前後的 modules window
    """
    # 找到 detect 的 index
    detect_idx = None
    for i, m in enumerate(modules_info):
        if m['name'] == detect_name:
            detect_idx = i
            break

    if detect_idx is None:
        return [], []

    # 取得前後 window
    start_idx = max(0, detect_idx - before)
    end_idx = min(len(modules_info), detect_idx + after + 1)

    before_window = modules_info[start_idx:detect_idx]
    after_window = modules_info[detect_idx + 1:end_idx]

    return before_window, after_window


def get_param_prefix_stats(model):
    """
    取得參數前綴統計（到第二層）
    """
    prefix_stats = defaultdict(lambda: {'count': 0, 'numel': 0})

    for name, param in model.named_parameters():
        # 取得前兩層 prefix
        parts = name.split('.')
        if len(parts) >= 2:
            prefix = '.'.join(parts[:2]) + '.*'
        else:
            prefix = parts[0] + '.*'

        prefix_stats[prefix]['count'] += 1
        prefix_stats[prefix]['numel'] += param.numel()

    # 排序
    sorted_stats = sorted(prefix_stats.items(), key=lambda x: x[0])
    return sorted_stats


def get_bn_modules(model):
    """
    取得所有 BatchNorm 模組
    """
    bn_modules = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # 嘗試推斷 top-level index
            parts = name.split('.')
            top_level_idx = None
            for part in parts:
                if part.isdigit():
                    top_level_idx = int(part)
                    break

            bn_modules.append({
                'name': name,
                'type': type(module).__name__,
                'top_level_idx': top_level_idx,
                'num_features': module.num_features if hasattr(module, 'num_features') else None
            })

    return bn_modules


def load_model(cfg, weights, device, img_size):
    """載入模型"""
    from models.yolo import Model

    device = torch.device(device if device != 'cpu' and torch.cuda.is_available() else 'cpu')

    # 載入權重檔取得設定
    ckpt = None
    if weights and Path(weights).exists():
        print(f"Loading checkpoint: {weights}")
        ckpt = torch.load(weights, map_location=device, weights_only=False)

        # 優先用 cfg 參數，否則從 checkpoint 取得
        if cfg is None and 'model' in ckpt:
            if hasattr(ckpt['model'], 'yaml'):
                cfg_from_ckpt = ckpt['model'].yaml
                print(f"Using cfg from checkpoint")
            else:
                raise ValueError("No cfg provided and checkpoint doesn't contain yaml config")

    if cfg is None:
        raise ValueError("Must provide --cfg or valid --weights with embedded config")

    # 建立模型
    print(f"Creating model from cfg: {cfg}")
    model = Model(cfg, ch=3, nc=80).to(device)

    # 載入權重（如果有）
    if ckpt is not None:
        if 'model' in ckpt:
            state_dict = ckpt['model'].float().state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
            if isinstance(state_dict, dict):
                # 過濾不匹配的層
                model_state = model.state_dict()
                filtered_state = {k: v for k, v in state_dict.items()
                                  if k in model_state and v.shape == model_state[k].shape}
                model.load_state_dict(filtered_state, strict=False)
                print(f"Loaded {len(filtered_state)}/{len(model_state)} parameters from checkpoint")
        elif 'ema' in ckpt:
            state_dict = ckpt['ema'].float().state_dict()
            model_state = model.state_dict()
            filtered_state = {k: v for k, v in state_dict.items()
                              if k in model_state and v.shape == model_state[k].shape}
            model.load_state_dict(filtered_state, strict=False)
            print(f"Loaded {len(filtered_state)}/{len(model_state)} parameters from EMA")

    model.eval()
    return model, device


def main():
    parser = argparse.ArgumentParser(description='YOLOv7-Tiny Model Structure Inspector')
    parser.add_argument('--cfg', '--model-cfg', type=str, default=None,
                        help='Model config yaml file')
    parser.add_argument('--weights', type=str, default=None,
                        help='Pretrained weights file (.pt)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu, 0, 1, ...)')
    parser.add_argument('--img', type=int, default=320,
                        help='Image size')
    parser.add_argument('--out', type=str, default='out/model_inspect',
                        help='Output directory')

    args = parser.parse_args()

    # 建立輸出目錄
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # 環境資訊
    print("=" * 70)
    print("YOLOv7-Tiny Model Structure Inspector")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Git commit: {get_git_info()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    print(f"Config file: {args.cfg}")
    print(f"Weights file: {args.weights}")
    print(f"Device: {args.device}")
    print(f"Image size: {args.img}")
    print(f"Output directory: {out_dir}")
    print("=" * 70)

    # 載入模型
    model, device = load_model(args.cfg, args.weights, args.device, args.img)

    print(f"\nModel loaded successfully")
    print(f"Total parameters: {count_parameters(model):,}")

    # =========================================================================
    # (A) Detect 層定位
    # =========================================================================
    print("\n" + "=" * 70)
    print("(A) Detect Layer Information")
    print("=" * 70)

    detect_info = find_detect_info(model)

    if detect_info:
        print(f"Detect module name: {detect_info['name']}")
        print(f"Detect module type: {detect_info['type']}")
        print(f"Detect module index: {detect_info['index']}")
        print(f"Detect num params: {detect_info['num_params']:,}")

        # 輸出 JSON（不含 module 物件）
        detect_json = {k: v for k, v in detect_info.items() if k != 'module'}
        detect_json_path = out_dir / 'detect_info.json'
        with open(detect_json_path, 'w') as f:
            json.dump(detect_json, f, indent=2)
        print(f"\nSaved to: {detect_json_path}")
    else:
        print("ERROR: Detect module not found!")
        detect_json = {'error': 'Detect module not found'}
        with open(out_dir / 'detect_info.json', 'w') as f:
            json.dump(detect_json, f, indent=2)

    # =========================================================================
    # (B) 模組樹與鄰近結構
    # =========================================================================
    print("\n" + "=" * 70)
    print("(B) Module Tree and Neighborhood")
    print("=" * 70)

    # 取得所有模組
    modules_info = get_all_modules_info(model)
    print(f"Total modules: {len(modules_info)}")

    # 輸出完整模組樹
    modules_full_path = out_dir / 'modules_full.txt'
    with open(modules_full_path, 'w') as f:
        f.write("# YOLOv7-Tiny Module Tree\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total modules: {len(modules_info)}\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Index':<6} {'Name':<50} {'Type':<25} {'Params':>15}\n")
        f.write("=" * 100 + "\n")
        for i, m in enumerate(modules_info):
            f.write(f"{i:<6} {m['name']:<50} {m['type']:<25} {m['num_params']:>15,}\n")
    print(f"Saved modules_full.txt: {modules_full_path}")

    # 取得 top-level modules
    top_level = get_top_level_modules(model)
    print(f"\nTop-level modules: {len(top_level)}")
    for m in top_level[:10]:
        print(f"  {m['name']}: {m['type']} ({m['num_params']:,} params)")
    if len(top_level) > 10:
        print(f"  ... and {len(top_level) - 10} more")

    # 輸出 Detect 前後 window
    if detect_info:
        before_window, after_window = get_modules_window(
            modules_info, detect_info['name'], before=30, after=5
        )

        modules_window_path = out_dir / 'modules_window.csv'
        with open(modules_window_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['position', 'index', 'name', 'type', 'num_params'])

            for i, m in enumerate(before_window):
                writer.writerow([f'before_{len(before_window)-i}', i, m['name'], m['type'], m['num_params']])

            # 寫入 Detect 本身
            detect_in_list = next((m for m in modules_info if m['name'] == detect_info['name']), None)
            if detect_in_list:
                writer.writerow(['DETECT', '-', detect_in_list['name'], detect_in_list['type'], detect_in_list['num_params']])

            for i, m in enumerate(after_window):
                writer.writerow([f'after_{i+1}', i, m['name'], m['type'], m['num_params']])

        print(f"\nSaved modules_window.csv: {modules_window_path}")
        print(f"  Before Detect: {len(before_window)} modules")
        print(f"  After Detect: {len(after_window)} modules")

    # =========================================================================
    # (C) 參數命名與分組線索
    # =========================================================================
    print("\n" + "=" * 70)
    print("(C) Parameter Prefix Statistics")
    print("=" * 70)

    prefix_stats = get_param_prefix_stats(model)

    # 輸出 CSV
    param_prefix_path = out_dir / 'param_prefix_stats.csv'
    with open(param_prefix_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prefix', 'count', 'numel'])
        for prefix, stats in prefix_stats:
            writer.writerow([prefix, stats['count'], stats['numel']])

    print(f"Saved param_prefix_stats.csv: {param_prefix_path}")
    print(f"\nTop 15 parameter prefixes:")
    for prefix, stats in prefix_stats[:15]:
        print(f"  {prefix:<30} count={stats['count']:<4} numel={stats['numel']:>12,}")
    if len(prefix_stats) > 15:
        print(f"  ... and {len(prefix_stats) - 15} more")

    # BatchNorm 模組
    bn_modules = get_bn_modules(model)
    bn_modules_path = out_dir / 'bn_modules.csv'
    with open(bn_modules_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'type', 'top_level_idx', 'num_features'])
        for m in bn_modules:
            writer.writerow([m['name'], m['type'], m['top_level_idx'], m['num_features']])

    print(f"\nSaved bn_modules.csv: {bn_modules_path}")
    print(f"Total BatchNorm modules: {len(bn_modules)}")

    # =========================================================================
    # (D) 摘要
    # =========================================================================
    print("\n" + "=" * 70)
    print("(D) Summary")
    print("=" * 70)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'git_commit': get_git_info(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cfg': str(args.cfg),
        'weights': str(args.weights),
        'device': str(device),
        'img_size': args.img,
        'total_params': count_parameters(model),
        'total_modules': len(modules_info),
        'top_level_modules': len(top_level),
        'bn_modules': len(bn_modules),
        'detect_info': detect_json if detect_info else None
    }

    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary.json: {summary_path}")

    print("\n" + "=" * 70)
    print("DONE! All outputs saved to:", out_dir)
    print("=" * 70)

    # 輸出關鍵資訊供使用者確認
    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR TWO-STAGE UNFREEZE")
    print("=" * 70)

    if detect_info:
        print(f"\n1. Detect Layer:")
        print(f"   Name: {detect_info['name']}")
        print(f"   Type: {detect_info['type']}")
        print(f"   Index: {detect_info['index']}")

        # 找出 Neck 和 Backbone 的分界
        # 通常 backbone 結束於某個 SPPCSPC 或類似層
        print(f"\n2. Suggested Layer Groups (based on index {detect_info['index']}):")

        if detect_info['index'] is not None:
            detect_idx = detect_info['index']
            print(f"   - Head (Detect): model.{detect_idx}")
            print(f"   - Neck (FPN/PAN): model.{detect_idx//2} ~ model.{detect_idx-1} (估計)")
            print(f"   - Backbone: model.0 ~ model.{detect_idx//2-1} (估計)")
            print("\n   NOTE: 請參考 modules_full.txt 確認實際邊界")

    return 0


if __name__ == '__main__':
    sys.exit(main())
