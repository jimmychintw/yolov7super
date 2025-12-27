#!/usr/bin/env python3
"""
YOLOv7 Trainable Parameters Inspection Tool

用途：Stage1/Stage2 驗收工具
- 印出每個 model.{i} 的 trainable 參數量、總參數量
- 印出 BN 模組狀態（train/eval）
- 驗證凍結設定是否正確

Usage:
    python tools/print_trainable.py --weights /path/to/weights.pt --device 0
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from models.yolo import Model
from utils.torch_utils import select_device


def get_top_level_index(name: str) -> int:
    """
    從參數名稱提取 top-level index
    支援 'model.{i}.' 和 'model.model.{i}.' 兩種格式
    """
    parts = name.split('.')

    # model.model.{i}.xxx 格式
    if len(parts) >= 3 and parts[0] == 'model' and parts[1] == 'model':
        try:
            return int(parts[2])
        except ValueError:
            return -1

    # model.{i}.xxx 格式
    if len(parts) >= 2 and parts[0] == 'model':
        try:
            return int(parts[1])
        except ValueError:
            return -1

    return -1


def analyze_trainable_params(model: nn.Module) -> dict:
    """
    分析每個 top-level module 的參數狀態

    Returns:
        dict: {index: {'trainable': int, 'frozen': int, 'total': int}}
    """
    stats = {}

    for name, param in model.named_parameters():
        idx = get_top_level_index(name)
        if idx < 0:
            continue

        if idx not in stats:
            stats[idx] = {'trainable': 0, 'frozen': 0, 'total': 0, 'trainable_count': 0, 'frozen_count': 0}

        numel = param.numel()
        stats[idx]['total'] += numel

        if param.requires_grad:
            stats[idx]['trainable'] += numel
            stats[idx]['trainable_count'] += 1
        else:
            stats[idx]['frozen'] += numel
            stats[idx]['frozen_count'] += 1

    return stats


def analyze_bn_status(model: nn.Module) -> dict:
    """
    分析 BatchNorm 模組的 train/eval 狀態

    Returns:
        dict: {index: {'train': int, 'eval': int, 'affine_trainable': int, 'affine_frozen': int}}
    """
    stats = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue

        idx = get_top_level_index(name)
        if idx < 0:
            continue

        if idx not in stats:
            stats[idx] = {'train': 0, 'eval': 0, 'affine_trainable': 0, 'affine_frozen': 0}

        # Check training mode
        if module.training:
            stats[idx]['train'] += 1
        else:
            stats[idx]['eval'] += 1

        # Check affine parameters
        if module.weight is not None:
            if module.weight.requires_grad:
                stats[idx]['affine_trainable'] += 1
            else:
                stats[idx]['affine_frozen'] += 1

    return stats


def print_trainable_summary(model: nn.Module, verbose: bool = False):
    """
    印出 trainable 參數摘要
    """
    param_stats = analyze_trainable_params(model)
    bn_stats = analyze_bn_status(model)

    print("\n" + "=" * 100)
    print("YOLOv7-Tiny Trainable Parameters Summary")
    print("=" * 100)

    # 定義區域
    regions = {
        'BACKBONE_EARLY': range(0, 23),
        'BACKBONE_LATE': range(23, 38),
        'NECK': range(38, 74),
        'HEAD': range(74, 100),  # 包含 Detect
    }

    # 統計各區域
    region_totals = {r: {'trainable': 0, 'frozen': 0, 'total': 0} for r in regions}

    if verbose:
        print("\n{:>6} {:>12} {:>12} {:>12} {:>10} {:>10}".format(
            "Index", "Trainable", "Frozen", "Total", "BN train", "BN eval"))
        print("-" * 80)

    for idx in sorted(param_stats.keys()):
        stats = param_stats[idx]
        bn = bn_stats.get(idx, {'train': 0, 'eval': 0})

        if verbose:
            print("{:>6} {:>12,} {:>12,} {:>12,} {:>10} {:>10}".format(
                idx, stats['trainable'], stats['frozen'], stats['total'],
                bn['train'], bn['eval']))

        # 累加到區域
        for region_name, idx_range in regions.items():
            if idx in idx_range:
                region_totals[region_name]['trainable'] += stats['trainable']
                region_totals[region_name]['frozen'] += stats['frozen']
                region_totals[region_name]['total'] += stats['total']
                break

    # 印出區域摘要
    print("\n" + "-" * 80)
    print("Region Summary:")
    print("-" * 80)
    print("{:<20} {:>12} {:>12} {:>12} {:>10}".format(
        "Region", "Trainable", "Frozen", "Total", "Status"))
    print("-" * 80)

    total_trainable = 0
    total_frozen = 0
    total_params = 0

    for region_name in ['BACKBONE_EARLY', 'BACKBONE_LATE', 'NECK', 'HEAD']:
        rt = region_totals[region_name]
        status = "✓ TRAINABLE" if rt['trainable'] > 0 else "✗ FROZEN"
        if rt['trainable'] > 0 and rt['frozen'] > 0:
            status = "⚠ PARTIAL"

        print("{:<20} {:>12,} {:>12,} {:>12,} {:>10}".format(
            region_name, rt['trainable'], rt['frozen'], rt['total'], status))

        total_trainable += rt['trainable']
        total_frozen += rt['frozen']
        total_params += rt['total']

    print("-" * 80)
    print("{:<20} {:>12,} {:>12,} {:>12,}".format(
        "TOTAL", total_trainable, total_frozen, total_params))

    # BN 狀態摘要
    print("\n" + "-" * 80)
    print("BatchNorm Status Summary:")
    print("-" * 80)

    for region_name, idx_range in regions.items():
        train_count = sum(bn_stats.get(i, {}).get('train', 0) for i in idx_range)
        eval_count = sum(bn_stats.get(i, {}).get('eval', 0) for i in idx_range)
        affine_trainable = sum(bn_stats.get(i, {}).get('affine_trainable', 0) for i in idx_range)
        affine_frozen = sum(bn_stats.get(i, {}).get('affine_frozen', 0) for i in idx_range)

        if train_count + eval_count > 0:
            print("{:<20} BN: {:>2} train, {:>2} eval | Affine: {:>2} trainable, {:>2} frozen".format(
                region_name, train_count, eval_count, affine_trainable, affine_frozen))

    print("=" * 100)

    return param_stats, bn_stats


def main():
    parser = argparse.ArgumentParser(description='Print trainable parameters summary')
    parser.add_argument('--weights', type=str, required=True, help='Path to weights file')
    parser.add_argument('--cfg', type=str, default='', help='Model config (optional, will use from checkpoint)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed per-layer info')
    parser.add_argument('--nc', type=int, default=80, help='Number of classes')
    args = parser.parse_args()

    device = select_device(args.device)

    # 載入模型
    print(f"\nLoading weights from: {args.weights}")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)

    if 'model' in ckpt:
        # Checkpoint format
        model = ckpt['model'].float()
        print(f"Loaded from checkpoint (epoch {ckpt.get('epoch', 'N/A')})")
    else:
        # Direct model
        model = ckpt.float()

    model.to(device)
    model.eval()  # 預設 eval 模式以正確顯示 BN 狀態

    # 分析並印出
    print_trainable_summary(model, verbose=args.verbose)


if __name__ == '__main__':
    main()
