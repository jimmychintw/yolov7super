"""
YOLOv7 Stage-aware Optimizer Parameter Groups Builder

用途：
- 為 Stage1/Stage2 微調建立分組的 optimizer param groups
- 支援不同 scope 使用不同的 learning rate

Stage1 分組：
- HEAD: lr = lr0 * mult_head (default 1.0)
- NECK: lr = lr0 * mult_neck (default 0.3)

Stage2 分組：
- HEAD: lr = lr0 * mult_head (default 1.0)
- NECK: lr = lr0 * mult_neck (default 0.3)
- LATE_BACKBONE: lr = lr0 * mult_backbone (default 0.05)
"""

import logging
import torch.nn as nn
from typing import Dict, List, Set, Tuple
from utils.freeze_by_index import get_top_level_index, YOLOV7_TINY_SCOPES

logger = logging.getLogger(__name__)


def build_stage_param_groups(
    model: nn.Module,
    lr0: float,
    weight_decay: float,
    mult_head: float = 1.0,
    mult_neck: float = 0.3,
    mult_backbone: float = 0.05,
    include_late_backbone: bool = False
) -> List[Dict]:
    """
    為 Stage 微調建立分組的 optimizer param groups

    Args:
        model: PyTorch 模型
        lr0: 基礎 learning rate
        weight_decay: weight decay 值
        mult_head: HEAD 區域的 LR 倍率
        mult_neck: NECK 區域的 LR 倍率
        mult_backbone: LATE_BACKBONE 區域的 LR 倍率 (Stage2 用)
        include_late_backbone: 是否包含 LATE_BACKBONE (Stage2=True, Stage1=False)

    Returns:
        List[Dict]: optimizer param_groups 列表
    """
    # 定義 scope 範圍
    head_indices = YOLOV7_TINY_SCOPES['HEAD']
    neck_indices = YOLOV7_TINY_SCOPES['NECK']
    late_indices = YOLOV7_TINY_SCOPES['BACKBONE_LATE']

    # 收集各 scope 的參數
    head_params = []
    neck_params = []
    late_params = []

    head_numel = 0
    neck_numel = 0
    late_numel = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 跳過 frozen 參數

        idx = get_top_level_index(name)

        if idx in head_indices:
            head_params.append(param)
            head_numel += param.numel()
        elif idx in neck_indices:
            neck_params.append(param)
            neck_numel += param.numel()
        elif idx in late_indices and include_late_backbone:
            late_params.append(param)
            late_numel += param.numel()

    # 建立 param groups
    param_groups = []

    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': lr0 * mult_head,
            'weight_decay': weight_decay,
            'name': 'head'
        })

    if neck_params:
        param_groups.append({
            'params': neck_params,
            'lr': lr0 * mult_neck,
            'weight_decay': weight_decay,
            'name': 'neck'
        })

    if late_params:
        param_groups.append({
            'params': late_params,
            'lr': lr0 * mult_backbone,
            'weight_decay': weight_decay,
            'name': 'late_backbone'
        })

    # 印出統計
    logger.info("=" * 60)
    logger.info("Stage Optimizer Parameter Groups Summary")
    logger.info("=" * 60)
    logger.info(f"Base LR: {lr0}, Weight Decay: {weight_decay}")
    logger.info("-" * 60)
    logger.info(f"{'Group':<15} {'Params':<10} {'Elements':<15} {'LR':<12} {'Multiplier':<10}")
    logger.info("-" * 60)

    if head_params:
        logger.info(f"{'HEAD':<15} {len(head_params):<10} {head_numel:<15,} {lr0 * mult_head:<12.6f} {mult_head:<10}")
    if neck_params:
        logger.info(f"{'NECK':<15} {len(neck_params):<10} {neck_numel:<15,} {lr0 * mult_neck:<12.6f} {mult_neck:<10}")
    if late_params:
        logger.info(f"{'LATE_BACKBONE':<15} {len(late_params):<10} {late_numel:<15,} {lr0 * mult_backbone:<12.6f} {mult_backbone:<10}")

    total_params = len(head_params) + len(neck_params) + len(late_params)
    total_numel = head_numel + neck_numel + late_numel
    logger.info("-" * 60)
    logger.info(f"{'TOTAL':<15} {total_params:<10} {total_numel:<15,}")
    logger.info("=" * 60)

    return param_groups


def build_stage1_param_groups(
    model: nn.Module,
    lr0: float,
    weight_decay: float,
    mult_head: float = 1.0,
    mult_neck: float = 0.3
) -> List[Dict]:
    """
    Stage1 專用：只包含 HEAD + NECK

    Args:
        model: 模型
        lr0: 基礎 LR
        weight_decay: weight decay
        mult_head: HEAD LR 倍率
        mult_neck: NECK LR 倍率

    Returns:
        optimizer param_groups
    """
    logger.info("[Stage1] Building optimizer groups for HEAD + NECK")
    return build_stage_param_groups(
        model=model,
        lr0=lr0,
        weight_decay=weight_decay,
        mult_head=mult_head,
        mult_neck=mult_neck,
        include_late_backbone=False
    )


def build_stage2_param_groups(
    model: nn.Module,
    lr0: float,
    weight_decay: float,
    mult_head: float = 1.0,
    mult_neck: float = 0.3,
    mult_backbone: float = 0.05
) -> List[Dict]:
    """
    Stage2 專用：包含 HEAD + NECK + LATE_BACKBONE

    Args:
        model: 模型
        lr0: 基礎 LR
        weight_decay: weight decay
        mult_head: HEAD LR 倍率
        mult_neck: NECK LR 倍率
        mult_backbone: LATE_BACKBONE LR 倍率

    Returns:
        optimizer param_groups
    """
    logger.info("[Stage2] Building optimizer groups for HEAD + NECK + LATE_BACKBONE")
    return build_stage_param_groups(
        model=model,
        lr0=lr0,
        weight_decay=weight_decay,
        mult_head=mult_head,
        mult_neck=mult_neck,
        mult_backbone=mult_backbone,
        include_late_backbone=True
    )


def get_warmup_scale(epoch_in_stage: int, warmup_epochs: int) -> float:
    """
    計算 warmup 縮放係數

    用於 stage 續跑時重新 warmup，避免 LR 太大造成崩潰

    Args:
        epoch_in_stage: 當前 stage 內的 epoch 數 (從 0 開始)
        warmup_epochs: warmup 總 epoch 數

    Returns:
        float: 縮放係數 (0.0 ~ 1.0)
    """
    if warmup_epochs <= 0:
        return 1.0

    if epoch_in_stage >= warmup_epochs:
        return 1.0

    # 線性 warmup: 0 -> 1
    return (epoch_in_stage + 1) / warmup_epochs


def apply_warmup_scale_to_optimizer(optimizer, scale: float):
    """
    將 warmup 縮放係數套用到 optimizer 的所有 param groups

    Args:
        optimizer: PyTorch optimizer
        scale: 縮放係數
    """
    for pg in optimizer.param_groups:
        if 'initial_lr' in pg:
            pg['lr'] = pg['initial_lr'] * scale
        else:
            # 首次呼叫時記錄 initial_lr
            pg['initial_lr'] = pg['lr']
            pg['lr'] = pg['initial_lr'] * scale


def print_stage_summary(
    stage_name: str,
    trainable_scopes: List[str],
    frozen_scopes: List[str],
    param_groups: List[Dict],
    warmup_epochs: int
):
    """
    印出 Stage 訓練摘要

    Args:
        stage_name: Stage 名稱
        trainable_scopes: 可訓練的 scope 名稱列表
        frozen_scopes: 凍結的 scope 名稱列表
        param_groups: optimizer param groups
        warmup_epochs: warmup epoch 數
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"STAGE TRAINING SUMMARY: {stage_name}")
    logger.info("=" * 70)

    logger.info(f"\nTrainable Scopes: {', '.join(trainable_scopes)}")
    logger.info(f"Frozen Scopes: {', '.join(frozen_scopes)}")

    # Scope index 範圍
    logger.info("\nScope Index Ranges:")
    for scope in trainable_scopes:
        if scope.upper() in YOLOV7_TINY_SCOPES:
            indices = sorted(YOLOV7_TINY_SCOPES[scope.upper()])
            logger.info(f"  {scope}: model.{min(indices)} ~ model.{max(indices)}")

    for scope in frozen_scopes:
        if scope.upper() in YOLOV7_TINY_SCOPES:
            indices = sorted(YOLOV7_TINY_SCOPES[scope.upper()])
            logger.info(f"  {scope}: model.{min(indices)} ~ model.{max(indices)} [FROZEN]")

    # Optimizer groups
    logger.info("\nOptimizer Parameter Groups:")
    total_params = 0
    for pg in param_groups:
        name = pg.get('name', 'unnamed')
        lr = pg.get('lr', 0)
        n_params = sum(p.numel() for p in pg['params'])
        total_params += n_params
        logger.info(f"  {name}: {n_params:,} params, lr={lr:.6f}")

    logger.info(f"\nTotal Trainable Parameters: {total_params:,}")
    logger.info(f"Warmup Epochs: {warmup_epochs}")
    logger.info("=" * 70 + "\n")
