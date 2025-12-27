"""
YOLOv7 Freeze by Index Utilities

提供最小侵入且可重用的凍結工具：
- set_requires_grad_by_top_indices: 根據 top-level index 設定 requires_grad
- set_bn_eval_by_top_indices: 根據 top-level index 設定 BN 為 eval 模式

支援兩種 prefix 格式：
- model.{i}.xxx
- model.model.{i}.xxx
"""

import logging
import torch.nn as nn
from typing import Set, Optional

logger = logging.getLogger(__name__)


def get_top_level_index(name: str) -> int:
    """
    從 module/parameter 名稱提取 top-level index

    支援格式：
    - 'model.{i}.xxx' -> i
    - 'model.model.{i}.xxx' -> i
    - 其他 -> -1

    Args:
        name: module 或 parameter 的完整名稱

    Returns:
        int: top-level index，無法解析時返回 -1
    """
    parts = name.split('.')

    # model.model.{i}.xxx 格式 (DDP 或某些容器包裝)
    if len(parts) >= 3 and parts[0] == 'model' and parts[1] == 'model':
        try:
            return int(parts[2])
        except ValueError:
            return -1

    # model.{i}.xxx 格式 (標準格式)
    if len(parts) >= 2 and parts[0] == 'model':
        try:
            return int(parts[1])
        except ValueError:
            return -1

    # 直接 {i}.xxx 格式 (Sequential 內部)
    if len(parts) >= 1:
        try:
            return int(parts[0])
        except ValueError:
            return -1

    return -1


def set_requires_grad_by_top_indices(
    model: nn.Module,
    idx_set: Set[int],
    requires_grad: bool,
    verbose: bool = False
) -> int:
    """
    根據 top-level index 設定參數的 requires_grad

    Args:
        model: PyTorch 模型
        idx_set: 要操作的 top-level index 集合
        requires_grad: True=可訓練, False=凍結
        verbose: 是否印出詳細資訊

    Returns:
        int: 被修改的參數數量
    """
    modified_count = 0
    modified_numel = 0

    for name, param in model.named_parameters():
        idx = get_top_level_index(name)
        if idx in idx_set:
            if param.requires_grad != requires_grad:
                param.requires_grad = requires_grad
                modified_count += 1
                modified_numel += param.numel()
                if verbose:
                    status = "TRAINABLE" if requires_grad else "FROZEN"
                    logger.info(f"  [{status}] {name} ({param.numel():,} params)")

    action = "unfroze" if requires_grad else "froze"
    logger.info(f"set_requires_grad_by_top_indices: {action} {modified_count} params "
                f"({modified_numel:,} elements) in indices {sorted(idx_set)}")

    return modified_count


def set_bn_eval_by_top_indices(
    model: nn.Module,
    idx_set: Set[int],
    freeze_affine: bool = True,
    verbose: bool = False
) -> int:
    """
    根據 top-level index 將 BatchNorm 設為 eval 模式

    Args:
        model: PyTorch 模型
        idx_set: 要操作的 top-level index 集合
        freeze_affine: 是否同時凍結 BN 的 weight/bias (gamma/beta)
        verbose: 是否印出詳細資訊

    Returns:
        int: 被修改的 BN 模組數量
    """
    modified_count = 0

    for name, module in model.named_modules():
        if not isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            continue

        idx = get_top_level_index(name)
        if idx in idx_set:
            # 設定 eval 模式 (固定 running_mean/running_var)
            module.eval()

            # 凍結 affine 參數 (gamma/beta)
            if freeze_affine:
                if module.weight is not None:
                    module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False

            modified_count += 1
            if verbose:
                affine_status = "affine=frozen" if freeze_affine else "affine=trainable"
                logger.info(f"  [BN EVAL] {name} ({affine_status})")

    logger.info(f"set_bn_eval_by_top_indices: set {modified_count} BN modules to eval mode "
                f"in indices {sorted(idx_set)}, freeze_affine={freeze_affine}")

    return modified_count


def set_bn_train_by_top_indices(
    model: nn.Module,
    idx_set: Set[int],
    unfreeze_affine: bool = True,
    verbose: bool = False
) -> int:
    """
    根據 top-level index 將 BatchNorm 設為 train 模式

    Args:
        model: PyTorch 模型
        idx_set: 要操作的 top-level index 集合
        unfreeze_affine: 是否同時解凍 BN 的 weight/bias
        verbose: 是否印出詳細資訊

    Returns:
        int: 被修改的 BN 模組數量
    """
    modified_count = 0

    for name, module in model.named_modules():
        if not isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            continue

        idx = get_top_level_index(name)
        if idx in idx_set:
            # 設定 train 模式 (更新 running_mean/running_var)
            module.train()

            # 解凍 affine 參數
            if unfreeze_affine:
                if module.weight is not None:
                    module.weight.requires_grad = True
                if module.bias is not None:
                    module.bias.requires_grad = True

            modified_count += 1
            if verbose:
                affine_status = "affine=trainable" if unfreeze_affine else "affine=frozen"
                logger.info(f"  [BN TRAIN] {name} ({affine_status})")

    logger.info(f"set_bn_train_by_top_indices: set {modified_count} BN modules to train mode "
                f"in indices {sorted(idx_set)}, unfreeze_affine={unfreeze_affine}")

    return modified_count


# ============================================================================
# YOLOv7-Tiny 專用的 scope 定義
# ============================================================================

# YOLOv7-Tiny 結構分組 (基於 model inspection)
YOLOV7_TINY_SCOPES = {
    'HEAD': set(range(74, 78)),           # {74, 75, 76, 77} - 最後 3 conv + IDetect
    'NECK': set(range(38, 74)),           # {38..73} - FPN+PAN
    'BACKBONE_LATE': set(range(23, 38)),  # {23..37} - 晚期骨幹
    'BACKBONE_EARLY': set(range(0, 23)),  # {0..22} - 早期骨幹
}


def get_scope_indices(scope_names: list) -> Set[int]:
    """
    根據 scope 名稱取得對應的 index 集合

    Args:
        scope_names: scope 名稱列表，如 ['HEAD', 'NECK']

    Returns:
        Set[int]: 所有指定 scope 的 index 聯集
    """
    result = set()
    for name in scope_names:
        if name.upper() in YOLOV7_TINY_SCOPES:
            result.update(YOLOV7_TINY_SCOPES[name.upper()])
        else:
            logger.warning(f"Unknown scope name: {name}")
    return result


def apply_stage1_freeze(model: nn.Module, verbose: bool = False):
    """
    Stage1: Neck Tune - 只訓練 NECK+HEAD，Backbone 全凍結

    - Trainable: NECK + HEAD (38-77)
    - Frozen: BACKBONE_EARLY + BACKBONE_LATE (0-37)
    - BN: Backbone 的 BN eval() + 凍結 affine
    """
    logger.info("=" * 60)
    logger.info("Applying Stage1 Freeze: NECK+HEAD trainable, Backbone frozen")
    logger.info("=" * 60)

    # 1. 先將全 model requires_grad=True（避免繼承上次狀態）
    for param in model.parameters():
        param.requires_grad = True
    logger.info("Reset all parameters to requires_grad=True")

    # 2. 凍結 Backbone (0-37)
    backbone_indices = get_scope_indices(['BACKBONE_EARLY', 'BACKBONE_LATE'])
    set_requires_grad_by_top_indices(model, backbone_indices, requires_grad=False, verbose=verbose)

    # 3. 固定 Backbone BN eval + freeze affine
    set_bn_eval_by_top_indices(model, backbone_indices, freeze_affine=True, verbose=verbose)

    logger.info("Stage1 freeze applied successfully")
    logger.info("=" * 60)


def apply_stage2_freeze(model: nn.Module, verbose: bool = False):
    """
    Stage2: Late Backbone Tune - 解凍 LATE_BACKBONE + NECK + HEAD

    - Trainable: HEAD + NECK + BACKBONE_LATE (23-77)
    - Frozen: BACKBONE_EARLY (0-22)
    - BN: EARLY BN eval + 凍結 affine
    """
    logger.info("=" * 60)
    logger.info("Applying Stage2 Freeze: LATE_BACKBONE+NECK+HEAD trainable, EARLY frozen")
    logger.info("=" * 60)

    # 1. 先將全 model requires_grad=True
    for param in model.parameters():
        param.requires_grad = True
    logger.info("Reset all parameters to requires_grad=True")

    # 2. 只凍結 EARLY Backbone (0-22)
    early_indices = get_scope_indices(['BACKBONE_EARLY'])
    set_requires_grad_by_top_indices(model, early_indices, requires_grad=False, verbose=verbose)

    # 3. EARLY BN eval + freeze affine
    set_bn_eval_by_top_indices(model, early_indices, freeze_affine=True, verbose=verbose)

    logger.info("Stage2 freeze applied successfully")
    logger.info("=" * 60)


def apply_stage2_bn_phase(model: nn.Module, epoch_in_stage: int, bn_phase1_epochs: int,
                          bn_unfreeze_late: bool = False, verbose: bool = False):
    """
    Stage2 BN Phase 切換

    - Phase1 (epoch < bn_phase1_epochs): LATE BN 也 eval（求穩）
    - Phase2 (epoch >= bn_phase1_epochs): LATE BN 可選切回 train

    Args:
        model: 模型
        epoch_in_stage: 當前 stage 內的 epoch 數
        bn_phase1_epochs: Phase1 持續的 epoch 數
        bn_unfreeze_late: Phase2 時是否讓 LATE BN 回到 train
        verbose: 詳細輸出
    """
    late_indices = get_scope_indices(['BACKBONE_LATE'])

    if epoch_in_stage < bn_phase1_epochs:
        # Phase1: LATE BN 保持 eval
        set_bn_eval_by_top_indices(model, late_indices, freeze_affine=False, verbose=verbose)
        logger.info(f"Stage2 BN Phase1 (epoch {epoch_in_stage} < {bn_phase1_epochs}): LATE BN = eval")
    else:
        # Phase2: 可選擇讓 LATE BN 回到 train
        if bn_unfreeze_late:
            set_bn_train_by_top_indices(model, late_indices, unfreeze_affine=True, verbose=verbose)
            logger.info(f"Stage2 BN Phase2 (epoch {epoch_in_stage} >= {bn_phase1_epochs}): LATE BN = train")
        else:
            set_bn_eval_by_top_indices(model, late_indices, freeze_affine=False, verbose=verbose)
            logger.info(f"Stage2 BN Phase2 (epoch {epoch_in_stage} >= {bn_phase1_epochs}): LATE BN = eval (conservative)")
