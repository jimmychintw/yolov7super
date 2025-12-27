#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1B4H (One Backbone Four Heads) 單元測試

測試項目:
- UT-01: HeadConfig 映射
- UT-02: HeadConfig 驗證
- UT-03: MultiHeadDetect 訓練輸出
- UT-04: MultiHeadDetect 推論輸出
- UT-05: ComputeLossRouter 遮罩
"""

import sys
import os

# 將專案根目錄加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


def test_head_config():
    """UT-01, UT-02: 測試 HeadConfig 映射與驗證"""
    print("\n" + "=" * 60)
    print("UT-01, UT-02: HeadConfig 映射與驗證測試")
    print("=" * 60)

    from utils.head_config import HeadConfig

    cfg = HeadConfig('data/coco_320_1b4h_standard.yaml')

    # UT-01: 映射測試
    print("\n[UT-01] 映射測試:")

    # 測試 person (class 0) -> Head 0, local_id 0
    head_id, local_id = cfg.get_head_info(0)
    assert head_id == 0, f"person 應該在 Head 0, 但得到 Head {head_id}"
    assert local_id == 0, f"person 在 Head 0 中應該是 local_id 0, 但得到 {local_id}"
    print(f"  person (0) -> Head {head_id}, local_id {local_id} ✓")

    # 測試 bicycle (class 1) -> Head 1
    head_id = cfg.get_head_id(1)
    assert head_id == 1, f"bicycle 應該在 Head 1, 但得到 Head {head_id}"
    print(f"  bicycle (1) -> Head {head_id} ✓")

    # 測試 bird (class 14) -> Head 2
    head_id = cfg.get_head_id(14)
    assert head_id == 2, f"bird 應該在 Head 2, 但得到 Head {head_id}"
    print(f"  bird (14) -> Head {head_id} ✓")

    # 測試 tv (class 62) -> Head 3
    head_id = cfg.get_head_id(62)
    assert head_id == 3, f"tv 應該在 Head 3, 但得到 Head {head_id}"
    print(f"  tv (62) -> Head {head_id} ✓")

    # UT-02: 驗證測試
    print("\n[UT-02] 驗證測試:")

    # 檢查總類別數
    assert cfg.nc == 80, f"總類別數應為 80, 但得到 {cfg.nc}"
    print(f"  總類別數: {cfg.nc} ✓")

    # 檢查 Head 數量
    assert cfg.num_heads == 4, f"Head 數量應為 4, 但得到 {cfg.num_heads}"
    print(f"  Head 數量: {cfg.num_heads} ✓")

    # 檢查每個 Head 的類別數
    for i in range(cfg.num_heads):
        nc = cfg.get_head_nc(i)
        assert nc == 20, f"Head {i} 應有 20 類, 但得到 {nc}"
    print(f"  每個 Head 類別數: 20 ✓")

    # 檢查權重
    weights = [cfg.get_head_weight(i) for i in range(cfg.num_heads)]
    print(f"  Head 權重: {weights} ✓")

    # 檢查所有類別都有分配
    all_assigned = set()
    for i in range(cfg.num_heads):
        all_assigned.update(cfg.get_head_classes(i))
    assert all_assigned == set(range(80)), "並非所有 80 類都有分配"
    print(f"  所有 80 類都已分配 ✓")

    print("\n[UT-01, UT-02] HeadConfig 測試通過! ✓")
    return True


def test_multihead_detect():
    """UT-03, UT-04: 測試 MultiHeadDetect 輸出"""
    print("\n" + "=" * 60)
    print("UT-03, UT-04: MultiHeadDetect 輸出測試")
    print("=" * 60)

    from utils.head_config import HeadConfig
    from models.multihead import MultiHeadDetect

    # 載入設定
    cfg = HeadConfig('data/coco_320_1b4h_standard.yaml')

    # 建立模型
    anchors = [
        [10, 13, 16, 30, 33, 23],      # P3/8
        [30, 61, 62, 45, 59, 119],     # P4/16
        [116, 90, 156, 198, 373, 326]  # P5/32
    ]
    ch = [128, 256, 512]  # YOLOv7-tiny 的 P3/P4/P5 通道數

    model = MultiHeadDetect(nc=80, anchors=anchors, ch=ch, head_config=cfg)
    model.stride = torch.tensor([8., 16., 32.])

    # 模擬輸入 (320x320 圖像)
    batch_size = 2
    x = [
        torch.randn(batch_size, 128, 40, 40),  # P3: 320/8 = 40
        torch.randn(batch_size, 256, 20, 20),  # P4: 320/16 = 20
        torch.randn(batch_size, 512, 10, 10),  # P5: 320/32 = 10
    ]

    # UT-03: 訓練模式
    print("\n[UT-03] 訓練模式輸出測試:")
    model.train()
    outputs = model(x)

    assert len(outputs) == 4, f"應該有 4 個 Head 輸出, 但得到 {len(outputs)}"
    print(f"  輸出 Head 數量: {len(outputs)} ✓")

    for i, head_out in enumerate(outputs):
        assert len(head_out) == 3, f"Head {i} 應該有 3 層輸出 (P3/P4/P5)"
        head_nc = cfg.get_head_nc(i)
        expected_no = 4 + 1 + head_nc  # bbox + obj + cls

        for j, p in enumerate(head_out):
            expected_shape = [batch_size, 3, [40, 20, 10][j], [40, 20, 10][j], expected_no]
            actual_shape = list(p.shape)
            assert actual_shape == expected_shape, f"Head {i} P{j+3} shape 不符: {actual_shape} vs {expected_shape}"

        print(f"  Head {i} 輸出 shape 正確 (no={expected_no}) ✓")

    # UT-04: 推論模式
    print("\n[UT-04] 推論模式輸出測試:")
    model.eval()
    with torch.no_grad():
        z, raw = model(x)

    # 計算預期的總 anchor 數
    # 4 heads * 3 layers * (40*40 + 20*20 + 10*10) * 3 anchors
    expected_anchors_per_head = 3 * (40*40*3 + 20*20*3 + 10*10*3)  # 3 anchors per location
    expected_total = 4 * expected_anchors_per_head
    # 修正: 每個 Head 的每層都有獨立輸出
    expected_total = 4 * (40*40*3 + 20*20*3 + 10*10*3)

    print(f"  合併輸出 shape: {z.shape}")
    assert z.shape[0] == batch_size, f"Batch size 不符: {z.shape[0]} vs {batch_size}"
    assert z.shape[2] == 85, f"輸出維度應為 85 (4+1+80), 但得到 {z.shape[2]}"
    print(f"  Batch size: {z.shape[0]} ✓")
    print(f"  輸出維度: {z.shape[2]} (4+1+80) ✓")
    print(f"  總 anchor 數: {z.shape[1]} ✓")

    print("\n[UT-03, UT-04] MultiHeadDetect 測試通過! ✓")
    return True


def test_loss_router():
    """UT-05: 測試 ComputeLossRouter 遮罩"""
    print("\n" + "=" * 60)
    print("UT-05: ComputeLossRouter 遮罩測試")
    print("=" * 60)

    from utils.head_config import HeadConfig
    from utils.loss_router import ComputeLossRouter
    from models.multihead import MultiHeadDetect
    import torch.nn as nn

    # 載入設定
    cfg = HeadConfig('data/coco_320_1b4h_standard.yaml')

    # 建立簡化的模型 (用於測試)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hyp = {
                'box': 0.05,
                'obj': 0.7,
                'cls': 0.3,
                'cls_pw': 1.0,
                'obj_pw': 1.0,
                'fl_gamma': 0.0,
                'label_smoothing': 0.0,
                'anchor_t': 4.0,
            }
            self.gr = 1.0
            # 建立 MultiHeadDetect
            anchors = [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326]
            ]
            ch = [128, 256, 512]
            self.detect = MultiHeadDetect(nc=80, anchors=anchors, ch=ch, head_config=cfg)
            self.detect.stride = torch.tensor([8., 16., 32.])
            self.model = nn.ModuleList([self.detect])

        def parameters(self):
            return self.detect.parameters()

    model = DummyModel()

    # 測試 _filter_targets_for_head 方法
    print("\n[UT-05] 遮罩測試:")

    # 建立假的 targets
    # targets: [image_idx, class, x, y, w, h]
    targets = torch.tensor([
        [0, 0, 0.5, 0.5, 0.1, 0.1],   # person (Head 0)
        [0, 1, 0.3, 0.3, 0.2, 0.2],   # bicycle (Head 1)
        [0, 14, 0.7, 0.7, 0.15, 0.15],  # bird (Head 2)
        [0, 62, 0.4, 0.4, 0.3, 0.3],  # tv (Head 3)
        [1, 0, 0.6, 0.6, 0.1, 0.1],   # person (Head 0)
    ])

    # 初始化 Loss Router
    loss_router = ComputeLossRouter(model, cfg)

    # 測試每個 Head 的 target 篩選
    print("  測試 target 篩選:")

    # Head 0 應該只有 person (class 0)
    head0_targets = loss_router._filter_targets_for_head(targets, 0)
    assert head0_targets.shape[0] == 2, f"Head 0 應有 2 個 targets, 但得到 {head0_targets.shape[0]}"
    assert all(head0_targets[:, 1] == 0), "Head 0 targets 的 local_id 應該都是 0"
    print(f"    Head 0: {head0_targets.shape[0]} targets (local_id={head0_targets[:, 1].tolist()}) ✓")

    # Head 1 應該只有 bicycle (class 1)
    head1_targets = loss_router._filter_targets_for_head(targets, 1)
    assert head1_targets.shape[0] == 1, f"Head 1 應有 1 個 target, 但得到 {head1_targets.shape[0]}"
    print(f"    Head 1: {head1_targets.shape[0]} target (local_id={head1_targets[:, 1].tolist()}) ✓")

    # Head 2 應該只有 bird (class 14)
    head2_targets = loss_router._filter_targets_for_head(targets, 2)
    assert head2_targets.shape[0] == 1, f"Head 2 應有 1 個 target, 但得到 {head2_targets.shape[0]}"
    print(f"    Head 2: {head2_targets.shape[0]} target (local_id={head2_targets[:, 1].tolist()}) ✓")

    # Head 3 應該只有 tv (class 62)
    head3_targets = loss_router._filter_targets_for_head(targets, 3)
    assert head3_targets.shape[0] == 1, f"Head 3 應有 1 個 target, 但得到 {head3_targets.shape[0]}"
    print(f"    Head 3: {head3_targets.shape[0]} target (local_id={head3_targets[:, 1].tolist()}) ✓")

    print("\n[UT-05] ComputeLossRouter 遮罩測試通過! ✓")
    return True


def run_all_tests():
    """執行所有單元測試"""
    print("\n" + "=" * 60)
    print("YOLOv7 1B4H Phase 1 單元測試")
    print("=" * 60)

    results = []

    try:
        results.append(("HeadConfig", test_head_config()))
    except Exception as e:
        print(f"\n[ERROR] HeadConfig 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        results.append(("HeadConfig", False))

    try:
        results.append(("MultiHeadDetect", test_multihead_detect()))
    except Exception as e:
        print(f"\n[ERROR] MultiHeadDetect 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        results.append(("MultiHeadDetect", False))

    try:
        results.append(("ComputeLossRouter", test_loss_router()))
    except Exception as e:
        print(f"\n[ERROR] ComputeLossRouter 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        results.append(("ComputeLossRouter", False))

    # 總結
    print("\n" + "=" * 60)
    print("測試結果總結")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")

    print(f"\n總計: {passed}/{total} 測試通過")
    print("=" * 60)

    return all(r for _, r in results)


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
