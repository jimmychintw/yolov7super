# models/multihead_attention.py
"""
MultiHeadDetectAttention - 1B4H 多頭檢測層 + CBAM 注意力機制

Phase 3: Task-Aware Attention
在每個 Head 進行卷積之前，先通過 CBAM 注意力模組過濾特徵。
每個 Head 擁有獨立的 CBAM，可學習專屬於該類別群組的注意力權重。

設計原則：
1. 繼承自 MultiHeadDetect，避免重複程式碼
2. 使用 super() 調用父類邏輯
3. 輸出格式與父類一致，確保相容性

參考文件: SDD v1.2, PRD v0.6
"""

import torch
import torch.nn as nn

from models.multihead import MultiHeadDetect
from models.attention import CBAM


class MultiHeadDetectAttention(MultiHeadDetect):
    """
    YOLOv7 1B4H 多頭檢測層 + CBAM 注意力機制

    繼承自 MultiHeadDetect，在特徵進入各 Head 卷積層之前，
    先通過獨立的 CBAM 模組進行注意力加權。

    每個 Head 有獨立的 CBAM 組（對應 P3/P4/P5），
    可學習過濾掉與該 Head 無關的背景雜訊。

    預期效果：
    - Head 0 (人物) 的 CBAM 會抑制車輛、動物等特徵
    - Head 1 (交通工具) 的 CBAM 會抑制人物、食物等特徵
    - 減少跨類別的 False Positive
    """

    def __init__(self, nc=80, anchors=(), ch=(), head_config=None, full_head=False):
        """
        初始化 MultiHeadDetectAttention

        Args:
            nc: 總類別數 (80)
            anchors: Anchor 設定 [[P3], [P4], [P5]]
            ch: 各層輸入通道數 [128, 256, 512]
            head_config: HeadConfig 實例
            full_head: 是否使用 full-size head (輸出維度 = 5 + nc)
        """
        # 先調用父類初始化
        super().__init__(nc, anchors, ch, head_config, full_head)

        # 為每個 Head 建立獨立的 CBAM 模組組
        # ch = [128, 256, 512] 對應 P3, P4, P5
        self.attentions = nn.ModuleList()
        for _ in range(self.num_heads):
            # 每個 Head 有 nl 層 CBAM (對應 P3, P4, P5)
            head_cbams = nn.ModuleList([CBAM(c) for c in ch])
            self.attentions.append(head_cbams)

    def _forward_train(self, x):
        """
        訓練模式前向傳播 (覆寫父類方法)

        先通過 CBAM 過濾特徵，再進行卷積運算。

        Returns:
            outputs: 列表，每個元素是一個 Head 的 P3/P4/P5 輸出
                     [[z0_p3, z0_p4, z0_p5], [z1_p3, z1_p4, z1_p5], ...]
                     每個 tensor shape: [bs, na, ny, nx, no]
        """
        outputs = []

        for head_id, (head_layers, attn_layers) in enumerate(zip(self.heads, self.attentions)):
            head_outputs = []
            head_no = self.head_no[head_id]

            for i, (layer, attn, xi) in enumerate(zip(head_layers, attn_layers, x)):
                bs, _, ny, nx = xi.shape

                # 先通過 CBAM 注意力過濾
                xi_filtered = attn(xi)

                # 再通過卷積運算
                yi = layer(xi_filtered)

                # reshape: [bs, na*no, ny, nx] → [bs, na, no, ny, nx]
                yi = yi.view(bs, self.na, head_no, ny, nx)

                # permute: [bs, na, no, ny, nx] → [bs, na, ny, nx, no]
                yi = yi.permute(0, 1, 3, 4, 2).contiguous()

                head_outputs.append(yi)

            outputs.append(head_outputs)

        return outputs

    def _forward_inference(self, x):
        """
        推論模式前向傳播 (覆寫父類方法)

        先通過 CBAM 過濾特徵，再進行卷積和 bbox 解碼，
        最後將各 Head 的 local class scores 映射回 global 80 類並合併。

        Returns:
            (z, outputs): 合併後的預測結果和原始輸出
                z shape: [bs, total_anchors, 85]
        """
        all_predictions = []
        outputs = []  # 保留原始輸出供除錯

        for head_id, (head_layers, attn_layers) in enumerate(zip(self.heads, self.attentions)):
            head_nc = self.head_nc[head_id]
            head_no = self.head_no[head_id]
            head_outputs = []

            # 取得該 Head 負責的類別列表
            if self.head_config is not None:
                head_classes = self.head_config.get_head_classes(head_id)
            else:
                # 預設: 均分
                start = head_id * (self.nc // self.num_heads)
                head_classes = list(range(start, start + head_nc))

            for i, (layer, attn, xi) in enumerate(zip(head_layers, attn_layers, x)):
                bs, _, ny, nx = xi.shape

                # 先通過 CBAM 注意力過濾
                xi_filtered = attn(xi)

                # 再通過卷積運算
                yi = layer(xi_filtered)

                # reshape & permute: [bs, na*no, ny, nx] → [bs, na, ny, nx, no]
                yi = yi.view(bs, self.na, head_no, ny, nx)
                yi = yi.permute(0, 1, 3, 4, 2).contiguous()

                head_outputs.append(yi)

                # 建立 grid (延遲初始化)
                if self.grid[i].shape[2:4] != yi.shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(yi.device)

                # Sigmoid 激活
                y = yi.sigmoid()

                # 解碼 bounding box
                xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]

                # 分離 objectness 和 class scores
                obj = y[..., 4:5]  # [bs, na, ny, nx, 1]
                cls_local = y[..., 5:]  # [bs, na, ny, nx, head_nc]

                # 建立 80 維的 cls tensor，初始化為 0
                cls_global = torch.zeros(
                    bs, self.na, ny, nx, self.nc,
                    dtype=cls_local.dtype, device=cls_local.device
                )

                # 將 local class scores 映射到對應的 global positions
                for local_id, global_id in enumerate(head_classes):
                    cls_global[..., global_id] = cls_local[..., local_id]

                # 合併: [bs, na, ny, nx, 85]
                pred = torch.cat([xy, wh, obj, cls_global], dim=-1)

                # 展平: [bs, na*ny*nx, 85]
                pred = pred.view(bs, -1, self.nc + 5)

                all_predictions.append(pred)

            outputs.append(head_outputs)

        # 合併所有 Head 和所有層的預測: [bs, total_anchors, 85]
        z = torch.cat(all_predictions, dim=1)

        return (z, outputs)


# 單獨執行時的測試
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    from utils.head_config import HeadConfig

    print("=== Testing MultiHeadDetectAttention ===\n")

    # 載入設定
    cfg = HeadConfig('data/coco_320_1b4h_standard.yaml')
    print(f"HeadConfig: {cfg}\n")

    # 建立模型
    anchors = [
        [10, 13, 16, 30, 33, 23],      # P3/8
        [30, 61, 62, 45, 59, 119],     # P4/16
        [116, 90, 156, 198, 373, 326]  # P5/32
    ]
    ch = [128, 256, 512]  # YOLOv7-tiny 的 P3/P4/P5 通道數

    model = MultiHeadDetectAttention(nc=80, anchors=anchors, ch=ch, head_config=cfg)
    model.stride = torch.tensor([8., 16., 32.])

    # 檢查繼承關係
    print(f"Is instance of MultiHeadDetect: {isinstance(model, MultiHeadDetect)}")
    print(f"Number of heads: {model.num_heads}")
    print(f"Number of attention modules per head: {len(model.attentions[0])}")

    # 統計參數
    total_params = sum(p.numel() for p in model.parameters())
    attn_params = sum(p.numel() for attn_list in model.attentions for attn in attn_list for p in attn.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Attention parameters: {attn_params:,} ({100*attn_params/total_params:.2f}%)")

    # 測試輸入
    x = [
        torch.randn(2, 128, 40, 40),  # P3: 320/8 = 40
        torch.randn(2, 256, 20, 20),  # P4: 320/16 = 20
        torch.randn(2, 512, 10, 10),  # P5: 320/32 = 10
    ]

    # 測試訓練模式
    model.train()
    outputs = model(x)
    print(f"\n=== 訓練模式測試 ===")
    print(f"輸出 Head 數量: {len(outputs)}")
    for i, head_out in enumerate(outputs):
        print(f"Head {i}:")
        for j, p in enumerate(head_out):
            print(f"  P{j+3} shape: {p.shape}")  # 預期: [2, 3, H, W, no]

    # 測試推論模式
    model.eval()
    with torch.no_grad():
        z, raw = model(x)
    print(f"\n=== 推論模式測試 ===")
    print(f"合併輸出 shape: {z.shape}")  # 預期: [2, total_anchors, 85]

    # 計算總 anchor 數
    total_anchors = 0
    for head_out in raw:
        for p in head_out:
            total_anchors += p.shape[1] * p.shape[2] * p.shape[3]
    print(f"預期總 anchors: {total_anchors}")
    print(f"實際輸出 anchors: {z.shape[1]}")

    # 驗證輸出格式與父類一致
    parent_model = MultiHeadDetect(nc=80, anchors=anchors, ch=ch, head_config=cfg)
    parent_model.stride = torch.tensor([8., 16., 32.])
    parent_model.eval()
    with torch.no_grad():
        z_parent, _ = parent_model(x)

    assert z.shape == z_parent.shape, f"Shape mismatch: {z.shape} vs {z_parent.shape}"
    print(f"\n輸出格式驗證: OK (與 MultiHeadDetect 相同)")

    print("\n=== 所有測試通過 ===")
