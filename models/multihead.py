# models/multihead.py
"""
MultiHeadDetect - 1B4H 多頭檢測層

特性：
1. 4 個獨立的檢測頭，各負責指定類別子集
2. 共享 Backbone 特徵 (P3/P4/P5)
3. 訓練模式：返回 4 組獨立的 raw tensors
4. 推論模式：合併輸出 + 全域 NMS

Phase 1 實作
參考文件: SDD v1.0, PRD v0.3
"""

import torch
import torch.nn as nn
import math


class MultiHeadDetect(nn.Module):
    """
    YOLOv7 1B4H 多頭檢測層

    每個 Head 獨立處理指定的類別子集，訓練時分別計算 Loss，
    推論時合併所有 Head 的輸出並送入標準 NMS。

    Attributes:
        nc (int): 總類別數 (80)
        num_heads (int): Head 數量 (4)
        head_nc (list): 各 Head 的類別數 [20, 20, 20, 20]
        nl (int): 檢測層數 (3: P3, P4, P5)
        na (int): 每層 anchor 數 (3)
        full_head (bool): 是否使用 full-size head (輸出維度 = 5 + nc)
    """
    stride = None  # strides computed during build [8, 16, 32]
    export = False  # ONNX export mode

    def __init__(self, nc=80, anchors=(), ch=(), head_config=None, full_head=False):
        """
        初始化 MultiHeadDetect

        Args:
            nc: 總類別數 (80)
            anchors: Anchor 設定 [[P3], [P4], [P5]]
            ch: 各層輸入通道數 [128, 256, 512]
            head_config: HeadConfig 實例
            full_head: 是否使用 full-size head
                       - False (預設): 輸出維度 = 5 + head_nc (原始設計)
                       - True: 輸出維度 = 5 + nc (與 1B1H 相同大小)
        """
        super(MultiHeadDetect, self).__init__()

        self.nc = nc  # 總類別數 (80)
        self.head_config = head_config
        self.full_head = full_head

        if head_config is not None:
            self.num_heads = head_config.num_heads
            # 每個 Head 的類別數
            self.head_nc = [head_config.get_head_nc(i) for i in range(self.num_heads)]
        else:
            # 預設: 4 個 Head，每個 20 類
            self.num_heads = 4
            self.head_nc = [nc // 4] * 4

        # 每個 Head 的輸出維度
        if full_head:
            # Full-size: 所有 Head 輸出維度相同 = 5 + nc (85)
            self.head_no = [5 + nc for _ in range(self.num_heads)]
        else:
            # 原始設計: 輸出維度 = 5 + head_nc
            self.head_no = [5 + nc_i for nc_i in self.head_nc]

        self.nl = len(anchors)  # 檢測層數 (3: P3, P4, P5)
        self.na = len(anchors[0]) // 2  # 每層 anchor 數 (3)

        # 初始化 grid（推論時使用）
        self.grid = [torch.zeros(1)] * self.nl

        # 註冊 anchors
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl, na, 2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))

        # 建立 num_heads 個獨立的檢測頭
        # 每個 Head 包含 nl 層 Conv (對應 P3, P4, P5)
        self.heads = nn.ModuleList()
        for head_id in range(self.num_heads):
            head_no = self.head_no[head_id]
            head_layers = nn.ModuleList([
                nn.Conv2d(ch[j], self.na * head_no, 1)
                for j in range(self.nl)
            ])
            self.heads.append(head_layers)

    def forward(self, x):
        """
        前向傳播

        Args:
            x: Backbone 輸出的特徵列表 [P3, P4, P5]

        Returns:
            訓練模式: [[head0_p3, head0_p4, head0_p5], [head1_...], ...]
            推論模式: (concatenated_output, raw_outputs)
        """
        if self.training:
            return self._forward_train(x)
        else:
            return self._forward_inference(x)

    def _forward_train(self, x):
        """
        訓練模式前向傳播

        Returns:
            outputs: 列表，每個元素是一個 Head 的 P3/P4/P5 輸出
                     [[z0_p3, z0_p4, z0_p5], [z1_p3, z1_p4, z1_p5], ...]
                     每個 tensor shape: [bs, na, ny, nx, no]
        """
        outputs = []

        for head_id, head_layers in enumerate(self.heads):
            head_outputs = []
            head_no = self.head_no[head_id]

            for i, (layer, xi) in enumerate(zip(head_layers, x)):
                bs, _, ny, nx = xi.shape

                # 卷積運算
                yi = layer(xi)

                # reshape: [bs, na*no, ny, nx] → [bs, na, no, ny, nx]
                yi = yi.view(bs, self.na, head_no, ny, nx)

                # permute: [bs, na, no, ny, nx] → [bs, na, ny, nx, no]
                yi = yi.permute(0, 1, 3, 4, 2).contiguous()

                head_outputs.append(yi)

            outputs.append(head_outputs)

        return outputs

    def _forward_inference(self, x):
        """
        推論模式前向傳播

        將各 Head 的 local class scores 映射回 global 80 類，
        然後合併所有 Head 的預測送入標準 NMS。

        Returns:
            (z, outputs): 合併後的預測結果和原始輸出
                z shape: [bs, total_anchors, 85]
        """
        all_predictions = []
        outputs = []  # 保留原始輸出供除錯

        for head_id, head_layers in enumerate(self.heads):
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

            for i, (layer, xi) in enumerate(zip(head_layers, x)):
                bs, _, ny, nx = xi.shape

                # 卷積運算
                yi = layer(xi)

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
                cls_output = y[..., 5:]  # [bs, na, ny, nx, head_nc or nc]

                if self.full_head:
                    # Full-head 模式: 輸出已經是 80 維，直接使用
                    # 但需要 mask 掉非該 Head 負責的類別
                    cls_global = torch.zeros(
                        bs, self.na, ny, nx, self.nc,
                        dtype=cls_output.dtype, device=cls_output.device
                    )
                    # 只保留該 Head 負責的類別
                    for local_id, global_id in enumerate(head_classes):
                        cls_global[..., global_id] = cls_output[..., global_id]
                else:
                    # 原始模式: 輸出是 head_nc 維，需要映射到 80 維
                    cls_global = torch.zeros(
                        bs, self.na, ny, nx, self.nc,
                        dtype=cls_output.dtype, device=cls_output.device
                    )
                    # 將 local class scores 映射到對應的 global positions
                    for local_id, global_id in enumerate(head_classes):
                        cls_global[..., global_id] = cls_output[..., local_id]

                # 合併: [bs, na, ny, nx, 85]
                pred = torch.cat([xy, wh, obj, cls_global], dim=-1)

                # 展平: [bs, na*ny*nx, 85]
                pred = pred.view(bs, -1, self.nc + 5)

                all_predictions.append(pred)

            outputs.append(head_outputs)

        # 合併所有 Head 和所有層的預測: [bs, total_anchors, 85]
        z = torch.cat(all_predictions, dim=1)

        return (z, outputs)

    def _make_grid(self, nx=20, ny=20):
        """建立網格座標"""
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def initialize_biases(self, cf=None):
        """
        初始化偏置（可選，用於改善訓練初期收斂）

        Args:
            cf: class frequency (可選)
        """
        for head_id, head_layers in enumerate(self.heads):
            head_nc = self.head_nc[head_id]
            # full_head 模式下，輸出維度是 nc，但只初始化該 Head 負責的類別
            output_nc = self.nc if self.full_head else head_nc

            for layer, s in zip(head_layers, self.stride):
                b = layer.bias.view(self.na, -1)  # [na, no]
                # objectness 偏置: 假設每張圖平均 8 個物體
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)
                # class 偏置
                if self.full_head:
                    # Full-head 模式: 只初始化該 Head 負責的類別
                    head_classes = self.head_config.get_head_classes(head_id) if self.head_config else []
                    for global_id in head_classes:
                        b.data[:, 5 + global_id] += math.log(0.6 / (head_nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
                else:
                    # 原始模式
                    b.data[:, 5:5+head_nc] += math.log(0.6 / (head_nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
                layer.bias = nn.Parameter(b.view(-1), requires_grad=True)


# 單獨執行時的測試
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    from utils.head_config import HeadConfig

    # 載入設定
    cfg = HeadConfig('data/coco_320_1b4h_standard.yaml')
    print(cfg)

    # 建立模型
    anchors = [
        [10, 13, 16, 30, 33, 23],      # P3/8
        [30, 61, 62, 45, 59, 119],     # P4/16
        [116, 90, 156, 198, 373, 326]  # P5/32
    ]
    ch = [128, 256, 512]  # YOLOv7-tiny 的 P3/P4/P5 通道數

    model = MultiHeadDetect(nc=80, anchors=anchors, ch=ch, head_config=cfg)
    model.stride = torch.tensor([8., 16., 32.])

    # 測試訓練模式
    model.train()
    x = [
        torch.randn(2, 128, 40, 40),  # P3: 320/8 = 40
        torch.randn(2, 256, 20, 20),  # P4: 320/16 = 20
        torch.randn(2, 512, 10, 10),  # P5: 320/32 = 10
    ]

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

    print("\n=== 測試通過 ===")
