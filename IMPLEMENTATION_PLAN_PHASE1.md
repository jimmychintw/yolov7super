# YOLOv7 1B4H Phase 1 實作計畫

**版本**: v1.0
**日期**: 2025-11-30
**狀態**: Planning
**參考文件**: PRD v0.3, SDD v1.0

---

## 1. 實作目標

在 **non-OTA** 環境下驗證 1B4H (One Backbone Four Heads) 架構的有效性。

| 項目 | 說明 |
|------|------|
| **Baseline** | non-OTA mAP@0.5 = 0.385 |
| **目標** | 驗證 1B4H 架構可正常訓練，Loss 正常下降 |
| **環境** | 320x320, batch=64, epochs=100, non-OTA |

---

## 2. 模組依賴關係

```
                    ┌─────────────────────────┐
                    │ coco_320_1b4h_standard  │  (Task 5)
                    │        .yaml            │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │     HeadConfig          │  (Task 1)
                    │  utils/head_config.py   │
                    └───────────┬─────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
┌───────────▼───────────┐ ┌─────▼─────────────┐ ┌───▼───────────────┐
│   MultiHeadDetect     │ │ ComputeLossRouter │ │   train.py        │
│ models/multihead.py   │ │ utils/loss_router │ │   (整合)          │
│      (Task 2)         │ │     (Task 3)      │ │   (Task 4)        │
└───────────────────────┘ └───────────────────┘ └───────────────────┘
```

**實作順序**: Task 5 → Task 1 → Task 2 → Task 3 → Task 4

---

## 3. 詳細實作任務

### Task 5: 標準分類設定檔

**檔案**: `data/coco_320_1b4h_standard.yaml`

```yaml
# YOLOv7 1B4H 標準分類設定檔
# 將 COCO 80 類依語意分為 4 個 Head

nc: 80  # 總類別數
heads: 4  # 檢測頭數量
grouping: standard  # 分類方式

# 資料集路徑（繼承自 coco320.yaml）
train: ./coco320/images/train2017
val: ./coco320/images/val2017

# 類別名稱（COCO 80 類）
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush']

head_assignments:
  head_0:
    name: "人物與配件"
    classes: [0, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
    weight: 1.0
    nc: 20  # 該 Head 負責的類別數

  head_1:
    name: "交通工具與日常物品"
    classes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 72, 73, 74, 75, 76, 77, 78, 79]
    weight: 1.2
    nc: 20

  head_2:
    name: "動物與食物"
    classes: [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
    weight: 1.5
    nc: 20

  head_3:
    name: "家具與電子產品"
    classes: [13, 29, 30, 45, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
    weight: 1.1
    nc: 20
```

**驗證**: 確保 80 類都有被分配，且無重複。

---

### Task 1: 設定檔解析模組

**檔案**: `utils/head_config.py`

```python
# utils/head_config.py
"""
HeadConfig - 1B4H 設定檔解析模組

功能：
1. 載入並解析 head_assignments YAML
2. 建立 global_id → head_id 映射
3. 建立 global_id → local_id 映射
4. 驗證設定檔完整性
"""

import yaml
from pathlib import Path


class HeadConfig:
    """1B4H 設定檔管理器"""

    def __init__(self, config_path: str):
        """
        初始化 HeadConfig

        Args:
            config_path: YAML 設定檔路徑
        """
        self.config_path = Path(config_path)
        self._load_config()
        self._build_mappings()
        self._validate()

    def _load_config(self):
        """載入 YAML 設定檔"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.nc = self.config['nc']  # 總類別數 (80)
        self.num_heads = self.config['heads']  # Head 數量 (4)
        self.head_assignments = self.config['head_assignments']

    def _build_mappings(self):
        """建立映射表"""
        # global_id → head_id
        self.head_map = {}
        # global_id → local_id (0-19)
        self.local_id_map = {}
        # head_id → weight
        self.weights = []
        # head_id → [global_ids]
        self.head_classes = []

        for head_id in range(self.num_heads):
            head_key = f'head_{head_id}'
            head_info = self.head_assignments[head_key]

            classes = head_info['classes']
            weight = head_info.get('weight', 1.0)

            self.weights.append(weight)
            self.head_classes.append(classes)

            for local_id, global_id in enumerate(classes):
                self.head_map[global_id] = head_id
                self.local_id_map[global_id] = local_id

    def _validate(self):
        """驗證設定檔完整性"""
        # 檢查是否所有 80 類都有被分配
        assigned_classes = set(self.head_map.keys())
        expected_classes = set(range(self.nc))

        missing = expected_classes - assigned_classes
        if missing:
            raise ValueError(f"缺少類別分配: {sorted(missing)}")

        extra = assigned_classes - expected_classes
        if extra:
            raise ValueError(f"無效的類別 ID: {sorted(extra)}")

        # 檢查是否有重複分配
        all_classes = []
        for head_id in range(self.num_heads):
            all_classes.extend(self.head_classes[head_id])

        if len(all_classes) != len(set(all_classes)):
            raise ValueError("存在重複的類別分配")

    def get_head_id(self, global_id: int) -> int:
        """
        取得類別所屬的 Head ID

        Args:
            global_id: COCO 類別 ID (0-79)

        Returns:
            head_id: 所屬 Head ID (0-3)
        """
        return self.head_map[global_id]

    def get_local_id(self, global_id: int) -> int:
        """
        將 global_id 轉換為該 Head 內的 local_id

        Args:
            global_id: COCO 類別 ID (0-79)

        Returns:
            local_id: Head 內的類別 ID (0-19)
        """
        return self.local_id_map[global_id]

    def get_head_info(self, global_id: int) -> tuple:
        """
        取得類別的完整資訊

        Args:
            global_id: COCO 類別 ID (0-79)

        Returns:
            (head_id, local_id): Head ID 和 Local ID
        """
        return self.get_head_id(global_id), self.get_local_id(global_id)

    def get_head_weight(self, head_id: int) -> float:
        """
        取得指定 Head 的 Loss 權重

        Args:
            head_id: Head ID (0-3)

        Returns:
            weight: Loss 權重
        """
        return self.weights[head_id]

    def get_head_classes(self, head_id: int) -> list:
        """
        取得指定 Head 負責的所有類別

        Args:
            head_id: Head ID (0-3)

        Returns:
            classes: Global class ID 列表
        """
        return self.head_classes[head_id]

    def get_head_nc(self, head_id: int) -> int:
        """
        取得指定 Head 的類別數量

        Args:
            head_id: Head ID (0-3)

        Returns:
            nc: 類別數量
        """
        return len(self.head_classes[head_id])

    def __repr__(self):
        return f"HeadConfig(nc={self.nc}, heads={self.num_heads}, path={self.config_path})"
```

**測試方式**:
```python
# tests/test_head_config.py
from utils.head_config import HeadConfig

cfg = HeadConfig('data/coco_320_1b4h_standard.yaml')
assert cfg.get_head_id(0) == 0  # person → Head 0
assert cfg.get_local_id(0) == 0  # person → local_id 0
assert cfg.get_head_weight(2) == 1.5  # Head 2 weight
print("HeadConfig 測試通過！")
```

---

### Task 2: 多頭檢測模組

**檔案**: `models/multihead.py`

```python
# models/multihead.py
"""
MultiHeadDetect - 1B4H 多頭檢測層

特性：
1. 4 個獨立的檢測頭，各負責 20 類
2. 共享 Backbone 特徵 (P3/P4/P5)
3. 訓練模式：返回 4 組獨立的 raw tensors
4. 推論模式：合併輸出 + 全域 NMS
"""

import torch
import torch.nn as nn
import math


class MultiHeadDetect(nn.Module):
    """
    YOLOv7 1B4H 多頭檢測層

    每個 Head 獨立處理指定的類別子集
    """
    stride = None  # 各層的 stride [8, 16, 32]
    export = False  # 是否為 ONNX 匯出模式

    def __init__(self, nc=80, anchors=(), ch=(), head_config=None):
        """
        初始化 MultiHeadDetect

        Args:
            nc: 總類別數 (80)
            anchors: Anchor 設定 [[P3], [P4], [P5]]
            ch: 各層輸入通道數 [128, 256, 512]
            head_config: HeadConfig 實例
        """
        super().__init__()

        self.nc = nc  # 總類別數 (80)
        self.head_config = head_config
        self.num_heads = head_config.num_heads if head_config else 4

        # 每個 Head 的類別數
        self.head_nc = [head_config.get_head_nc(i) for i in range(self.num_heads)]

        # 每個 Head 的輸出維度 = 4(bbox) + 1(obj) + nc_i(cls)
        self.head_no = [4 + 1 + nc_i for nc_i in self.head_nc]

        self.nl = len(anchors)  # 檢測層數 (3: P3, P4, P5)
        self.na = len(anchors[0]) // 2  # 每層 anchor 數 (3)

        # 初始化 grid（推論時使用）
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl

        # 註冊 anchors
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl, na, 2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))

        # 建立 4 個獨立的檢測頭
        # 每個 Head 包含 nl 層 Conv (對應 P3, P4, P5)
        self.heads = nn.ModuleList()
        for head_id in range(self.num_heads):
            head_no = self.head_no[head_id]
            head_layers = nn.ModuleList([
                nn.Conv2d(ch[j], self.na * head_no, 1)
                for j in range(self.nl)
            ])
            self.heads.append(head_layers)

        # 初始化權重
        self._initialize_biases()

    def _initialize_biases(self):
        """初始化偏置（類似原始 Detect）"""
        for head_id, head_layers in enumerate(self.heads):
            head_nc = self.head_nc[head_id]
            for layer, s in zip(head_layers, self.stride):
                b = layer.bias.view(self.na, -1)
                # objectness 偏置
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)
                # class 偏置
                b.data[:, 5:5+head_nc] += math.log(0.6 / (head_nc - 0.99))
                layer.bias = nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        """
        前向傳播

        Args:
            x: Backbone 輸出的特徵列表 [P3, P4, P5]

        Returns:
            訓練模式: [[head0_outputs], [head1_outputs], ...]
            推論模式: (concatenated_output, x)
        """
        if self.training:
            return self._forward_train(x)
        else:
            return self._forward_inference(x)

    def _forward_train(self, x):
        """
        訓練模式前向傳播

        Returns:
            outputs: 列表，每個元素是一個 Head 的輸出
                     [[z0_p3, z0_p4, z0_p5], [z1_p3, z1_p4, z1_p5], ...]
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

        Returns:
            (z, x): 合併後的預測結果和原始特徵
        """
        all_predictions = []

        for head_id, head_layers in enumerate(self.heads):
            head_nc = self.head_nc[head_id]
            head_no = self.head_no[head_id]
            head_classes = self.head_config.get_head_classes(head_id)

            for i, (layer, xi) in enumerate(zip(head_layers, x)):
                bs, _, ny, nx = xi.shape

                # 卷積運算
                yi = layer(xi)

                # reshape & permute
                yi = yi.view(bs, self.na, head_no, ny, nx)
                yi = yi.permute(0, 1, 3, 4, 2).contiguous()

                # 建立 grid
                if self.grid[i].shape[2:4] != yi.shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                # Sigmoid 激活
                yi = yi.sigmoid()

                # 解碼 bounding box
                xy = (yi[..., :2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                wh = (yi[..., 2:4] * 2) ** 2 * self.anchor_grid[i]

                # 擴展類別預測到 80 維
                # 原始: [bs, na, ny, nx, head_no] 其中 cls 部分是 head_nc 維
                # 目標: [bs, na, ny, nx, 85] 其中 cls 部分是 80 維
                obj = yi[..., 4:5]  # objectness
                cls_local = yi[..., 5:]  # local class scores (head_nc 維)

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

        # 合併所有 Head 的預測: [bs, total_anchors, 85]
        z = torch.cat(all_predictions, dim=1)

        return (z, x)

    def _make_grid(self, nx, ny, i):
        """建立網格座標"""
        d = self.anchors.device
        yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid
```

---

### Task 3: 損失路由器

**檔案**: `utils/loss_router.py`

```python
# utils/loss_router.py
"""
ComputeLossRouter - 1B4H 損失路由器

功能：
1. 將 targets 依據 HeadConfig 分配到對應的 Head
2. 將 global_class_id 轉換為 local_class_id
3. 計算各 Head 的 Loss 並加權求和
4. 實現「隱式負樣本挖掘」
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import bbox_iou
from utils.general import xywh2xyxy


def smooth_BCE(eps=0.1):
    """Label smoothing for BCE"""
    return 1.0 - 0.5 * eps, 0.5 * eps


class ComputeLossRouter:
    """1B4H 損失路由器"""

    def __init__(self, model, head_config, autobalance=False):
        """
        初始化 ComputeLossRouter

        Args:
            model: YOLOv7 模型
            head_config: HeadConfig 實例
            autobalance: 是否自動平衡各層 Loss
        """
        self.device = next(model.parameters()).device
        self.head_config = head_config
        self.num_heads = head_config.num_heads

        # 取得超參數
        h = model.hyp

        # BCE loss with label smoothing
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))

        # Focal Loss
        g = h.get('fl_gamma', 0.0)
        if g > 0:
            self.BCEcls = FocalLoss(nn.BCEWithLogitsLoss(reduction='none'), g)
            self.BCEobj = FocalLoss(nn.BCEWithLogitsLoss(reduction='none'), g)
        else:
            self.BCEcls = nn.BCEWithLogitsLoss(reduction='none')
            self.BCEobj = nn.BCEWithLogitsLoss(reduction='none')

        # 取得檢測層參數
        det = model.model[-1]  # MultiHeadDetect 層
        self.na = det.na  # anchor 數量
        self.nc = det.nc  # 總類別數 (80)
        self.nl = det.nl  # 檢測層數 (3)
        self.anchors = det.anchors
        self.stride = det.stride
        self.head_nc = det.head_nc  # 各 Head 類別數

        # Loss 權重
        self.box_weight = h['box']
        self.obj_weight = h['obj']
        self.cls_weight = h['cls']

        # 各層的 balance 權重 (P3, P4, P5)
        self.balance = {3: [4.0, 1.0, 0.4]}.get(self.nl, [4.0, 1.0, 0.25, 0.06, 0.02])

        self.autobalance = autobalance
        self.gr = 1.0  # IoU ratio

    def __call__(self, predictions, targets):
        """
        計算 1B4H Loss

        Args:
            predictions: MultiHeadDetect 的輸出
                         [[head0_p3, head0_p4, head0_p5], [head1_...], ...]
            targets: 標籤 [N, 6] (image_idx, class, x, y, w, h)

        Returns:
            loss: 總 Loss
            loss_items: [box_loss, obj_loss, cls_loss, total_loss]
        """
        # 初始化 Loss 累加器
        total_lbox = torch.zeros(1, device=self.device)
        total_lobj = torch.zeros(1, device=self.device)
        total_lcls = torch.zeros(1, device=self.device)

        # 遍歷每個 Head
        for head_id in range(self.num_heads):
            head_preds = predictions[head_id]  # [p3, p4, p5] for this head
            head_weight = self.head_config.get_head_weight(head_id)
            head_classes = set(self.head_config.get_head_classes(head_id))
            head_nc = self.head_nc[head_id]

            # 篩選屬於此 Head 的 targets
            # targets: [N, 6] (image_idx, class, x, y, w, h)
            if targets.shape[0] > 0:
                mask = torch.tensor([
                    int(t[1].item()) in head_classes
                    for t in targets
                ], dtype=torch.bool, device=self.device)
                head_targets = targets[mask].clone()

                # 轉換 global_class_id → local_class_id
                if head_targets.shape[0] > 0:
                    for i in range(head_targets.shape[0]):
                        global_id = int(head_targets[i, 1].item())
                        local_id = self.head_config.get_local_id(global_id)
                        head_targets[i, 1] = local_id
            else:
                head_targets = targets

            # 計算此 Head 的 Loss
            lbox, lobj, lcls = self._compute_head_loss(
                head_preds, head_targets, head_id, head_nc
            )

            # 加權累加
            total_lbox += lbox * head_weight
            total_lobj += lobj * head_weight
            total_lcls += lcls * head_weight

        # 乘以超參數權重
        total_lbox *= self.box_weight
        total_lobj *= self.obj_weight
        total_lcls *= self.cls_weight

        # 計算總 Loss
        bs = predictions[0][0].shape[0]  # batch size
        loss = total_lbox + total_lobj + total_lcls

        return loss * bs, torch.cat([total_lbox, total_lobj, total_lcls, loss]).detach()

    def _compute_head_loss(self, head_preds, targets, head_id, head_nc):
        """
        計算單一 Head 的 Loss

        Args:
            head_preds: 此 Head 的預測 [p3, p4, p5]
            targets: 已轉換為 local_id 的 targets
            head_id: Head ID
            head_nc: 此 Head 的類別數

        Returns:
            lbox, lobj, lcls: Box/Obj/Cls Loss
        """
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        lcls = torch.zeros(1, device=self.device)

        # 建立 targets 分配
        tcls, tbox, indices, anchors = self._build_targets(head_preds, targets)

        # 遍歷各檢測層 (P3, P4, P5)
        for i, pi in enumerate(head_preds):
            # pi shape: [bs, na, ny, nx, no]
            # no = 4 + 1 + head_nc

            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0])  # objectness target

            n = b.shape[0]  # 此層的 target 數量

            if n:
                # 取得對應位置的預測
                ps = pi[b, a, gj, gi]  # [n, no]

                # Box regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat([pxy, pwh], dim=1)

                # CIoU Loss
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()

                # Objectness target (用 IoU 作為 soft label)
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)

                # Classification Loss
                if head_nc > 1:
                    # 建立 one-hot target
                    t = torch.full_like(ps[:, 5:], self.cn)
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t).mean()

            # Objectness Loss (包含負樣本)
            obji = self.BCEobj(pi[..., 4], tobj).mean()
            lobj += obji * self.balance[i]

        return lbox, lobj, lcls

    def _build_targets(self, preds, targets):
        """
        建立 targets 分配（簡化版，非 OTA）

        Args:
            preds: Head 預測 [p3, p4, p5]
            targets: [N, 6] (image_idx, class, x, y, w, h)

        Returns:
            tcls, tbox, indices, anchors: 各層的 targets
        """
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []

        gain = torch.ones(7, device=self.device)
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        g = 0.5  # offset
        off = torch.tensor([
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
        ], device=self.device).float() * g

        for i in range(self.nl):
            anchors_i = self.anchors[i]
            gain[2:6] = torch.tensor(preds[i].shape)[[3, 2, 3, 2]]

            t = targets * gain

            if nt:
                # 篩選符合 anchor ratio 的 targets
                r = t[:, :, 4:6] / anchors_i[:, None]
                j = torch.max(r, 1. / r).max(2)[0] < 4.0  # anchor ratio threshold
                t = t[j]

                # Offset
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack([torch.ones_like(j), j, k, l, m])
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # 提取 targets
            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T

            a = t[:, 6].long()
            indices.append((b, a, gj.clamp_(0, gain[3].long() - 1), gi.clamp_(0, gain[2].long() - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors_i[a])
            tcls.append(c)

        return tcls, tbox, indices, anch


class FocalLoss(nn.Module):
    """Focal Loss wrapper"""
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        return loss.mean()
```

---

### Task 4: 整合到訓練流程

#### 4.1 修改 `train.py`

**位置**: 參數解析區域 (~第 500 行)

```python
# 新增參數
parser.add_argument('--heads', type=int, default=0, help='Number of detection heads (0=disabled, 4=1B4H)')
parser.add_argument('--head-config', type=str, default='', help='Path to head configuration file')
```

**位置**: Loss 初始化區域 (~第 300 行)

```python
# 原始程式碼:
# compute_loss_ota = ComputeLossOTA(model)
# compute_loss = ComputeLoss(model)

# 修改為:
if opt.heads > 0 and opt.head_config:
    from utils.head_config import HeadConfig
    from utils.loss_router import ComputeLossRouter
    head_config = HeadConfig(opt.head_config)
    compute_loss = ComputeLossRouter(model, head_config)
    compute_loss_ota = None  # 1B4H 暫不支援 OTA
    LOGGER.info(f'Using 1B{opt.heads}H architecture with {opt.head_config}')
else:
    compute_loss_ota = ComputeLossOTA(model)
    compute_loss = ComputeLoss(model)
```

**位置**: Loss 計算區域 (~第 362 行)

```python
# 原始程式碼:
# if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
#     loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)
# else:
#     loss, loss_items = compute_loss(pred, targets.to(device))

# 修改為:
if opt.heads > 0:
    # 1B4H 模式
    loss, loss_items = compute_loss(pred, targets.to(device))
elif 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
    loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)
else:
    loss, loss_items = compute_loss(pred, targets.to(device))
```

#### 4.2 修改 `models/yolo.py`

**位置**: import 區域 (~第 1 行)

```python
# 新增 import
from models.multihead import MultiHeadDetect
```

**位置**: parse_model 函數 (~第 790 行)

```python
# 原始程式碼:
# elif m in [Detect, IDetect, IAuxDetect, IBin, IKeypoint]:

# 修改為:
elif m in [Detect, IDetect, IAuxDetect, IBin, IKeypoint, MultiHeadDetect]:
    args.append([ch[x] for x in f])
    if m is MultiHeadDetect:
        # 需要傳入 head_config
        pass  # head_config 在 train.py 中設定
```

#### 4.3 建立模型設定檔

**檔案**: `cfg/training/yolov7-tiny-1b4h.yaml`

```yaml
# YOLOv7-tiny 1B4H 模型設定
# 沿用原始 backbone，替換 head 為 MultiHeadDetect

nc: 80  # 類別數 (實際由 head_config 控制分配)
depth_multiple: 1.0
width_multiple: 1.0

# 使用與 yolov7-tiny 相同的 anchors
anchors:
  - [10,13, 16,30, 33,23]     # P3/8
  - [30,61, 62,45, 59,119]    # P4/16
  - [116,90, 156,198, 373,326] # P5/32

# Backbone: 與 yolov7-tiny.yaml 完全相同
backbone:
  # ... (複製自 yolov7-tiny.yaml)

# Head: 使用 MultiHeadDetect
head:
  # ... (前面的層與 yolov7-tiny.yaml 相同)

  # 最後一層改為 MultiHeadDetect
  - [[74,75,76], 1, MultiHeadDetect, [nc, anchors]]
```

---

## 4. 驗收測試

### 4.1 單元測試

| 測試 ID | 測試項目 | 驗收標準 |
|---------|---------|---------|
| UT-01 | HeadConfig 映射 | `get_head_id(0) == 0`, `get_local_id(0) == 0` |
| UT-02 | HeadConfig 驗證 | 80 類都有分配，無重複 |
| UT-03 | MultiHeadDetect 訓練輸出 | 返回 4 組 tensor |
| UT-04 | MultiHeadDetect 推論輸出 | 返回合併的 [bs, total_anchors, 85] |
| UT-05 | ComputeLossRouter 遮罩 | Head 0 不會訓練 Head 1 的類別 |

### 4.2 整合測試

| 測試 ID | 測試項目 | 指令 | 驗收標準 |
|---------|---------|------|---------|
| IT-01 | 訓練啟動 | 見下方 | 無 crash，Loss 開始下降 |
| IT-02 | 10 epochs 訓練 | 見下方 | Loss 穩定下降 |
| IT-03 | 推論測試 | `python detect.py ...` | NMS 正常，輸出正確 |

**IT-01 訓練指令 (non-OTA, 10 epochs 快速驗證)**:
```bash
python train.py \
    --img-size 320 320 \
    --batch-size 64 \
    --epochs 10 \
    --data data/coco_320_1b4h_standard.yaml \
    --cfg cfg/training/yolov7-tiny-1b4h.yaml \
    --weights '' \
    --hyp data/hyp.scratch.tiny.noota.yaml \
    --device 0 \
    --workers 16 \
    --project runs/train \
    --name p1_tiny320_1b4h_standard_noota_10ep_test \
    --exist-ok \
    --noautoanchor \
    --cache-images \
    --heads 4 \
    --head-config data/coco_320_1b4h_standard.yaml
```

**IT-02 完整訓練指令 (non-OTA, 100 epochs)**:
```bash
python train.py \
    --img-size 320 320 \
    --batch-size 64 \
    --epochs 100 \
    --data data/coco_320_1b4h_standard.yaml \
    --cfg cfg/training/yolov7-tiny-1b4h.yaml \
    --weights '' \
    --hyp data/hyp.scratch.tiny.noota.yaml \
    --device 0 \
    --workers 16 \
    --save_period 25 \
    --project runs/train \
    --name p1_tiny320_1b4h_standard_noota_100ep \
    --exist-ok \
    --noautoanchor \
    --cache-images \
    --heads 4 \
    --head-config data/coco_320_1b4h_standard.yaml
```

**對照組 (Baseline non-OTA, 已完成 mAP@0.5=0.385)**:
```bash
# 已完成，結果在 runs/train/noota_100ep2/
# mAP@0.5 = 0.385, mAP@0.5:0.95 = 0.226
```

### 命名規則說明

```
{phase}_{model}{resolution}_{architecture}_{grouping}_{loss}_{epochs}_{note}
```

| 欄位 | 說明 | 範例 |
|------|------|------|
| phase | 開發階段 | `p1` (Phase 1), `p2`, `p3` |
| model | 模型類型 | `tiny` |
| resolution | 圖像尺寸 | `320`, `640` |
| architecture | 架構類型 | `baseline`, `1b4h`, `1b8h` |
| grouping | 分類方式 | `standard`, `geometry` |
| loss | Loss 類型 | `noota`, `ota` |
| epochs | 訓練週期 | `100ep` |
| note | 備註 | `test`, `v2` |

**輸出目錄範例**:
| 實驗 | 目錄名稱 |
|------|---------|
| Phase 1 快速測試 | `runs/train/p1_tiny320_1b4h_standard_noota_10ep_test/` |
| Phase 1 完整訓練 | `runs/train/p1_tiny320_1b4h_standard_noota_100ep/` |
| Phase 2 幾何分類 | `runs/train/p2_tiny320_1b4h_geometry_noota_100ep/` |
| Phase 5 最終訓練 | `runs/train/p5_tiny640_1b4h_standard_ota_300ep/` |

---

## 5. 檔案清單

| 類型 | 檔案路徑 | 狀態 |
|------|---------|------|
| 新建 | `utils/head_config.py` | 待實作 |
| 新建 | `models/multihead.py` | 待實作 |
| 新建 | `utils/loss_router.py` | 待實作 |
| 新建 | `data/coco_320_1b4h_standard.yaml` | 待實作 |
| 新建 | `cfg/training/yolov7-tiny-1b4h.yaml` | 待實作 |
| 修改 | `train.py` | 新增參數和 Loss 選擇邏輯 |
| 修改 | `models/yolo.py` | 新增 MultiHeadDetect import 和 parse |
| 新建 | `tests/test_head_config.py` | 待實作 |
| 新建 | `tests/test_multihead.py` | 待實作 |

---

## 6. 風險與緩解

| 風險 | 緩解措施 |
|------|---------|
| Local ID 映射錯誤 | 單元測試 UT-01, UT-02 |
| 推論時類別合併錯誤 | 單元測試 UT-04，視覺化驗證 |
| Loss 計算邏輯錯誤 | 與 Baseline ComputeLoss 對比 |
| CUDA OOM | 監控 GPU 記憶體，必要時減小 batch size |

---

## 7. 時程規劃

| 階段 | 內容 |
|------|------|
| 階段 1 | Task 5 + Task 1 (設定檔 + HeadConfig) |
| 階段 2 | Task 2 (MultiHeadDetect) |
| 階段 3 | Task 3 (ComputeLossRouter) |
| 階段 4 | Task 4 (整合 train.py) |
| 階段 5 | 單元測試 + 整合測試 |
| 階段 6 | 100 epochs 完整訓練 |

---

*文件版本: v1.0*
*建立日期: 2025-11-30*
