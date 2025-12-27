# YOLOv7 1B4H 架構產品需求文檔

**版本**: v0.7
**日期**: 2025-12-01
**狀態**: Planning

---

## 1. 執行摘要

本專案旨在透過 **多檢測頭架構 (1 Backbone + N Heads)** 提升 YOLOv7 物件偵測精度。將 80 類物件依據不同策略分配到多個獨立檢測頭，解決不同類別間的特徵衝突問題。

### 核心設計原則

1. **向下相容**：原始功能完全保留，不修改現有程式碼邏輯
2. **外部參數控制**：所有新功能用 `--` 訓練參數啟用
3. **漸進式開發**：每個功能獨立，可單獨啟用或組合

---

## 2. 分類策略

### 2.1 標準分類 (Standard Grouping)

依據**語意/功能**進行人工分組。

| Head | 類別名稱 | 數量 | 特點 | class_id |
|------|---------|------|------|----------|
| Head 0 | 人物與配件 | 20 類 | 複雜姿態、小物件 | 0, 24-28, 31-44 |
| Head 1 | 交通工具與日常物品 | 20 類 | 剛性、規則形狀 | 1-12, 72-79 |
| Head 2 | 動物與食物 | 20 類 | 可變形、有生命 | 13-23, 45-53 |
| Head 3 | 家具與電子產品 | 20 類 | 室內、靜態 | 29-30, 54-71 |

### 2.2 幾何分類 (Geometry Grouping)

依據 **K-Means 聚類**自動分組，考慮物件的長寬比與面積。

- **輸入**: COCO 標籤的 `(width/height)` 比率與 `area`
- **方法**: K-Means Clustering (k=4 或 k=8)
- **特性**: 不同解析度 (320/640) 需分別產生

### 2.3 分類方式比較

| 分類方式 | 依據 | 用途 | 設定檔來源 |
|---------|------|------|-----------|
| 標準分類 | 語意/功能 | Head 分配、Loss Router | 手動定義 |
| 幾何分類 | 長寬比、面積 | Anchor 優化、增強策略 | K-Means 自動產生 |

---

## 3. 設定檔規範

### 3.1 命名規則

```
data/coco_{解析度}_{架構}_{分類方式}.yaml
```

### 3.2 設定檔列表

| 設定檔 | 說明 |
|--------|------|
| `coco_320_1b4h_standard.yaml` | 320x320, 4 Heads, 標準分類 |
| `coco_320_1b4h_geometry.yaml` | 320x320, 4 Heads, 幾何分類 |
| `coco_640_1b4h_standard.yaml` | 640x640, 4 Heads, 標準分類 |
| `coco_640_1b4h_geometry.yaml` | 640x640, 4 Heads, 幾何分類 |
| `coco_320_1b8h_standard.yaml` | 320x320, 8 Heads, 標準分類 |
| `coco_640_1b8h_standard.yaml` | 640x640, 8 Heads, 標準分類 |

### 3.3 設定檔格式

```yaml
# 範例：coco_320_1b4h_standard.yaml
nc: 80  # 總類別數
heads: 4  # 檢測頭數量
grouping: standard  # 分類方式

head_assignments:
  head_0:
    name: "人物與配件"
    classes: [0, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
    weight: 1.0

  head_1:
    name: "交通工具與日常物品"
    classes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 72, 73, 74, 75, 76, 77, 78, 79]
    weight: 1.2

  head_2:
    name: "動物與食物"
    classes: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 45, 46, 47, 48, 49, 50, 51, 52, 53]
    weight: 1.5

  head_3:
    name: "家具與電子產品"
    classes: [29, 30, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
    weight: 1.1
```

---

## 4. 命令列參數設計

### 4.1 新增參數

| 參數 | 類型 | 說明 |
|------|------|------|
| `--heads N` | int | 指定檢測頭數量 (4, 8, ...) |
| `--standard-grouping` | flag | 使用標準分類（語意分組）|
| `--geometry-grouping` | flag | 使用幾何分類（K-Means）|
| `--head-config` | str | 指定分類設定檔路徑 |
| `--head-attention` | str | 啟用 Attention 機制 (cbam, se, ...) |
| `--rl-augment` | flag | 啟用 RL 超參數優化 |
| `--test-batch-size` | int | 驗證階段 batch size（解決 1B4H 推論 OOM）|
| `--transfer-weights` | flag | 啟用智慧權重遷移（載入 Backbone/Neck，捨棄 Head）|

### 4.2 使用範例

```bash
# 原始訓練（完全不變）
python train.py --data data/coco320.yaml --img 320 \
    --cfg cfg/training/yolov7-tiny.yaml --batch-size 64 --epochs 100

# 1B4H + 標準分類 (需使用 --test-batch-size 避免驗證 OOM)
python train.py --data data/coco320.yaml --img 320 \
    --cfg cfg/training/yolov7-tiny.yaml --batch-size 384 --test-batch-size 64 --epochs 100 \
    --heads 4 --standard-grouping --head-config data/coco_320_1b4h_standard.yaml

# 1B4H + 幾何分類
python train.py --data data/coco320.yaml --img 320 \
    --cfg cfg/training/yolov7-tiny.yaml --batch-size 64 --epochs 100 \
    --heads 4 --geometry-grouping --head-config data/coco_320_1b4h_geometry.yaml

# 1B8H + 標準分類
python train.py --data data/coco320.yaml --img 320 \
    --cfg cfg/training/yolov7-tiny.yaml --batch-size 64 --epochs 100 \
    --heads 8 --standard-grouping --head-config data/coco_320_1b8h_standard.yaml

# 1B4H + 標準分類 + Attention
python train.py --data data/coco320.yaml --img 320 \
    --cfg cfg/training/yolov7-tiny.yaml --batch-size 64 --epochs 100 \
    --heads 4 --standard-grouping --head-config data/coco_320_1b4h_standard.yaml \
    --head-attention cbam

# 1B4H + 標準分類 + RL 超參數優化
python train.py --data data/coco320.yaml --img 320 \
    --cfg cfg/training/yolov7-tiny.yaml --batch-size 64 --epochs 100 \
    --heads 4 --standard-grouping --head-config data/coco_320_1b4h_standard.yaml \
    --rl-augment

# 1B4H + 標準分類 + 載入 1B1H 預訓練權重 (推薦)
python train.py --weights runs/train/noota_100ep2/weights/best.pt \
    --data data/coco320.yaml --img 320 \
    --cfg cfg/training/yolov7-tiny-1b4h.yaml --batch-size 384 --test-batch-size 64 --epochs 100 \
    --heads 4 --head-config data/coco_320_1b4h_standard.yaml \
    --transfer-weights
```

---

## 5. 訓練策略 (Training Strategy)

### 5.1 權重遷移機制 (Weight Transfer)

為解決 1B4H 架構冷啟動收斂慢的問題，支援從 1B1H 模型遷移權重。

- **來源**: 1B1H 預訓練模型（如 `yolov7-tiny.pt` 或自訓練版本）
- **目標**: 1B4H 模型
- **邏輯**:
  1. 載入來源 `.pt` 檔案
  2. 比對來源與目標的層名稱 (Layer Names) 與形狀 (Shapes)
  3. **保留**: 形狀完全匹配的層（即 Backbone + Neck）
  4. **捨棄**: 形狀不匹配的層（即 Detect Head，因為 1B1H 輸出通道數與 1B4H 不同）
  5. **初始化**: 被捨棄的 Head 層採用預設初始化

### 5.2 預期效果

| 訓練方式 | Epoch 1 mAP | 收斂速度 |
|----------|-------------|----------|
| 從零開始 | ~0.01 | 慢 |
| 權重遷移 | ~0.15+ | 快 |

### 5.3 自動學習率縮放 (Auto LR Scaling)

為充分利用 RTX 5090 大顯存優勢，支援大 Batch Size (如 128, 384) 訓練。

**啟用方式**: 需明確加上 `--auto-lr` 參數才會啟用，預設為關閉。

```bash
# 啟用 Auto LR Scaling
python train.py --batch-size 384 --auto-lr ...

# 不啟用（預設行為，使用原始 LR）
python train.py --batch-size 384 ...
```

#### 問題背景

- 原生 YOLOv7 針對 `nbs = 64` (nominal batch size) 優化
- 若加大 Batch 但不調整 LR，會導致收斂過慢
- 實測: Batch 384 + LR 0.01 需要 400 epochs 才能達到 mAP 0.377

#### 策略: 平方根縮放法則 (Square Root Scaling Rule)

採用平方根縮放法則，相較於線性縮放更穩定，適合多頭架構訓練。

- **基準 (Nominal Batch Size)**: `nbs = 64`
- **公式**: `New LR = Old LR × √(Current Batch / nbs)`
- **Warmup 延長**: 當 `batch_size >= 128` 時，自動將 `warmup_epochs` 延長至 5.0

#### 縮放範例

| Batch Size | Scale Factor | Original LR | Scaled LR | Warmup Epochs |
|------------|--------------|-------------|-----------|---------------|
| 64 | 1.0 | 0.01 | 0.01 | 3.0 |
| 128 | 1.41 | 0.01 | 0.0141 | 5.0 |
| 256 | 2.0 | 0.01 | 0.02 | 5.0 |
| 384 | 2.45 | 0.01 | 0.0245 | 5.0 |

#### 預期效果

| 訓練策略 | Batch | LR | 100ep mAP | 收斂特性 |
|----------|-------|-----|-----------|----------|
| 原生 (無縮放) | 384 | 0.01 | ~0.35 | 慢，需 400+ epochs |
| Auto LR Scaling | 384 | 0.0245 | ~0.38+ | 快，100 epochs 達標 |

---

## 6. 系統架構

### 6.1 原始架構 (Baseline)

```
Input → Backbone (CSPDarknet) → Neck (PANet) → Detect Head → Output
                                                    ↓
                                              80 類預測
```

### 6.2 1B4H 架構

```
Input → Backbone (CSPDarknet) → Neck (PANet) → P3/P4/P5 特徵
                                                    ↓
                                            ┌──────────────┐
                                            │ Loss Router  │
                                            └──────────────┘
                                                    ↓
                         ┌──────────┬──────────┬──────────┬──────────┐
                         ↓          ↓          ↓          ↓
                      Head 0    Head 1     Head 2     Head 3
                     (20類)    (20類)     (20類)     (20類)
                         ↓          ↓          ↓          ↓
                         └──────────┴──────────┴──────────┘
                                            ↓
                                      80 類預測
```

### 6.3 1B4H + Attention 架構（未來）

```
Input → Backbone → Neck → P3/P4/P5 特徵
                              ↓
                    ┌─────────────────┐
                    │   Loss Router   │
                    └─────────────────┘
                              ↓
         ┌────────────┬────────────┬────────────┬────────────┐
         ↓            ↓            ↓            ↓
      CBAM 0       CBAM 1       CBAM 2       CBAM 3
         ↓            ↓            ↓            ↓
      Head 0       Head 1       Head 2       Head 3
         ↓            ↓            ↓            ↓
         └────────────┴────────────┴────────────┘
                              ↓
                        80 類預測
```

### 6.4 推論階段：全域合併 NMS

為確保跨檢測頭 (Cross-Head) 的預測結果一致性，推論階段採用**「全域合併 NMS (Global Concatenation NMS)」**策略。

#### 處理流程

**Step 1: 前向傳播 (Forward)**

4 個 Head 平行運算，各自輸出預測張量：

```
Head 0 Output: [Batch, Anchors_0, 85]
Head 1 Output: [Batch, Anchors_1, 85]
Head 2 Output: [Batch, Anchors_2, 85]
Head 3 Output: [Batch, Anchors_3, 85]
```

**Step 2: 張量拼接 (Concatenation)**

將所有 Head 的輸出在維度 1 (Anchors 維度) 進行拼接：

```
Combined Output: [Batch, Total_Anchors, 85]
```

其中 `Total_Anchors = Anchors_0 + Anchors_1 + Anchors_2 + Anchors_3`

**Step 3: 全域 NMS (Global NMS)**

將拼接後的張量送入標準 YOLOv7 `non_max_suppression` 函數。

#### 優勢

- **跨 Head 去重**：自動處理跨 Head 的重疊框（例如 Head 0 和 Head 2 同時偵測到同一物體）
- **依據 Confidence Score 保留最佳結果**：不需要額外的後處理邏輯
- **相容性**：使用原始 YOLOv7 的 NMS 函數，無需修改

---

## 7. 開發模組

### 7.1 核心模組

| 模組 | 檔案 | 功能 | 階段 |
|------|------|------|------|
| MultiHeadDetect | `models/multihead.py` | 多檢測頭架構 | Phase 1 |
| ComputeLossRouter | `utils/loss_router.py` | 損失路由器 | Phase 1 |
| HeadConfig | `utils/head_config.py` | 設定檔載入 | Phase 1 |
| WeightTransfer | `utils/weight_transfer.py` | 智慧權重遷移 | Phase 1 |
| GeometryGrouping | `utils/geometry_grouping.py` | K-Means 分群 | Phase 2 |
| CBAM | `models/attention.py` | Channel + Spatial Attention | Phase 3 |
| MultiHeadDetectAttention | `models/multihead_attention.py` | Attention 增強多頭檢測 | Phase 3 |
| RLAugment | `utils/rl_augment.py` | RL 超參數優化 | Phase 4 |

### 7.2 設定檔

| 檔案 | 功能 | 階段 |
|------|------|------|
| `data/coco_320_1b4h_standard.yaml` | 320 標準分類設定 | Phase 1 |
| `data/coco_640_1b4h_standard.yaml` | 640 標準分類設定 | Phase 1 |
| `data/coco_320_1b4h_geometry.yaml` | 320 幾何分類設定 | Phase 2 |
| `data/coco_640_1b4h_geometry.yaml` | 640 幾何分類設定 | Phase 2 |
| `cfg/training/yolov7-tiny-1b4h-attn.yaml` | 1B4H + Attention 模型架構 | Phase 3 |

---

## 8. 實施階段

### Phase 1: 標準分類 1B4H 驗證（優先）

- **目標**: 驗證 1B4H 架構有效性
- **環境**: non-OTA + 320x320（快速迭代）
- **對照**: v1.0-baseline (mAP@0.5 = 0.385)
- **產出**: MultiHeadDetect, ComputeLossRouter, 標準分類設定檔

### Phase 2: 幾何分類實作（順序待定）

- **目標**: 實作 K-Means 自動分群
- **產出**: GeometryGrouping 模組, 幾何分類設定檔

### Phase 3: Task-Aware Attention 機制

- **目標**: 解決 Head 受到背景雜訊干擾的問題
- **方法**: 在每個 Head 之前引入 **CBAM (Convolutional Block Attention Module)**
- **架構**: `MultiHeadDetectAttention` 繼承自 `MultiHeadDetect`
- **預期效果**:
  - Head 0 (人物) 的 Attention 會自動過濾掉 Head 1 (車輛) 的特徵
  - 提升各 Head 的 Precision 與 Recall
  - 減少跨類別的 False Positive
- **產出**:
  - `models/attention.py` (CBAM 模組)
  - `models/multihead_attention.py` (MultiHeadDetectAttention)
  - `cfg/training/yolov7-tiny-1b4h-attn.yaml`

### Phase 4: RL 超參數優化（順序待定）

- **目標**: 針對每個 Head 自動搜索最佳增強策略
- **產出**: RLAugment 模組, 各 Head 增強策略

### Phase 5: 正式訓練

- **環境**: OTA + 640x640
- **目標**: 整合所有優化，達成最終精度目標

---

## 9. 風險與緩解

| 風險 | 可能性 | 嚴重性 | 緩解措施 |
|------|--------|--------|----------|
| Head 間類別不平衡 | 高 | 高 | 在 Loss Router 中引入 weight 加權 |
| 多 Head 增加記憶體使用 | 中 | 中 | 優化共享 Backbone 特徵 |
| Attention 導致推論延遲 | 中 | 低 | 使用輕量級 Attention 或 TensorRT 優化 |
| 驗證階段 OOM | 高 | 高 | 使用 `--test-batch-size` 分離訓練/驗證 batch size |

---

## 10. 變更歷史

| 版本 | 日期 | 變更內容 |
|------|------|----------|
| v0.1 | 2025-11-30 | 初版，整合架構設計 |
| v0.2 | 2025-11-30 | 新增向下相容原則、參數設計、設定檔規範、分階段實施計畫 |
| v0.3 | 2025-11-30 | 新增推論階段全域合併 NMS 策略 |
| v0.4 | 2025-11-30 | 新增 `--test-batch-size` 參數解決 1B4H 驗證階段 OOM 問題 |
| v0.5 | 2025-11-30 | 新增 `--transfer-weights` 參數和訓練策略章節（智慧權重遷移）|
| v0.6 | 2025-12-01 | 更新 Phase 3 Task-Aware Attention 設計，新增 CBAM 和 MultiHeadDetectAttention 模組規範 |
| v0.7 | 2025-12-01 | 新增 5.3 自動學習率縮放 (Auto LR Scaling) 策略，採用平方根縮放法則 |
