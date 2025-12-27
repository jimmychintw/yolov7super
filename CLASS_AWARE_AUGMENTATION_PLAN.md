# Stochastic Class-Aware Augmentation 實作計劃

> **文件版本**: v2.0 (已根據審核報告修正)
> **建立日期**: 2025-12-05
> **修訂日期**: 2025-12-05
> **狀態**: 已修正

---

## 修訂記錄

| 版本 | 日期 | 修改內容 |
|------|------|----------|
| v1.0 | 2025-12-05 | 初版 (Weighted Average 方案) |
| v2.0 | 2025-12-05 | 重大修正：改用 Stochastic Round-Robin 策略 |

### v2.0 主要修正

根據審核報告 `Class-Aware Augmentation Plan - 審核與修正建議報告.md` 修正：

1. **廢棄** Weighted Average 方案 (會導致特徵稀釋)
2. **採用** Stochastic Round-Robin 策略
3. **新增** Mosaic-Level Decision 處理
4. **新增** Dynamic Label Filtering (可選)

---

## 1. 專案背景

### 1.1 YOLOv7 1B4H 架構

```
                    ┌─── Head 0 (Tall/垂直物體)
                    │
Image → Backbone ───┼─── Head 1 (Square/方形物體)
                    │
                    ├─── Head 2 (Square/方形物體)
                    │
                    └─── Head 3 (Wide/水平物體)
```

- **1 Backbone**: 共享特徵提取
- **4 Heads**: 依據物體幾何形狀分組，各自負責偵測特定類別

### 1.2 Head 分組配置

來源: `data/coco_320_1b4h_geometry.yaml`

| Head | 名稱 | 類別數 | 代表物體 | Avg Aspect Ratio |
|------|------|--------|----------|------------------|
| Head 0 | Tall (Vertical) | 12 | person, traffic light, fire hydrant | 0.56 |
| Head 1 | Square (Central) | 28 | bicycle, bench, backpack | 0.93 |
| Head 2 | Square (Central) | 26 | car, dog, chair | 1.26 |
| Head 3 | Wide (Horizontal) | 14 | airplane, bus, train | 1.92 |

---

## 2. 已完成工作

### 2.1 Phase 4: RL-based HPO 搜索

使用 Optuna 對每個 Head 進行 30 trials 的超參數搜索。

**搜索配置**:
- Backbone: `backbone_elite_0.435.pt` (預訓練)
- Proxy epochs: 10
- Freeze layers: 50 (只訓練 Head)
- 評估指標: 各 Head 負責類別的 mAP@0.5

**搜索空間**:
| 參數 | 範圍 |
|------|------|
| degrees | 0.0 ~ 45.0 |
| flipud | 0.0 ~ 0.5 |
| fliplr | 0.0 ~ 0.5 |
| shear | 0.0 ~ 5.0 |
| mixup | 0.0 ~ 0.2 |

### 2.2 HPO 搜索結果

**最佳超參數** (已下載至本機):

```
data/hyp.head0.best.yaml  - Head 0 最佳超參數
data/hyp.head1.best.yaml  - Head 1 最佳超參數
data/hyp.head2.best.yaml  - Head 2 最佳超參數
data/hyp.head3.best.yaml  - Head 3 最佳超參數
```

**Optuna 資料庫** (完整搜索歷史):

```
temp/rl_augment_head0.db
temp/rl_augment_head1.db
temp/rl_augment_head2.db
temp/rl_augment_head3.db
```

### 2.3 HPO 結果比較

#### Augmentation 參數對比表

```
┌─────────────┬─────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│    參數     │ Default │   Head 0     │   Head 1     │   Head 2     │   Head 3     │
│             │         │ (Tall/垂直)  │ (Square/方)  │ (Square/方)  │ (Wide/水平)  │
├─────────────┼─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ degrees     │   0.00  │   2.42       │   0.08       │   0.84       │   0.07       │
│ flipud      │   0.00  │   0.01       │   0.08       │   0.03       │   0.04       │
│ fliplr      │   0.50  │   0.39       │   0.26       │   0.23       │   0.19       │
│ shear       │   0.00  │   0.14       │   1.61       │   0.13       │   3.01       │
│ mixup       │   0.05  │   0.03       │   0.17       │   0.09       │   0.01       │
└─────────────┴─────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

#### 關鍵發現

| Head | 物體特性 | 最佳策略 |
|------|----------|----------|
| **Head 0** | 垂直物體 (人、紅綠燈) | 高旋轉 (2.42°)，適度 fliplr |
| **Head 1** | 方形物體 (自行車、背包) | 高 mixup (0.17)，高 shear (1.61°) |
| **Head 2** | 方形物體 (汽車、狗) | 中等平衡設定 |
| **Head 3** | 水平物體 (飛機、公車) | 極低旋轉 (0.07°)，最高 shear (3.01°)，最低 mixup |

---

## 3. 問題分析

### 3.1 核心挑戰

```
現有架構限制:

Image → [單一 Augmentation] → Backbone → 4 Heads
              ↑
        所有 Head 看到相同的增強圖像
```

### 3.2 ~~v1.0 方案缺陷~~ (已廢棄)

~~原方案使用 Weighted Average 混合參數~~

```
❌ 問題: 加權平均導致「特徵稀釋」

例如:
- Head 0 需要 2.42° 旋轉 (訓練旋轉魯棒性)
- Head 1 需要 0.08° 旋轉 (保持水平特性)
- 平均 = 1.25°

結果:
- Head 0: 1.25° 不足以訓練出旋轉魯棒性
- Head 1: 1.25° 已破壞水平特性
→ 兩個 Head 都沒有得到最佳訓練！
```

---

## 4. 方案 C2-v2: Stochastic Class-Aware Augmentation

### 4.1 核心概念

**不做參數混合**，而是根據圖中物體分布「擲骰子」，選擇**其中一套**參數作為當前圖片的策略。

```
v1.0 (廢棄):  混合所有 Head 參數 → 平庸化
v2.0 (採用):  隨機選一個 Head 的參數 → 原汁原味
```

### 4.2 演算法設計

#### 4.2.1 Stochastic Round-Robin 策略

```python
import random

# 各 Head 最佳參數 (來自 HPO)
HEAD_PARAMS = {
    0: {'degrees': 2.42, 'flipud': 0.01, 'fliplr': 0.39, 'shear': 0.14, 'mixup': 0.03},
    1: {'degrees': 0.08, 'flipud': 0.08, 'fliplr': 0.26, 'shear': 1.61, 'mixup': 0.17},
    2: {'degrees': 0.84, 'flipud': 0.03, 'fliplr': 0.23, 'shear': 0.13, 'mixup': 0.09},
    3: {'degrees': 0.07, 'flipud': 0.04, 'fliplr': 0.19, 'shear': 3.01, 'mixup': 0.01},
}

def select_stochastic_policy(head_counts):
    """
    依機率隨機選擇一個 Head 當作「主導者」

    Args:
        head_counts: [3, 2, 0, 0] 表示圖中有 3 個 Head0 物體, 2 個 Head1 物體

    Returns:
        dict: 選中 Head 的「原汁原味」參數
    """
    total = sum(head_counts)

    if total == 0:
        # 無標籤時隨機選
        active_head = random.randint(0, 3)
    else:
        # 依機率選擇 (物體數越多的 Head 越有可能被選中)
        active_head = random.choices(range(4), weights=head_counts, k=1)[0]

    return HEAD_PARAMS[active_head], active_head
```

#### 4.2.2 Mosaic-Level Decision

YOLOv7 使用 Mosaic 增強時，會合併 4 張圖片。策略必須在 Mosaic 層級決定。

```python
def load_mosaic_with_class_aware(self, index):
    """
    修改後的 load_mosaic，支援 Class-Aware Augmentation
    """
    # 原始邏輯: 載入 4 張圖片
    indices = [index] + random.choices(self.indices, k=3)

    # 統計 4 張圖片的總類別分布
    total_head_counts = [0, 0, 0, 0]
    for idx in indices:
        labels = self.labels[idx]
        for label in labels:
            class_id = int(label[0])
            head_id = self.class_to_head.get(class_id, 0)
            total_head_counts[head_id] += 1

    # 決定 Global Policy (對整個 Mosaic 圖)
    policy, active_head = select_stochastic_policy(total_head_counts)

    # ... 原有的 Mosaic 合成邏輯 ...

    # 使用選中的 Policy 進行 random_perspective
    img, labels = random_perspective(
        img, labels,
        degrees=policy['degrees'],
        shear=policy['shear'],
        # ... 其他參數
    )

    return img, labels, active_head  # 回傳 active_head 供後續使用
```

#### 4.2.3 範例計算

```
圖片 A: 包含 3 個 person (Head 0), 2 個 car (Head 2)
  → head_counts = [3, 0, 2, 0]
  → weights = [0.6, 0.0, 0.4, 0.0]
  → 60% 機率選 Head 0 策略 (2.42° 旋轉)
  → 40% 機率選 Head 2 策略 (0.84° 旋轉)
  → 不會得到 1.79° 的折衷值！

圖片 B: 包含 1 個 airplane (Head 3), 1 個 bus (Head 3)
  → head_counts = [0, 0, 0, 2]
  → 100% 選 Head 3 策略 (0.07° 旋轉)

圖片 C: 包含各類物體混合 [2, 2, 2, 2]
  → 25% 機率選任一 Head
  → 每次訓練都是「完整」的某一套策略
```

### 4.3 進階功能: Dynamic Label Filtering (可選)

當選中「大旋轉」策略時，保護其他 Head 的標籤。

```python
def apply_label_filtering(labels, policy, active_head, rotation_threshold=5.0):
    """
    當旋轉角度超過閾值時，過濾掉非主導 Head 的標籤

    目的: 防止 Head 1 (水平物體) 在大旋轉時學習錯誤特徵
    """
    if policy['degrees'] <= rotation_threshold:
        return labels  # 小旋轉不需過濾

    filtered_labels = []
    for label in labels:
        class_id = int(label[0])
        head_id = class_to_head.get(class_id, 0)

        if head_id == active_head:
            # 主導 Head 的標籤保留
            filtered_labels.append(label)
        else:
            # 非主導 Head 的標籤設為 ignore (或移除)
            # 這樣該物體不會參與 Loss 計算
            pass  # 移除

    return np.array(filtered_labels)
```

**效果**:
- 選中 Head 0 策略 (2.42° 旋轉)
- 圖中的 car (Head 2) 標籤被移除
- 車的圖像雖然歪了，但不會被當作正樣本訓練
- Head 2 不會學到「歪的車」

### 4.4 實作計劃

#### 4.4.1 需修改/新增的檔案

| 檔案 | 操作 | 內容 |
|------|------|------|
| `utils/class_aware_augment.py` | **新增** | Stochastic Policy 選擇邏輯 |
| `utils/datasets.py` | **修改** | 整合到 `load_mosaic`, `__getitem__` |
| `train.py` | **修改** | 新增 `--class-aware-aug` 參數 |
| `data/hyp.head_params.yaml` | **新增** | 各 Head 最佳參數配置 |

#### 4.4.2 實作步驟

```
Step 1: 建立 Head 參數配置檔
        → data/hyp.head_params.yaml

Step 2: 實作 StochasticClassAwareAugmentation 類別
        → utils/class_aware_augment.py

Step 3: 修改 load_mosaic
        → utils/datasets.py: 統計 4 張圖的類別，選擇 Policy

Step 4: 修改 random_perspective 調用
        → 傳入選中的 Policy 參數

Step 5: (可選) 實作 Dynamic Label Filtering

Step 6: 新增訓練參數
        → train.py: --class-aware-aug

Step 7: 單元測試
        → 驗證策略選擇邏輯正確
```

### 4.5 配置檔案設計

#### data/hyp.head_params.yaml

```yaml
# Stochastic Class-Aware Augmentation 參數配置
# 來源: HPO 搜索結果 (2025-12-05)
# 策略: 隨機選擇其中一套，不做混合

head_0:  # Tall (Vertical) - person, traffic light, etc.
  degrees: 2.4246
  flipud: 0.0114
  fliplr: 0.3877
  shear: 0.1397
  mixup: 0.0316

head_1:  # Square (Central) - bicycle, bench, etc.
  degrees: 0.0796
  flipud: 0.0770
  fliplr: 0.2581
  shear: 1.6101
  mixup: 0.1749

head_2:  # Square (Central) - car, dog, etc.
  degrees: 0.8416
  flipud: 0.0258
  fliplr: 0.2318
  shear: 0.1311
  mixup: 0.0948

head_3:  # Wide (Horizontal) - airplane, bus, etc.
  degrees: 0.0677
  flipud: 0.0395
  fliplr: 0.1938
  shear: 3.0063
  mixup: 0.0127

# Dynamic Label Filtering 設定
label_filtering:
  enabled: false  # 預設關閉，可選開啟
  rotation_threshold: 5.0  # 旋轉超過此角度時過濾非主導 Head 標籤
```

---

## 5. 風險與考量

### 5.1 「偷看答案」疑慮

| 階段 | 使用資訊 | 是否合理 |
|------|----------|----------|
| HPO 搜索 | val set mAP | ✅ 標準做法 |
| 訓練時 | train labels | ✅ Augmentation 本就可用 labels |
| 推論時 | 無 labels | ✅ 推論不做 augmentation |

**結論**: 這是合理的訓練策略，類似 AutoAugment 方法論。

### 5.2 Stochastic 策略的特性

| 特性 | 說明 |
|------|------|
| **優點** | 每個 Head 都能獲得完整的最佳策略訓練 |
| **優點** | 避免折衷參數導致的「兩頭都不好」 |
| **風險** | 訓練過程可能有較大 variance |
| **緩解** | 長期訓練會收斂；可用 EMA 平滑 |

### 5.3 Dynamic Label Filtering 風險

| 風險 | 緩解措施 |
|------|----------|
| 過度過濾導致樣本不足 | 設定合理的 rotation_threshold |
| 增加實作複雜度 | 預設關閉，驗證基本功能後再開啟 |

---

## 6. 實驗計劃

### 6.1 對照實驗設計

| 實驗 | Augmentation | 說明 |
|------|--------------|------|
| Baseline | Default hyp | 對照基準 |
| Exp-A | Merged hyp | 簡單平均 (參考用) |
| **Exp-B** | **Stochastic Class-Aware** | **主要驗證目標** |
| Exp-C | Stochastic + Label Filtering | 進階功能驗證 |

### 6.2 評估指標

- mAP@0.5 (整體)
- mAP@0.5 per Head (分組)
- 訓練收斂速度
- 各類別 AP 變化
- 訓練 loss variance

### 6.3 訓練配置

```bash
# Baseline
python train.py --hyp data/hyp.scratch.tiny.yaml ...

# Exp-A: Merged (參考)
python train.py --hyp data/hyp.merged.yaml ...

# Exp-B: Stochastic Class-Aware (主要)
python train.py --hyp data/hyp.scratch.tiny.yaml \
                --class-aware-aug \
                --head-params data/hyp.head_params.yaml ...

# Exp-C: + Label Filtering
python train.py --hyp data/hyp.scratch.tiny.yaml \
                --class-aware-aug \
                --head-params data/hyp.head_params.yaml \
                --label-filtering ...
```

---

## 7. 時程規劃

| 階段 | 任務 | 預估工時 |
|------|------|----------|
| 1 | 建立配置檔 (`hyp.head_params.yaml`) | 0.5h |
| 2 | 實作 `StochasticClassAwareAugmentation` 類別 | 1.5h |
| 3 | 修改 `load_mosaic` 統計類別 | 1h |
| 4 | 整合 Policy 到 `random_perspective` | 1h |
| 5 | 修改 `train.py` 參數 | 0.5h |
| 6 | 單元測試 | 1h |
| 7 | (可選) Dynamic Label Filtering | 1.5h |
| 8 | 對照實驗 | 依 GPU 時數 |

---

## 8. 待確認事項

- [x] ~~是否先跑 Merged (Exp-A) 作為快速驗證？~~ → 建議同時跑
- [x] ~~Class-Aware 權重是否需要 smoothing？~~ → v2.0 改用 Stochastic，不需要
- [x] ~~是否需要支援 mosaic 的 class-aware？~~ → 是，已納入設計
- [ ] 是否啟用 Dynamic Label Filtering？
- [ ] 實驗要跑多少 epochs？

---

## 9. 附錄

### A. 相關檔案路徑

```
/Users/jimmy/Projects/Yolov7fast/
├── data/
│   ├── hyp.scratch.tiny.yaml      # Default 超參數
│   ├── hyp.head0.best.yaml        # Head 0 HPO 結果
│   ├── hyp.head1.best.yaml        # Head 1 HPO 結果
│   ├── hyp.head2.best.yaml        # Head 2 HPO 結果
│   ├── hyp.head3.best.yaml        # Head 3 HPO 結果
│   └── coco_320_1b4h_geometry.yaml # Head 分組配置
├── temp/
│   ├── rl_augment_head0.db        # Optuna 完整記錄
│   ├── rl_augment_head1.db
│   ├── rl_augment_head2.db
│   └── rl_augment_head3.db
└── utils/
    ├── datasets.py                 # 待修改
    └── rl_augment.py              # HPO 搜索腳本
```

### B. 審核歷史

- **v1.0 審核**: 發現 Weighted Average 缺陷
- **v2.0 修正**: 改用 Stochastic Round-Robin 策略

### C. 參考資料

- [AutoAugment: Learning Augmentation Strategies from Data](https://arxiv.org/abs/1805.09501)
- [RandAugment: Practical Automated Data Augmentation](https://arxiv.org/abs/1909.13719)
- Optuna Documentation: https://optuna.readthedocs.io/
- 審核報告: `Class-Aware Augmentation Plan - 審核與修正建議報告.md`

---

**文件結束**
