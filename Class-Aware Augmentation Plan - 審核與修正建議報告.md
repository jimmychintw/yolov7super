# Class-Aware Augmentation Plan - 審核與修正建議報告

審核日期: 2025-12-05

審核對象: CLASS_AWARE_AUGMENTATION_PLAN.md (v1.0)

## 1. 總體評價

- **優點**: 架構清晰，RL 搜索結果分析透徹，準確識別了「單一增強策略」的瓶頸。
- **缺點**: 核心演算法 (Weighted Average) 存在邏輯缺陷，可能導致「所有 Head 都吃不飽」的平庸化結果。且未充分考慮 Mosaic 增強的實作限制。

## 2. 重大缺陷 (Critical Issues)

### 🔴 缺陷 1: 加權平均 (Weighted Average) 導致特徵稀釋

問題描述:

文檔建議將不同 Head 的最佳參數進行加權平均。例如 Head 0 需要 2.42° 旋轉，Head 1 需要 0.08°。如果兩者同時出現，平均結果約 1.25°。

影響:

- 對於 Head 0：`1.25°` 不足以訓練出對旋轉的魯棒性。
- 對於 Head 1：`1.25°` 已經破壞了剛體的水平特性。
- **結果**: 兩個 Head 都沒有得到最佳訓練，甚至 Head 1 被迫學習錯誤特徵。

✅ 修正方案: 隨機輪盤策略 (Stochastic Round-Robin)

不進行參數混合，而是根據權重 「擲骰子」 選擇其中 一套 參數作為當前 Batch 的策略。

```
import random

# 修正後的邏輯示意
def select_aug_policy(head_counts):
    # head_counts: [3, 2, 0, 0] (3人, 2車)
    # weights: [0.6, 0.4, 0.0, 0.0]
    
    # 依機率隨機選擇一個 Head 當作「主導者」
    # random.choices 回傳的是 list，取 [0]
    active_head = random.choices(range(4), weights=head_counts, k=1)[0]
    
    # 直接回傳該 Head 的「原汁原味」參數，不打折
    # HEAD_PARAMS 是預先載入的 dict
    return HEAD_PARAMS[active_head]
```

### 🔴 缺陷 2: 忽略 Mosaic/Mixup 的全局性

問題描述:

YOLOv7 的 augment_hsv 和 random_perspective (旋轉/剪切) 通常是在 load_mosaic 之後對 整張合成圖 進行的。

LoadImagesAndLabels 的 __getitem__ 邏輯需要處理的是一組索引，而不僅僅是單張圖。

✅ 修正方案: Mosaic-Level Decision

必須在 load_mosaic 內部或調用前，統計這 4 張圖片的總類別分佈，決定出一個 Global Policy，然後應用於 random_perspective。

## 3. 進階建議 (Optional Improvements)

### 💡 建議 1: 動態標籤過濾 (Dynamic Label Filtering)

為了保護 Head 1 (車) 不受 Head 0 (人) 的大旋轉策略傷害。

**邏輯**:

- 當系統選中 **Head 0 策略** (執行大旋轉) 時。
- 檢查圖片中的 **Head 1 類別 (車)**。
- 如果旋轉角度 > 閾值 (例如 5度)，則將這些車的標籤 **移除** (或設為 ignore)。
- **效果**: 雖然圖裡的車歪了，但 Loss Router 不會把它當作正樣本訓練，而是當作背景。這能有效防止 Head 1 學壞。

### 💡 建議 2: 參數平滑化

如果擔心隨機策略跳動太大，可以使用 `EMA (Exponential Moving Average)` 更新全域參數，但這會增加複雜度，建議優先使用隨機策略。

## 4. 修正後的實作路徑 (Action Items)

1. **廢棄** `compute_weighted_params` 函數。
2. **實作** `select_stochastic_policy` (隨機選擇) 函數。
3. **修改 `utils/datasets.py`**:
   - 在 `load_mosaic` 中統計 4 張圖的總類別。
   - 決定 Policy。
   - 將 Policy 傳入 `random_perspective`。
4. **更新文檔**：將方案 C2 改為 **"Stochastic Class-Aware Augmentation"**。