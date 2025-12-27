# 1B4H 漸進式解凍訓練計畫

*更新時間: 2025-12-05*

---

## 最新數據分析 (2025-12-05)

### 成長動能對比

| 架構 | 訓練方式 | 每 50ep 成長率 | 最終 mAP@0.5 |
|------|----------|----------------|--------------|
| 1B1H | From Scratch | **+1-3%** | **0.4335** |
| 1B4H | Transfer (freeze 50) | +0.1-0.2% | 0.4293 |

### mAP@0.5 里程碑對比

| Epoch | 1B1H | 1B4H 幾何 | 差距 | 說明 |
|-------|------|-----------|------|------|
| 50 | 0.3136 | **0.4181** | -10.5% | 1B4H 起點高 |
| 100 | 0.3514 | **0.4205** | -6.9% | 1B1H 追趕中 |
| 200 | 0.3806 | **0.4228** | -4.2% | 差距縮小 |
| 300 | 0.4046 | **0.4244** | -2.0% | 繼續追趕 |
| 400 | **0.4274** | 0.4270 | **交叉點** | 1B1H 反超！ |
| 499 | **0.4335** | 0.4293 | **+0.4%** | 1B1H 勝出 |

### 階段成長率對比

| 階段 | 1B1H 成長 | 1B4H 幾何成長 |
|------|-----------|---------------|
| 50→100 | **+3.8%** | +0.2% |
| 100→150 | **+1.6%** | +0.2% |
| 150→200 | **+1.3%** | +0.0% |
| 200→250 | **+1.1%** | +0.1% |
| 250→300 | **+1.3%** | +0.0% |
| 300→350 | **+1.1%** | +0.2% |
| 350→400 | **+1.2%** | +0.1% |
| 400→450 | +0.6% | +0.2% |
| 450→499 | +0.1% | +0.0% |

### 關鍵發現

```
1B1H:  起點低 (0.31) → 動能強 → 最終高 (0.4335)
1B4H:  起點高 (0.42) → 動能弱 → 最終低 (0.4293)
```

**交叉點在 Epoch ~400**，之後 1B1H 反超！

**根本原因**：1B4H freeze 50 層導致 Backbone 無法針對 4-Head 架構優化。

---

## 策略選項

### 策略 A：階段式解凍 (Gradual Unfreeze) - 原方案

```
Stage 1: Epoch 0-100,   freeze 50 層
Stage 2: Epoch 100-200, freeze 30 層
Stage 3: Epoch 200-300, freeze 10 層
Stage 4: Epoch 300-500, freeze 0 層
```

**優點**：漸進式，風險較低
**缺點**：需要多次 resume 訓練

### 策略 B：從 500ep 續訓 + 完全解凍

```bash
python train.py \
    --weights runs/train/1b4h_geo_500ep/weights/last.pt \
    --freeze 0 \           # 完全解凍
    --epochs 300 \         # 額外 300 epochs
    --lr0 0.001 \          # 較低起始 LR
    ...
```

**優點**：簡單直接
**風險**：可能 catastrophic forgetting

### 策略 C：From Scratch 1B4H (最激進)

```bash
python train.py \
    --weights '' \         # 不使用預訓練
    --freeze 0 \           # 不凍結
    --epochs 800 \         # 更長訓練
    ...
```

**優點**：動能應該與 1B1H 相當
**缺點**：訓練時間長，起點低

### 策略 D：Warm Restart (推薦嘗試)

```bash
python train.py \
    --weights runs/train/1b4h_geo_500ep/weights/best.pt \
    --freeze 20 \          # 只凍結前 20 層
    --epochs 200 \
    --lr0 0.005 \          # 中等 LR
    ...
```

**優點**：平衡風險與收益

---

## 建議優先順序

| 優先級 | 策略 | 預期提升 | 時間成本 | 風險 |
|--------|------|----------|----------|------|
| 1 | **D: Warm Restart (freeze 20)** | +0.5-1% | 200 ep | 低 |
| 2 | B: 續訓 + freeze 0 | +0.3-0.8% | 300 ep | 中 |
| 3 | A: 階段式解凍 | +0.5-1% | 需重跑 | 低 |
| 4 | C: From Scratch | +0.5%? | 800 ep | 高 |

---

## 問題背景

目前 1B4H 實驗都使用 `--freeze 50` 凍結 backbone，結果：
- **優點**：快速收斂到 mAP ≈ 0.42
- **缺點**：天花板低，無法超越 1B1H 的 0.4335

直接 unfreeze 會導致 mAP 從 0.42 暴跌到 0.14，因為 learning rate 太大破壞了已學好的權重。

## 解決方案：漸進式解凍

從靠近 Head 的深層開始，逐步解凍更淺的層，每階段使用更低的 learning rate。

```
YOLOv7-tiny 層結構：
┌─────────────────────────────────────────────────────────────┐
│  層 0-10:   淺層特徵（邊緣、顏色、紋理）    ← Stage 4 解凍  │
│  層 11-30:  中層特徵（形狀、部件）          ← Stage 3 解凍  │
│  層 31-50:  深層特徵（物體、語意）          ← Stage 2 解凍  │
│  層 51+:    Head 層（分類、回歸）           ← Stage 1 已解凍 │
└─────────────────────────────────────────────────────────────┘
```

---

## 訓練計畫

### 總覽

| Stage | Freeze | lr0 | Epochs | 累計 | 說明 |
|-------|--------|-----|--------|------|------|
| 1 | 50 | 0.01 | 100 | 100 | 只訓練 Head，快速收斂 |
| 2 | 30 | 0.001 | 100 | 200 | 解凍深層 (31-50)，適應 4H |
| 3 | 10 | 0.0005 | 100 | 300 | 解凍中層 (11-30) |
| 4 | 0 | 0.0001 | 200 | 500 | 全層微調 |

### Hyperparameter 檔案

| Stage | 檔案 | lr0 |
|-------|------|-----|
| 1 | `hyp.scratch.tiny.noota.yaml` | 0.01 |
| 2 | `hyp.scratch.tiny.noota.stage2.yaml` | 0.001 |
| 3 | `hyp.scratch.tiny.noota.stage3.yaml` | 0.0005 |
| 4 | `hyp.scratch.tiny.noota.stage4.yaml` | 0.0001 |

---

## 執行指令

### Stage 2 從目前 500ep 結果續訓 (推薦)

```bash
cd /workspace/Yolov7fast && source venv/bin/activate && \
python3 train.py \
    --img-size 320 320 \
    --batch-size 64 \
    --test-batch-size 64 \
    --epochs 200 \
    --weights runs/train/20251205_1b4h_geo_classaug_f50_500ep2/weights/best.pt \
    --freeze 20 \
    --data data/coco320.yaml \
    --cfg cfg/training/yolov7-tiny-1b4h.yaml \
    --hyp data/hyp.scratch.tiny.noota.yaml \
    --device 0 \
    --workers 16 \
    --project runs/train \
    --name 20251206_1b4h_geo_classaug_f20_200ep_stage2 \
    --noautoanchor \
    --cache-images \
    --heads 4 \
    --head-config data/coco_320_1b4h_geometry.yaml \
    --class-aware-aug \
    --head-params data/hyp.head_params.yaml
```

---

## 預期結果

| Stage | 預期 mAP | 說明 |
|-------|----------|------|
| Stage 1 (freeze 50) | 0.429 | 目前 500ep 結果 |
| Stage 2 (freeze 20) | 0.435+ | 解凍更多層，應該超越 1B1H |
| Stage 3 (freeze 10) | 0.440+ | 接近最佳 |
| Stage 4 (freeze 0) | **0.445+** | 全層微調 |

---

## 成功指標

| 階段 | 目標 mAP@0.5 | vs 1B1H |
|------|--------------|---------|
| 目前 (500ep freeze50) | 0.4293 | -0.4% |
| Stage 2 (200ep freeze20) | 0.435+ | 持平 |
| Stage 3 (200ep freeze10) | 0.440+ | **+0.7%** |

---

## 監控指標

每個 Stage 結束時檢查：

1. **mAP 是否上升**：若下降超過 0.01，可能 lr 太大
2. **Loss 曲線**：應該平穩下降，不應該有跳動
3. **各 Head 的 AP**：確保沒有某個 Head 崩掉

---

*計畫建立時間: 2025-12-02*
*更新時間: 2025-12-05 - 加入成長動能分析*
