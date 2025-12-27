# 專案進度報告

## 目前狀態：策略 A 實作完成，準備對比實驗 ⭐

---

## 策略 A（停火協議）實作完成 (2025-12-23)

### 功能說明

**核心概念**：「那是別人的獵物，我不要開槍」
- 當一個 Head 看到不屬於自己的物體時，**不當作負樣本訓練**
- 消除梯度衝突，提升 Backbone 學習效率
- 目標：讓 1B4H 突破 plateau，超越 1B1H (0.4353)

### 實作狀態

✅ train.py 新增 `--ignore-other-heads` 參數
✅ ComputeLossRouter 實作 ignore_mask 機制
✅ 單元測試全部通過
✅ 向後相容（Default = False）

### 對比實驗指令

**實驗組（策略 A）**：
```bash
python train.py --img-size 320 320 --batch-size 64 --epochs 500 \
    --weights runs/train/20251201_1b1h_500ep_bs128/weights/best.pt \
    --transfer-weights --freeze 50 \
    --data data/coco320.yaml --cfg cfg/training/yolov7-tiny-1b4h.yaml \
    --hyp data/hyp.scratch.tiny.noota.yaml --device 0 --workers 16 \
    --project runs/train --name 1b4h_strategy_a_500ep \
    --noautoanchor --cache-images --heads 4 \
    --head-config data/coco_320_1b4h_anticonfusion.yaml \
    --ignore-other-heads
```

**對照組（原有模式）**：
```bash
python train.py --img-size 320 320 --batch-size 64 --epochs 500 \
    --weights runs/train/20251201_1b1h_500ep_bs128/weights/best.pt \
    --transfer-weights --freeze 50 \
    --data data/coco320.yaml --cfg cfg/training/yolov7-tiny-1b4h.yaml \
    --hyp data/hyp.scratch.tiny.noota.yaml --device 0 --workers 16 \
    --project runs/train --name 1b4h_original_500ep \
    --noautoanchor --cache-images --heads 4 \
    --head-config data/coco_320_1b4h_anticonfusion.yaml
```

**差異**：只有 `--ignore-other-heads` 一個參數

### 預期結果

| 訓練 | 預期 mAP@0.5 | 說明 |
|------|-------------|------|
| 1B1H Baseline | 0.4353 | 當前冠軍 |
| Hybrid (原有) | 0.4320 | 最佳 1B4H，plateau |
| **Strategy A** | **0.44+** | 目標：突破 plateau |

### 相關檔案

- 實作：`utils/loss_router.py`, `train.py`
- 測試：`temp/test_strategy_a.py`
- 說明：`temp/strategy_a_usage.md`

---

## Freeze 策略比較實驗 (2025-12-04)

### 目標
比較不同 freeze 層數對 1B4H 訓練效果的影響，找出最佳訓練策略

### 訓練中

| 伺服器 | 訓練 | 進度 | mAP@0.5 | 狀態 |
|--------|------|------|---------|------|
| **9950** | Unfreeze f74→f50 200ep | 0/200 | - | 🆕 剛啟動 |
| **99502** | freeze29 200ep | 42/200 | 0.3959 | 🔄 進行中 |

### 已完成

| 訓練 | Epochs | mAP@0.5 | mAP@0.5:0.95 | P | R |
|------|--------|---------|--------------|---|---|
| **1B1H Baseline** | 500 | **0.4335** | **0.2661** | 0.6321 | 0.3986 |
| freeze74 (Head only) | 200 | 0.4251 | 0.2554 | 0.5939 | 0.4025 |
| freeze50 (部分Neck+Head) | 200 | 0.4282 | 0.2537 | 0.5779 | 0.4139 |

### 關鍵發現

1. **freeze74/50 天花板**：200ep 後 mAP@0.5 約 0.425，距離 1B1H (0.4335) 還差 ~0.008
2. **Gradual Unfreeze 策略**：從 freeze74 best.pt 繼續，解凍到 freeze50，使用更低 LR (0.005)
3. **freeze29 收斂較慢**：可訓練參數多 (70%)，需要更多 epochs

### Hyp 配置

| 檔案 | lr0 | lrf | 用途 |
|------|-----|-----|------|
| `hyp.gradual_unfreeze_f74to50.yaml` | 0.005 | 0.01 | freeze74→freeze50 |
| `hyp.gradual_unfreeze_f74to29.yaml` | 0.001 | 0.1 | freeze74→freeze29 |

---

## 訓練動態分析報告 (2025-12-03)

### 已完成分析產出

| 檔案 | 說明 |
|------|------|
| `temp/analysis_mAP_comparison.png` | mAP@0.5 曲線對比圖 |
| `temp/analysis_loss_comparison.png` | 四種 Loss 對比圖 |
| `temp/analysis_mAP_gain_rate.png` | 每 100 epochs mAP 增量柱狀圖 |
| `temp/analysis_precision_recall.png` | Precision/Recall 曲線 |
| `temp/analysis_summary.csv` | 摘要 CSV |
| `temp/analysis_full_epochs.csv` | 完整 epoch 數據 CSV |
| `temp/training_analysis.py` | 分析腳本 |

### 訓練結果摘要

| Experiment | Epochs | mAP@100 | Best mAP@0.5 | Best Epoch | Final mAP |
|------------|--------|---------|--------------|------------|-----------|
| 1B1H 500ep | 500 | 0.3508 | **0.4353** | 481 | 0.4335 |
| Geo 200ep | 200 | 0.4232 | 0.4283 | 193 | 0.4282 |
| Hybrid 500ep | 500 | 0.4206 | 0.4320 | 445 | 0.4310 |
| Anti 500ep | 500 | 0.4231 | 0.4317 | 459 | 0.4309 |

### 關鍵發現

1. **1B4H 初期收斂快**：ep10 達到 0.41+，比 1B1H 快 10 倍
2. **1B4H plateau 問題**：ep100 後 mAP 增量 < 0.005/100ep
3. **1B1H 持續成長**：每 100 epochs 穩定增加 0.02-0.03
4. **Anti Obj Loss 最低**：0.14 vs 0.21-0.23，定位更精準
5. **最終 1B1H 勝出**：0.4353 > Hybrid 0.4320 > Anti 0.4317 > Geo 0.4283

---

## 伺服器訓練狀態 (2025-12-23)

### 285 (ssh -p 45897 root@173.239.88.241 -L 8080:localhost:8080)
| 訓練 | 進度 | mAP@0.5 | 狀態 |
|------|------|---------|------|
| - | - | - | ⏳ 待執行策略 A 對比實驗 |

### 已淘汰伺服器
- 7950, 9950, 99502（所有訓練已完成並同步）

---

## 待執行計畫

### 1B4H 漸進式解凍訓練（待執行）⭐ 重點實驗

**目的**：突破 freeze backbone 的天花板限制，讓 1B4H 超越 1B1H (0.4353)

**問題**：
- Freeze 50 層：快速收斂到 0.42，但天花板低
- 直接 Unfreeze：mAP 從 0.42 暴跌到 0.14（lr 太大）

**解法**：漸進式解凍 + 遞減 learning rate

| Stage | Freeze | lr0 | Epochs | 累計 | 預期 mAP |
|-------|--------|-----|--------|------|----------|
| 1 | 50 | 0.01 | 100 | 100 | 0.42 |
| 2 | 30 | 0.001 | 100 | 200 | 0.43 |
| 3 | 10 | 0.0005 | 100 | 300 | 0.435 |
| 4 | 0 | 0.0001 | 200 | 500 | **0.44+** |

**Hyp 檔案**：
- Stage 1: `hyp.scratch.tiny.noota.yaml` (lr=0.01)
- Stage 2: `hyp.scratch.tiny.noota.stage2.yaml` (lr=0.001)
- Stage 3: `hyp.scratch.tiny.noota.stage3.yaml` (lr=0.0005)
- Stage 4: `hyp.scratch.tiny.noota.stage4.yaml` (lr=0.0001)

**完整計畫**：[GRADUAL_UNFREEZE_PLAN.md](GRADUAL_UNFREEZE_PLAN.md)

---

### 1B4H AntiConfusion 分類訓練 ✅ 已完成

基於 1B1H 500ep 混淆矩陣分析，設計「反混淆」分類策略。

**設定檔**: `data/coco_320_1b4h_anticonfusion.yaml`

**分類特點**:
| Head | 類別數 | 樣本數 | 佔比 | 說明 |
|------|--------|--------|------|------|
| Head 0 | 1 | 262,465 | 30.5% | person 專用 |
| Head 1 | 26 | 199,806 | 23.2% | car, motorcycle, bird, couch... |
| Head 2 | 26 | 198,854 | 23.1% | bus, cat, sheep, chair, knife... |
| Head 3 | 27 | 198,876 | 23.1% | bicycle, truck, dog, cow, fork... |

**核心優勢**: 19 對高混淆類別全部分開（car↔truck, cat↔dog, fork↔knife 等）

**訓練指令**:
```bash
python train.py --img-size 320 320 --batch-size 64 --epochs 500 \
    --weights runs/train/20251201_1b1h_500ep_bs128/weights/best.pt \
    --transfer-weights --freeze 50 \
    --data data/coco320.yaml --cfg cfg/training/yolov7-tiny-1b4h.yaml \
    --hyp data/hyp.scratch.tiny.noota.yaml --device 0 --workers 16 \
    --project runs/train --name 1b4h_anticonfusion_500ep \
    --noautoanchor --cache-images --heads 4 \
    --head-config data/coco_320_1b4h_anticonfusion.yaml
```

**相關文件**: [混淆矩陣與分類策略.md](混淆矩陣與分類策略.md)

---

### 1B4H Hybrid Balanced 監控（進行中 @9950）

**目前狀態**: ep 297/500，mAP@0.5 = 0.4258

**診斷結果** (2025-12-02):
- 最近 30 epochs mAP 波動僅 0.002
- 趨勢斜率 ≈ 0（無上升趨勢）
- Loss 仍緩慢下降，但 mAP 不動
- **判斷：深陷泥淖 (plateau)**

**監控計畫**: 讓它跑到 ep 220，觀察是否突破

| 情況 | mAP@0.5 @ ep220 | 行動 |
|------|-----------------|------|
| 突破 | > 0.430 | 繼續跑到 500 |
| 小幅上升 | 0.425-0.430 | 考慮繼續 |
| 持平 | 0.420-0.425 | 停止，換跑 AntiConfusion |
| 下降 | < 0.420 | 停止，已過擬合 |

---

## 1B4H 訓練結果比較 (2025-12-02)

| 訓練名稱 | 分類方式 | Epochs | Best mAP@0.5 | 狀態 |
|----------|----------|--------|--------------|------|
| 1B1H 500ep | 無分類 | 500 | **0.4353** | ✅ 完成 |
| 1B4H Standard | 語意分類 | 100 | 0.4263 | ✅ 完成 |
| 1B4H Geometry | 幾何分類 | 200 | 0.4283 | ✅ 完成 |
| 1B4H Hybrid Balanced | 混合分類 | 500 | 訓練中 | 🔄 進行中 |
| **1B4H AntiConfusion** | **反混淆分類** | 500 | - | ⏳ 待執行 |

### Epoch 100 公平比較

| 分類方式 | mAP@0.5 @ ep100 | vs 1B1H |
|----------|-----------------|---------|
| 1B4H Standard | 0.4259 | +21.4% |
| 1B4H Geometry | 0.4232 | +20.6% |
| 1B4H Hybrid | 0.4206 | +19.9% |
| 1B1H | 0.3508 | baseline |

---

### 1B4H 訓練初步結果 (2025-11-30)

| Epoch | OTA | non-OTA | 1B4H non-OTA | 1B4H vs non-OTA |
|-------|-----|---------|--------------|-----------------|
| 10 | 0.247 | 0.216 | 0.144 | 67% |
| 17 | 0.279 | 0.273 | 0.191 | 70% |

**觀察**: 1B4H 從零開始訓練收斂較慢，mAP 約為 baseline 的 70%。需要實作 `--transfer-weights` 從預訓練模型載入 Backbone/Neck。

### Baseline 訓練結果對比 (2025-11-28)

| 版本 | 訓練時間 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 狀態 |
|------|---------|---------|--------------|-----------|--------|------|
| **non-OTA** | 1.81 小時 | 0.385 | 0.226 | **0.568** | 0.355 | ✅ 完成 |
| **OTA** | 10.65 小時 | **0.414** | **0.251** | 0.558 | **0.400** | ✅ 完成 |

詳細分析請見 [OTA_ANALYSIS_REPORT.md](OTA_ANALYSIS_REPORT.md)

---

## 訓練詳細結果

### non-OTA 訓練結果 (noota_100ep2)

- **訓練目錄**：`runs/train/noota_100ep2`
- **最終 mAP@0.5**：0.385
- **最終 mAP@0.5:0.95**：0.226
- **Precision**：0.568
- **Recall**：0.355
- **訓練時間**：1.81 小時
- **訓練速度**：~5.76 it/s
- **GPU 利用率**：~90%

### OTA 訓練結果 (ota_100ep4)

- **訓練目錄**：`runs/train/ota_100ep4`
- **最終 mAP@0.5**：0.414
- **最終 mAP@0.5:0.95**：0.251
- **Precision**：0.558
- **Recall**：0.400
- **訓練時間**：10.65 小時
- **訓練速度**：~0.97 it/s
- **GPU 利用率**：~13%

---

## 效能分析結果 (2025-11-27)

**ComputeLossOTA 是訓練緩慢的主要原因**

| 指標 | ComputeLossOTA | ComputeLoss | 改善 |
|------|----------------|-------------|------|
| **loss計算** | 1016.03 ms | **11.22 ms** | **90.5x 更快** |
| forward | 50.20 ms | 50.24 ms | 相同 |
| backward | 103.39 ms | 103.60 ms | 相同 |
| **Total/iter** | 1174.06 ms | **170.89 ms** | **6.9x 更快** |
| **GPU利用率** | 13.1% | **90.0%** | 從超低變正常 |

---

## 已完成項目

| 日期 | 項目 | 說明 |
|------|------|------|
| 2025-11-25 | 專案初始化 | 建立 Python 3.12 venv |
| 2025-11-25 | GitHub 設定 | 建立遠端倉庫 jimmychintw/Yolov7fast |
| 2025-11-25 | 基礎程式碼 | 從 jimmychintw/yolov7 複製 YOLOv7 原始碼 |
| 2025-11-25 | 開發規範 | 建立 CLAUDE.md 定義開發規則 |
| 2025-11-25 | 進度追蹤 | 建立 progress.md 進度報告機制 |
| 2025-11-26 | COCO 資料集 | 確認本地有 320x320 版本 (5.9GB, 118287 張) |
| 2025-11-26 | 多解析度支援 | 建立 coco320/480/640 目錄結構與設定檔 |
| 2025-11-26 | 文檔更新 | 重寫 README.md 為簡潔版本 |
| 2025-11-27 | vast.ai 環境 | 新 instance 設定完成 (RTX 5090 + PyTorch 2.8.0) |
| 2025-11-27 | 程式碼修正 | 修正 test.py 硬編碼 annotations 路徑問題 |
| 2025-11-27 | 設定文檔 | 重寫 VAST_SETUP.md 為一鍵設定指南 |
| 2025-11-27 | 效能分析 | 找出 ComputeLossOTA 是瓶頸 |
| 2025-11-28 | PyTorch 2.8 相容 | 修正所有 torch.load 加入 weights_only=False |
| 2025-11-28 | non-OTA 訓練 | 完成 100 epochs，mAP@0.5 = 0.385 |
| 2025-11-28 | OTA 訓練 | 完成 100 epochs，mAP@0.5 = 0.414 |
| 2025-11-28 | 分析報告 | 撰寫完整 OTA vs non-OTA 分析報告 |
| 2025-11-29 | 新主機設定 | 租用新 vast.ai instance，完成環境設定 |
| 2025-11-29 | 虛擬環境 | 建立 venv，更新 VAST_SETUP.md 加入虛擬環境步驟 |
| 2025-11-29 | 資料同步 | 上傳 coco.zip (4.8GB)，同步 runs/ 訓練結果 (1.3GB) |

---

## vast.ai 遠端環境

```
SSH: ssh -p 45897 root@173.239.88.241 -L 8080:localhost:8080
主機名: 285
GPU: RTX 5090 (32GB VRAM)
PyTorch: 2.8.0+cu128 (支援 Blackwell sm_120)
venv: /workspace/Yolov7fast/venv
tmux session: vast (4 windows: train, cpu, gpu, terminal)
```

---

## 訓練指令參考

```bash
# 在 vast.ai 遠端執行
cd /workspace/Yolov7fast

# non-OTA (快速訓練，~1.8 小時)
python train.py --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml \
    --hyp data/hyp.scratch.tiny.noota.yaml --batch-size 64 --epochs 100 \
    --weights '' --noautoanchor

# OTA (標準訓練，~10-12 小時)
python train.py --data data/coco320.yaml --img 320 --cfg cfg/training/yolov7-tiny.yaml \
    --hyp data/hyp.scratch.tiny.yaml --batch-size 64 --epochs 100 \
    --weights '' --noautoanchor
```

---

## 變更歷史

### 2025-12-03 (訓練動態分析)
- 完成 1B1H vs 1B4H (Geo/Hybrid/Anti) 訓練動態分析
- 生成分析圖表：mAP 曲線、Loss 曲線、mAP 增量、P/R 曲線
- 生成 CSV 數據：summary.csv, full_epochs.csv
- 建立分析腳本 `temp/training_analysis.py`
- 確認 AntiConfusion 500ep 已完成 (7950)
- 確認 Geometry 500ep 進行中 ep404/500 (9950)
- 關鍵發現：1B4H freeze 50 層導致 plateau，1B1H 最終勝出 0.4353

### 2025-12-02 (混淆矩陣分析與反混淆分類)
- 分析 1B1H 500ep 混淆矩陣，識別 19 對高混淆類別
- 設計「反混淆」分類策略，確保混淆類別分到不同 Head
- 建立 `data/coco_320_1b4h_anticonfusion.yaml` 設定檔
- 撰寫 `混淆矩陣與分類策略.md` 分析報告
- 建立四組訓練比較圖 `training_4_comparison.png`
- 更新 progress.md 加入待執行計畫

### 2025-11-30 (1B4H Phase 1 實作)
- 建立 PRD v0.3 和 SDD v1.0 規格文件
- 完成 Phase 1 實作計畫 (IMPLEMENTATION_PLAN_PHASE1.md)
- **新增模組:**
  - `utils/head_config.py` - HeadConfig 設定檔解析模組
  - `models/multihead.py` - MultiHeadDetect 多頭檢測層
  - `utils/loss_router.py` - ComputeLossRouter 損失路由器
- **新增設定檔:**
  - `data/coco_320_1b4h_standard.yaml` - 標準分類設定 (4 Heads x 20 類)
  - `cfg/training/yolov7-tiny-1b4h.yaml` - 1B4H 模型架構
- **修改檔案:**
  - `train.py` - 新增 --heads, --head-config 參數
  - `models/yolo.py` - 支援 MultiHeadDetect
- **新增測試:**
  - `tests/test_1b4h.py` - 單元測試 (UT-01 ~ UT-05)
- **待執行:** 在 vast.ai 上執行單元測試和整合測試

### 2025-11-29
- 租用新 vast.ai instance (RTX 5090)
- 執行一鍵設定腳本，安裝 PyTorch 2.8.0+cu128
- 建立虛擬環境 /workspace/Yolov7fast/venv
- 更新 VAST_SETUP.md 加入虛擬環境建立步驟
- 上傳 coco.zip (4.8GB) 到 Server
- 同步 runs/ 訓練結果 (1.3GB, 10 個實驗)
- GitHub 版本同步（以本機為準，force push）

### 2025-11-28 (下午 - 分析報告)
- OTA 100 epochs 訓練完成，mAP@0.5 = 0.414
- 撰寫完整 OTA vs non-OTA 分析報告 (OTA_ANALYSIS_REPORT.md)
- 更新 progress.md 加入完整對比結果

### 2025-11-28 (凌晨)
- non-OTA 100 epochs 訓練完成，mAP@0.5 = 0.385
- 修正 PyTorch 2.8 相容性問題 (torch.load weights_only)
- 啟動 OTA 100 epochs 訓練

### 2025-11-27 (下午 - 效能分析)
- 建立效能分析計劃 PERFORMANCE_ANALYSIS_PLAN_V2.md
- 建立 CUDA Events 剖析工具 tests/profile_training_loop.py
- 發現 ComputeLossOTA 佔用 86.5% 訓練時間
- 建立 hyp.scratch.tiny.noota.yaml (關閉 OTA Loss)
- 驗證：關閉 OTA 後訓練速度提升 6.9x，GPU 利用率從 13% 提升到 90%

### 2025-11-27 (上午)
- 租用新 vast.ai instance (RTX 5090)
- 設定 SSH key 連線
- 安裝 PyTorch 2.8.0 + CUDA 12.8 (支援 Blackwell 架構)
- 安裝所有依賴套件
- Clone 專案到 /workspace/Yolov7fast
- 建立 tmux 環境 (train, cpu, gpu, terminal)
- 啟動 TensorBoard (port 6006)
- 刪除空的 coco/ 目錄
- 修正 test.py 硬編碼 annotations 路徑（改為從 data yaml 自動推導）
- 重寫 VAST_SETUP.md 為一鍵設定指南

### 2025-11-26
- 修正 COCO 資料集路徑設定（移除硬編碼）
- 確認本地資料集為 320x320 版本
- 重命名 coco → coco320
- 建立多解析度支援：coco320.yaml, coco480.yaml, coco640.yaml
- 重寫 README.md 為簡潔版本
- 建立 VAST_SETUP.md 設定文檔

### 2025-11-25
- 專案建立
- GitHub 倉庫初始化：https://github.com/jimmychintw/Yolov7fast
- 匯入 YOLOv7 基礎程式碼（107 個檔案）
- 建立 CLAUDE.md 開發規範（6 條規定）
- 建立 progress.md 進度追蹤機制
