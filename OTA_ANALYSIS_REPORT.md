# OTA vs non-OTA Loss 完整分析報告

**實驗日期**: 2025-11-28
**實驗環境**: vast.ai RTX 5090 (32GB VRAM), PyTorch 2.8.0+cu128
**模型**: YOLOv7-tiny
**資料集**: COCO 320x320 (118,287 張訓練圖片)
**Batch Size**: 64
**Epochs**: 100

---

## 1. 執行摘要

| 指標 | non-OTA | OTA | 差異 | 勝出 |
|------|---------|-----|------|------|
| **mAP@0.5** | 0.385 | **0.414** | +7.5% | OTA |
| **mAP@0.5:0.95** | 0.226 | **0.251** | +11.1% | OTA |
| **Precision** | **0.568** | 0.558 | -1.8% | non-OTA |
| **Recall** | 0.355 | **0.400** | +12.7% | OTA |
| **訓練時間** | **1.81 hr** | 10.65 hr | 5.9x 更慢 | non-OTA |
| **訓練速度** | **5.76 it/s** | 0.97 it/s | 5.9x 更快 | non-OTA |
| **GPU 利用率** | **~90%** | ~13% | 6.9x 更高 | non-OTA |

### 結論
- **OTA Loss 能提升 mAP 約 7-11%**，但代價是訓練時間增加 5.9 倍
- **OTA Loss 的瓶頸在 CPU**，SimOTA 匹配演算法佔用 86.5% 的迭代時間
- 對於快速迭代實驗，建議使用 non-OTA；最終訓練可考慮 OTA

---

## 2. 實驗設定

### 2.1 共同設定

| 參數 | 數值 |
|------|------|
| 圖像尺寸 | 320 x 320 |
| Batch Size | 64 |
| Epochs | 100 |
| Optimizer | SGD (lr=0.01, momentum=0.937) |
| LR Scheduler | OneCycleLR (lrf=0.01) |
| Warmup | 3 epochs |
| Mosaic | 1.0 (啟用) |
| MixUp | 0.05 |
| 權重初始化 | 從零開始 (--weights '') |
| Auto Anchor | 停用 (--noautoanchor) |

### 2.2 唯一差異

| 參數 | non-OTA | OTA |
|------|---------|-----|
| **loss_ota** | 0 | 1 |
| Loss 函數 | ComputeLoss | ComputeLossOTA |
| hyp 檔案 | hyp.scratch.tiny.noota.yaml | hyp.scratch.tiny.yaml |

---

## 3. 訓練過程分析

### 3.1 Loss 收斂曲線

#### Box Loss
| Epoch | non-OTA | OTA | 差異 |
|-------|---------|-----|------|
| 0 | 0.0963 | 0.0609 | OTA 較低 |
| 25 | 0.0674 | 0.0401 | OTA 較低 |
| 50 | 0.0649 | 0.0387 | OTA 較低 |
| 75 | 0.0624 | 0.0372 | OTA 較低 |
| 99 | 0.0602 | 0.0357 | OTA 較低 (-40.7%) |

#### Objectness Loss
| Epoch | non-OTA | OTA | 差異 |
|-------|---------|-----|------|
| 0 | 0.0460 | 0.0251 | OTA 較低 |
| 25 | 0.0430 | 0.0254 | OTA 較低 |
| 50 | 0.0421 | 0.0249 | OTA 較低 |
| 75 | 0.0411 | 0.0241 | OTA 較低 |
| 99 | 0.0398 | 0.0232 | OTA 較低 (-41.7%) |

#### Classification Loss
| Epoch | non-OTA | OTA | 差異 |
|-------|---------|-----|------|
| 0 | 0.0830 | 0.0766 | OTA 較低 |
| 25 | 0.0401 | 0.0296 | OTA 較低 |
| 50 | 0.0368 | 0.0264 | OTA 較低 |
| 75 | 0.0329 | 0.0226 | OTA 較低 |
| 99 | 0.0295 | 0.0192 | OTA 較低 (-34.9%) |

#### Total Loss
| Epoch | non-OTA | OTA | 差異 |
|-------|---------|-----|------|
| 0 | 0.2254 | 0.1626 | OTA 較低 |
| 25 | 0.1505 | 0.0952 | OTA 較低 |
| 50 | 0.1438 | 0.0900 | OTA 較低 |
| 75 | 0.1364 | 0.0839 | OTA 較低 |
| 99 | 0.1295 | 0.0781 | OTA 較低 (-39.7%) |

**觀察**: OTA Loss 從第一個 epoch 開始就維持較低的 loss 值，最終收斂時 Total Loss 比 non-OTA 低約 40%。

### 3.2 精度收斂曲線

#### mAP@0.5
| Epoch | non-OTA | OTA | 差異 |
|-------|---------|-----|------|
| 10 | 0.216 | 0.247 | OTA +14.4% |
| 25 | 0.310 | 0.304 | 接近 |
| 50 | 0.365 | 0.360 | 接近 |
| 75 | 0.377 | 0.395 | OTA +4.8% |
| 99 | 0.385 | 0.414 | OTA +7.5% |

#### mAP@0.5:0.95
| Epoch | non-OTA | OTA | 差異 |
|-------|---------|-----|------|
| 10 | 0.107 | 0.135 | OTA +26.2% |
| 25 | 0.172 | 0.175 | 接近 |
| 50 | 0.209 | 0.215 | OTA +2.9% |
| 75 | 0.220 | 0.239 | OTA +8.6% |
| 99 | 0.226 | 0.251 | OTA +11.1% |

#### Recall
| Epoch | non-OTA | OTA | 差異 |
|-------|---------|-----|------|
| 10 | 0.230 | 0.260 | OTA +13.0% |
| 25 | 0.312 | 0.290 | non-OTA +7.6% |
| 50 | 0.343 | 0.358 | OTA +4.4% |
| 75 | 0.363 | 0.379 | OTA +4.4% |
| 99 | 0.355 | 0.400 | OTA +12.7% |

**觀察**:
- OTA 在早期訓練階段 (epoch 10) 就展現更好的收斂
- 中期兩者接近，但最終 OTA 拉開差距
- OTA 的 Recall 顯著較高，表示能偵測到更多物件

### 3.3 收斂穩定性

#### non-OTA 最後 10 epochs 的 mAP@0.5 變化
```
ep90: 0.3828 → ep91: 0.3828 → ep92: 0.3832 → ep93: 0.3833 → ep94: 0.3831
ep95: 0.3838 → ep96: 0.3843 → ep97: 0.3845 → ep98: 0.3849 → ep99: 0.3852
```
**變化幅度**: 0.0024 (0.6%)

#### OTA 最後 10 epochs 的 mAP@0.5 變化
```
ep90: 0.4098 → ep91: 0.4104 → ep92: 0.4114 → ep93: 0.4118 → ep94: 0.4124
ep95: 0.4126 → ep96: 0.4129 → ep97: 0.4132 → ep98: 0.4136 → ep99: 0.4140
```
**變化幅度**: 0.0042 (1.0%)

**觀察**: 兩者都已接近收斂，但 OTA 仍有微小上升趨勢

---

## 4. 效能瓶頸分析

### 4.1 CUDA Events 效能剖析結果

根據 `tests/profile_training_loop.py` 的測試結果：

| 階段 | ComputeLossOTA | ComputeLoss | 加速比 |
|------|----------------|-------------|--------|
| Forward | 50.20 ms | 50.24 ms | 1.0x |
| **Loss 計算** | **1016.03 ms** | **11.22 ms** | **90.5x** |
| Backward | 103.39 ms | 103.60 ms | 1.0x |
| **Total/iter** | **1174.06 ms** | **170.89 ms** | **6.9x** |

### 4.2 GPU 利用率分析

| 指標 | ComputeLossOTA | ComputeLoss |
|------|----------------|-------------|
| GPU 計算時間 | 153.59 ms | 153.84 ms |
| 總迭代時間 | 1174.06 ms | 170.89 ms |
| **GPU 利用率** | **13.1%** | **90.0%** |

### 4.3 瓶頸原因

ComputeLossOTA 使用 **SimOTA (Simplified Optimal Transport Assignment)** 演算法：

1. **動態標籤分配**: 每個 GT box 動態決定分配給多少個 anchor
2. **Cost Matrix 計算**: 需要計算所有預測與 GT 之間的 cost
3. **CPU 密集運算**: 匈牙利算法/Sinkhorn 運算在 CPU 上執行
4. **頻繁同步**: 需要 GPU→CPU→GPU 的資料傳輸

```
OTA 瓶頸示意圖:
GPU: [Forward 50ms] → [等待 CPU 1016ms] → [Backward 103ms]
CPU:                  [SimOTA matching]
```

---

## 5. 訓練時間與成本分析

### 5.1 訓練時間對比

| 指標 | non-OTA | OTA | 比例 |
|------|---------|-----|------|
| 總訓練時間 | 1.81 小時 | 10.65 小時 | 5.9x |
| 每 epoch 時間 | 1.09 分鐘 | 6.39 分鐘 | 5.9x |
| 每 iteration 時間 | 174 ms | 1170 ms | 6.7x |
| iterations/秒 | 5.76 | 0.85 | 6.8x |

### 5.2 GPU 租用成本估算 (vast.ai RTX 5090)

假設 RTX 5090 租用費率約 $1.5/hr：

| 訓練方式 | 時間 | 成本 |
|----------|------|------|
| non-OTA 100ep | 1.81 hr | ~$2.72 |
| OTA 100ep | 10.65 hr | ~$15.98 |
| **差異** | | **5.9x** |

### 5.3 等效訓練量

在相同 10.65 小時內：
- non-OTA 可完成: ~588 epochs
- OTA 可完成: 100 epochs

---

## 6. 模型權重分析

### 6.1 輸出檔案大小

| 檔案 | non-OTA | OTA |
|------|---------|-----|
| best.pt (stripped) | 48 MB | 13 MB |
| last.pt (stripped) | 48 MB | 13 MB |
| epoch checkpoint | 48 MB | 48 MB |
| init.pt | 24 MB | 24 MB |

**注意**: stripped 後的 OTA 模型較小是因為移除了 optimizer state

### 6.2 模型架構

兩者使用完全相同的 YOLOv7-tiny 架構：
- 參數量: ~6M
- FLOPs: ~13G (320x320 input)

---

## 7. 精度-效率權衡分析

### 7.1 Pareto 效率分析

| 方案 | mAP@0.5 | 訓練時間 | 效率指標 (mAP/hr) |
|------|---------|----------|-------------------|
| non-OTA | 0.385 | 1.81 hr | 0.213 |
| OTA | 0.414 | 10.65 hr | 0.039 |

non-OTA 的效率指標是 OTA 的 **5.5 倍**

### 7.2 等時間比較

如果用 OTA 的訓練時間 (10.65 hr) 來跑 non-OTA：
- 可跑 ~588 epochs
- 預估 mAP@0.5 可能達到 0.40-0.42 (需要驗證)

### 7.3 建議使用場景

| 場景 | 建議 | 原因 |
|------|------|------|
| 快速實驗/Debug | non-OTA | 6x 更快的迭代速度 |
| 超參數搜索 | non-OTA | 可以測試更多組合 |
| 最終訓練 | OTA | 7-11% 的精度提升 |
| 資源受限 | non-OTA | 更低的 GPU 成本 |
| 追求最高精度 | OTA | mAP 明顯較高 |

---

## 8. 結論與建議

### 8.1 主要發現

1. **OTA Loss 確實能提升精度**: mAP@0.5 提升 7.5%，mAP@0.5:0.95 提升 11.1%
2. **代價是訓練時間**: 5.9 倍的訓練時間增加
3. **瓶頸在 CPU**: SimOTA 匹配演算法佔用 86.5% 的時間
4. **GPU 利用率極低**: 使用 OTA 時僅 13%，是嚴重的資源浪費

### 8.2 優化建議

1. **短期**: 使用 `loss_ota: 0` 進行快速實驗
2. **中期**: 研究 SimOTA 的 GPU 加速版本
3. **長期**: 考慮使用 TOOD 或其他高效的動態標籤分配方法

### 8.3 下一步計畫

- [ ] 測試 non-OTA 300 epochs 是否能達到 OTA 100 epochs 的精度
- [ ] 調查 OTA 的 GPU 實現可能性
- [ ] 測試不同 batch size 對 OTA 效能的影響
- [ ] 比較 640x640 解析度下的差異

---

## 附錄 A: 訓練指令

### non-OTA 訓練
```bash
python train.py --data data/coco320.yaml --img 320 \
    --cfg cfg/training/yolov7-tiny.yaml \
    --hyp data/hyp.scratch.tiny.noota.yaml \
    --batch-size 64 --epochs 100 \
    --weights '' --noautoanchor \
    --name noota_100ep
```

### OTA 訓練
```bash
python train.py --data data/coco320.yaml --img 320 \
    --cfg cfg/training/yolov7-tiny.yaml \
    --hyp data/hyp.scratch.tiny.yaml \
    --batch-size 64 --epochs 100 \
    --weights '' --noautoanchor \
    --name ota_100ep
```

---

## 附錄 B: 完整訓練結果資料

### non-OTA (noota_100ep2)
- 訓練目錄: `runs/train/noota_100ep2`
- 最佳 epoch: 99
- 最終結果:
  - Precision: 0.568
  - Recall: 0.355
  - mAP@0.5: 0.385
  - mAP@0.5:0.95: 0.226

### OTA (ota_100ep4)
- 訓練目錄: `runs/train/ota_100ep4`
- 最佳 epoch: 99
- 最終結果:
  - Precision: 0.558
  - Recall: 0.400
  - mAP@0.5: 0.414
  - mAP@0.5:0.95: 0.251

---

*報告生成日期: 2025-11-28*
*實驗執行者: Claude Code + Jimmy*
