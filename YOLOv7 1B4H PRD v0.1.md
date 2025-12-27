

# YOLOv7 "Super-Optimized" 1B4H 架構產品需求文檔



項目名稱: YOLOv7-tiny 1B4H Architecture with Task-Aware Attention & RL Optimization

版本: v4.0 (Final Architecture)

日期: 2025-11-30

狀態: Approved for Implementation

硬體基準: AMD Ryzen 9 9950X3D + NVIDIA RTX 5090

------



## 1. 執行摘要 (Executive Summary)



本專案旨在突破物件偵測模型 (YOLOv7) 的精度天花板。透過將單一檢測任務解耦為 **四個獨立的幾何/語意子任務 (1B4H)**，並引入 **任務感知注意力機制 (Task-Aware Attention)** 與 **自動化特徵工程 (RL-Augment)**，我們預期在不顯著增加模型複雜度的前提下，將 COCO mAP 提升至 **50%** 大關（640x640 解析度）。

這是一套 **「軟硬體垂直整合」** 的解決方案：軟體上利用分治策略解決特徵衝突，硬體上利用 3D V-Cache 解決 SimOTA 運算瓶頸。

------



## 2. 核心目標與指標 (Objectives & Metrics)





### 2.1 主要目標



1. **架構解耦**：消除不同類別間的幾何衝突（如「人」與「車」長寬比的拉鋸）。
2. **特徵純淨化**：透過 Attention 機制，實現每個 Head 的特徵零雜訊。
3. **數據最佳化**：針對不同 Head 的物理特性，自動搜索最佳數據增強策略。



### 2.2 關鍵績效指標 (KPIs)



| **指標**         | **基準 (Baseline: OTA 640)** | **目標 (Target)** | **預期增幅**     | **備註**                  |
| ---------------- | ---------------------------- | ----------------- | ---------------- | ------------------------- |
| **mAP@0.5**      | ~47.5%                       | **> 50.0%**       | **+2.5% ~ 3.0%** | 主要來自架構與數據策略    |
| **mAP@0.5:0.95** | ~30.0%                       | **> 32.0%**       | +2.0%            | 主要來自 K-Means 幾何分群 |
| **訓練時間**     | ~35 hr (未優化)              | **< 20 hr**       | 節省 40%         | 依賴 X3D 綁核優化         |
| **推論 FPS**     | ~180 FPS (3090)              | **> 140 FPS**     | 下降 < 25%       | 保持實時推論能力          |

------



## 3. 系統架構設計 (System Architecture)





### 3.1 骨幹網路 (Backbone & Neck)



- **結構**: 標準 YOLOv7-tiny (CSPDarknet + PANet)。
- **功能**: 提取 P3, P4, P5 三種尺度的共享特徵圖。
- **狀態**: 在 Phase 0 (搜索階段) 凍結，Phase 1 (正式訓練) 解凍。



### 3.2 檢測頭架構：1B4H Strategy B + Attention



採用 **「完全獨立頭 + 任務感知濾鏡」** 設計。

1. **CBAM Attention Modules (守門員)**
   - **位置**: Neck 輸出後，Head 輸入前。
   - **數量**: 4 組（對應 4 個 Head）。
   - **功能**:
     - **Channel Attention**: 選擇關鍵特徵通道（學會「看什麼」）。
     - **Spatial Attention**: 產生空間遮罩（學會「看哪裡」），過濾背景雜訊。
2. **Independent Heads (執行者)**
   - **結構**: 4 個完全獨立的 Detect Head，不共享 Box/Obj 分支。
   - **優勢**: 徹底解決 Strategy A 的幾何回歸衝突。



### 3.3 訓練機制：Loss Router (損失路由器)



- **機制**: One-Pass Training (單次前向傳播)。
- **邏輯**:
  - 輸入：完整 COCO Batch。
  - 路由：根據 `class_id` 判斷該物體屬於哪個 Head。
  - 遮罩：對 Head X 來說，不屬於它的物體視為 **背景 (Negative Sample)**。
  - **效益**: 實現隱式負樣本挖掘 (Implicit Negative Mining)，大幅降低 False Positive。

------



## 4. 優化流水線 (Optimization Pipeline)



本計畫將訓練分為兩個階段：**Phase 0 (戰略搜索)** 與 **Phase 1 (決戰訓練)**。



### 4.1 Phase 0: 搜索與定義 (Non-OTA 模式)



*利用 non-OTA 的高速度 (6x Faster) 進行快速迭代*。



#### 步驟 A: 幾何感知自動分群 (Geometry-Aware Grouping)



- **方法**: K-Means Clustering (k=4)。
- **輸入**: COCO 所有標籤的 `(width/height)` 比率與 `area`。
- **產出**: 4 個 Head 的類別分配表。確保同一個 Head 內的物體形狀高度相似。
- **目的**: 最大化 Anchor 的匹配效率。



#### 步驟 B: 強化學習數據增強 (RL-Based AutoAugment)



- **方法**: 使用 Optuna/RL 針對每個 Head 進行超參數搜索。
- **搜尋空間**:
  - `degrees` (旋轉): Head 0 (人) 預期高，Head 1 (車) 預期低。
  - `flipud` (上下翻轉): Head 3 (物) 預期高，Head 1 (車) 預期禁止。
- **產出**: 4 份 `augment_policy.yaml`。



### 4.2 Phase 1: 正式訓練 (OTA 模式)



*啟用 OTA (`loss_ota: 1`) 以獲得最終的高精度*。

- **輸入**: Phase 0 產出的「分組表」與「增強策略」。
- **數據載入**: `DataLoader` 根據圖片所屬的 Head，動態應用對應的增強策略 (Stochastic Policy Mixing)。

------



## 5. 硬體與調度策略 (Hardware Strategy)



針對 OTA 訓練中的 SimOTA CPU 瓶頸，採用 **拓樸感知 (Topology-Aware)** 調度。



### 5.1 硬體規格



- **CPU**: **AMD Ryzen 9 9950X3D** (16C/32T, 128MB L3 Cache)。
- **GPU**: **NVIDIA RTX 5090** (32GB VRAM)。
- **RAM**: 192GB DDR5 (全量快取數據集)。



### 5.2 綁核策略 (Core Affinity)



利用 `taskset` 與 `worker_init_fn` 進行物理隔離：

- **CCD 0 (V-Cache CCD)**:
  - **分配給**: Python Main Process (SimOTA 計算)。
  - **邏輯核心**: `0-7` & `16-23` (基於 `lscpu` 拓樸確認)。
  - **目的**: 確保 Cost Matrix 計算全程在 L3 Cache 內，解決 Latency 瓶頸。
- **CCD 1 (High-Freq CCD)**:
  - **分配給**: DataLoader Workers。
  - **邏輯核心**: `8-15` & `24-31`。
  - **目的**: 負責圖片解碼與增強，避免汙染 CCD 0 的 Cache。

------



## 6. 實施路線圖 (Implementation Roadmap)



| **階段**   | **任務模組** | **估計工時** | **關鍵產出**                                     |
| ---------- | ------------ | ------------ | ------------------------------------------------ |
| **Week 1** | **架構開發** | 3 Days       | `MultiHeadDetect` (w/ CBAM), `ComputeLossRouter` |
|            | **硬體驗證** | 1 Day        | 確認 9950X3D 拓樸，測試綁核腳本                  |
| **Week 2** | **幾何分群** | 0.5 Day      | K-Means 分組表 (`data/coco_kmeans.yaml`)         |
|            | **RL 搜索**  | 2 Days       | 4 組增強策略 (`hyp.head0.yaml`...)               |
| **Week 3** | **正式訓練** | 2 Days       | 100 Epochs Full Training (640x640)               |
|            | **評估分析** | 1 Day        | Grad-CAM 熱力圖、消融實驗報告                    |

------



## 7. 風險評估 (Risk Assessment)



| **風險項目**                     | **可能性** | **嚴重性** | **緩解措施**                                                 |
| -------------------------------- | ---------- | ---------- | ------------------------------------------------------------ |
| **640x640 訓練時間過長 (>30hr)** | 中         | 中         | 加大 Batch Size (利用 5090 32GB VRAM) 以減少 SimOTA 頻率。   |
| **Head 間類別不平衡**            | 高         | 高         | 在 Loss Router 中引入 `Class Balancing Weight`，對樣本少的 Head 加權。 |
| **Attention 導致推論延遲**       | 中         | 低         | 若 FPS 不達標，採用 TensorRT 優化或 RepVGG 簡化 Attention 結構。 |

------



## 8. 結論 (Conclusion)



本 PRD 定義了一個極具野心的架構升級計畫。我們不再僅僅是訓練一個模型，而是構建了一套 **「自我優化的物件偵測系統」**。

從底層的 **硬體快取優化**，到中層的 **架構解耦與注意力機制**，再到上層的 **數據策略搜索**，每一個環節都經過精心設計以消除瓶頸。這將使 YOLOv7-tiny 在保持輕量級優勢的同時，展現出足以比肩中大型模型的檢測精度，具備極高的學術發表與產品應用價值。