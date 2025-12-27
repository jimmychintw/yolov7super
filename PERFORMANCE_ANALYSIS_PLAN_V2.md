# YOLOv7 訓練效能完整分析計畫書 V2

## 整合版 - 結合 Claude + Gemini 的優點

---

## 核心問題

| 現象 | 數據 |
|------|------|
| 純 GPU 推理時間 | ~66.7 ms (batch=384) |
| 實際訓練迭代時間 | ~1170 ms |
| **差距** | **17.5 倍** |
| GPU 利用率 | 僅 34% |

**目標**：找出這 17.5 倍差距的來源，並恢復 RTX 5090 應有的速度。

---

## Phase 0: 環境記錄（執行前必做）

```bash
# 在 vast.ai 上執行，記錄完整環境資訊
echo "=== GPU Info ===" && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "=== PyTorch ===" && python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
echo "=== CPU ===" && cat /proc/cpuinfo | grep "model name" | head -1 && nproc
echo "=== Memory ===" && free -h
echo "=== Disk ===" && df -h /workspace
```

---

## Phase 1: Ghost Buster（抓出隱形延遲）

### 1.1 檢查 train.py 中的「拖油瓶」設定

```python
# 必須確認這些設定：

# ✅ 應該開啟
torch.backends.cudnn.benchmark = True

# ❌ 必須關閉（會造成巨大延遲）
torch.autograd.set_detect_anomaly(False)  # 確認是 False 或沒有這行

# ✅ DataLoader 應有
pin_memory = True
```

### 1.2 檢查 Training Loop 內的同步點

**搜尋以下「隱形殺手」**：

```python
# 這些操作會強制 GPU 等待 CPU，必須移除或降低頻率：

loss.item()           # 每次呼叫都會同步
tensor.cpu()          # 強制同步
print(tensor)         # 強制同步
tensor.numpy()        # 強制同步
```

### 1.3 檢查 Logging 開銷

```python
# 暫時註解掉這些（確認不是 logging 造成的延遲）：
# tb_writer.add_scalar(...)
# wandb.log(...)
# plot_images(...)
# cv2.imwrite(...)
```

### 1.4 建立最小測試腳本

```python
"""
tests/minimal_training_test.py
最小化訓練測試 - 排除所有干擾因素
"""
import torch
import time
from models.yolo import Model
from utils.loss import ComputeLossOTA

# 建立模型
model = Model('cfg/training/yolov7-tiny-320.yaml').cuda()
model.train()

# 假資料
imgs = torch.randn(384, 3, 320, 320).cuda()
targets = torch.zeros(1000, 6).cuda()  # 假 targets

# Loss function
compute_loss = ComputeLossOTA(model)

# Warmup
for _ in range(5):
    pred = model(imgs)
    loss, _ = compute_loss(pred, targets.to(model.device))
    loss.backward()

torch.cuda.synchronize()

# 測量
n_iter = 20
start = time.perf_counter()
for _ in range(n_iter):
    pred = model(imgs)
    loss, _ = compute_loss(pred, targets)
    loss.backward()
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"純 GPU 訓練（無 DataLoader）: {elapsed/n_iter*1000:.1f} ms/iter")
print(f"預期完整訓練應接近此數值")
```

### 1.5 驗收標準

| 測試 | 預期結果 |
|------|----------|
| 最小測試腳本 | < 100 ms/iter |
| 若 > 500 ms | train.py 有問題需要修復 |

---

## Phase 2: 復現 6x 差異（驗證 CPU 瓶頸）

### 2.1 測試矩陣

| 測試組 | Mosaic | MixUp | Cache | Workers | 預期時間 |
|--------|--------|-------|-------|---------|----------|
| A | ON | ON | ON | 8 | ~100-150 ms |
| B | OFF | OFF | ON | 8 | ~20-30 ms |
| C | ON | ON | OFF | 8 | 可能更慢 |
| D | OFF | OFF | OFF | 8 | 比 B 慢 |

### 2.2 執行命令

```bash
# Group A: Mosaic ON
python train.py --img-size 320 320 --batch-size 384 --epochs 1 \
  --data data/coco320.yaml --cfg cfg/training/yolov7-tiny-320.yaml \
  --weights '' --hyp data/hyp.scratch.tiny.yaml \
  --device 0 --workers 8 --cache-images --noautoanchor \
  --project runs/phase2 --name group_a_mosaic_on

# Group B: Mosaic OFF
python train.py --img-size 320 320 --batch-size 384 --epochs 1 \
  --data data/coco320.yaml --cfg cfg/training/yolov7-tiny-320.yaml \
  --weights '' --hyp data/hyp.scratch.tiny.nomosaic.yaml \
  --device 0 --workers 8 --cache-images --noautoanchor \
  --project runs/phase2 --name group_b_mosaic_off
```

### 2.3 驗收標準

| 結果 | 解讀 |
|------|------|
| A / B ≈ 4-6x | ✅ CPU 瓶頸確認，A 計畫有效 |
| A / B ≈ 1-2x | ❓ 需要進一步分析 |
| A ≈ B 都很慢 | ❌ 有其他系統問題 |

---

## Phase 3: 細粒度效能剖析（使用 CUDA Events）

### 3.1 植入 Profiler 到 train.py

```python
"""
在 train.py 的 training loop 中植入以下代碼
"""
import torch

class CUDAProfiler:
    def __init__(self):
        self.timings = {
            'data_loading': [],
            'data_transfer': [],
            'forward': [],
            'loss': [],
            'backward': [],
            'optimizer': [],
        }
        self.start_events = {}
        self.end_events = {}

    def start(self, name):
        if name not in self.start_events:
            self.start_events[name] = torch.cuda.Event(enable_timing=True)
            self.end_events[name] = torch.cuda.Event(enable_timing=True)
        self.start_events[name].record()

    def end(self, name):
        self.end_events[name].record()

    def sync_and_record(self):
        torch.cuda.synchronize()
        for name in self.start_events:
            elapsed = self.start_events[name].elapsed_time(self.end_events[name])
            self.timings[name].append(elapsed)

    def report(self, skip_first=5):
        print("\n" + "="*50)
        print("Profile Result - CUDA Event Timing")
        print("="*50)
        total = 0
        for name, times in self.timings.items():
            if len(times) > skip_first:
                times = times[skip_first:]
                avg = sum(times) / len(times)
                total += avg
                print(f"{name:20s}: {avg:8.2f} ms")
        print("-"*50)
        print(f"{'Total':20s}: {total:8.2f} ms")

        # GPU 利用率估算
        gpu_time = (self.timings.get('forward', [0])[-1] +
                    self.timings.get('backward', [0])[-1])
        if total > 0:
            print(f"{'GPU Utilization Est':20s}: {gpu_time/total*100:6.1f}%")
        print("="*50)


# 在 training loop 中使用：
profiler = CUDAProfiler()

for i, (imgs, targets, paths, _) in enumerate(dataloader):
    # 1. Data Loading (這個要特殊處理，因為是 CPU 等待時間)
    profiler.start('data_transfer')
    imgs = imgs.to(device, non_blocking=True).float() / 255.0
    targets = targets.to(device)
    profiler.end('data_transfer')

    # 2. Forward
    profiler.start('forward')
    with torch.cuda.amp.autocast():
        pred = model(imgs)
    profiler.end('forward')

    # 3. Loss
    profiler.start('loss')
    with torch.cuda.amp.autocast():
        loss, loss_items = compute_loss(pred, targets)
    profiler.end('loss')

    # 4. Backward
    profiler.start('backward')
    scaler.scale(loss).backward()
    profiler.end('backward')

    # 5. Optimizer
    profiler.start('optimizer')
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    profiler.end('optimizer')

    # 每 10 步同步一次記錄
    if i % 10 == 0:
        profiler.sync_and_record()

    if i >= 100:
        break

profiler.report()
```

### 3.2 測量 DataLoader 等待時間

```python
"""
特別測量 DataLoader 的等待時間（CPU 準備資料的時間）
"""
import time

data_wait_times = []

data_iter = iter(dataloader)
for i in range(100):
    t0 = time.perf_counter()
    batch = next(data_iter)  # 這裡會等待 CPU 準備好資料
    t1 = time.perf_counter()
    data_wait_times.append((t1 - t0) * 1000)

print(f"DataLoader Wait Time: {sum(data_wait_times[5:])/len(data_wait_times[5:]):.2f} ms")
```

### 3.3 預期輸出格式

```
[Profile Result - Avg of 100 steps]
-----------------------------------
1. Data Loading (CPU Wait):  85.4 ms  <-- 如果這很大，A計畫有效
2. Data Transfer (PCIe):      2.1 ms
3. Forward (GPU):            12.5 ms
4. Loss Calc (CPU/GPU):       4.2 ms  <-- 如果這很大，優化 SimOTA
5. Backward/Opt (GPU):       28.6 ms
-----------------------------------
Total Step Time:            132.8 ms
GPU Utilization Est.:       31%
```

---

## Phase 4: __getitem__ 內部分段測時

### 4.1 測試腳本

```python
"""
tests/profile_getitem_detailed.py
"""
import time
import numpy as np
from utils.datasets import LoadImagesAndLabels
import yaml

def profile_getitem(dataset, n_samples=100):
    """直接測量 dataset.__getitem__ 的時間"""
    times = []

    for i in range(n_samples):
        idx = np.random.randint(len(dataset))
        t0 = time.perf_counter()
        _ = dataset[idx]
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = times[5:]  # skip warmup
    print(f"__getitem__ time: {np.mean(times):.3f} ± {np.std(times):.3f} ms")
    return np.mean(times)

# 測試
with open('data/hyp.scratch.tiny.yaml') as f:
    hyp = yaml.safe_load(f)

# Mosaic ON
hyp['mosaic'] = 1.0
dataset_on = LoadImagesAndLabels('coco320/images/train2017', 320, 384,
                                  augment=True, hyp=hyp, cache_images=True)
time_on = profile_getitem(dataset_on)

# Mosaic OFF
hyp['mosaic'] = 0.0
dataset_off = LoadImagesAndLabels('coco320/images/train2017', 320, 384,
                                   augment=True, hyp=hyp, cache_images=True)
time_off = profile_getitem(dataset_off)

print(f"\nMosaic ON/OFF ratio: {time_on/time_off:.2f}x")
```

---

## Phase 5: load_mosaic 內部分段測時

### 5.1 修改 datasets.py 加入計時

```python
# 在 utils/datasets.py 的 load_mosaic 函數中加入計時點

def load_mosaic(self, index):
    """帶計時的 load_mosaic"""
    import time
    timings = {}

    t_total = time.perf_counter()

    # 1. 選擇索引
    t0 = time.perf_counter()
    indices = [index] + random.choices(range(self.n), k=3)
    timings['select_indices'] = time.perf_counter() - t0

    # 2. 載入 4 張圖像
    t0 = time.perf_counter()
    imgs = [self.load_image(i) for i in indices]
    timings['load_4_images'] = time.perf_counter() - t0

    # 3. 建立畫布
    t0 = time.perf_counter()
    s = self.img_size
    img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
    timings['create_canvas'] = time.perf_counter() - t0

    # 4. 放置圖像
    t0 = time.perf_counter()
    # ... 放置邏輯 ...
    timings['place_images'] = time.perf_counter() - t0

    # 5. 合併 labels
    t0 = time.perf_counter()
    # ... labels 處理 ...
    timings['process_labels'] = time.perf_counter() - t0

    # 6. random_perspective
    t0 = time.perf_counter()
    img4, labels4 = random_perspective(...)
    timings['random_perspective'] = time.perf_counter() - t0

    timings['total'] = time.perf_counter() - t_total

    # 輸出（可選）
    if hasattr(self, '_profile_count'):
        self._profile_count += 1
        if self._profile_count % 100 == 0:
            for k, v in timings.items():
                print(f"  {k}: {v*1000:.3f} ms")

    return img4, labels4
```

---

## Phase 6: random_perspective 深入分析

### 6.1 比較不同插值方法

```python
"""
tests/benchmark_interpolation.py
"""
import cv2
import numpy as np
import time

# 建立測試圖像
img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
M = np.eye(2, 3, dtype=np.float32)
M[0, 0] = 0.8  # scale
M[1, 1] = 0.8

methods = {
    'INTER_NEAREST': cv2.INTER_NEAREST,
    'INTER_LINEAR': cv2.INTER_LINEAR,
    'INTER_AREA': cv2.INTER_AREA,
    'INTER_CUBIC': cv2.INTER_CUBIC,
}

for name, method in methods.items():
    # warmup
    for _ in range(10):
        cv2.warpAffine(img, M, (320, 320), flags=method)

    # benchmark
    t0 = time.perf_counter()
    for _ in range(100):
        cv2.warpAffine(img, M, (320, 320), flags=method)
    elapsed = time.perf_counter() - t0

    print(f"{name:15s}: {elapsed/100*1000:.3f} ms")
```

---

## Phase 7: 決策樹

根據 Phase 3 的數據，決定下一步：

```
IF T_Data > 50% of Total:
    → 執行 A 計畫 (Micro-Collate / GPU Augmentation)

ELIF T_Loss > 30% of Total:
    → 優化 ComputeLossOTA (SimOTA)

ELIF T_Forward < 10% of Total AND Total > 100ms:
    → 考慮 torch.compile 或增大 batch size

ELIF T_Transfer > 20% of Total:
    → 檢查 pin_memory, non_blocking 設定

ELSE:
    → 模型太小，考慮增大模型或圖像尺寸
```

---

## 結果記錄表

### Phase 1 結果

| 檢查項目 | 狀態 | 備註 |
|----------|------|------|
| cudnn.benchmark | ☐ ON / ☐ OFF | |
| detect_anomaly | ☐ OFF / ☐ ON | |
| pin_memory | ☐ ON / ☐ OFF | |
| loss.item() 頻率 | | |
| logging 頻率 | | |

### Phase 2 結果

| 測試組 | 速度 (ms/it) | 相對於 B |
|--------|--------------|----------|
| A (Mosaic ON) | | |
| B (Mosaic OFF) | | 1.0x |

### Phase 3 結果

| 階段 | Mosaic ON | Mosaic OFF | 差異 |
|------|-----------|------------|------|
| Data Loading | ms | ms | x |
| Data Transfer | ms | ms | x |
| Forward | ms | ms | x |
| Loss | ms | ms | x |
| Backward | ms | ms | x |
| **Total** | ms | ms | x |

### Phase 4 結果

| 測量項目 | Mosaic ON | Mosaic OFF |
|----------|-----------|------------|
| __getitem__ | ms | ms |
| collate_fn | ms | ms |

### Phase 5 結果 (load_mosaic 內部)

| 步驟 | 時間 (ms) | 佔比 |
|------|-----------|------|
| select_indices | | |
| load_4_images | | |
| create_canvas | | |
| place_images | | |
| process_labels | | |
| random_perspective | | |
| **Total** | | 100% |

---

## 執行順序檢查表

- [ ] Phase 0: 環境記錄
- [ ] Phase 1: Ghost Buster
  - [ ] 1.1 檢查 train.py 設定
  - [ ] 1.2 檢查同步點
  - [ ] 1.3 檢查 logging
  - [ ] 1.4 執行最小測試
- [ ] Phase 2: 復現 6x 差異
  - [ ] Group A (Mosaic ON)
  - [ ] Group B (Mosaic OFF)
- [ ] Phase 3: 細粒度剖析
  - [ ] 植入 CUDA Profiler
  - [ ] 測量各階段時間
- [ ] Phase 4: __getitem__ 分析
- [ ] Phase 5: load_mosaic 分析
- [ ] Phase 6: random_perspective 分析
- [ ] Phase 7: 根據數據決定方向

---

*計畫書 V2 - 整合 Claude + Gemini 建議*
*建立日期：2025-11-27*
