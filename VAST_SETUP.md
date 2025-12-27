# Vast.ai 遠端訓練環境設定指南

## 🔄 每次重啟 Instance 的設定流程

### 快速檢查清單

| 步驟 | 動作 | 需要時間 |
|------|------|----------|
| 1 | 設定環境變數 | 10 秒 |
| 2 | 一鍵安裝腳本 | 3-5 分鐘 |
| 3 | 資料集還原 | 視情況 |
| 4 | 連線開始工作 | - |

---

## Step 1: 設定環境變數（本機執行）

```bash
# 根據新 instance 的連線資訊修改
export VAST_HOST="root@<IP>"
export VAST_PORT="<Port>"

# 範例：
export VAST_HOST="root@116.122.206.233"
export VAST_PORT="21024"
```

---

## Step 2: 一鍵安裝腳本（本機執行）

```bash
ssh -p $VAST_PORT $VAST_HOST -o StrictHostKeyChecking=no 'bash -s' << 'EOF'
set -e
echo "=== 開始設定 vast.ai 環境 ==="

# 1. Clone 專案
echo "[1/6] Clone/更新 YOLOv7fast 專案..."
cd /workspace
if [ ! -d "Yolov7fast" ]; then
    git clone https://github.com/jimmychintw/Yolov7fast.git
else
    cd Yolov7fast && git pull
fi
cd /workspace/Yolov7fast

# 2. 建立虛擬環境
echo "[2/6] 建立虛擬環境..."
python3 -m venv venv
source venv/bin/activate

# 3. 升級 pip
echo "[3/6] 升級 pip..."
pip install -U pip setuptools wheel -q

# 4. 安裝 PyTorch 2.8.0 + CUDA 12.8
echo "[4/6] 安裝 PyTorch 2.8.0 (CUDA 12.8)..."
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128 -q

# 5. 安裝其他依賴
echo "[5/6] 安裝其他依賴套件..."
pip install -q \
    matplotlib opencv-python Pillow PyYAML requests scipy tqdm \
    tensorboard torch-tb-profiler pandas seaborn ipython psutil thop pycocotools

# 6. 建立 tmux 環境
echo "[6/6] 建立 tmux 環境..."
tmux kill-server 2>/dev/null || true
tmux new -d -s vast -n train
tmux new-window -t vast -n cpu
tmux new-window -t vast -n gpu
tmux new-window -t vast -n terminal
tmux send-keys -t vast:cpu 'htop' Enter
tmux send-keys -t vast:gpu 'watch -n 1 nvidia-smi' Enter

# 驗證
echo ""
echo "=== 設定完成！==="
source venv/bin/activate
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo "venv: $(which python)"
tmux ls
EOF
```

---

## Step 3: 還原資料集



### 從本機上傳 (如果 server 上尚未有對應的 dataset)
```bash
# 上傳 coco.zip
使用 rsync 
最後解壓縮，放在與本機相同的對應目錄中
```

---

## Step 4: 連線並開始工作

```bash
# SSH 連線
ssh -p $VAST_PORT $VAST_HOST

# 進入 tmux
tmux attach -t vast

# 進入專案目錄並啟用虛擬環境
cd /workspace/Yolov7fast
source venv/bin/activate
```

---

## 📋 精簡版（複製貼上用）

```bash
# === 每次新 instance 執行 ===

# 1. 設定變數（改成你的）
export VAST_HOST="root@116.122.206.233"
export VAST_PORT="21024"

# 2. 一鍵設定（約 3-5 分鐘）
ssh -p $VAST_PORT $VAST_HOST -o StrictHostKeyChecking=no 'bash -s' << 'SETUP'
cd /workspace && git clone https://github.com/jimmychintw/Yolov7fast.git 2>/dev/null || (cd Yolov7fast && git pull)
cd /workspace/Yolov7fast
python3 -m venv venv && source venv/bin/activate
pip install -U pip setuptools wheel -q
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 -q
pip install -q matplotlib opencv-python Pillow PyYAML requests scipy tqdm tensorboard pandas seaborn psutil thop pycocotools
tmux kill-server 2>/dev/null; tmux new -d -s vast -n train; tmux new-window -t vast -n terminal
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
SETUP

# 3. 連線
ssh -p $VAST_PORT $VAST_HOST -t "tmux attach -t vast"
```

---

## 套件版本（RTX 5090 專用）

| 套件 | 版本 | 說明 |
|------|------|------|
| Python | 3.12 | vast.ai 預裝 |
| PyTorch | 2.8.0+cu128 | 支援 Blackwell (sm_120) |
| torchvision | 0.23.0+cu128 | |
| CUDA | 12.8 | PyTorch wheel 內建 |

**重要**：RTX 5090 使用 Blackwell 架構 (sm_120)，需要 PyTorch 2.8.0+ 和 CUDA 12.8+

---

## Tmux 環境

### Session 結構
```
vast (session)
├── train     - 訓練任務
├── cpu       - htop CPU 監控
├── gpu       - nvidia-smi GPU 監控
└── terminal  - 一般操作
```

### 快捷鍵
| 按鍵 | 功能 |
|------|------|
| `Ctrl+b` → `n` | 下一個 window |
| `Ctrl+b` → `p` | 上一個 window |
| `Ctrl+b` → `0-3` | 跳到指定 window |
| `Ctrl+b` → `d` | Detach（離開但不關閉） |

---

## 常用指令

```bash
# 檢查 GPU 狀態
ssh -p $VAST_PORT $VAST_HOST "nvidia-smi"

# 檢查 tmux
ssh -p $VAST_PORT $VAST_HOST "tmux ls"

# 進入 tmux session
ssh -p $VAST_PORT $VAST_HOST -t "tmux attach -t vast"

# 查看訓練輸出
ssh -p $VAST_PORT $VAST_HOST "tmux capture-pane -t vast:train -p | tail -20"
```

---

## 備份與還原

### 備份訓練結果到 Google Drive
- vast.ai 控制台 → 點 → (Sync) 按鈕

### 從 Google Drive 還原
- vast.ai 控制台 → 點 ☁️ (Copy) 按鈕

### 手動下載訓練結果
```bash
scp -P $VAST_PORT $VAST_HOST:/workspace/Yolov7fast/runs/train/*/weights/best.pt ./
```

---

## 注意事項

1. **SSH Key**：每次租用新 instance 都需要重新添加 SSH key
2. **Instance 重啟**：tmux session 會消失，需重新設定
3. **費用**：記得用完要停止 instance
4. **資料集**：建議用 Google Drive 備份，避免重複上傳

---

*最後更新：2025-11-29*
