

```
# Role
You are a Senior AI Research Engineer specializing in AutoML and Hyperparameter Optimization.
The project is "YOLOv7 1B4H" (One Backbone Four Heads).
We have completed Phase 2 (Geometry Grouping) and have a `data/coco_320_1b4h_geometry.yaml`.
We also have a strong pre-trained backbone: `backbone_elite_0.435.pt`.

# Task
Implement **Phase 4: RL-based Data Augmentation Search**.
Create a Python script using **Optuna** to automatically search for the best data augmentation hyperparameters (`degrees`, `flipud`, `shear`, `mixup`) for *each* specific Head.

# Dependencies
- `optuna`
- `pyyaml`

# Implementation Details

## 1. Create `utils/rl_augment.py`
This is the main search engine.

### Key Logic:
1.  **Load Geometry Config**: Read `data/coco_320_1b4h_geometry.yaml` to identify which classes belong to `Head X`.
2.  **Define Search Space**:
    - `degrees`: 0.0 ~ 45.0
    - `flipud`: 0.0 ~ 0.5 (probability)
    - `fliplr`: 0.0 ~ 0.5
    - `shear`: 0.0 ~ 5.0
    - `mixup`: 0.0 ~ 0.2
3.  **Proxy Task Execution**:
    - For each Optuna trial, generate a temporary `hyp.trial_X.yaml`.
    - Run `train.py` with the following **FIXED** settings for speed:
        - `--weights backbone_elite_0.435.pt` (Strong Backbone)
        - `--transfer-weights` (Reset Head)
        - `--freeze 50` (Freeze Backbone & Neck - Speed up 3x)
        - `--epochs 10` (Short run to see trends)
        - `--head-config data/coco_320_1b4h_geometry.yaml`
        - `--batch-size 128`
4.  **Reward Calculation (CRITICAL)**:
    - The metric must be the **mAP@0.5 of ONLY the classes belonging to the target Head**.
    - *Challenge*: Standard `train.py` outputs global mAP.
    - *Solution*: You must modify `train.py` or `val.py` (or parse the log) to calculate/extract the mAP of specific classes.
    - *Recommendation*: Add a helper function in `rl_augment.py` to parse `results.txt` or `best_fitness` if you modify `train.py` to allow filtering validation by classes. Or simply parse the console output if `train.py` prints class-wise AP.
    - *Simpler approach for this task*: Let's modify `train.py` / `val.py` slightly to support a `--save-per-class-ap` flag, writing a JSON file that `rl_augment.py` can read.

## 2. Necessary Modifications

### Modify `train.py` / `test.py` (or `val.py`)
- Add a mechanism to export per-class AP (Average Precision) after validation.
- Example: Save `runs/train/exp/class_ap.json` containing `{class_id: ap_val, ...}`.
- This allows `rl_augment.py` to compute the weighted average mAP for *only* the classes in Head X.

## 3. Usage Interface
The script should be runnable via CLI:
```bash
python utils/rl_augment.py --head 0 --trials 30  # Optimize Head 0
python utils/rl_augment.py --head 1 --trials 30  # Optimize Head 1
```

## 4. Output

- Save the best hyperparameters for each head into:
  - `data/hyp.head0.best.yaml`
  - `data/hyp.head1.best.yaml`
  - ...

# Goal

I want to run `python utils/rl_augment.py --head 0` and have it automatically find that Head 0 (Vertical objects) benefits from higher rotation, while Head 1 (Horizontal objects) prefers zero rotation.

```
---

### 💡 給您的操作建議 (Human Instructions)

1.  **安裝 Optuna**:
    在您的環境中執行：`pip install optuna`。
2.  **執行順序**:
    * 先把上面的 Prompt 丟給 Claude Code，讓它寫程式。
    * 程式寫好後，**請先跑一次 `python utils/rl_augment.py --head 0 --trials 1` (測試跑)**，確認它真的能讀到 mAP 並且產出檔案。
    * 確認無誤後，您可以開 **4 個終端機視窗 (Tmux/Screen)**，同時跑 4 個 Head 的搜索（您的 5090 夠力，可以平行跑）。
        * Window 1: `python utils/rl_augment.py --head 0`
        * Window 2: `python utils/rl_augment.py --head 1` ...

祝您的自動化搜索順利！這一步完成後，您就擁有針對每個幾何形狀最佳化的「完美食譜」了。
```