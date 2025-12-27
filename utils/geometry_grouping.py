import json
import numpy as np
from sklearn.cluster import KMeans
import yaml
import os
from pathlib import Path

def generate_geometry_config(
    json_path='coco320/annotations/instances_train2017.json',
    output_path='data/coco_320_1b4h_geometry.yaml',
    n_heads=4
):
    """
    執行 K-Means 幾何分群並生成 YOLOv7 1B4H 設定檔
    """

    # 檢查路徑
    if not os.path.exists(json_path):
        print(f"Error: COCO annotations not found at {json_path}")
        print("Please check your dataset path.")
        return

    print(f"Loading COCO annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. 建立類別映射與資料結構
    print("Analyzing class geometries...")

    # COCO category_id (1-90) -> YOLO train_id (0-79)
    coco_id_to_train_id = {}
    id_to_name = {}

    # 確保按照 ID 排序，以符合 YOLO 的 0-79 順序
    categories = sorted(data['categories'], key=lambda x: x['id'])
    for idx, cat in enumerate(categories):
        coco_id_to_train_id[cat['id']] = idx
        id_to_name[idx] = cat['name']

    # 儲存每個類別的所有長寬比 (Width / Height)
    # 我們使用 log(w/h) 來確保 1/2 (高瘦) 和 2/1 (矮胖) 在數學距離上是對稱的
    class_ratios = {i: [] for i in range(80)}

    # 2. 遍歷所有標註框
    for ann in data['annotations']:
        # 忽略沒有 bbox 的標註 (如僅有 segmentation)
        if 'bbox' not in ann:
            continue

        cat_id = ann['category_id']
        if cat_id not in coco_id_to_train_id:
            continue

        train_id = coco_id_to_train_id[cat_id]
        w, h = ann['bbox'][2], ann['bbox'][3]

        # 避免除以零或極端數據
        if w > 1 and h > 1:
            ratio = w / h
            class_ratios[train_id].append(np.log(ratio))

    # 3. 計算每個類別的「平均幾何特徵」
    # 我們對 80 個類別進行分群，而不是對所有框分群
    X = []
    valid_classes = []

    for i in range(80):
        if len(class_ratios[i]) > 0:
            # 取中位數或是平均值
            avg_log_ratio = np.mean(class_ratios[i])
            X.append([avg_log_ratio])
            valid_classes.append(i)
        else:
            print(f"Warning: Class {i} ({id_to_name[i]}) has no boxes. Defaulting to square.")
            X.append([0.0]) # 預設為方形
            valid_classes.append(i)

    # 4. 執行 K-Means 分群
    print(f"Running K-Means clustering (k={n_heads})...")
    kmeans = KMeans(n_clusters=n_heads, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    # 5. 整理結果：根據長寬比排序 Head ID
    # 我們希望 Head ID 有物理意義：
    # Head 0: 最瘦長 (Smallest Ratio) -> ... -> Head 3: 最扁平 (Largest Ratio)
    sorted_indices = np.argsort(centers.flatten())

    # 建立映射：KMeans Cluster ID -> Sorted Head ID
    cluster_to_head = {old_id: new_id for new_id, old_id in enumerate(sorted_indices)}

    # 準備 YAML 資料結構
    head_assignments = {}

    print("-" * 60)
    print(f"{'Head ID':<10} {'Avg Ratio':<12} {'Description':<15} {'Count':<5} {'Examples'}")
    print("-" * 60)

    for new_head_id in range(n_heads):
        # 找出原本的 cluster id
        old_cluster_id = sorted_indices[new_head_id]

        # 找出屬於這個 Head 的類別
        classes_in_head = [valid_classes[i] for i, label in enumerate(labels) if label == old_cluster_id]
        classes_in_head.sort()

        # 計算實際平均長寬比 (還原 log)
        avg_ratio_val = np.exp(centers[old_cluster_id][0])

        # 自動生成描述
        if avg_ratio_val < 0.75: desc = "Tall (Vertical)"
        elif avg_ratio_val > 1.3: desc = "Wide (Horizontal)"
        else: desc = "Square (Central)"

        head_key = f'head_{new_head_id}'
        head_assignments[head_key] = {
            'name': f"{desc}_Group",
            'classes': classes_in_head,
            'weight': 1.0,  # 預設權重，後續可用 RL 調整
            'comment': f"Avg Ratio: {avg_ratio_val:.2f}"
        }

        # 顯示預覽
        example_names = [id_to_name[c] for c in classes_in_head[:4]]
        examples_str = ", ".join(example_names) + "..."
        print(f"{new_head_id:<10} {avg_ratio_val:<12.2f} {desc:<15} {len(classes_in_head):<5} {examples_str}")

    print("-" * 60)

    # 6. 輸出 YAML 檔案
    yaml_data = {
        'nc': 80,
        'heads': n_heads,
        'grouping': 'geometry_kmeans',
        'generated_by': 'utils/geometry_grouping.py',
        'head_assignments': head_assignments
    }

    # 確保目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=None)

    print(f"\n✅ Configuration saved to: {output_path}")
    print("Next Step: Run training with this config using:")
    print(f"python train.py --head-config {output_path} ...")

if __name__ == "__main__":
    generate_geometry_config()
