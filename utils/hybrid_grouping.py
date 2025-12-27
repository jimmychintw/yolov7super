"""
混合分群策略 (方案四)：幾何相似度 + 共現相似度 + 譜聚類

此腳本結合兩種分群特徵：
1. 幾何相似度：基於類別平均長寬比 log(w/h) 的距離
2. 共現相似度：基於同一張圖片中同時出現的類別關聯

使用譜聚類 (Spectral Clustering) 來處理圖結構的相似度矩陣，
這比 K-Means 更能捕捉複雜的非凸形狀關係。

使用方法:
    python utils/hybrid_grouping.py

輸出:
    data/coco_320_1b4h_hybrid.yaml
"""

import json
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
import yaml
import os
from collections import defaultdict


def balance_clusters(labels, hybrid_sim, n_heads, n_classes, target_range=(15, 25)):
    """
    平衡各群組的類別數量

    透過將邊界類別從過大的群組移動到過小的群組來達成平衡。
    邊界類別 = 與目標群組有較高相似度的類別

    Args:
        labels: 原始分群標籤 (n_classes,)
        hybrid_sim: 混合相似度矩陣 (n_classes, n_classes)
        n_heads: Head 數量
        n_classes: 類別總數
        target_range: 目標類別數範圍 (min, max)

    Returns:
        balanced_labels: 平衡後的分群標籤
    """
    balanced_labels = labels.copy()
    min_size, max_size = target_range
    ideal_size = n_classes // n_heads

    max_iterations = 100
    for iteration in range(max_iterations):
        # 計算各群組大小
        cluster_sizes = {h: 0 for h in range(n_heads)}
        for label in balanced_labels:
            cluster_sizes[label] += 1

        # 找出過大和過小的群組
        oversized = [h for h, s in cluster_sizes.items() if s > max_size]
        undersized = [h for h, s in cluster_sizes.items() if s < min_size]

        if not oversized or not undersized:
            break

        # 從最大的群組移動到最小的群組
        largest = max(oversized, key=lambda h: cluster_sizes[h])
        smallest = min(undersized, key=lambda h: cluster_sizes[h])

        # 找出 largest 群組中與 smallest 群組最相似的類別 (邊界類別)
        classes_in_largest = [i for i, l in enumerate(balanced_labels) if l == largest]
        classes_in_smallest = [i for i, l in enumerate(balanced_labels) if l == smallest]

        best_class = None
        best_similarity = -1

        for cls in classes_in_largest:
            # 計算該類別與 smallest 群組的平均相似度
            if classes_in_smallest:
                avg_sim = np.mean([hybrid_sim[cls, c] for c in classes_in_smallest])
            else:
                avg_sim = 0

            if avg_sim > best_similarity:
                best_similarity = avg_sim
                best_class = cls

        if best_class is not None:
            balanced_labels[best_class] = smallest

    return balanced_labels


def compute_geometry_similarity(annotations, coco_id_to_train_id, n_classes=80):
    """
    計算幾何相似度矩陣

    基於每個類別的平均 log(w/h)，計算類別間的相似度。
    使用 RBF kernel: sim(i,j) = exp(-gamma * |log_ratio_i - log_ratio_j|^2)
    """
    # 收集每個類別的所有長寬比
    class_ratios = {i: [] for i in range(n_classes)}

    for ann in annotations:
        if 'bbox' not in ann:
            continue
        cat_id = ann['category_id']
        if cat_id not in coco_id_to_train_id:
            continue
        train_id = coco_id_to_train_id[cat_id]
        w, h = ann['bbox'][2], ann['bbox'][3]
        if w > 1 and h > 1:
            class_ratios[train_id].append(np.log(w / h))

    # 計算每個類別的平均 log ratio
    avg_ratios = np.zeros(n_classes)
    for i in range(n_classes):
        if len(class_ratios[i]) > 0:
            avg_ratios[i] = np.mean(class_ratios[i])
        else:
            avg_ratios[i] = 0.0  # 預設方形

    # 計算相似度矩陣 (使用 RBF kernel)
    gamma = 2.0  # 控制衰減速度
    geom_sim = np.zeros((n_classes, n_classes))

    for i in range(n_classes):
        for j in range(n_classes):
            dist = (avg_ratios[i] - avg_ratios[j]) ** 2
            geom_sim[i, j] = np.exp(-gamma * dist)

    return geom_sim, avg_ratios


def compute_cooccurrence_similarity(annotations, coco_id_to_train_id, n_classes=80):
    """
    計算共現相似度矩陣

    基於同一張圖片中同時出現的類別，建立共現次數矩陣，
    然後正規化為相似度。
    """
    # 建立圖片 -> 類別列表的映射
    image_classes = defaultdict(set)

    for ann in annotations:
        if 'bbox' not in ann:
            continue
        cat_id = ann['category_id']
        if cat_id not in coco_id_to_train_id:
            continue
        train_id = coco_id_to_train_id[cat_id]
        image_id = ann['image_id']
        image_classes[image_id].add(train_id)

    # 計算共現次數矩陣
    cooccur_count = np.zeros((n_classes, n_classes))
    class_count = np.zeros(n_classes)  # 每個類別出現的圖片數

    for img_id, classes in image_classes.items():
        classes_list = list(classes)
        for c in classes_list:
            class_count[c] += 1
        # 記錄共現
        for i, c1 in enumerate(classes_list):
            for c2 in classes_list[i:]:
                cooccur_count[c1, c2] += 1
                if c1 != c2:
                    cooccur_count[c2, c1] += 1

    # 正規化：使用 Jaccard 相似度
    # Jaccard(A, B) = |A ∩ B| / |A ∪ B| = cooccur / (count_A + count_B - cooccur)
    cooccur_sim = np.zeros((n_classes, n_classes))

    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                cooccur_sim[i, j] = 1.0
            else:
                union = class_count[i] + class_count[j] - cooccur_count[i, j]
                if union > 0:
                    cooccur_sim[i, j] = cooccur_count[i, j] / union
                else:
                    cooccur_sim[i, j] = 0.0

    return cooccur_sim, cooccur_count


def generate_hybrid_config(
    json_path='coco320/annotations/instances_train2017.json',
    output_path='data/coco_320_1b4h_hybrid.yaml',
    n_heads=4,
    alpha=0.5,  # 幾何權重
    beta=0.5,   # 共現權重
    balance=False,  # 是否平衡類別數
    verbose=True
):
    """
    執行混合分群並生成 YOLOv7 1B4H 設定檔

    Args:
        json_path: COCO 標註檔路徑
        output_path: 輸出 YAML 路徑
        n_heads: Head 數量
        alpha: 幾何相似度權重 (0-1)
        beta: 共現相似度權重 (0-1)
        balance: 是否啟用類別數平衡 (目標每個 Head 15-25 類)
        verbose: 是否輸出詳細資訊
    """
    # 正規化權重
    total = alpha + beta
    alpha = alpha / total
    beta = beta / total

    if verbose:
        print(f"混合分群參數：")
        print(f"  - 幾何權重 (α): {alpha:.2f}")
        print(f"  - 共現權重 (β): {beta:.2f}")
        print(f"  - Head 數量: {n_heads}")
        print(f"  - 類別數平衡: {'啟用' if balance else '停用'}")
        print()

    # 檢查路徑
    if not os.path.exists(json_path):
        print(f"Error: COCO annotations not found at {json_path}")
        print("Please check your dataset path.")
        return None

    print(f"Loading COCO annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 建立類別映射
    coco_id_to_train_id = {}
    id_to_name = {}
    categories = sorted(data['categories'], key=lambda x: x['id'])
    for idx, cat in enumerate(categories):
        coco_id_to_train_id[cat['id']] = idx
        id_to_name[idx] = cat['name']

    n_classes = len(categories)
    print(f"Found {n_classes} categories, {len(data['annotations'])} annotations")

    # 計算兩種相似度矩陣
    print("\n[1/4] 計算幾何相似度矩陣...")
    geom_sim, avg_ratios = compute_geometry_similarity(
        data['annotations'], coco_id_to_train_id, n_classes
    )

    print("[2/4] 計算共現相似度矩陣...")
    cooccur_sim, cooccur_count = compute_cooccurrence_similarity(
        data['annotations'], coco_id_to_train_id, n_classes
    )

    # 混合相似度矩陣
    print("[3/4] 計算混合相似度矩陣...")
    hybrid_sim = alpha * geom_sim + beta * cooccur_sim

    # 確保對稱性和正定性
    hybrid_sim = (hybrid_sim + hybrid_sim.T) / 2
    np.fill_diagonal(hybrid_sim, 1.0)

    # 執行譜聚類
    print(f"[4/4] 執行譜聚類 (n_clusters={n_heads})...")

    # 使用預計算的相似度矩陣
    clustering = SpectralClustering(
        n_clusters=n_heads,
        affinity='precomputed',
        random_state=42,
        n_init=10,
        assign_labels='kmeans'
    )
    labels = clustering.fit_predict(hybrid_sim)

    # 顯示原始分群結果
    if verbose:
        original_sizes = [sum(labels == h) for h in range(n_heads)]
        print(f"  原始分群大小: {original_sizes}")

    # 平衡類別數 (如果啟用)
    if balance:
        print("[4.5/4] 執行類別數平衡...")
        # 目標範圍：80/4 = 20，允許 ±5
        target_range = (n_classes // n_heads - 5, n_classes // n_heads + 5)
        labels = balance_clusters(labels, hybrid_sim, n_heads, n_classes, target_range)
        if verbose:
            balanced_sizes = [sum(labels == h) for h in range(n_heads)]
            print(f"  平衡後分群大小: {balanced_sizes}")

    # 整理結果：根據每個群組的平均幾何比例排序
    cluster_avg_ratio = {}
    for cluster_id in range(n_heads):
        classes_in_cluster = [i for i, label in enumerate(labels) if label == cluster_id]
        if classes_in_cluster:
            cluster_avg_ratio[cluster_id] = np.mean([avg_ratios[c] for c in classes_in_cluster])
        else:
            cluster_avg_ratio[cluster_id] = 0.0

    # 按平均比例排序
    sorted_clusters = sorted(cluster_avg_ratio.keys(), key=lambda x: cluster_avg_ratio[x])
    cluster_to_head = {old: new for new, old in enumerate(sorted_clusters)}

    # 準備 YAML 結構
    head_assignments = {}

    if verbose:
        print()
        print("=" * 80)
        print(f"{'Head':<6} {'Avg Ratio':<12} {'Type':<18} {'Count':<6} {'Top Classes'}")
        print("=" * 80)

    for new_head_id in range(n_heads):
        old_cluster_id = sorted_clusters[new_head_id]
        classes_in_head = sorted([i for i, label in enumerate(labels) if label == old_cluster_id])

        avg_ratio_val = np.exp(cluster_avg_ratio[old_cluster_id])

        # 自動生成描述
        if avg_ratio_val < 0.7:
            desc = "Tall_Vertical"
        elif avg_ratio_val < 0.9:
            desc = "Slight_Tall"
        elif avg_ratio_val > 1.4:
            desc = "Wide_Horizontal"
        elif avg_ratio_val > 1.1:
            desc = "Slight_Wide"
        else:
            desc = "Square_Central"

        # 計算這個 head 內的平均共現強度
        avg_cooccur = 0
        count = 0
        for i, c1 in enumerate(classes_in_head):
            for c2 in classes_in_head[i+1:]:
                avg_cooccur += cooccur_count[c1, c2]
                count += 1
        avg_cooccur = avg_cooccur / count if count > 0 else 0

        head_key = f'head_{new_head_id}'
        head_assignments[head_key] = {
            'name': f"{desc}_Group",
            'classes': [int(c) for c in classes_in_head],  # 確保是 Python int
            'weight': 1.0,
            'avg_ratio': float(round(avg_ratio_val, 3)),  # 轉為 Python float
            'avg_cooccurrence': float(round(avg_cooccur, 1)),  # 轉為 Python float
            'comment': f"Ratio: {avg_ratio_val:.2f}, Avg Co-occur: {avg_cooccur:.1f}"
        }

        if verbose:
            example_names = [id_to_name[c] for c in classes_in_head[:5]]
            examples_str = ", ".join(example_names)
            if len(classes_in_head) > 5:
                examples_str += "..."
            print(f"{new_head_id:<6} {avg_ratio_val:<12.3f} {desc:<18} {len(classes_in_head):<6} {examples_str}")

    if verbose:
        print("=" * 80)

    # 輸出 YAML
    yaml_data = {
        'nc': n_classes,
        'heads': n_heads,
        'grouping': 'hybrid_spectral_balanced' if balance else 'hybrid_spectral',
        'alpha': round(alpha, 2),
        'beta': round(beta, 2),
        'balanced': balance,
        'generated_by': 'utils/hybrid_grouping.py',
        'head_assignments': head_assignments
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=None, allow_unicode=True)

    print(f"\n✅ Configuration saved to: {output_path}")
    print("\n下一步：使用此設定檔進行訓練:")
    print(f"python train.py \\")
    print(f"    --weights backbone_elite.pt \\")
    print(f"    --transfer-weights \\")
    print(f"    --cfg cfg/training/yolov7-tiny-1b4h-strategy-b.yaml \\")
    print(f"    --head-config {output_path} \\")
    print(f"    --data data/coco320.yaml \\")
    print(f"    --heads 4 \\")
    print(f"    --freeze 50 \\")
    print(f"    --epochs 200 \\")
    print(f"    --batch-size 64 \\")
    print(f"    --name 1b4h_hybrid_freeze")

    return yaml_data


def analyze_grouping_quality(json_path, head_config_path):
    """
    分析分群品質：計算群內一致性和群間區分度
    """
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    with open(head_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 建立類別映射
    coco_id_to_train_id = {}
    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    for idx, cat in enumerate(categories):
        coco_id_to_train_id[cat['id']] = idx

    n_classes = len(categories)

    # 計算相似度矩陣
    geom_sim, _ = compute_geometry_similarity(
        coco_data['annotations'], coco_id_to_train_id, n_classes
    )
    cooccur_sim, _ = compute_cooccurrence_similarity(
        coco_data['annotations'], coco_id_to_train_id, n_classes
    )

    # 分析每個 head
    print("\n分群品質分析:")
    print("=" * 60)

    for head_key, head_info in config['head_assignments'].items():
        classes = head_info['classes']
        n = len(classes)

        # 群內幾何一致性
        intra_geom = 0
        intra_cooccur = 0
        count = 0
        for i, c1 in enumerate(classes):
            for c2 in classes[i+1:]:
                intra_geom += geom_sim[c1, c2]
                intra_cooccur += cooccur_sim[c1, c2]
                count += 1

        if count > 0:
            intra_geom /= count
            intra_cooccur /= count

        print(f"{head_key}: {n} classes")
        print(f"  群內幾何一致性: {intra_geom:.4f}")
        print(f"  群內共現一致性: {intra_cooccur:.4f}")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='混合分群：幾何 + 共現 + 譜聚類')
    parser.add_argument('--json', default='coco320/annotations/instances_train2017.json',
                        help='COCO annotations JSON path')
    parser.add_argument('--output', default='data/coco_320_1b4h_hybrid.yaml',
                        help='Output YAML path')
    parser.add_argument('--heads', type=int, default=4, help='Number of heads')
    parser.add_argument('--alpha', type=float, default=0.5, help='Geometry weight')
    parser.add_argument('--beta', type=float, default=0.5, help='Co-occurrence weight')
    parser.add_argument('--balance', action='store_true', help='Enable class count balancing (target: 15-25 per head)')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing config')

    args = parser.parse_args()

    if args.analyze and os.path.exists(args.output):
        analyze_grouping_quality(args.json, args.output)
    else:
        generate_hybrid_config(
            json_path=args.json,
            output_path=args.output,
            n_heads=args.heads,
            alpha=args.alpha,
            beta=args.beta,
            balance=args.balance
        )
