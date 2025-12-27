# utils/class_aware_augment.py
"""
Stochastic Class-Aware Augmentation 模組

功能：
1. 載入每個 Head 的最佳增強參數
2. 統計圖片中各 Head 的類別分布
3. 使用隨機輪盤策略選擇增強參數
4. (可選) 動態標籤過濾

來源: HPO 搜索結果 (2025-12-05)
參考: CLASS_AWARE_AUGMENTATION_PLAN.md v2.0

使用方法:
    python train.py --class-aware-aug --head-params data/hyp.head_params.yaml
"""

import random
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


class StochasticClassAwareAugmentation:
    """
    隨機類別感知增強策略

    根據圖片中物體的 Head 分布，使用隨機輪盤法選擇一個 Head 的參數。
    避免加權平均造成的「所有 Head 都吃不飽」問題。

    Attributes:
        head_params (dict): 每個 Head 的增強參數 (degrees, flipud, fliplr, shear, mixup)
        head_map (dict): global_class_id → head_id 映射
        num_heads (int): Head 數量 (預設 4)
        label_filtering (dict): 動態標籤過濾設定
    """

    # 預設參數鍵值
    PARAM_KEYS = ['degrees', 'flipud', 'fliplr', 'shear', 'mixup']

    def __init__(self,
                 head_params_path: str,
                 head_config=None,
                 head_map: Dict[int, int] = None):
        """
        初始化 Stochastic Class-Aware Augmentation

        Args:
            head_params_path: hyp.head_params.yaml 路徑
            head_config: HeadConfig 實例 (可選，用於取得 head_map)
            head_map: 直接提供 global_id → head_id 映射 (可選)

        Raises:
            FileNotFoundError: 參數檔案不存在
            ValueError: 參數格式錯誤
        """
        self.head_params_path = Path(head_params_path)
        self.num_heads = 4

        # 載入 Head 參數
        self._load_head_params()

        # 設定 head_map
        if head_config is not None:
            self.head_map = head_config.head_map
        elif head_map is not None:
            self.head_map = head_map
        else:
            raise ValueError("必須提供 head_config 或 head_map")

        # 統計資訊
        self.selection_stats = {i: 0 for i in range(self.num_heads)}
        self.total_selections = 0

    def _load_head_params(self):
        """載入 Head 增強參數"""
        if not self.head_params_path.exists():
            raise FileNotFoundError(f"Head 參數檔案不存在: {self.head_params_path}")

        with open(self.head_params_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self.head_params = {}
        for head_id in range(self.num_heads):
            head_key = f'head_{head_id}'
            if head_key not in config:
                raise ValueError(f"參數檔案缺少 {head_key} 區段")

            params = config[head_key]
            # 驗證必要參數
            for key in self.PARAM_KEYS:
                if key not in params:
                    raise ValueError(f"{head_key} 缺少參數: {key}")

            self.head_params[head_id] = {
                'degrees': float(params['degrees']),
                'flipud': float(params['flipud']),
                'fliplr': float(params['fliplr']),
                'shear': float(params['shear']),
                'mixup': float(params['mixup']),
            }

        # 載入標籤過濾設定 (可選功能)
        self.label_filtering = config.get('label_filtering', {
            'enabled': False,
            'rotation_threshold': 5.0
        })

    def count_head_distribution(self, labels: np.ndarray) -> List[int]:
        """
        統計標籤中各 Head 的物體數量

        Args:
            labels: 標籤陣列，shape (N, 5) 或 (N, 6)
                    第一欄是 class_id

        Returns:
            head_counts: 長度為 num_heads 的列表，每個元素是該 Head 的物體數量
        """
        head_counts = [0] * self.num_heads

        if labels is None or len(labels) == 0:
            return head_counts

        for label in labels:
            class_id = int(label[0])
            if class_id in self.head_map:
                head_id = self.head_map[class_id]
                head_counts[head_id] += 1

        return head_counts

    def select_policy(self, head_counts: List[int]) -> Tuple[Dict, int]:
        """
        使用隨機輪盤法選擇增強策略

        根據各 Head 的物體數量作為權重，隨機選擇一個 Head 作為「主導者」，
        並返回該 Head 的完整參數。

        Args:
            head_counts: 各 Head 的物體數量列表

        Returns:
            (params, selected_head): 選中的參數字典和 Head ID
        """
        total = sum(head_counts)

        if total == 0:
            # 沒有物體時，隨機選擇
            selected_head = random.randint(0, self.num_heads - 1)
        else:
            # 使用物體數量作為權重進行隨機選擇
            selected_head = random.choices(
                range(self.num_heads),
                weights=head_counts,
                k=1
            )[0]

        # 更新統計
        self.selection_stats[selected_head] += 1
        self.total_selections += 1

        return self.head_params[selected_head].copy(), selected_head

    def select_policy_for_mosaic(self,
                                  labels_list: List[np.ndarray]) -> Tuple[Dict, int]:
        """
        為 Mosaic 增強選擇策略

        統計 4 張圖片的總類別分布，然後選擇一個全局策略。

        Args:
            labels_list: 4 張圖片的標籤列表

        Returns:
            (params, selected_head): 選中的參數字典和 Head ID
        """
        # 合併統計 4 張圖片的 Head 分布
        total_head_counts = [0] * self.num_heads

        for labels in labels_list:
            counts = self.count_head_distribution(labels)
            for i in range(self.num_heads):
                total_head_counts[i] += counts[i]

        return self.select_policy(total_head_counts)

    def should_filter_label(self,
                            class_id: int,
                            selected_head: int,
                            actual_rotation: float) -> bool:
        """
        判斷是否應該過濾標籤 (動態標籤過濾)

        當選擇了大旋轉策略時，過濾掉那些對旋轉敏感的類別。

        Args:
            class_id: 物體類別 ID
            selected_head: 選中的 Head ID
            actual_rotation: 實際施加的旋轉角度

        Returns:
            True 表示應該過濾此標籤
        """
        if not self.label_filtering.get('enabled', False):
            return False

        threshold = self.label_filtering.get('rotation_threshold', 5.0)

        # 只有當旋轉超過閾值時才考慮過濾
        if abs(actual_rotation) <= threshold:
            return False

        # 檢查此物體所屬的 Head
        if class_id not in self.head_map:
            return False

        object_head = self.head_map[class_id]

        # 如果物體的 Head 與選中的策略 Head 不同
        # 且物體的 Head 期望低旋轉 (Head 1 或 Head 3)
        if object_head != selected_head:
            # Head 1 和 Head 3 期望低旋轉 (degrees < 0.1)
            if object_head in [1, 3]:
                return True

        return False

    def get_stats_summary(self) -> str:
        """
        獲取統計摘要

        Returns:
            統計資訊字串
        """
        if self.total_selections == 0:
            return "尚無選擇統計"

        lines = ["Class-Aware Augmentation 統計:"]
        for head_id in range(self.num_heads):
            count = self.selection_stats[head_id]
            pct = count / self.total_selections * 100
            lines.append(f"  Head {head_id}: {count} ({pct:.1f}%)")
        lines.append(f"  總計: {self.total_selections}")

        return "\n".join(lines)

    def __repr__(self):
        return f"StochasticClassAwareAugmentation(path='{self.head_params_path}')"


# 單獨執行時的測試
if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from utils.head_config import HeadConfig

    # 測試用參數
    head_params_path = 'data/hyp.head_params.yaml'
    head_config_path = 'data/coco_320_1b4h_geometry.yaml'

    print("=== Stochastic Class-Aware Augmentation 測試 ===\n")

    try:
        # 載入 HeadConfig
        head_cfg = HeadConfig(head_config_path)
        print(f"載入 HeadConfig: {head_config_path}")

        # 初始化 Class-Aware Augmentation
        caa = StochasticClassAwareAugmentation(
            head_params_path=head_params_path,
            head_config=head_cfg
        )
        print(f"載入 Head 參數: {head_params_path}")

        # 印出各 Head 參數
        print("\n各 Head 增強參數:")
        for head_id, params in caa.head_params.items():
            print(f"  Head {head_id}: degrees={params['degrees']:.4f}, "
                  f"shear={params['shear']:.4f}, mixup={params['mixup']:.4f}")

        # 模擬測試
        print("\n=== 模擬測試 (1000 次) ===")

        # 模擬標籤分布
        test_cases = [
            ([10, 5, 3, 2], "以 Head 0 為主"),
            ([2, 15, 3, 0], "以 Head 1 為主"),
            ([1, 1, 20, 1], "以 Head 2 為主"),
            ([0, 2, 2, 16], "以 Head 3 為主"),
            ([5, 5, 5, 5], "平均分布"),
        ]

        for head_counts, desc in test_cases:
            # 重置統計
            caa.selection_stats = {i: 0 for i in range(4)}
            caa.total_selections = 0

            # 執行 1000 次選擇
            for _ in range(1000):
                caa.select_policy(head_counts)

            print(f"\n{desc} {head_counts}:")
            for head_id in range(4):
                pct = caa.selection_stats[head_id] / 10
                print(f"  Head {head_id}: {pct:.1f}%")

        print("\n=== 測試通過 ===")

    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
