# utils/head_config.py
"""
HeadConfig - 1B4H 設定檔解析模組

功能：
1. 載入並解析 head_assignments YAML
2. 建立 global_id → head_id 映射
3. 建立 global_id → local_id 映射
4. 驗證設定檔完整性

Phase 1 實作
參考文件: SDD v1.0, PRD v0.3
"""

import yaml
from pathlib import Path


class HeadConfig:
    """
    1B4H 設定檔管理器

    負責解析 YAML 設定檔，建立類別到 Head 的映射關係。

    Attributes:
        nc (int): 總類別數 (80)
        num_heads (int): Head 數量 (4)
        head_map (dict): global_class_id → head_id 映射
        local_id_map (dict): global_class_id → local_class_id 映射
        weights (list): 每個 Head 的 Loss 權重
        head_classes (list): 每個 Head 負責的類別列表
        full_head (bool): 是否使用 full-size head (輸出維度 = 5 + nc)
    """

    def __init__(self, config_path: str, full_head: bool = None):
        """
        初始化 HeadConfig

        Args:
            config_path: YAML 設定檔路徑
            full_head: 是否使用 full-size head
                       - None: 從 YAML 讀取 (預設 False)
                       - True/False: 覆蓋 YAML 設定

        Raises:
            FileNotFoundError: 設定檔不存在
            ValueError: 設定檔格式錯誤或類別分配有誤
        """
        self.config_path = Path(config_path)
        self._full_head_override = full_head  # 儲存覆蓋值
        self._load_config()
        self._build_mappings()
        self._validate()

    def _load_config(self):
        """載入 YAML 設定檔"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"設定檔不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 基本參數
        self.nc = self.config.get('nc', 80)  # 總類別數
        self.num_heads = self.config.get('heads', 4)  # Head 數量
        self.grouping = self.config.get('grouping', 'standard')  # 分類方式

        # full_head 設定：優先使用覆蓋值，否則從 YAML 讀取
        if self._full_head_override is not None:
            self.full_head = self._full_head_override
        else:
            self.full_head = self.config.get('full_head', False)

        # Head 分配設定
        if 'head_assignments' not in self.config:
            raise ValueError("設定檔缺少 head_assignments 區段")
        self.head_assignments = self.config['head_assignments']

    def _build_mappings(self):
        """建立映射表"""
        # global_id → head_id
        self.head_map = {}

        # global_id → local_id (0 ~ nc_per_head-1)
        self.local_id_map = {}

        # head_id → weight
        self.weights = []

        # head_id → [global_ids]
        self.head_classes = []

        # head_id → name
        self.head_names = []

        # head_id → nc (該 Head 的類別數)
        self.head_nc_list = []

        for head_id in range(self.num_heads):
            head_key = f'head_{head_id}'

            if head_key not in self.head_assignments:
                raise ValueError(f"設定檔缺少 {head_key} 區段")

            head_info = self.head_assignments[head_key]

            # 取得該 Head 的類別列表
            classes = head_info.get('classes', [])
            if not classes:
                raise ValueError(f"{head_key} 的 classes 為空")

            # 取得權重 (預設 1.0)
            weight = head_info.get('weight', 1.0)

            # 取得名稱
            name = head_info.get('name', f'Head {head_id}')

            self.weights.append(weight)
            self.head_classes.append(classes)
            self.head_names.append(name)
            self.head_nc_list.append(len(classes))

            # 建立映射
            for local_id, global_id in enumerate(classes):
                if global_id in self.head_map:
                    existing_head = self.head_map[global_id]
                    raise ValueError(
                        f"類別 {global_id} 重複分配: head_{existing_head} 和 head_{head_id}"
                    )
                self.head_map[global_id] = head_id
                self.local_id_map[global_id] = local_id

    def _validate(self):
        """驗證設定檔完整性"""
        # 檢查是否所有類別都有被分配
        assigned_classes = set(self.head_map.keys())
        expected_classes = set(range(self.nc))

        missing = expected_classes - assigned_classes
        if missing:
            raise ValueError(f"缺少類別分配: {sorted(missing)}")

        extra = assigned_classes - expected_classes
        if extra:
            raise ValueError(f"無效的類別 ID (超出 0-{self.nc-1} 範圍): {sorted(extra)}")

        # 統計資訊
        total_classes = sum(len(classes) for classes in self.head_classes)
        if total_classes != self.nc:
            raise ValueError(
                f"類別總數不符: 分配了 {total_classes} 類，預期 {self.nc} 類"
            )

    def get_head_id(self, global_id: int) -> int:
        """
        取得類別所屬的 Head ID

        Args:
            global_id: COCO 類別 ID (0-79)

        Returns:
            head_id: 所屬 Head ID (0-3)

        Raises:
            KeyError: 無效的類別 ID
        """
        if global_id not in self.head_map:
            raise KeyError(f"無效的類別 ID: {global_id}")
        return self.head_map[global_id]

    def get_local_id(self, global_id: int) -> int:
        """
        將 global_id 轉換為該 Head 內的 local_id

        Args:
            global_id: COCO 類別 ID (0-79)

        Returns:
            local_id: Head 內的類別 ID (0-19)

        Raises:
            KeyError: 無效的類別 ID
        """
        if global_id not in self.local_id_map:
            raise KeyError(f"無效的類別 ID: {global_id}")
        return self.local_id_map[global_id]

    def get_head_info(self, global_id: int) -> tuple:
        """
        取得類別的完整資訊

        Args:
            global_id: COCO 類別 ID (0-79)

        Returns:
            (head_id, local_id): Head ID 和 Local ID 的元組
        """
        return self.get_head_id(global_id), self.get_local_id(global_id)

    def get_head_weight(self, head_id: int) -> float:
        """
        取得指定 Head 的 Loss 權重

        Args:
            head_id: Head ID (0-3)

        Returns:
            weight: Loss 權重
        """
        if head_id < 0 or head_id >= self.num_heads:
            raise IndexError(f"無效的 Head ID: {head_id}")
        return self.weights[head_id]

    def get_head_classes(self, head_id: int) -> list:
        """
        取得指定 Head 負責的所有類別

        Args:
            head_id: Head ID (0-3)

        Returns:
            classes: Global class ID 列表
        """
        if head_id < 0 or head_id >= self.num_heads:
            raise IndexError(f"無效的 Head ID: {head_id}")
        return self.head_classes[head_id].copy()

    def get_head_nc(self, head_id: int) -> int:
        """
        取得指定 Head 的類別數量

        Args:
            head_id: Head ID (0-3)

        Returns:
            nc: 類別數量
        """
        if head_id < 0 or head_id >= self.num_heads:
            raise IndexError(f"無效的 Head ID: {head_id}")
        return self.head_nc_list[head_id]

    def get_head_name(self, head_id: int) -> str:
        """
        取得指定 Head 的名稱

        Args:
            head_id: Head ID (0-3)

        Returns:
            name: Head 名稱
        """
        if head_id < 0 or head_id >= self.num_heads:
            raise IndexError(f"無效的 Head ID: {head_id}")
        return self.head_names[head_id]

    def summary(self) -> str:
        """
        產生設定摘要

        Returns:
            str: 設定摘要字串
        """
        lines = [
            f"HeadConfig Summary",
            f"==================",
            f"Config: {self.config_path}",
            f"Total classes: {self.nc}",
            f"Number of heads: {self.num_heads}",
            f"Grouping: {self.grouping}",
            f"",
        ]

        for head_id in range(self.num_heads):
            name = self.head_names[head_id]
            nc = self.head_nc_list[head_id]
            weight = self.weights[head_id]
            classes = self.head_classes[head_id]
            lines.append(f"Head {head_id}: {name}")
            lines.append(f"  - Classes: {nc}")
            lines.append(f"  - Weight: {weight}")
            lines.append(f"  - IDs: {classes[:5]}...{classes[-2:]}" if nc > 7 else f"  - IDs: {classes}")
            lines.append("")

        return "\n".join(lines)

    def __repr__(self):
        return f"HeadConfig(nc={self.nc}, heads={self.num_heads}, grouping='{self.grouping}', path='{self.config_path}')"

    def __str__(self):
        return self.summary()


# 單獨執行時的測試
if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'data/coco_320_1b4h_standard.yaml'

    try:
        cfg = HeadConfig(config_path)
        print(cfg.summary())

        # 測試映射
        print("\n=== 映射測試 ===")
        test_ids = [0, 1, 14, 29, 60, 79]
        for gid in test_ids:
            head_id, local_id = cfg.get_head_info(gid)
            print(f"Global ID {gid:2d} → Head {head_id}, Local ID {local_id:2d}")

        print("\n=== 驗證通過 ===")

    except Exception as e:
        print(f"錯誤: {e}")
        sys.exit(1)
