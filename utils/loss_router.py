# utils/loss_router.py
"""
ComputeLossRouter - 1B4H 損失路由器

功能：
1. 將 targets 依據 HeadConfig 分配到對應的 Head
2. 將 global_class_id 轉換為 local_class_id
3. 計算各 Head 的 Loss 並加權求和
4. 實現「隱式負樣本挖掘」- 不屬於該 Head 的類別自動成為背景

Phase 1 實作 (non-OTA 版本)
參考文件: SDD v1.0, PRD v0.3
"""

import torch
import torch.nn as nn
from utils.general import bbox_iou
from utils.loss import smooth_BCE, FocalLoss, is_parallel


class ComputeLossRouter:
    """
    1B4H 損失路由器

    將多頭檢測的輸出與 targets 進行匹配，計算各 Head 的 Loss 並加權求和。
    採用「隱式負樣本挖掘」策略：每個 Head 只訓練其負責的類別，
    其他類別的物體對該 Head 而言是背景（負樣本）。
    """

    def __init__(self, model, head_config, autobalance=False):
        """
        初始化 ComputeLossRouter

        Args:
            model: YOLOv7 模型
            head_config: HeadConfig 實例
            autobalance: 是否自動平衡各層 Loss
        """
        self.device = next(model.parameters()).device
        self.head_config = head_config
        self.num_heads = head_config.num_heads

        # 取得超參數
        h = model.hyp
        self.hyp = h

        # 策略 A：是否忽略其他 Head 的物體
        self.ignore_other_heads = h.get('ignore_other_heads', False)
        # Soft ignore weight: 0 = hard ignore (完全排除), >0 = soft ignore (降低權重)
        self.ignore_weight = h.get('ignore_weight', 0.0)

        # BCE loss with label smoothing
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))

        # BCE Loss
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=self.device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=self.device))

        # Focal Loss (if gamma > 0)
        g = h.get('fl_gamma', 0.0)
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        self.BCEcls = BCEcls
        self.BCEobj = BCEobj

        # 取得檢測層參數
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]

        self.na = det.na  # anchor 數量 (3)
        self.nc = det.nc  # 總類別數 (80)
        self.nl = det.nl  # 檢測層數 (3)
        self.anchors = det.anchors  # [nl, na, 2]
        self.stride = det.stride if hasattr(det, 'stride') else None

        # 各 Head 的類別數
        self.head_nc = det.head_nc  # [20, 20, 20, 20]

        # 檢查是否使用 full_head 模式
        self.full_head = getattr(det, 'full_head', False)

        # 建立各 Head 負責的類別列表 (用於 full_head 模式)
        self.head_classes = []
        for head_id in range(self.num_heads):
            classes = head_config.get_head_classes(head_id)
            self.head_classes.append(classes)

        # 各層的 balance 權重 (P3, P4, P5)
        self.balance = {3: [4.0, 1.0, 0.4]}.get(self.nl, [4.0, 1.0, 0.25, 0.06, 0.02])

        self.autobalance = autobalance
        self.ssi = 0  # stride 16 index
        self.gr = 1.0  # IoU ratio for objectness

        # 建立類別到 Head 的快速查找表 (用於 GPU 加速)
        self.class_to_head = torch.zeros(self.nc, dtype=torch.long, device=self.device)
        self.class_to_local = torch.zeros(self.nc, dtype=torch.long, device=self.device)
        for global_id in range(self.nc):
            head_id, local_id = head_config.get_head_info(global_id)
            self.class_to_head[global_id] = head_id
            self.class_to_local[global_id] = local_id

    def __call__(self, predictions, targets):
        """
        計算 1B4H Loss

        Args:
            predictions: MultiHeadDetect 的訓練輸出
                         [[head0_p3, head0_p4, head0_p5], [head1_...], ...]
                         每個 tensor shape: [bs, na, ny, nx, no]
            targets: 標籤 [N, 6] (image_idx, class, x, y, w, h)
                     其中 x, y, w, h 是歸一化座標 (0-1)

        Returns:
            loss: 總 Loss (乘以 batch size)
            loss_items: [box_loss, obj_loss, cls_loss, total_loss] (detached)
        """
        device = targets.device

        # 初始化 Loss 累加器
        total_lbox = torch.zeros(1, device=device)
        total_lobj = torch.zeros(1, device=device)
        total_lcls = torch.zeros(1, device=device)

        # 遍歷每個 Head
        for head_id in range(self.num_heads):
            head_preds = predictions[head_id]  # [p3, p4, p5] for this head
            head_weight = self.head_config.get_head_weight(head_id)
            head_nc = self.head_nc[head_id]

            # 篩選屬於此 Head 的 targets 並轉換 class ID
            head_targets = self._filter_targets_for_head(targets, head_id)

            # 計算此 Head 的 Loss
            lbox, lobj, lcls = self._compute_head_loss(
                head_preds, head_targets, head_id, head_nc, all_targets=targets
            )

            # 加權累加
            total_lbox += lbox * head_weight
            total_lobj += lobj * head_weight
            total_lcls += lcls * head_weight

        # 乘以超參數權重
        total_lbox *= self.hyp['box']
        total_lobj *= self.hyp['obj']
        total_lcls *= self.hyp['cls']

        # 計算總 Loss
        bs = predictions[0][0].shape[0]  # batch size
        loss = total_lbox + total_lobj + total_lcls

        return loss * bs, torch.cat([total_lbox, total_lobj, total_lcls, loss]).detach()

    def _filter_targets_for_head(self, targets, head_id):
        """
        篩選屬於指定 Head 的 targets，並轉換 class ID

        Args:
            targets: [N, 6] (image_idx, class, x, y, w, h)
            head_id: Head ID (0-3)

        Returns:
            head_targets: [M, 6] 篩選後的 targets，class 已轉換為 local ID
        """
        if targets.shape[0] == 0:
            return targets

        # 取得各 target 的 class ID
        class_ids = targets[:, 1].long()

        # 找出屬於此 Head 的 targets
        target_heads = self.class_to_head[class_ids]
        mask = (target_heads == head_id)

        if not mask.any():
            # 此 Head 沒有對應的 targets
            return torch.zeros((0, 6), device=targets.device, dtype=targets.dtype)

        # 篩選 targets
        head_targets = targets[mask].clone()

        # 轉換 global class ID → local class ID
        global_ids = head_targets[:, 1].long()
        local_ids = self.class_to_local[global_ids]
        head_targets[:, 1] = local_ids.float()

        return head_targets

    def _compute_head_loss(self, head_preds, targets, head_id, head_nc, all_targets=None):
        """
        計算單一 Head 的 Loss

        Args:
            head_preds: 此 Head 的預測 [p3, p4, p5]
                        每個 tensor shape: [bs, na, ny, nx, no]
                        no = 4 + 1 + head_nc
            targets: 已篩選並轉換為 local_id 的 targets [M, 6]
            head_id: Head ID
            head_nc: 此 Head 的類別數 (20)
            all_targets: 完整的 targets（未篩選，用於策略 A）

        Returns:
            lbox, lobj, lcls: Box/Obj/Cls Loss (未乘以超參數權重)
        """
        device = head_preds[0].device
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        lcls = torch.zeros(1, device=device)

        # 建立 targets 到 anchor 的分配
        tcls, tbox, indices, anchors = self._build_targets(head_preds, targets, head_id)

        # 遍歷各檢測層 (P3, P4, P5)
        for i, pi in enumerate(head_preds):
            # pi shape: [bs, na, ny, nx, no]
            # no = 4 + 1 + head_nc

            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # objectness target

            n = b.shape[0]  # 此層的 target 數量

            if n:
                # 取得對應位置的預測
                ps = pi[b, a, gj, gi]  # [n, no]

                # Box regression (CIoU Loss)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat([pxy, pwh], dim=1)  # [n, 4]

                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # [n]
                lbox += (1.0 - iou).mean()

                # Objectness target (用 IoU 作為 soft label)
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)

                # Classification Loss
                if head_nc > 1:
                    if self.full_head:
                        # Full-head 模式: 輸出是 80 維，只計算該 Head 負責的類別
                        head_classes = self.head_classes[head_id]
                        cls_pred = ps[:, 5:]  # [n, 80]

                        # 只取該 Head 負責的類別的預測
                        cls_pred_head = cls_pred[:, head_classes]  # [n, head_nc]

                        # 建立 one-hot target with label smoothing
                        t = torch.full_like(cls_pred_head, self.cn, device=device)  # [n, head_nc]
                        t[range(n), tcls[i]] = self.cp
                        lcls += self.BCEcls(cls_pred_head, t)
                    else:
                        # 原始模式: 輸出是 head_nc 維
                        # 建立 one-hot target with label smoothing
                        t = torch.full_like(ps[:, 5:], self.cn, device=device)  # [n, head_nc]
                        t[range(n), tcls[i]] = self.cp
                        lcls += self.BCEcls(ps[:, 5:], t)

            # Objectness Loss (包含負樣本 - 重要！)
            # 即使此層沒有正樣本，也要計算 obj loss（全是負樣本）
            if self.ignore_other_heads and all_targets is not None:
                # 策略 A：忽略其他 Head 的物體位置
                ignore_mask = self._get_ignore_mask(i, head_preds[i], head_id, all_targets)
                valid_positions = ~ignore_mask  # [bs, na, ny, nx]

                if self.ignore_weight > 0:
                    # Soft ignore: 被忽略的位置仍計算 loss，但權重較低
                    n_valid = valid_positions.sum().item()
                    n_ignored = ignore_mask.sum().item()

                    if n_valid > 0 and n_ignored > 0:
                        # 分別計算兩部分的 loss
                        loss_valid = self.BCEobj(pi[..., 4][valid_positions], tobj[valid_positions])
                        loss_ignored = self.BCEobj(pi[..., 4][ignore_mask], tobj[ignore_mask])
                        # 加權平均：ignored positions 的貢獻降低
                        total_n = n_valid + n_ignored
                        obji = (loss_valid * n_valid + loss_ignored * n_ignored * self.ignore_weight) / total_n
                    elif n_valid > 0:
                        obji = self.BCEobj(pi[..., 4][valid_positions], tobj[valid_positions])
                    elif n_ignored > 0:
                        obji = self.BCEobj(pi[..., 4][ignore_mask], tobj[ignore_mask]) * self.ignore_weight
                    else:
                        obji = torch.tensor(0.0, device=device)
                else:
                    # Hard ignore: 完全排除被忽略的位置
                    obji = self.BCEobj(pi[..., 4][valid_positions], tobj[valid_positions])
            else:
                # 原有行為：計算所有位置的 obj loss
                obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]

        return lbox, lobj, lcls

    def _build_targets(self, preds, targets, head_id):
        """
        建立 targets 到 anchor 的分配 (簡化版，非 OTA)

        與原始 ComputeLoss.build_targets 邏輯相同，
        但使用此 Head 的 anchors 和預測。

        Args:
            preds: Head 預測 [p3, p4, p5]
            targets: [M, 6] (image_idx, local_class, x, y, w, h)
            head_id: Head ID

        Returns:
            tcls, tbox, indices, anchors: 各層的 targets 分配
        """
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []

        gain = torch.ones(7, device=targets.device).long()
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)

        if nt:
            targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # [na, nt, 7]

        g = 0.5  # offset bias
        off = torch.tensor([
            [0, 0],
            [1, 0], [0, 1], [-1, 0], [0, -1],
        ], device=targets.device).float() * g

        for i in range(self.nl):
            anchors_i = self.anchors[i]  # [na, 2]
            gain[2:6] = torch.tensor(preds[i].shape)[[3, 2, 3, 2]]  # [nx, ny, nx, ny]

            # Match targets to anchors
            if nt:
                t = targets * gain  # 縮放到 grid 座標

                # 根據 anchor ratio 篩選
                r = t[:, :, 4:6] / anchors_i[:, None]  # wh ratio [na, nt, 2]
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # [na, nt]
                t = t[j]  # 篩選後的 targets

                # 計算 offset
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0] if nt else torch.zeros((0, 7), device=targets.device)
                offsets = 0

            # 提取 targets 資訊
            if t.shape[0]:
                b, c = t[:, :2].long().T  # image, class
                gxy = t[:, 2:4]  # grid xy
                gwh = t[:, 4:6]  # grid wh
                gij = (gxy - offsets).long()
                gi, gj = gij.T  # grid indices

                a = t[:, 6].long()  # anchor indices
                indices.append((
                    b,
                    a,
                    gj.clamp_(0, gain[3] - 1),
                    gi.clamp_(0, gain[2] - 1)
                ))
                tbox.append(torch.cat((gxy - gij, gwh), 1))  # box (相對於 grid cell)
                anch.append(anchors_i[a])  # 對應的 anchors
                tcls.append(c)  # class (local ID)
            else:
                indices.append((
                    torch.zeros(0, dtype=torch.long, device=targets.device),
                    torch.zeros(0, dtype=torch.long, device=targets.device),
                    torch.zeros(0, dtype=torch.long, device=targets.device),
                    torch.zeros(0, dtype=torch.long, device=targets.device)
                ))
                tbox.append(torch.zeros((0, 4), device=targets.device))
                anch.append(torch.zeros((0, 2), device=targets.device))
                tcls.append(torch.zeros(0, dtype=torch.long, device=targets.device))

        return tcls, tbox, indices, anch

    def _get_ignore_mask(self, layer_idx, pred, head_id, all_targets):
        """
        策略 A：找出當前 Head 應該 Ignore 的位置（其他 Head 的獵物）

        Args:
            layer_idx: 檢測層 index (0=P3, 1=P4, 2=P5)
            pred: 當前層的預測 [bs, na, ny, nx, no]
            head_id: 當前 Head ID
            all_targets: 所有 targets [N, 6] (image_idx, global_class, x, y, w, h)

        Returns:
            ignore_mask: [bs, na, ny, nx] bool tensor
                         True = 該位置有其他 Head 的物體，應 ignore
                         False = 該位置可以計算 loss
        """
        device = pred.device
        bs, na, ny, nx = pred.shape[:4]

        # 初始化 ignore_mask，預設全為 False（可以計算 loss）
        ignore_mask = torch.zeros(bs, na, ny, nx, dtype=torch.bool, device=device)

        if all_targets.shape[0] == 0:
            return ignore_mask

        # 1. 篩選出「不屬於當前 Head」的 targets
        class_ids = all_targets[:, 1].long()
        target_heads = self.class_to_head[class_ids]
        other_head_mask = (target_heads != head_id)

        if not other_head_mask.any():
            # 沒有其他 Head 的物體
            return ignore_mask

        other_targets = all_targets[other_head_mask]  # [M, 6]

        # 2. 將其他 Head 的 targets 分配到 grid 位置
        # 使用與 _build_targets 類似的邏輯，但只需要位置信息

        # 計算 gain（grid size）
        gain = torch.tensor([1, 1, nx, ny, nx, ny], device=device, dtype=torch.float)
        t = other_targets * gain  # [M, 6], 縮放到 grid 座標

        # 對於每個 target，找到對應的 grid cell
        if t.shape[0] > 0:
            # 取得 grid 座標
            b = t[:, 0].long()  # image index
            gxy = t[:, 2:4]     # grid xy
            gwh = t[:, 4:6]     # grid wh

            # 計算中心點所在的 grid cell
            gij = gxy.long()
            gi, gj = gij[:, 0].clamp_(0, nx - 1), gij[:, 1].clamp_(0, ny - 1)

            # 對於每個 anchor，標記該位置為 ignore（向量化版本）
            # 這裡簡化處理：所有 anchor 都標記（保守策略）
            # 使用 broadcasting 一次性設定所有 anchor（避免 Python loop）
            num_targets = len(b)
            anchor_indices = torch.arange(na, device=device).unsqueeze(0).expand(num_targets, -1)  # [M, na]
            b_expanded = b.unsqueeze(1).expand(-1, na)  # [M, na]
            gj_expanded = gj.unsqueeze(1).expand(-1, na)  # [M, na]
            gi_expanded = gi.unsqueeze(1).expand(-1, na)  # [M, na]

            # 一次性標記所有位置（GPU 平行化）
            ignore_mask[b_expanded, anchor_indices, gj_expanded, gi_expanded] = True

        return ignore_mask


# 單獨執行時的測試
if __name__ == '__main__':
    print("ComputeLossRouter 模組載入測試")
    print("請在整合測試中驗證完整功能")
