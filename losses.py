"""DOA 分类的损失函数。"""

import torch
import torch.nn as nn
import numpy as np


class DOALoss(nn.Module):
    """带有可选标签平滑的交叉熵损失。

    参数
    ----------
    num_classes : int
        方位角分箱数量。
    label_smoothing : float
        标签平滑系数（0 = 不使用平滑）。
    anti_confusion_weight : float
        前后混淆惩罚权重（0 = 不使用）。
    circular_soft_label_weight : float
        circular soft label loss 的权重（0 = 不使用）。
    circular_kappa : float
        von-Mises 软标签集中度，越大越接近 one-hot。
    """

    def __init__(
        self,
        num_classes: int = 72,
        label_smoothing: float = 0.0,
        anti_confusion_weight: float = 0.0,
        circular_soft_label_weight: float = 0.0,
        circular_kappa: float = 4.0,
        front_back_aux_weight: float = 0.0,
        front_back_focus_weight: float = 0.0,
        front_back_focus_window_deg: float = 20.0,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')
        self.front_back_ce = nn.CrossEntropyLoss(reduction='none')
        self.num_classes = num_classes
        self.anti_confusion_weight = anti_confusion_weight
        self.circular_soft_label_weight = float(circular_soft_label_weight)
        self.circular_kappa = float(circular_kappa)
        self.front_back_aux_weight = float(front_back_aux_weight)
        self.front_back_focus_weight = float(front_back_focus_weight)
        self.front_back_focus_window_deg = float(front_back_focus_window_deg)

        # 预计算每个bin的中心角（弧度）用于构造圆周软标签。
        centers_deg = -180.0 + (torch.arange(num_classes).float() + 0.5) * (360.0 / num_classes)
        centers_rad = torch.deg2rad(centers_deg)
        self.register_buffer("bin_centers_rad", centers_rad, persistent=False)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        front_back_logits: torch.Tensor = None,
        front_back_targets: torch.Tensor = None,
        front_back_focus_distance_deg: torch.Tensor = None,
    ) -> dict:
        """
        参数:
            logits:  ``[B, num_classes]``
            targets: ``[B]`` -- 整数类别索引

        返回:
            标量损失张量。
        """
        # 基础交叉熵
        ce_loss = self.ce(logits, targets)  # [B]
        total_loss = ce_loss

        if self.anti_confusion_weight > 0:
            # 计算预测的bin和真实bin的差距
            pred_bins = logits.argmax(dim=-1)  # [B]
            bin_diff = torch.abs(pred_bins - targets)

            # 处理圆周包裹：例如 bin 0 和 bin 71 实际上距离是1
            bin_diff = torch.minimum(bin_diff, self.num_classes - bin_diff)

            # 检测前后混淆：bin差距约为 num_classes/2 (对应180°)
            # 例如 72个bin时，相差36个bin = 180°
            half_bins = self.num_classes / 2
            is_opposite = torch.abs(bin_diff - half_bins) < (self.num_classes * 0.15)  # ±15%容差

            # 对前后混淆的样本施加额外惩罚
            confusion_penalty = is_opposite.float() * self.anti_confusion_weight
            total_loss = total_loss * (1.0 + confusion_penalty)

        # Circular soft label loss（可选）
        if self.circular_soft_label_weight > 0:
            # 取目标bin对应中心角，构造von-Mises软标签分布
            centers = self.bin_centers_rad.to(logits.device)
            target_angles = centers[targets]                               # [B]
            all_centers = centers.unsqueeze(0)                             # [1, C]
            delta = all_centers - target_angles.unsqueeze(1)              # [B, C]
            soft_targets = torch.exp(self.circular_kappa * torch.cos(delta))
            soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)

            log_probs = torch.log_softmax(logits, dim=-1)
            circular_loss = -(soft_targets * log_probs).sum(dim=-1)       # [B]
            total_loss = total_loss + self.circular_soft_label_weight * circular_loss

        sample_weights = None
        if self.front_back_focus_weight > 0 and front_back_focus_distance_deg is not None:
            within_window = front_back_focus_distance_deg <= self.front_back_focus_window_deg
            sample_weights = 1.0 + within_window.float() * self.front_back_focus_weight
            total_loss = total_loss * sample_weights

        front_back_loss_value = None
        if (
            self.front_back_aux_weight > 0
            and front_back_logits is not None
            and front_back_targets is not None
        ):
            fb_loss = self.front_back_ce(front_back_logits, front_back_targets)
            if sample_weights is not None:
                fb_loss = fb_loss * sample_weights
            front_back_loss_value = fb_loss.mean()
            total_loss = total_loss.mean() + self.front_back_aux_weight * front_back_loss_value
        else:
            total_loss = total_loss.mean()

        return {
            "total": total_loss,
            "classification": ce_loss.mean().item(),
            "front_back": None if front_back_loss_value is None else front_back_loss_value.item(),
        }


class MultiTaskDOALoss(nn.Module):
    """多任务DOA损失：结合分类loss和回归loss。

    分类loss：帮助模型进行粗定位（识别方向区域）
    回归loss：帮助模型进行精定位（预测准确角度）

    参数
    ----------
    num_classes : int
        方位角分箱数量。
    label_smoothing : float
        分类loss的标签平滑系数。
    regression_weight : float
        回归loss的权重（分类loss权重固定为1.0）。
    use_angular_loss : bool
        是否使用考虑圆周性质的角度损失（而非简单MSE）。
    """

    def __init__(
        self,
        num_classes: int = 72,
        label_smoothing: float = 0.0,
        regression_weight: float = 0.5,
        use_angular_loss: bool = True,
        anti_confusion_weight: float = 0.0,
        front_back_aux_weight: float = 0.0,
    ):
        super().__init__()
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.regression_weight = regression_weight
        self.use_angular_loss = use_angular_loss
        self.num_classes = num_classes
        self.anti_confusion_weight = float(anti_confusion_weight)
        self.front_back_aux_weight = float(front_back_aux_weight)
        self.front_back_ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_angle: torch.Tensor,
        target_label: torch.Tensor,
        target_angle: torch.Tensor,
        front_back_logits: torch.Tensor = None,
        front_back_targets: torch.Tensor = None,
    ) -> dict:
        """
        参数:
            pred_logits:   ``[B, num_classes]`` - 分类logits
            pred_angle:    ``[B]`` - 预测角度（弧度，范围 [-π, π]）
            target_label:  ``[B]`` - 目标类别索引
            target_angle:  ``[B]`` - 目标角度（弧度，范围 [-π, π]）

        返回:
            dict: 包含 'total', 'classification', 'regression' 三个key的字典
        """
        # 1. 分类loss
        cls_loss = self.classification_loss(pred_logits, target_label)

        if self.anti_confusion_weight > 0:
            pred_bins = pred_logits.argmax(dim=-1)
            bin_diff = torch.abs(pred_bins - target_label)
            bin_diff = torch.minimum(bin_diff, self.num_classes - bin_diff)
            half_bins = self.num_classes / 2
            is_opposite = torch.abs(bin_diff - half_bins) < (self.num_classes * 0.15)
            confusion_penalty = is_opposite.float() * self.anti_confusion_weight
            cls_loss = cls_loss * (1.0 + confusion_penalty.mean())

        # 2. 回归loss
        if self.use_angular_loss:
            # 角度感知的损失：考虑圆周性质
            # 使用 1 - cos(Δθ) 作为损失，范围 [0, 2]
            # 当预测准确时loss=0，预测相反时loss=2
            angle_diff = pred_angle - target_angle
            reg_loss = (1 - torch.cos(angle_diff)).mean()
        else:
            # 简单的MSE loss
            reg_loss = nn.functional.mse_loss(pred_angle, target_angle)

        # 3. 总损失
        total_loss = cls_loss + self.regression_weight * reg_loss
        fb_loss_value = None
        if (
            self.front_back_aux_weight > 0
            and front_back_logits is not None
            and front_back_targets is not None
        ):
            fb_loss = self.front_back_ce(front_back_logits, front_back_targets)
            fb_loss_value = fb_loss
            total_loss = total_loss + self.front_back_aux_weight * fb_loss

        return {
            'total': total_loss,
            'classification': cls_loss.item(),
            'regression': reg_loss.item(),
            'front_back': None if fb_loss_value is None else fb_loss_value.item(),
        }


class DOAVectorRegressionLoss(nn.Module):
    """二维单位向量回归损失，用于水平面 DOA baseline。"""

    def __init__(self, front_back_aux_weight: float = 0.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.front_back_aux_weight = float(front_back_aux_weight)
        self.front_back_ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(
        self,
        pred_vec: torch.Tensor,
        target_vec: torch.Tensor,
        front_back_logits: torch.Tensor = None,
        front_back_targets: torch.Tensor = None,
    ) -> dict:
        reg_loss = self.mse(pred_vec, target_vec)
        total_loss = reg_loss
        fb_loss_value = None
        if (
            self.front_back_aux_weight > 0
            and front_back_logits is not None
            and front_back_targets is not None
        ):
            fb_loss = self.front_back_ce(front_back_logits, front_back_targets)
            fb_loss_value = fb_loss
            total_loss = total_loss + self.front_back_aux_weight * fb_loss

        return {
            "total": total_loss,
            "classification": reg_loss.item(),
            "front_back": None if fb_loss_value is None else fb_loss_value.item(),
        }


class PureRegressionDOALoss(nn.Module):
    """纯回归 DOA 损失：二维单位向量方向回归 + 可选 front/back 辅助。"""

    def __init__(self, front_back_aux_weight: float = 0.0):
        super().__init__()
        self.front_back_aux_weight = float(front_back_aux_weight)
        self.front_back_ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(
        self,
        pred_vec: torch.Tensor,
        target_vec: torch.Tensor,
        front_back_logits: torch.Tensor = None,
        front_back_targets: torch.Tensor = None,
    ) -> dict:
        pred_vec = nn.functional.normalize(pred_vec, dim=-1)
        target_vec = nn.functional.normalize(target_vec, dim=-1)
        cos_sim = torch.sum(pred_vec * target_vec, dim=-1).clamp(-1.0, 1.0)
        reg_loss = (1.0 - cos_sim).mean()

        total_loss = reg_loss
        fb_loss_value = None
        if (
            self.front_back_aux_weight > 0
            and front_back_logits is not None
            and front_back_targets is not None
        ):
            fb_loss = self.front_back_ce(front_back_logits, front_back_targets)
            fb_loss_value = fb_loss
            total_loss = total_loss + self.front_back_aux_weight * fb_loss

        return {
            "total": total_loss,
            "regression": reg_loss.item(),
            "front_back": None if fb_loss_value is None else fb_loss_value.item(),
        }
