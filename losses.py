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
    """

    def __init__(
        self,
        num_classes: int = 72,
        label_smoothing: float = 0.0,
        anti_confusion_weight: float = 0.0,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')
        self.num_classes = num_classes
        self.anti_confusion_weight = anti_confusion_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        参数:
            logits:  ``[B, num_classes]``
            targets: ``[B]`` -- 整数类别索引

        返回:
            标量损失张量。
        """
        # 基础交叉熵
        ce_loss = self.ce(logits, targets)  # [B]

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
            ce_loss = ce_loss * (1.0 + confusion_penalty)

        return ce_loss.mean()


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
    ):
        super().__init__()
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.regression_weight = regression_weight
        self.use_angular_loss = use_angular_loss

    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_angle: torch.Tensor,
        target_label: torch.Tensor,
        target_angle: torch.Tensor,
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

        return {
            'total': total_loss,
            'classification': cls_loss.item(),
            'regression': reg_loss.item(),
        }
