"""独立评估（测试集），包含指标报告和可选的混淆矩阵可视化。"""

import os

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from metrics import DOAMetrics
from utils.visualization import save_confusion_matrix
from utils.angle import bins_to_angles


class Evaluator:
    """在给定数据集上评估已训练的模型。

    参数
    ----------
    model : nn.Module
    dataloader : DataLoader
    cfg : Config
    logger : logging.Logger
    """

    def __init__(self, model, dataloader, cfg, logger):
        self.model = model
        self.dataloader = dataloader
        self.cfg = cfg
        self.logger = logger

        self.device = torch.device(
            cfg.train.device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

        m = cfg.model
        self.metrics = DOAMetrics(
            num_classes=m.num_classes,
            azimuth_range=tuple(m.azimuth_range),
        )
        self.use_amp = cfg.train.amp and self.device.type == "cuda"

    @torch.no_grad()
    def evaluate(self, save_cm: bool = True) -> dict:
        """运行评估并返回指标。

        参数:
            save_cm: 若为 ``True``，则保存混淆矩阵图片。

        返回:
            指标名称到数值的字典。
        """
        self.metrics.reset()

        for batch in self.dataloader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            labels = batch["azimuth_label"]

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                out = self.model(batch)

            logits_np = out["logits"].float().cpu().numpy()
            labels_np = labels.cpu().numpy()

            # 提取真实角度
            true_degs = batch.get("azimuth_deg")
            if true_degs is not None:
                if isinstance(true_degs, torch.Tensor):
                    true_degs = true_degs.cpu().numpy()
                else:
                    true_degs = np.array(true_degs)

            # 提取回归预测（如果有）
            pred_degs = None
            if "angle" in out:
                # 从弧度转为度
                pred_angle_rad = out["angle"].float().cpu()
                pred_degs = torch.rad2deg(pred_angle_rad).numpy()

            self.metrics.update(logits_np, labels_np, true_degs, pred_degs)

        results = self.metrics.compute()

        self.logger.info("=== 评估结果 ===")
        for k, v in results.items():
            self.logger.info(f"  {k}: {v:.4f}")

        if save_cm:
            cm = self.metrics.confusion_matrix()
            m = self.cfg.model
            num_classes = m.num_classes
            angles = bins_to_angles(
                np.arange(num_classes), num_classes, tuple(m.azimuth_range)
            )
            labels_str = [f"{a:.0f}°" for a in angles]
            cm_path = os.path.join(self.cfg.output.log_dir, "confusion_matrix.png")
            save_confusion_matrix(cm, cm_path, labels=labels_str)
            self.logger.info(f"  混淆矩阵已保存至 {cm_path}")

        return results
