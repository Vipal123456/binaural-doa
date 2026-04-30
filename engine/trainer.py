"""训练循环，包含 AMP 混合精度、梯度裁剪、学习率调度和检查点保存。"""

import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from losses import DOALoss, MultiTaskDOALoss, DOAVectorRegressionLoss, PureRegressionDOALoss
from metrics import DOAMetrics
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import TBWriter
from utils.early_stopping import EarlyStopping
from dataset.augmentation import BinauralAugmentation


class Trainer:
    """封装训练循环。

    参数
    ----------
    model : nn.Module
        DOA 网络模型。
    train_loader : DataLoader
        训练数据加载器。
    val_loader : DataLoader
        验证数据加载器。
    cfg : Config
        完整配置对象。
    logger : logging.Logger
        用于控制台/文件输出的 Python 日志记录器。
    """

    def __init__(self, model, train_loader, val_loader, cfg, logger):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.logger = logger

        t = cfg.train
        m = cfg.model
        self.model_type = getattr(m, "type", "binaural_doa_net")

        # 设备
        self.device = torch.device(t.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 损失函数
        # 检查是否使用多任务loss（分类+回归）
        self.use_regression = getattr(m, 'use_regression', False)
        self.use_pure_regression = getattr(m, 'use_pure_regression', False)
        self.use_vector_regression = self.model_type == "sdel_doa_reg"
        if self.use_pure_regression:
            front_back_aux_weight = getattr(t, 'front_back_aux_weight', 0.0)
            self.criterion = PureRegressionDOALoss(
                front_back_aux_weight=front_back_aux_weight,
            )
            msg = "使用原生 DOA-Net 纯回归loss (cosine angle-vector)"
            if front_back_aux_weight > 0:
                msg += f" + front/back辅助头(weight={front_back_aux_weight})"
            self.logger.info(msg)
        elif self.use_vector_regression:
            front_back_aux_weight = getattr(t, 'front_back_aux_weight', 0.0)
            self.criterion = DOAVectorRegressionLoss(
                front_back_aux_weight=front_back_aux_weight,
            )
            msg = "使用 SDEL 向量回归loss"
            if front_back_aux_weight > 0:
                msg += f" + front/back辅助头(weight={front_back_aux_weight})"
            self.logger.info(msg)
        elif self.use_regression:
            regression_weight = getattr(t, 'regression_weight', 0.5)
            use_angular_loss = getattr(t, 'use_angular_loss', True)
            anti_confusion_weight = getattr(t, 'anti_confusion_weight', 0.0)
            front_back_aux_weight = getattr(t, 'front_back_aux_weight', 0.0)
            self.criterion = MultiTaskDOALoss(
                num_classes=m.num_classes,
                label_smoothing=t.label_smoothing,
                regression_weight=regression_weight,
                use_angular_loss=use_angular_loss,
                anti_confusion_weight=anti_confusion_weight,
                front_back_aux_weight=front_back_aux_weight,
            )
            self.logger.info(
                f"使用多任务loss (分类+回归): regression_weight={regression_weight}, "
                f"angular_loss={use_angular_loss}, anti_confusion={anti_confusion_weight}, "
                f"fbaux={front_back_aux_weight}"
            )
        else:
            # 获取前后消歧权重
            anti_confusion_weight = getattr(t, 'anti_confusion_weight', 0.0)
            circular_soft_label_weight = getattr(t, 'circular_soft_label_weight', 0.0)
            circular_kappa = getattr(t, 'circular_kappa', 4.0)
            front_back_aux_weight = getattr(t, 'front_back_aux_weight', 0.0)
            front_back_focus_weight = getattr(t, 'front_back_focus_weight', 0.0)
            front_back_focus_window_deg = getattr(t, 'front_back_focus_window_deg', 20.0)
            self.criterion = DOALoss(
                num_classes=m.num_classes,
                label_smoothing=t.label_smoothing,
                anti_confusion_weight=anti_confusion_weight,
                circular_soft_label_weight=circular_soft_label_weight,
                circular_kappa=circular_kappa,
                front_back_aux_weight=front_back_aux_weight,
                front_back_focus_weight=front_back_focus_weight,
                front_back_focus_window_deg=front_back_focus_window_deg,
            )
            msg = "使用分类loss"
            if anti_confusion_weight > 0:
                msg += f" + 前后消歧惩罚(weight={anti_confusion_weight})"
            if circular_soft_label_weight > 0:
                msg += (
                    f" + circular_soft_label(weight={circular_soft_label_weight},"
                    f" kappa={circular_kappa})"
                )
            if front_back_aux_weight > 0:
                msg += f" + front/back辅助头(weight={front_back_aux_weight})"
            if front_back_focus_weight > 0:
                msg += (
                    f" + 前后轴样本加权(weight={front_back_focus_weight},"
                    f" window={front_back_focus_window_deg}deg)"
                )
            self.logger.info(msg)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=t.lr,
            weight_decay=t.weight_decay,
        )

        # 学习率调度器
        if t.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t.cosine_T_max,
                eta_min=t.cosine_eta_min,
            )
        elif t.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=t.step_size,
                gamma=t.step_gamma,
            )
        else:
            self.scheduler = None

        # 混合精度训练
        self.use_amp = t.amp and self.device.type == "cuda"
        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)

        # 评估指标
        self.metrics = DOAMetrics(
            num_classes=m.num_classes,
            azimuth_range=tuple(m.azimuth_range),
        )

        # TensorBoard 日志
        self.tb_writer = TBWriter(log_dir=os.path.join(cfg.output.log_dir, "tb"))

        # 训练状态
        self.start_epoch = 0
        self.best_mae = float("inf")
        self.global_step = 0

        # 早停机制（如果配置中启用）
        self.early_stopping = None
        patience = int(getattr(t, 'early_stopping_patience', 0) or 0)
        if patience > 0:
            self.early_stopping = EarlyStopping(
                patience=patience,
                delta=0.01,  # 改善至少0.01度才算有效
                mode='min',  # MAE越小越好
                verbose=False  # 通过logger打印，不使用内置print
            )
            self.logger.info(f"早停机制已启用 (patience={patience})")

        # 数据增强（如果配置中启用）
        self.augmentation = None
        if hasattr(t, 'use_augmentation') and t.use_augmentation:
            aug_params = {
                'time_mask_max_frames': getattr(t, 'aug_time_mask', 20),
                'freq_mask_max_bins': getattr(t, 'aug_freq_mask', 8),
                'magnitude_std': getattr(t, 'aug_magnitude_std', 0.1),
                'ipd_noise_std': getattr(t, 'aug_ipd_std', 0.05),
                'ild_noise_std': getattr(t, 'aug_ild_std', 0.5),
                'prob': getattr(t, 'aug_prob', 0.5),
            }
            self.augmentation = BinauralAugmentation(**aug_params).to(self.device)
            self.logger.info(f"数据增强已启用: {aug_params}")

    # ------------------------------------------------------------------
    # 从检查点恢复训练
    # ------------------------------------------------------------------

    def resume(self, ckpt_path: str) -> None:
        """从检查点加载模型/优化器/调度器状态。"""
        self.logger.info(f"Resuming from {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path, map_location=str(self.device))
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_mae = ckpt.get("best_mae", float("inf"))
        self.global_step = ckpt.get("global_step", 0)
        self.logger.info(f"Resumed at epoch {self.start_epoch}, best MAE={self.best_mae:.2f}")

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """运行完整的训练循环。"""
        t = self.cfg.train
        for epoch in range(self.start_epoch, t.epochs):
            train_loss = self._train_one_epoch(epoch)
            self.logger.info(
                f"Epoch {epoch}/{t.epochs-1}  train_loss={train_loss:.4f}  "
                f"lr={self.optimizer.param_groups[0]['lr']:.6f}"
            )
            self.tb_writer.add_scalar("train/loss", train_loss, epoch)
            self.tb_writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            # 验证（跳过空验证集）
            is_best = False
            if epoch % t.val_interval == 0 and len(self.val_loader) > 0:
                val_results = self._validate(epoch)
                mae = val_results["mean_angular_error"]
                self.logger.info(
                    f"  val  acc={val_results['accuracy']:.4f}  "
                    f"top{self.metrics.top_k}_acc={val_results['top_k_accuracy']:.4f}  "
                    f"MAE={mae:.2f}°  median_AE={val_results['median_angular_error']:.2f}°"
                )
                for k, v in val_results.items():
                    self.tb_writer.add_scalar(f"val/{k}", v, epoch)

                # 保存最佳模型
                is_best = mae < self.best_mae
                if is_best:
                    self.best_mae = mae

                # 检查早停
                if self.early_stopping is not None:
                    should_stop = self.early_stopping(mae, epoch)
                    if should_stop:
                        self.logger.info(
                            f"\n早停触发！最佳 MAE={self.early_stopping.best_value:.2f}° "
                            f"已连续 {self.early_stopping.patience} 轮无改善。"
                        )
                        self._save(epoch, is_best)
                        self.tb_writer.close()
                        return  # 提前结束训练
            elif len(self.val_loader) == 0:
                self.logger.info("  (validation skipped - empty validation set)")

            self._save(epoch, is_best)

        self.tb_writer.close()
        self.logger.info("Training complete.")

    def _train_one_epoch(self, epoch: int) -> float:
        """执行一个训练轮次。返回平均损失值。"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        non_finite_batches = 0
        t = self.cfg.train

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)

            # 应用数据增强（如果启用）
            if self.augmentation is not None:
                batch = self.augmentation(batch)

            labels = batch["azimuth_label"]  # [B]

            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                out = self.model(batch)

                if self.use_pure_regression:
                    pred_vec = out["angle_vec"]
                    true_angle_deg = batch["azimuth_deg"].float()
                    true_angle_rad = torch.deg2rad(true_angle_deg)
                    target_vec = torch.stack(
                        [torch.sin(true_angle_rad), torch.cos(true_angle_rad)],
                        dim=-1,
                    ).float()
                    loss_dict = self.criterion(
                        pred_vec,
                        target_vec,
                        front_back_logits=out.get("front_back_logits"),
                        front_back_targets=batch.get("front_back_label"),
                    )
                    loss = loss_dict["total"]
                elif self.use_vector_regression:
                    pred_vec = out["angle_vec"]
                    true_angle_deg = batch["azimuth_deg"].float()
                    true_angle_rad = torch.deg2rad(true_angle_deg)
                    target_vec = torch.stack(
                        [torch.sin(true_angle_rad), torch.cos(true_angle_rad)],
                        dim=-1,
                    ).float()
                    loss_dict = self.criterion(
                        pred_vec,
                        target_vec,
                        front_back_logits=out.get("front_back_logits"),
                        front_back_targets=batch.get("front_back_label"),
                    )
                    loss = loss_dict["total"]
                elif self.use_regression:
                    # 多任务loss：需要分类label和回归angle
                    pred_logits = out["logits"]
                    pred_angle = out["angle"]

                    # 获取真实角度（弧度）
                    true_angle_deg = batch["azimuth_deg"]  # [B]
                    true_angle_rad = torch.deg2rad(true_angle_deg)  # [B], 转换为弧度

                    loss_dict = self.criterion(
                        pred_logits,
                        pred_angle,
                        labels,
                        true_angle_rad,
                        front_back_logits=out.get("front_back_logits"),
                        front_back_targets=batch.get("front_back_label"),
                    )
                    loss = loss_dict['total']
                else:
                    # 纯分类loss
                    loss_dict = self.criterion(
                        out["logits"],
                        labels,
                        front_back_logits=out.get("front_back_logits"),
                        front_back_targets=batch.get("front_back_label"),
                        front_back_focus_distance_deg=batch.get("front_back_focus_distance_deg"),
                    )
                    loss = loss_dict["total"]

            # 防止数值发散传播到参数：非有限loss直接跳过该batch
            if not torch.isfinite(loss):
                non_finite_batches += 1
                self.logger.warning(
                    f"  [Epoch {epoch}][{batch_idx}/{len(self.train_loader)}] "
                    f"non-finite loss detected ({loss.item()}); batch skipped"
                )
                self.optimizer.zero_grad(set_to_none=True)
                if non_finite_batches >= 20:
                    self.logger.warning(
                        f"Epoch {epoch}: non-finite batches reached {non_finite_batches}, "
                        "stopping this epoch early"
                    )
                    break
                continue

            self.scaler.scale(loss).backward()

            if t.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), t.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if batch_idx % t.log_interval == 0:
                if self.use_pure_regression:
                    extra = ""
                    if loss_dict.get("front_back") is not None:
                        extra = f" (reg={loss_dict['regression']:.4f}, fb={loss_dict['front_back']:.4f})"
                    self.logger.info(
                        f"  [Epoch {epoch}][{batch_idx}/{len(self.train_loader)}] "
                        f"loss={loss.item():.4f}{extra}"
                    )
                    self.tb_writer.add_scalar("train/step_loss_reg", loss_dict["regression"], self.global_step)
                    if loss_dict.get("front_back") is not None:
                        self.tb_writer.add_scalar("train/step_loss_fb", loss_dict["front_back"], self.global_step)
                elif self.use_vector_regression:
                    extra = ""
                    if loss_dict.get("front_back") is not None:
                        extra = f" (vec={loss_dict['classification']:.4f}, fb={loss_dict['front_back']:.4f})"
                    self.logger.info(
                        f"  [Epoch {epoch}][{batch_idx}/{len(self.train_loader)}] "
                        f"loss={loss.item():.4f}{extra}"
                    )
                    self.tb_writer.add_scalar("train/step_loss_vec", loss_dict["classification"], self.global_step)
                    if loss_dict.get("front_back") is not None:
                        self.tb_writer.add_scalar("train/step_loss_fb", loss_dict["front_back"], self.global_step)
                elif self.use_regression:
                    # 多任务loss：显示分类和回归loss
                    self.logger.info(
                        f"  [Epoch {epoch}][{batch_idx}/{len(self.train_loader)}] "
                        f"loss={loss.item():.4f} "
                        f"(cls={loss_dict['classification']:.4f}, reg={loss_dict['regression']:.4f}"
                        + (
                            f", fb={loss_dict['front_back']:.4f})"
                            if loss_dict.get('front_back') is not None else ")"
                        )
                    )
                    self.tb_writer.add_scalar("train/step_loss_cls", loss_dict['classification'], self.global_step)
                    self.tb_writer.add_scalar("train/step_loss_reg", loss_dict['regression'], self.global_step)
                    if loss_dict.get('front_back') is not None:
                        self.tb_writer.add_scalar("train/step_loss_fb", loss_dict['front_back'], self.global_step)
                else:
                    extra = ""
                    if loss_dict.get("front_back") is not None:
                        extra = f" (cls={loss_dict['classification']:.4f}, fb={loss_dict['front_back']:.4f})"
                    self.logger.info(
                        f"  [Epoch {epoch}][{batch_idx}/{len(self.train_loader)}] "
                        f"loss={loss.item():.4f}{extra}"
                    )
                    self.tb_writer.add_scalar("train/step_loss_cls", loss_dict["classification"], self.global_step)
                    if loss_dict.get("front_back") is not None:
                        self.tb_writer.add_scalar("train/step_loss_fb", loss_dict["front_back"], self.global_step)
                self.tb_writer.add_scalar("train/step_loss", loss.item(), self.global_step)

        return total_loss / max(num_batches, 1)

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        """运行验证并返回指标字典。"""
        self.model.eval()
        self.metrics.reset()

        for batch in self.val_loader:
            batch = self._to_device(batch)
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
            if "angle_vec" in out:
                pred_xy = out["angle_vec"].float().cpu()
                pred_degs = torch.rad2deg(torch.atan2(pred_xy[:, 0], pred_xy[:, 1])).numpy()
            elif "angle" in out:
                # 从弧度转为度
                pred_angle_rad = out["angle"].float().cpu()
                pred_degs = torch.rad2deg(pred_angle_rad).numpy()

            self.metrics.update(logits_np, labels_np, true_degs, pred_degs)

        return self.metrics.compute()

    # ------------------------------------------------------------------
    # 检查点保存
    # ------------------------------------------------------------------

    def _save(self, epoch: int, is_best: bool) -> None:
        """保存最新检查点（以及可选的最佳检查点）。"""
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_mae": self.best_mae,
            "global_step": self.global_step,
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()

        save_dir = self.cfg.output.save_dir
        save_checkpoint(state, save_dir, "latest.pth")

        if is_best:
            save_checkpoint(state, save_dir, "best.pth")
            self.logger.info(f"  ★ New best model saved (MAE={self.best_mae:.2f}°)")

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _to_device(self, batch: dict) -> dict:
        """将批次字典中的张量转移到训练设备上。"""
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out
