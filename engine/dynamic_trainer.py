"""Training loop for moving DOA sequence estimation."""

import os

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from metrics_dynamic import DynamicDOAMetrics
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import TBWriter


class DynamicTrainer:
    def __init__(self, model, train_loader, val_loader, cfg, logger):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.use_amp = cfg.train.amp and self.device.type == "cuda"
        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
        self.scheduler = None
        if cfg.train.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.train.cosine_T_max,
                eta_min=cfg.train.cosine_eta_min,
            )
        self.metrics = DynamicDOAMetrics(
            num_classes=cfg.model.num_classes,
            azimuth_range=tuple(cfg.model.azimuth_range),
        )
        self.target_metrics = DynamicDOAMetrics(
            num_classes=cfg.model.num_classes,
            azimuth_range=tuple(cfg.model.azimuth_range),
        )
        self.rendered_metrics = DynamicDOAMetrics(
            num_classes=cfg.model.num_classes,
            azimuth_range=tuple(cfg.model.azimuth_range),
        )
        self.tb_writer = TBWriter(log_dir=os.path.join(cfg.output.log_dir, "tb"))
        self.start_epoch = 0
        self.global_step = 0
        self.best_mae = float("inf")
        self.early_stopping_patience = int(getattr(cfg.train, "early_stopping_patience", 0) or 0)
        self.no_improve_epochs = 0
        self.front_back_aux_weight = float(getattr(cfg.train, "front_back_aux_weight", 0.0) or 0.0)
        if self.front_back_aux_weight > 0:
            self.logger.info(f"Using moving front/back auxiliary loss weight={self.front_back_aux_weight}")

    def resume(self, ckpt_path: str) -> None:
        ckpt = load_checkpoint(ckpt_path, map_location=str(self.device))
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.global_step = ckpt.get("global_step", 0)
        self.best_mae = ckpt.get("best_mae", float("inf"))
        self.no_improve_epochs = ckpt.get("no_improve_epochs", 0)

    def fit(self):
        for epoch in range(self.start_epoch, self.cfg.train.epochs):
            train_loss = self._train_one_epoch(epoch)
            self.logger.info(
                f"Epoch {epoch}/{self.cfg.train.epochs-1}  train_loss={train_loss:.4f}  "
                f"lr={self.optimizer.param_groups[0]['lr']:.6f}"
            )
            self.tb_writer.add_scalar("train/loss", train_loss, epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            val_results = self.validate()
            mae = val_results["frame_mae"]
            self.logger.info(
                f"  val  acc={val_results['frame_accuracy']:.4f}  "
                f"MAE={mae:.2f}°  "
                f"Acc@5°={val_results['acc_at_5deg']:.4f}  "
                f"Acc@10°={val_results['acc_at_10deg']:.4f}  "
                f"OppErr={val_results['opposite_error_rate']:.4f}  "
                f"median_AE={val_results['median_angular_error']:.2f}°"
            )
            if "target_frame_mae" in val_results and "rendered_frame_mae" in val_results:
                self.logger.info(
                    f"  val target_MAE={val_results['target_frame_mae']:.2f}° "
                    f"target_Acc@5={val_results['target_acc_at_5deg']:.4f} "
                    f"target_Acc@10={val_results['target_acc_at_10deg']:.4f} | "
                    f"rendered_MAE={val_results['rendered_frame_mae']:.2f}° "
                    f"rendered_Acc@5={val_results['rendered_acc_at_5deg']:.4f} "
                    f"rendered_Acc@10={val_results['rendered_acc_at_10deg']:.4f}"
                )
            for k, v in val_results.items():
                self.tb_writer.add_scalar(f"val/{k}", v, epoch)
            is_best = mae < self.best_mae
            if is_best:
                self.best_mae = mae
                self.no_improve_epochs = 0
                self.logger.info(f"  ★ New best model saved (MAE={self.best_mae:.2f}°)")
            else:
                self.no_improve_epochs += 1
                if self.early_stopping_patience > 0:
                    self.logger.info(
                        f"  no improvement: {self.no_improve_epochs}/{self.early_stopping_patience} "
                        f"(best val MAE={self.best_mae:.2f}°)"
                    )
            self._save(epoch, is_best)
            if self.early_stopping_patience > 0 and self.no_improve_epochs >= self.early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered: best val MAE did not improve for "
                    f"{self.early_stopping_patience} epochs."
                )
                break
        self.tb_writer.close()

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total = 0.0
        count = 0
        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)
            labels = batch["doa_labels"].long()
            self.optimizer.zero_grad()
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                out = self.model(batch)
                logits = out["doa_logits"]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                )
                if self.front_back_aux_weight > 0 and "front_back_logits" in out:
                    fb_labels = self._front_back_labels(batch["doa_angles"]).long()
                    fb_loss = F.cross_entropy(
                        out["front_back_logits"].reshape(-1, 2),
                        fb_labels.reshape(-1),
                    )
                    loss = loss + self.front_back_aux_weight * fb_loss
            if not torch.isfinite(loss):
                self.logger.warning(f"non-finite loss at batch {batch_idx}; skipped")
                continue
            self.scaler.scale(loss).backward()
            if self.cfg.train.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += float(loss.item())
            count += 1
            self.global_step += 1
            if batch_idx % self.cfg.train.log_interval == 0:
                self.logger.info(f"  [Epoch {epoch}][{batch_idx}/{len(self.train_loader)}] loss={loss.item():.4f}")
        return total / max(count, 1)

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        self.metrics.reset()
        self.target_metrics.reset()
        self.rendered_metrics.reset()
        for batch in self.val_loader:
            batch = self._to_device(batch)
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                out = self.model(batch)
            self.metrics.update(
                out["doa_logits"].float().cpu().numpy(),
                batch["doa_labels"].cpu().numpy(),
                batch["doa_angles"].float().cpu().numpy(),
                group_values={
                    "speed": batch.get("speed_bin", []),
                    "rt60": [self._bucket_float(x, [0.0, 0.3, 0.6]) for x in batch.get("rt60", [])],
                    "snr": [self._format_snr(x) for x in batch.get("snr", [])],
                },
            )
            logits_np = out["doa_logits"].float().cpu().numpy()
            self.target_metrics.update(
                logits_np,
                batch["target_labels"].cpu().numpy(),
                batch["target_angles"].float().cpu().numpy(),
            )
            self.rendered_metrics.update(
                logits_np,
                batch["rendered_labels"].cpu().numpy(),
                batch["rendered_angles"].float().cpu().numpy(),
            )
        results = self.metrics.compute()
        for prefix, metric_obj in (("target", self.target_metrics), ("rendered", self.rendered_metrics)):
            for key, value in metric_obj.compute().items():
                results[f"{prefix}_{key}"] = value
        return results

    @staticmethod
    def _bucket_float(value, edges):
        v = float(value)
        if v <= edges[0]:
            return f"{edges[0]:.1f}"
        for lo, hi in zip(edges[:-1], edges[1:]):
            if lo < v <= hi:
                return f"{lo:.1f}-{hi:.1f}"
        return f">{edges[-1]:.1f}"

    @staticmethod
    def _format_snr(value):
        v = float(value)
        return "clean" if v > 900 else f"{v:.0f}dB"

    @staticmethod
    def _front_back_labels(angles: torch.Tensor) -> torch.Tensor:
        wrapped = torch.remainder(angles + 180.0, 360.0) - 180.0
        front = (wrapped >= -90.0) & (wrapped <= 90.0)
        return (~front).long()

    def _save(self, epoch: int, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_mae": self.best_mae,
            "global_step": self.global_step,
            "no_improve_epochs": self.no_improve_epochs,
        }
        save_checkpoint(state, self.cfg.output.save_dir, "latest.pth")
        if is_best:
            save_checkpoint(state, self.cfg.output.save_dir, "best.pth")

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
