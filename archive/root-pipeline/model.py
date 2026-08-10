"""
model.py
--------
PyTorch Lightning module for Tell site segmentation using:
    - MA-Net architecture with EfficientNet-B3 encoder (pretrained ImageNet)
    - Focal Loss
    - Metrics: Accuracy, Precision, Recall, IoU (pixel-level, binary)
    - WandB logging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics import Accuracy, Precision, Recall, JaccardIndex
import wandb


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary Focal Loss for segmentation.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weight for foreground class. Set based on class distribution
               (e.g. if fg=20% of pixels, alpha ~ 0.75-0.8).
               Default 0.75 — adjust after running compute_class_distribution().
        gamma: Focusing parameter. Higher values down-weight easy examples.
               Default 2.0 (standard from Lin et al. 2017).
        reduction: 'mean' or 'sum'
    """

    def __init__(self, alpha: float = 0.95, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output, shape (N, 1, H, W)
            targets: Binary ground truth, shape (N, H, W), values {0, 1}
        """
        # Squeeze channel dim: (N, 1, H, W) -> (N, H, W)
        logits = logits.squeeze(1)

        # Binary cross entropy per pixel (no reduction)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )

        # p_t: probability of the true class
        probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        focal_loss = focal_weight * bce

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class TellSiteSegmentationModel(pl.LightningModule):
    """
    MA-Net with EfficientNet-B3 backbone for binary Tell site segmentation.

    Args:
        encoder_name: SMP encoder name. Default 'efficientnet-b3'.
        encoder_weights: Pretrained weights. Default 'imagenet'.
        focal_alpha: Alpha for FocalLoss. Tune after class distribution analysis.
        focal_gamma: Gamma for FocalLoss. Default 2.0.
        lr: Initial learning rate. Default 1e-4.
        weight_decay: AdamW weight decay. Default 1e-4.
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet-b3",
        encoder_weights: str = "imagenet",
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,  # Raw logits — loss handles activation
        )

        # Loss
        self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        # Metrics — binary, threshold 0.5
        metric_kwargs = dict(task="binary", threshold=0.5)
        self.train_acc       = Accuracy(**metric_kwargs)
        self.train_precision = Precision(**metric_kwargs)
        self.train_recall    = Recall(**metric_kwargs)
        self.train_iou       = JaccardIndex(**metric_kwargs)

        self.val_acc         = Accuracy(**metric_kwargs)
        self.val_precision   = Precision(**metric_kwargs)
        self.val_recall      = Recall(**metric_kwargs)
        self.val_iou         = JaccardIndex(**metric_kwargs)

        self.test_acc        = Accuracy(**metric_kwargs)
        self.test_precision  = Precision(**metric_kwargs)
        self.test_recall     = Recall(**metric_kwargs)
        self.test_iou        = JaccardIndex(**metric_kwargs)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        return self.model(x)

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------

    def _shared_step(self, batch):
        images, masks = batch           # images: (N,3,H,W), masks: (N,H,W)
        logits = self(images)           # (N,1,H,W)
        loss = self.criterion(logits, masks)
        preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).long()
        return loss, preds, masks

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss, preds, masks = self._shared_step(batch)

        self.train_acc(preds, masks)
        self.train_precision(preds, masks)
        self.train_recall(preds, masks)
        self.train_iou(preds, masks)

        self.log("train/loss",      loss,                  on_step=True,  on_epoch=True, prog_bar=True)
        self.log("train/acc",       self.train_acc,        on_step=False, on_epoch=True)
        self.log("train/precision", self.train_precision,  on_step=False, on_epoch=True)
        self.log("train/recall",    self.train_recall,     on_step=False, on_epoch=True)
        self.log("train/iou",       self.train_iou,        on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        loss, preds, masks = self._shared_step(batch)

        self.val_acc(preds, masks)
        self.val_precision(preds, masks)
        self.val_recall(preds, masks)
        self.val_iou(preds, masks)

        self.log("val/loss",      loss,               on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc",       self.val_acc,        on_step=False, on_epoch=True)
        self.log("val/precision", self.val_precision,  on_step=False, on_epoch=True)
        self.log("val/recall",    self.val_recall,     on_step=False, on_epoch=True)
        self.log("val/iou",       self.val_iou,        on_step=False, on_epoch=True, prog_bar=True)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_idx):
        loss, preds, masks = self._shared_step(batch)

        self.test_acc(preds, masks)
        self.test_precision(preds, masks)
        self.test_recall(preds, masks)
        self.test_iou(preds, masks)

        self.log("test/loss",      loss,                on_step=False, on_epoch=True)
        self.log("test/acc",       self.test_acc,       on_step=False, on_epoch=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test/recall",    self.test_recall,    on_step=False, on_epoch=True)
        self.log("test/iou",       self.test_iou,       on_step=False, on_epoch=True)

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,       #max_epochs
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
