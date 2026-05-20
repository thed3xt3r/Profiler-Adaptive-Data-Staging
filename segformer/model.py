import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, JaccardIndex

try:
    from transformers import SegformerForSemanticSegmentation
except ImportError as exc:
    raise ImportError(
        "transformers is required for SegFormer. Please install it in this environment."
    ) from exc


class ArcheoModel(pl.LightningModule):
    """Lightning module for archaeological site segmentation (SegFormer)."""

    def __init__(self, model_name, out_classes, config):
        super().__init__()
        self.config = config

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=out_classes,
            ignore_mismatched_sizes=True,
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

        metric_kwargs = dict(task="binary", threshold=0.5)
        self.train_iou = JaccardIndex(**metric_kwargs)
        self.train_acc = Accuracy(**metric_kwargs)
        self.val_iou = JaccardIndex(**metric_kwargs)
        self.val_acc = Accuracy(**metric_kwargs)
        self.test_iou = JaccardIndex(**metric_kwargs)
        self.test_acc = Accuracy(**metric_kwargs)

    def forward(self, image):
        outputs = self.model(pixel_values=image)
        return outputs.logits

    def shared_step(self, batch, stage):
        image = batch[0]
        if not torch.is_tensor(image):
            image = torch.tensor(image)
        image = image.float()

        mask = batch[1]
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask)
        mask = mask.float()

        logits_mask = self.forward(image)
        logits_mask = F.interpolate(
            logits_mask,
            size=mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        loss = self.loss_fn(logits_mask, mask)

        prob_mask = torch.sigmoid(logits_mask)
        pred_mask = (prob_mask > 0.5).long()

        pred_flat = pred_mask.squeeze(1)
        mask_flat = mask.squeeze(1).long()

        if stage == "train":
            self.train_iou(pred_flat, mask_flat)
            self.train_acc(pred_flat, mask_flat)
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                     batch_size=self.config["batch_size"])
            self.log("train/iou", self.train_iou, on_step=False, on_epoch=True, prog_bar=True,
                     batch_size=self.config["batch_size"])
            self.log("train/acc", self.train_acc, on_step=False, on_epoch=True,
                     batch_size=self.config["batch_size"])
        elif stage == "valid":
            self.val_iou(pred_flat, mask_flat)
            self.val_acc(pred_flat, mask_flat)
            self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                     batch_size=self.config["batch_size"])
            self.log("valid/iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True,
                     batch_size=self.config["batch_size"])
            self.log("valid/acc", self.val_acc, on_step=False, on_epoch=True,
                     batch_size=self.config["batch_size"])
        elif stage == "test":
            self.test_iou(pred_flat, mask_flat)
            self.test_acc(pred_flat, mask_flat)
            self.log("test/loss", loss, on_step=False, on_epoch=True,
                     batch_size=self.config["batch_size"])
            self.log("test/iou", self.test_iou, on_step=False, on_epoch=True,
                     batch_size=self.config["batch_size"])
            self.log("test/acc", self.test_acc, on_step=False, on_epoch=True,
                     batch_size=self.config["batch_size"])

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config["learning_rate"])
