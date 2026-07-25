import torch
import lightning.pytorch as pl
import segmentation_models_pytorch as smp
from torchmetrics import Accuracy, Precision, Recall, JaccardIndex


class ArcheoModel(pl.LightningModule):
    """Lightning module for archaeological site segmentation (deeplab)"""
    
    def __init__(self, arch, encoder_name, in_channels, out_classes, config, **kwargs):
        super().__init__()
        self.config = config
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, 
            classes=out_classes, **kwargs
        )
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        
        if config["loss"] == "jaccard":
            self.loss_fn = smp.losses.JaccardLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif config["loss"] == "dice":
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif config["loss"] == "focal":
            self.loss_fn = smp.losses.FocalLoss(mode=smp.losses.BINARY_MODE)

        # Metrics — binary, threshold 0.5
        metric_kwargs = dict(task="binary", threshold=0.5)
        self.train_iou = JaccardIndex(**metric_kwargs)
        self.train_acc = Accuracy(**metric_kwargs)

        self.val_iou = JaccardIndex(**metric_kwargs)
        self.val_acc = Accuracy(**metric_kwargs)

        self.test_iou = JaccardIndex(**metric_kwargs)
        self.test_acc = Accuracy(**metric_kwargs)

    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch[0]
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        
        mask = batch[1]
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0
        
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)
        
        # Predictions: sigmoid + threshold
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).long()
        
        # Squeeze channel dim for metrics: (B, 1, H, W) -> (B, H, W)
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
        return torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
