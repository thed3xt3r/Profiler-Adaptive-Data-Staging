import torch
import lightning.pytorch as pl
import segmentation_models_pytorch as smp
from torchmetrics import Accuracy, Precision, Recall, JaccardIndex, MatthewsCorrCoef


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
            # alpha weights the positive class. smp defaults to alpha=None (no
            # weighting), but only ~12% of pixels are sites (median 7%), so the
            # unweighted loss lets the model coast on predicting background.
            # The reference implementation uses 0.75; keep it configurable.
            self.loss_fn = smp.losses.FocalLoss(
                mode=smp.losses.BINARY_MODE,
                alpha=config.get("focal_alpha", None),
                gamma=config.get("focal_gamma", 2.0),
            )

        # Metrics — binary, threshold 0.5
        # IoU/accuracy are pixel-level; precision/recall/MCC are logged at the
        # same pixel level here (not the site-detection level the reference
        # study reports at — see the thesis's evaluation-metrics discussion),
        # so they are directly comparable to val_iou/val_acc above but not
        # directly to the reference paper's site-level precision/recall/MCC.
        metric_kwargs = dict(task="binary", threshold=0.5)
        self.train_iou = JaccardIndex(**metric_kwargs)
        self.train_acc = Accuracy(**metric_kwargs)
        self.train_precision = Precision(**metric_kwargs)
        self.train_recall = Recall(**metric_kwargs)
        self.train_mcc = MatthewsCorrCoef(**metric_kwargs)

        self.val_iou = JaccardIndex(**metric_kwargs)
        self.val_acc = Accuracy(**metric_kwargs)
        self.val_precision = Precision(**metric_kwargs)
        self.val_recall = Recall(**metric_kwargs)
        self.val_mcc = MatthewsCorrCoef(**metric_kwargs)

        self.test_iou = JaccardIndex(**metric_kwargs)
        self.test_acc = Accuracy(**metric_kwargs)
        self.test_precision = Precision(**metric_kwargs)
        self.test_recall = Recall(**metric_kwargs)
        self.test_mcc = MatthewsCorrCoef(**metric_kwargs)

    def forward(self, image):
        # Casini et al. standardise raw 0-255 data with smp's ImageNet statistics,
        # which nominally expect input_range [0,1]: the encoder input lands around
        # [118, 1136] rather than [-2.2, 2.7]. BatchNorm after the stem conv
        # absorbs most of that scale, which is why the reference still trains.
        # Rescaling first converges measurably faster (DeepLab reached 0.31 IoU at
        # epoch 2 instead of epoch 9) but is a DEVIATION from the reference, so it
        # is opt-in via --scale_input rather than the default.
        if self.config.get("scale_input", False):
            image = image.float() / 255.0
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
            self.train_precision(pred_flat, mask_flat)
            self.train_recall(pred_flat, mask_flat)
            self.train_mcc(pred_flat, mask_flat)
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                     batch_size=self.config["batch_size"])
            self.log("train/iou", self.train_iou, on_step=False, on_epoch=True, prog_bar=True,
                     batch_size=self.config["batch_size"])
            self.log("train/acc", self.train_acc, on_step=False, on_epoch=True,
                     batch_size=self.config["batch_size"])
            self.log("train/precision", self.train_precision, on_step=False, on_epoch=True,
                     batch_size=self.config["batch_size"])
            self.log("train/recall", self.train_recall, on_step=False, on_epoch=True,
                     batch_size=self.config["batch_size"])
            self.log("train/mcc", self.train_mcc, on_step=False, on_epoch=True,
                     batch_size=self.config["batch_size"])
        elif stage == "valid":
            self.val_iou(pred_flat, mask_flat)
            self.val_acc(pred_flat, mask_flat)
            self.val_precision(pred_flat, mask_flat)
            self.val_recall(pred_flat, mask_flat)
            self.val_mcc(pred_flat, mask_flat)
            self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                     batch_size=self.config["batch_size"])
            self.log("valid/iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True,
                     batch_size=self.config["batch_size"])
            self.log("valid/acc", self.val_acc, on_step=False, on_epoch=True,
                     batch_size=self.config["batch_size"])
            self.log("valid/precision", self.val_precision, on_step=False, on_epoch=True,
                     batch_size=self.config["batch_size"])
            self.log("valid/recall", self.val_recall, on_step=False, on_epoch=True,
                     batch_size=self.config["batch_size"])
            self.log("valid/mcc", self.val_mcc, on_step=False, on_epoch=True,
                     batch_size=self.config["batch_size"])
        elif stage == "test":
            self.test_iou(pred_flat, mask_flat)
            self.test_acc(pred_flat, mask_flat)
            self.test_precision(pred_flat, mask_flat)
            self.test_recall(pred_flat, mask_flat)
            self.test_mcc(pred_flat, mask_flat)
            self.log("test/loss", loss, on_step=False, on_epoch=True,
                     batch_size=self.config["batch_size"])
            self.log("test/iou", self.test_iou, on_step=False, on_epoch=True,
                     batch_size=self.config["batch_size"])
            self.log("test/acc", self.test_acc, on_step=False, on_epoch=True,
                     batch_size=self.config["batch_size"])
            self.log("test/precision", self.test_precision, on_step=False, on_epoch=True,
                     batch_size=self.config["batch_size"])
            self.log("test/recall", self.test_recall, on_step=False, on_epoch=True,
                     batch_size=self.config["batch_size"])
            self.log("test/mcc", self.test_mcc, on_step=False, on_epoch=True,
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
