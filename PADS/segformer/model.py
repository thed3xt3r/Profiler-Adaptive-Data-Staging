import torch
import torch.nn.functional as F
import lightning.pytorch as pl
import segmentation_models_pytorch as smp
from torch.nn import BCEWithLogitsLoss
from torchmetrics import Accuracy, JaccardIndex
from transformers import SegformerForSemanticSegmentation


class ArcheoModel(pl.LightningModule):
    """Lightning module for archaeological site segmentation using NVlabs' SegFormer
    via HuggingFace Transformers.

    The model loads a pretrained SegFormer variant (e.g. b0–b5) from the
    'nvidia/segformer-{variant}-finetuned-ade-512-512' checkpoint and replaces
    the classification head for binary segmentation (num_labels=out_classes).

    NOTE: The SegFormer decode head outputs logits at 1/4 of the input spatial
    resolution.  We upsample them back to the original size with bilinear
    interpolation so that the loss and metrics are computed at full resolution.
    """

    # Map short names used in config["encoder"] to the HuggingFace checkpoint
    # suffix.  Accepted values: "b0" .. "b5" or the legacy "mit_b0" .. "mit_b5".
    _VARIANT_MAP = {
        "mit_b0": "b0", "mit_b1": "b1", "mit_b2": "b2",
        "mit_b3": "b3", "mit_b4": "b4", "mit_b5": "b5",
        "b0": "b0", "b1": "b1", "b2": "b2",
        "b3": "b3", "b4": "b4", "b5": "b5",
    }

    def __init__(self, encoder_name, in_channels, out_classes, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        variant = self._VARIANT_MAP.get(encoder_name, "b0")
        pretrained_name = f"nvidia/segformer-{variant}-finetuned-ade-512-512"

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_name,
            num_labels=out_classes,
            ignore_mismatched_sizes=True,
        )

        # ImageNet normalisation buffers (SegFormer expects normalised input)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # --- Loss function ---
        # This used to hardcode BCEWithLogitsLoss and ignore config["loss"]
        # entirely, so the run logged 'loss: focal' while actually training on
        # unweighted BCE. On ~12%-positive data that biases hard toward
        # background, and it silently made segformer incomparable to
        # manet/deeplab, which do use focal. Honour the config like they do.
        if config["loss"] == "jaccard":
            self.loss_fn = smp.losses.JaccardLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif config["loss"] == "dice":
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif config["loss"] == "focal":
            self.loss_fn = smp.losses.FocalLoss(
                mode=smp.losses.BINARY_MODE,
                alpha=config.get("focal_alpha", 0.75),
                gamma=config.get("focal_gamma", 2.0),
            )
        elif config["loss"] == "bce":
            self.loss_fn = BCEWithLogitsLoss()
        else:
            raise ValueError(
                f"Unknown loss {config['loss']!r}; expected one of "
                "jaccard, dice, focal, bce."
            )

        # --- Metrics ---
        metric_kwargs = dict(task="binary", threshold=0.5)
        self.train_iou = JaccardIndex(**metric_kwargs)
        self.train_acc = Accuracy(**metric_kwargs)
        self.val_iou = JaccardIndex(**metric_kwargs)
        self.val_acc = Accuracy(**metric_kwargs)
        self.test_iou = JaccardIndex(**metric_kwargs)
        self.test_acc = Accuracy(**metric_kwargs)

    def forward(self, image):
        # NOTE: no /255 here on purpose. Unlike the manet/deeplab pipelines,
        # segformer's dataset._finalize() already rescales to [0, 1] before the
        # tensor reaches us, so the ImageNet statistics below are applied to
        # correctly-scaled input. Adding a rescale here would divide twice.
        image = (image - self.mean) / self.std
        outputs = self.model(pixel_values=image)
        logits = outputs.logits  # shape: (B, out_classes, H/4, W/4)

        # Upsample logits to the original input resolution
        logits = F.interpolate(
            logits,
            size=image.shape[2:],  # (H, W)
            mode="bilinear",
            align_corners=False,
        )
        return logits

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
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
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
