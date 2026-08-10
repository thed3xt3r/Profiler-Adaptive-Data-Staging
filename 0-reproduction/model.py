"""Bare-bones, faithful port of the model cell (Definitivo.ipynb cell 10) --
the Casini et al. reference notebook's ArcheoModel, loss, and metric
computation, restricted to the no-CORONA branch (single 3-channel input;
the corona-concatenation branch of cell 10's forward() is out of scope for
Bing 1k without CORONA).

ONE STRUCTURAL ADAPTATION (documented, not silent): the notebook reads
config["loss"] etc. off a mutable global `config` dict shared across cells.
A script has no such global notebook state, so config is passed into
__init__ explicitly instead. This changes nothing about the loss/metric
formulas themselves -- only how the same values reach the class.

SegFormer was never part of Definitivo.ipynb or Casini et al.'s own
architecture search (they compared U-Net/MA-Net x ResNet-18/EfficientNet-B3
x Dice/focal loss only -- no transformer). smp.create_model() cannot build a
transformer backbone, so there is no literal cell to port for it. What
follows applies the notebook's protocol -- same loss_fn construction, same
shared_step/shared_epoch_end using smp.metrics.get_stats +
iou_score(reduction="macro-imagewise"), same optimizer -- to a SegFormer
backbone built the same way 3-pads/segformer/model.py already does it. This is
a faithful extension of the reference PROTOCOL, not a port of a notebook
cell that doesn't exist.
"""
import torch
import lightning.pytorch as pl
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation


class ArcheoModel(pl.LightningModule):
    """Faithful port of Definitivo.ipynb cell 10, no-CORONA branch."""

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
        if config["loss"] == "dice":
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        if config["loss"] == "focal":
            self.loss_fn = smp.losses.FocalLoss(mode=smp.losses.BINARY_MODE)

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
        self.log_dict({f"{stage}/loss": loss.detach().item()}, batch_size=self.config["batch_size"])
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        pred_mask = pred_mask.permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        if stage == "train":
            self.log_dict({
                "train/batch-IOU-img": smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise"),
                "train/batch-IOU": smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro"),
            }, prog_bar=True, batch_size=self.config["batch_size"])

        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")

        metrics = {f"{stage}/IOU-img": per_image_iou, f"{stage}/IOU": dataset_iou}
        # on_step=False, on_epoch=True made explicit: this is called from
        # on_test_epoch_end (Lightning 2.x has no test_epoch_end(outputs) hook
        # any more, see module docstring), and trainer.test()'s returned dict
        # only reliably includes metrics logged as epoch-level from here --
        # left implicit, this silently produced no "test/IOU-img" key at all.
        self.log_dict(metrics, prog_bar=True, batch_size=self.config["batch_size"],
                       on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])


# Lightning 2.x removed training_epoch_end/validation_epoch_end/test_epoch_end
# in favour of hooks that receive no accumulated `outputs` (the same
# adaptation the reproduction appendix documents making on the workstation:
# "per-epoch aggregation was therefore reimplemented explicitly"). Buffer the
# per-step dicts ourselves and reduce them at epoch end via on_*_epoch_end.
class _EpochBufferedArcheoModel(ArcheoModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_outputs = []
        self._valid_outputs = []
        self._test_outputs = []

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch, "train")
        self._train_outputs.append(out)
        return out

    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, "valid")
        self._valid_outputs.append(out)
        return out

    def test_step(self, batch, batch_idx):
        out = self.shared_step(batch, "test")
        self._test_outputs.append(out)
        return out

    def on_train_epoch_end(self):
        self.shared_epoch_end(self._train_outputs, "train")
        self._train_outputs.clear()

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self._valid_outputs, "valid")
        self._valid_outputs.clear()

    def on_test_epoch_end(self):
        self.shared_epoch_end(self._test_outputs, "test")
        self._test_outputs.clear()


class SegformerArcheoModel(_EpochBufferedArcheoModel):
    """Same protocol as ArcheoModel (loss, metric, optimizer all identical),
    with a SegFormer backbone in place of smp.create_model() -- see module
    docstring for why. Constructed the same way 3-pads/segformer/model.py
    already builds SegFormer (HuggingFace pretrained checkpoint, bilinear
    upsample of the 1/4-resolution decode head back to input size)."""

    _VARIANT_MAP = {
        "mit_b0": "b0", "mit_b1": "b1", "mit_b2": "b2",
        "mit_b3": "b3", "mit_b4": "b4", "mit_b5": "b5",
        "b0": "b0", "b1": "b1", "b2": "b2", "b3": "b3", "b4": "b4", "b5": "b5",
    }

    def __init__(self, encoder_name, in_channels, out_classes, config):
        pl.LightningModule.__init__(self)
        self.config = config
        self._train_outputs, self._valid_outputs, self._test_outputs = [], [], []

        variant = self._VARIANT_MAP.get(encoder_name, "b0")
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            f"nvidia/segformer-{variant}-finetuned-ade-512-512",
            num_labels=out_classes, ignore_mismatched_sizes=True,
        )
        params = smp.encoders.get_preprocessing_params("resnet18")  # ImageNet stats, arch-independent
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        if config["loss"] == "jaccard":
            self.loss_fn = smp.losses.JaccardLoss(smp.losses.BINARY_MODE, from_logits=True)
        if config["loss"] == "dice":
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        if config["loss"] == "focal":
            self.loss_fn = smp.losses.FocalLoss(mode=smp.losses.BINARY_MODE)

    def forward(self, image):
        import torch.nn.functional as F
        image = (image - self.mean) / self.std
        logits = self.model(pixel_values=image).logits
        logits = F.interpolate(logits, size=image.shape[2:], mode="bilinear", align_corners=False)
        return logits
