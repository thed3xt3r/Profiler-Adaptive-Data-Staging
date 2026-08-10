"""Faithful port of the relevant cells of Definitivo.ipynb (Casini et al. 2023).

Scope: the "bing_1k" variant only -- no CORONA, no 2k, no filtering. That is
Table 1 Model 1, IoU 74.17 +/- 0.38.

Deliberately kept EXACTLY as the notebook:
  * no /255 -- the model standardises raw 0-255 data with ImageNet stats
  * smp.losses.FocalLoss(mode=BINARY), i.e. alpha=None
  * torch.optim.Adam (weight_decay=0), lr 1e-4
  * precision 32
  * 20 epochs
  * RandomCrop(512) -> Resize(256) for BOTH train and val
  * A.Flip (single op), RandomRotate90, RandomBrightnessContrast -- all p=0.25
  * metric via smp.metrics, reduction="macro-imagewise" (the notebook's IOU-img)
  * no ModelCheckpoint monitor -- the notebook evaluates the final-epoch weights

Adapted only where it cannot run otherwise:
  * PL 1.x {training,validation}_epoch_end no longer exist in PL 2.x, so outputs
    are accumulated manually in on_*_epoch_end
  * the notebook reads one train/sites dir with negs mixed in by filename; our
    copy splits originals/ and negs/. Concatenating originals-then-negs
    reproduces the notebook's sorted order, since "neg*" sorts after the
    uppercase site codes.
  * batch size is a flag: the notebook uses 32, which needs ~5 GB at fp32 and
    does not fit a 4 GB card.
"""

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import albumentations as A
from albumentations import pytorch

import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers

import segmentation_models_pytorch as smp

CONFIG = {
    "random_seed": 1234,
    "arch": "MAnet",
    "encoder": "efficientnet-b3",
    "weights": "imagenet",
    "loss": "focal",
    "learning_rate": 0.0001,
    "precision": 32,
    "epochs": 20,
    "batch_size": 32,
    "in_channels": 3,
}


# ---------------------------------------------------------------------------
# notebook cell 5 -- load_dataset
# ---------------------------------------------------------------------------
def load_dataset(path, seed, indices):
    o_img = os.path.join(path, "train/originals/sites")
    o_msk = os.path.join(path, "train/originals/masks")
    n_img = os.path.join(path, "train/negs/sites")
    n_msk = os.path.join(path, "train/negs/masks")

    items = [(f, o_img, o_msk) for f in sorted(os.listdir(o_img))]
    items += [(f, n_img, n_msk) for f in sorted(os.listdir(n_img))]
    items = np.asarray(items, dtype=object)
    print("total files:", len(items))

    valid_split = -int(len(indices) * 0.2)
    test_split = valid_split // 2
    train_items = items[indices[:valid_split]]
    val_items = items[indices[valid_split:test_split]]
    test_items = items[indices[test_split:]]
    print(f"train {len(train_items)}  val {len(val_items)}  test {len(test_items)}")
    return train_items, val_items, test_items


# ---------------------------------------------------------------------------
# notebook cell 6 -- transforms for dim_input == '1k', no corona
# ---------------------------------------------------------------------------
def _flip_like_old_albumentations(p=0.25):
    """Reproduce the removed A.Flip.

    Old A.Flip sampled d from {0, 1, -1} and applied a vertical, horizontal or
    both-axis flip respectively -- ONE of the three, not two independent flips.
    A.HorizontalFlip(p) + A.VerticalFlip(p) is a different distribution, so it
    is not a drop-in substitute.
    """
    return A.OneOf([
        A.VerticalFlip(p=1.0),
        A.HorizontalFlip(p=1.0),
        A.Sequential([A.VerticalFlip(p=1.0), A.HorizontalFlip(p=1.0)], p=1.0),
    ], p=p)


def esegui_trasformazioni():
    train_transform = A.Compose([
        A.RandomCrop(512, 512, p=1.0),
        _flip_like_old_albumentations(p=0.25), A.RandomRotate90(p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.25),
        A.Resize(256, 256),
        A.pytorch.ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.RandomCrop(512, 512, p=1.0),
        A.Resize(256, 256),
        A.pytorch.ToTensorV2(),
    ])
    return train_transform, val_transform


# ---------------------------------------------------------------------------
# notebook cell 8 -- ArcheoDataset
# ---------------------------------------------------------------------------
class ArcheoDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_filename, img_dir, mask_dir = self.items[idx]
        image_path = os.path.join(img_dir, image_filename)
        mask_path = os.path.join(mask_dir, image_filename.replace(".jpg", ".png"))

        image = Image.open(image_path)
        # masks are flipped because of qgis
        mask = ~np.array(Image.open(mask_path).convert("L"))
        mask = mask.astype("float")
        mask[mask > 0.0] = 1.0
        mask = np.expand_dims(mask, -1)

        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]

        transformed = self.transform(image=arr, mask=np.asarray(mask))
        image = transformed["image"]
        mask = transformed["mask"].permute(2, 0, 1)
        return image, mask, image_filename


# ---------------------------------------------------------------------------
# notebook cell 10 -- ArcheoModel
# ---------------------------------------------------------------------------
class ArcheoModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels,
            classes=out_classes, **kwargs)
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        if CONFIG["loss"] == "jaccard":
            self.loss_fn = smp.losses.JaccardLoss(smp.losses.BINARY_MODE, from_logits=True)
        if CONFIG["loss"] == "dice":
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        if CONFIG["loss"] == "focal":
            self.loss_fn = smp.losses.FocalLoss(mode=smp.losses.BINARY_MODE)

        self._buf = {"train": [], "valid": [], "test": []}

    def forward(self, image):
        # NOTE: no /255 here -- this is what the notebook does.
        image = (image - self.mean) / self.std
        return self.model(image)

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
        self.log_dict({f"{stage}/loss": loss.detach().item()},
                      batch_size=CONFIG["batch_size"])

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary")

        self._buf[stage].append((tp, fp, fn, tn))
        return {"loss": loss}

    def _epoch_end(self, stage):
        rows = self._buf[stage]
        if not rows:
            return
        tp = torch.cat([r[0] for r in rows])
        fp = torch.cat([r[1] for r in rows])
        fn = torch.cat([r[2] for r in rows])
        tn = torch.cat([r[3] for r in rows])
        self.log_dict({
            f"{stage}/IOU-img": smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise"),
            f"{stage}/IOU": smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro"),
        }, prog_bar=True, batch_size=CONFIG["batch_size"])
        self._buf[stage] = []

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def on_train_epoch_end(self):
        self._epoch_end("train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def on_validation_epoch_end(self):
        self._epoch_end("valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def on_test_epoch_end(self):
        self._epoch_end("test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=CONFIG["learning_rate"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=r"C:\Users\nabee\Source\Thesis\source")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    # The notebook uses a true batch of 32 with no accumulation. At fp32 that
    # needs ~5 GB (measured: b8 2.44 GB, b16 4.65 GB, b32 OOM), so on a 4 GB card
    # the closest achievable is batch 8 x accum 4: same gradient signal, but
    # BatchNorm sees 8 samples per forward instead of 32.
    ap.add_argument("--accumulate", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--precision", default="32")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--test_runs", type=int, default=10)
    args = ap.parse_args()

    CONFIG["batch_size"] = args.batch_size
    CONFIG["epochs"] = args.epochs
    CONFIG["precision"] = args.precision

    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(os.path.dirname(here), "runs", "paper_repro")
    os.makedirs(out, exist_ok=True)

    random.seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])
    torch.manual_seed(CONFIG["random_seed"])

    dataset_path = os.path.join(args.data_root, "bing_1k")
    n = len(os.listdir(os.path.join(dataset_path, "train/originals/sites"))) + \
        len(os.listdir(os.path.join(dataset_path, "train/negs/sites")))
    indices = np.arange(0, n)
    np.random.shuffle(indices)

    train_items, val_items, test_items = load_dataset(
        dataset_path, CONFIG["random_seed"], indices)
    train_tf, val_tf = esegui_trasformazioni()

    train_loader = DataLoader(
        ArcheoDataset(train_items, transform=train_tf),
        batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True,
        num_workers=args.num_workers)
    val_loader = DataLoader(
        ArcheoDataset(val_items, transform=val_tf),
        batch_size=CONFIG["batch_size"], shuffle=False, drop_last=False,
        num_workers=args.num_workers)

    model = ArcheoModel(CONFIG["arch"], encoder_name=CONFIG["encoder"],
                        encoder_weights=CONFIG["weights"],
                        in_channels=CONFIG["in_channels"], out_classes=1)

    print("\n".join(f"{k}: {v}" for k, v in CONFIG.items()))

    trainer = pl.Trainer(
        max_epochs=CONFIG["epochs"],
        precision=CONFIG["precision"],
        accumulate_grad_batches=args.accumulate,
        accelerator="gpu",
        devices=1,
        logger=pl_loggers.TensorBoardLogger(out),
        log_every_n_steps=1,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # notebook cell 16: ten test passes, each with a fresh random crop
    print("\n" + "=" * 60)
    print(f"TEST -- {args.test_runs} runs, random crop each time (notebook cell 16)")
    scores_img, scores_ds = [], []
    for j in range(args.test_runs):
        _, val_tf_j = esegui_trasformazioni()
        test_loader = DataLoader(
            ArcheoDataset(test_items, transform=val_tf_j),
            batch_size=CONFIG["batch_size"], shuffle=False, drop_last=False,
            num_workers=args.num_workers)
        m = trainer.test(model, dataloaders=test_loader, verbose=False)[0]
        scores_img.append(m["test/IOU-img"])
        scores_ds.append(m["test/IOU"])
        print(f"  run {j}: IOU-img={m['test/IOU-img']:.4f}  IOU={m['test/IOU']:.4f}")

    print()
    print(f"RESULT  IOU-img (paper Table 1 metric): "
          f"{np.mean(scores_img):.4f} +/- {np.std(scores_img):.4f}")
    print(f"RESULT  IOU     (pooled)              : "
          f"{np.mean(scores_ds):.4f} +/- {np.std(scores_ds):.4f}")
    print(f"paper Model 1 (Bing 1k, unfiltered)   : 0.7417 +/- 0.0038")


if __name__ == "__main__":
    main()
