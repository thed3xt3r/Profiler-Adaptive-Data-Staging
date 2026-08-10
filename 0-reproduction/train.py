"""Bare-bones, faithful port of Definitivo.ipynb's training + test-evaluation
cells (11, 13, 16, 17) -- the Casini et al. reference notebook -- restricted
to the Bing 1k / no-CORONA branch, run against all three of this thesis's
architectures.

Deliberately NOT the PADS pipeline: no early stopping, no policy selection,
no profiling, no --num_workers tuning. Fixed 20 epochs, num_workers=0,
default (final-epoch, unmonitored) checkpointing, exactly as the notebook
does it. This exists to answer one question -- does a maximally faithful
reproduction of the reference notebook reach reference-level IoU on this
cluster with the Bing 1k dataset? -- not to be a better pipeline.

ONE INTERFACE ADAPTATION (documented, not silent): the notebook hardcodes a
`modelli` list and loops over all configs in one script/session. This takes
--arch on the command line instead, one model per invocation, so each
architecture can be submitted as its own Slurm job. This changes nothing
about the training procedure itself.

SegFormer was never part of the reference notebook (see model.py's
docstring) -- included here as a protocol-faithful extension, not a literal
port, because this thesis needs the same question answered for all three
architectures it studies.

Usage: python train.py --arch {manet,deeplab,segformer} [--data_root PATH]
"""
import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers

from dataset import load_dataset, esegui_trasformazioni, ArcheoDataset
from model import _EpochBufferedArcheoModel, SegformerArcheoModel

# Monkey-patch Lightning's device capability check to bypass driver error on
# pre-Ampere GPUs (V100) -- same workaround 3-pads/*/train.py already uses,
# unrelated to the reference protocol itself.
try:
    import lightning_fabric.accelerators.cuda

    def _dummy_is_ampere_or_later(device):
        return False
    lightning_fabric.accelerators.cuda._is_ampere_or_later = _dummy_is_ampere_or_later
except (ImportError, AttributeError):
    pass


ARCH_CONFIG = {
    # arch/encoder values matched to what PADS/<model>/train.py already uses,
    # so this is an apples-to-apples comparison of protocol, not architecture.
    "manet":    {"arch": "MAnet",         "encoder": "efficientnet-b3"},
    "deeplab":  {"arch": "DeepLabV3Plus", "encoder": "resnet50"},
    "segformer": {"arch": None,           "encoder": "b0"},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=list(ARCH_CONFIG.keys()))
    parser.add_argument("--data_root", type=str, default=None,
                         help="Directory containing bing_1k/ (env: PADS_DATA_ROOT)")
    parser.add_argument("--project_root", type=str, default=None,
                         help="Output root for logs/checkpoints (env: CASINI_PROJECT_ROOT)")
    parser.add_argument("--epochs", type=int, default=20,
                         help="Override for smoke-testing only -- the faithful "
                              "replica value is 20 (cell 3's config['epochs']); "
                              "leave at the default for a real run.")
    parser.add_argument("--test_passes", type=int, default=10,
                         help="Override for smoke-testing only -- the faithful "
                              "replica value is 10 (cell 16's 10-pass average); "
                              "leave at the default for a real run.")
    return parser.parse_args()


def build_model(arch_key, config):
    if arch_key == "segformer":
        return SegformerArcheoModel(
            encoder_name=config["encoder"], in_channels=config["in_channels"],
            out_classes=1, config=config,
        )
    # _EpochBufferedArcheoModel, not the plain ArcheoModel: Lightning 2.x
    # removed the *_epoch_end(self, outputs) hooks the notebook's own
    # ArcheoModel relies on, so plain ArcheoModel never actually logs the
    # epoch-aggregated IOU metrics at all (training "succeeds" since
    # per-step loss logging doesn't need the hook, but silently produces no
    # valid/IOU-img, train/IOU-img, or test/IOU-img -- caught via a
    # KeyError on the missing test metric, not a crash during training).
    return _EpochBufferedArcheoModel(
        config["arch"], encoder_name=config["encoder"],
        encoder_weights=config["weights"], in_channels=config["in_channels"],
        out_classes=1, config=config,
    )


def load_model_from_checkpoint(arch_key, config, checkpoint_path):
    if arch_key == "segformer":
        model = SegformerArcheoModel(
            encoder_name=config["encoder"], in_channels=config["in_channels"],
            out_classes=1, config=config,
        )
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["state_dict"])
        return model
    return _EpochBufferedArcheoModel.load_from_checkpoint(
        checkpoint_path,
        arch=config["arch"], encoder_name=config["encoder"],
        encoder_weights=config["weights"], in_channels=config["in_channels"],
        out_classes=1, config=config,
    )


def main():
    args = parse_args()
    arch_key = args.arch
    arch_cfg = ARCH_CONFIG[arch_key]

    data_root = args.data_root or os.environ.get("PADS_DATA_ROOT") or os.path.expanduser("~/Thesis")
    project_root = args.project_root or os.environ.get("CASINI_PROJECT_ROOT") \
        or os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", arch_key)
    os.makedirs(project_root, exist_ok=True)

    # ---- config: Definitivo.ipynb cell 3, Bing 1k / no-CORONA values ----
    config = {
        "timestamp": datetime.now().strftime("%d-%m-%Y_%H%M%S"),
        "dataset_path": os.path.join(data_root, "bing_1k"),
        "checkpoint_path": project_root,
        "random_seed": 1234,
        "arch": arch_cfg["arch"],
        "encoder": arch_cfg["encoder"],
        "weights": "imagenet",
        "loss": "focal",
        "learning_rate": 0.0001,
        "precision": 32,
        "epochs": args.epochs,
        "batch_size": 32,
        "corona_path": "",
        "dim_input": "1k",
        "in_channels": 3,
    }

    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    print("============================================================")
    print(f"Casini reference replica -- arch={arch_key}")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("============================================================")

    # ---- cell 13: indices shuffled once, before load_dataset ----
    n_files = len(os.listdir(os.path.join(config["dataset_path"], "train", "originals", "sites"))) \
        + len(os.listdir(os.path.join(config["dataset_path"], "train", "negs", "sites")))
    indices = np.arange(0, n_files)
    np.random.shuffle(indices)

    path_lookup, train_files, val_files, test_files = load_dataset(
        config["dataset_path"], config["random_seed"], indices)
    train_transform, val_transform = esegui_trasformazioni()

    train_dataset = ArcheoDataset(train_files, path_lookup, transform=train_transform)
    val_dataset = ArcheoDataset(val_files, path_lookup, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,
        drop_last=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False,
        drop_last=False, num_workers=0)

    model = build_model(arch_key, config)

    # ---- cell 13: no early stopping, no monitor -- default (final-epoch)
    # checkpointing only, exactly as the notebook's Trainer (no
    # ModelCheckpoint callback passed at all) leaves it. ----
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        precision=config["precision"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=pl_loggers.TensorBoardLogger(config["checkpoint_path"]),
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    cfg_text = "\n".join([f"{key}: {config[key]}" for key in config])
    print("\nTraining Configuration:")
    print(cfg_text)
    trainer.logger.experiment.add_text(tag="config", text_string=cfg_text)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Locate whatever checkpoint Lightning's default callback saved.
    ckpt_dir = os.path.join(trainer.logger.log_dir, "checkpoints")
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not ckpt_files:
        raise RuntimeError(f"No checkpoint found in {ckpt_dir}")
    checkpoint_path = os.path.join(ckpt_dir, ckpt_files[0])
    print(f"\nUsing checkpoint: {checkpoint_path}")

    # ---- cell 16-17: 10 independent-random-crop test passes, averaged ----
    reload_model = load_model_from_checkpoint(arch_key, config, checkpoint_path)

    test_ious = []
    for j in range(args.test_passes):
        print(f"Risultato {j} del modello {arch_key}")
        _, test_val_transform = esegui_trasformazioni()
        test_dataset = ArcheoDataset(test_files, path_lookup, transform=test_val_transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config["batch_size"], shuffle=False,
            drop_last=False, num_workers=0)

        # precision=16 here (not config["precision"]) is preserved exactly
        # as authored in cell 16 -- the notebook's own test trainer really
        # does use half precision while training used full. Not "fixed"
        # here; this is a faithful replica, including its quirks.
        test_trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1, max_epochs=40, precision=16, logger=False,
        )
        test_metrics = test_trainer.test(reload_model, dataloaders=test_loader, verbose=True)
        test_ious.append(test_metrics[0]["test/IOU-img"])

    x = np.array(test_ious)
    print(f"\nLe statistiche sul test set per il modello {arch_key} sono:")
    print(f"media:{round(float(x.mean()), 4)} | min:{round(float(x.min()), 4)} | "
          f"max:{round(float(x.max()), 4)} | std:{round(float(x.std()), 4)}")

    import json
    summary_path = os.path.join(project_root, "casini_replica_result.json")
    with open(summary_path, "w") as handle:
        json.dump({
            "arch": arch_key, "test_iou_mean": float(x.mean()),
            "test_iou_min": float(x.min()), "test_iou_max": float(x.max()),
            "test_iou_std": float(x.std()), "test_iou_all_10": test_ious,
            "checkpoint": checkpoint_path,
        }, handle, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
