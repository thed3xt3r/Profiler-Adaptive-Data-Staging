import os
import random
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader

from dataset import ArcheoDataset, get_transforms, load_dataset
from model import ArcheoModel


torch.set_float32_matmul_precision("medium")


try:
    import lightning_fabric.accelerators.cuda

    def _dummy_is_ampere_or_later(device):
        return False

    lightning_fabric.accelerators.cuda._is_ampere_or_later = _dummy_is_ampere_or_later
except (ImportError, AttributeError):
    pass


class TrainingPlotCallback(pl.Callback):
    def __init__(self, save_path="training_curves.png"):
        self.save_path = save_path
        self.train_loss, self.val_loss = [], []
        self.train_iou, self.val_iou = [], []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        self._append(self.train_loss, metrics.get("train/loss_epoch"))
        self._append(self.train_iou, metrics.get("train/iou"))

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        self._append(self.val_loss, metrics.get("valid/loss"))
        self._append(self.val_iou, metrics.get("valid/iou"))

    def on_train_end(self, trainer, pl_module):
        if not self.train_loss:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        train_epochs = range(1, len(self.train_loss) + 1)
        val_epochs = range(1, len(self.val_loss) + 1)

        ax1.plot(train_epochs, self.train_loss, label="Train Loss")
        if self.val_loss:
            ax1.plot(val_epochs, self.val_loss, label="Val Loss")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        if self.train_iou:
            ax2.plot(train_epochs, self.train_iou, label="Train IoU")
        if self.val_iou:
            ax2.plot(val_epochs, self.val_iou, label="Val IoU")
        ax2.set_title("IoU (Jaccard Index)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("IoU")
        ax2.legend()
        ax2.grid(True)

        fig.suptitle("Training Curves — SegFormer B0", fontsize=14)
        plt.tight_layout()
        plt.savefig(self.save_path, dpi=150)
        plt.close(fig)
        print(f"\nTraining curves saved to: {self.save_path}")

    @staticmethod
    def _append(lst, value):
        if value is not None:
            lst.append(float(value))


def profile_model(model, dataloader, num_batches=5, device="cuda",
                  profile_dir="profiler_logs", wait=1, warmup=1, active=3):
    os.makedirs(profile_dir, exist_ok=True)
    model = model.to(device)
    model.eval()

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=lambda p: p.export_chrome_trace(os.path.join(profile_dir, "trace.json")),
    ) as prof:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break

                images = batch[0]
                if not torch.is_tensor(images):
                    images = torch.tensor(images)
                images = images.to(device).float()

                with record_function("model_forward"):
                    _ = model(images)

                prof.step()

    report_path = os.path.join(profile_dir, "profiler_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PROFILER REPORT — SegFormer B0\n")
        f.write("=" * 80 + "\n\n")
        f.write("Top 50 operations by CPU time:\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
        if device == "cuda":
            f.write("\n\nTop 50 operations by CUDA time:\n")
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
            f.write("\n\nTop 50 operations by CUDA memory usage:\n")
            f.write(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=50))

    print(f"\nDetailed report saved to: {report_path}")
    print(f"Chrome trace saved to: {os.path.join(profile_dir, 'trace.json')}")


def main():
    PROJECT_ROOT = os.path.expanduser("~/Thesis")
    PATH_LOG = os.path.join(PROJECT_ROOT, "checkpoints_segformer")
    PATH_DATASETS = PROJECT_ROOT

    config = {
        "timestamp": datetime.now().strftime("%d-%m-%Y_%H%M%S"),
        "dataset_path": os.path.join(PATH_DATASETS, "bing_1k"),
        "checkpoint_path": PATH_LOG,
        "random_seed": 1234,
        "model_name": "nvidia/segformer-b0-finetuned-ade-512-512",
        "loss": "focal",
        "learning_rate": 0.0001,
        "precision": 32,
        "epochs": 100,
        "batch_size": 32,
        "in_channels": 3,
        "patience": 15,
        "profile": os.environ.get("SEGFORMER_PROFILE", "0") == "1",
        "profile_wait": 1,
        "profile_warmup": 1,
        "profile_active": 5,
    }

    os.makedirs(config["checkpoint_path"], exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "segformer", "logs"), exist_ok=True)

    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["random_seed"])

    originals_count = len(sorted(os.listdir(os.path.join(config["dataset_path"], "train/originals/sites"))))
    negs_count = len(sorted(os.listdir(os.path.join(config["dataset_path"], "train/negs/sites"))))
    total_count = originals_count + negs_count
    indices = np.arange(0, total_count)

    (
        originals_images_dir,
        originals_masks_dir,
        negs_images_dir,
        negs_masks_dir,
        train_data,
        val_data,
        test_data,
    ) = load_dataset(config["dataset_path"], config["random_seed"], indices)

    train_transform, val_transform = get_transforms()

    train_dataset = ArcheoDataset(
        train_data,
        originals_images_dir,
        originals_masks_dir,
        negs_images_dir,
        negs_masks_dir,
        transform=train_transform,
    )
    val_dataset = ArcheoDataset(
        val_data,
        originals_images_dir,
        originals_masks_dir,
        negs_images_dir,
        negs_masks_dir,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    model = ArcheoModel(
        model_name=config["model_name"],
        out_classes=1,
        config=config,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config["checkpoint_path"],
        filename="segformer-b0-{epoch:02d}-{valid/iou:.4f}",
        monitor="valid/iou",
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    early_stopping = EarlyStopping(
        monitor="valid/iou",
        mode="max",
        patience=config["patience"],
        verbose=True,
    )
    plot_callback = TrainingPlotCallback(
        save_path=os.path.join(config["checkpoint_path"], "training_curves.png")
    )

    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        precision=config["precision"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=pl_loggers.TensorBoardLogger(config["checkpoint_path"]),
        log_every_n_steps=1,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback, early_stopping, plot_callback],
        deterministic=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if config["profile"]:
        try:
            best_ckpt = checkpoint_callback.best_model_path
            if best_ckpt:
                profiled_model = ArcheoModel.load_from_checkpoint(
                    best_ckpt,
                    model_name=config["model_name"],
                    out_classes=1,
                    config=config,
                )
            else:
                profiled_model = model

            profile_dir = os.path.join(config["checkpoint_path"], "profiler_results")
            profile_loader = DataLoader(
                val_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                drop_last=False,
                num_workers=0,
            )

            profile_model(
                profiled_model,
                profile_loader,
                num_batches=config["profile_wait"] + config["profile_warmup"] + config["profile_active"] + 2,
                device="cuda" if torch.cuda.is_available() else "cpu",
                profile_dir=profile_dir,
                wait=config["profile_wait"],
                warmup=config["profile_warmup"],
                active=config["profile_active"],
            )
        except Exception as e:
            print(f"[Profile] Warning: Could not profile model: {e}")

    print("All done!")


if __name__ == "__main__":
    main()
