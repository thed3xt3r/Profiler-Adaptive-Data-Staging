import argparse
import os
import random
import numpy as np
import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for HPC
import matplotlib.pyplot as plt

from dataset import load_dataset, get_transforms, ArcheoDataset
from model import ArcheoModel

# ---------------------------------------------------------------------------
# GPU compatibility workaround
# ---------------------------------------------------------------------------
torch.set_float32_matmul_precision('medium')

# Monkey-patch Lightning's device capability check to bypass driver error
try:
    import lightning_fabric.accelerators.cuda
    def _dummy_is_ampere_or_later(device):
        """Bypass device capability check for driver compatibility."""
        return False
    lightning_fabric.accelerators.cuda._is_ampere_or_later = _dummy_is_ampere_or_later
except (ImportError, AttributeError):
    pass  # Not needed if lightning_fabric is not installed or API changed


# ---------------------------------------------------------------------------
# Training Plot Callback
# ---------------------------------------------------------------------------

class TrainingPlotCallback(pl.Callback):
    """
    Collects loss and IoU each epoch and saves a PNG at the end of training.
    Output: <ckpt_dir>/training_curves.png
    """

    def __init__(self, save_path: str = "training_curves.png"):
        self.save_path = save_path
        self.train_loss, self.val_loss = [], []
        self.train_iou,  self.val_iou  = [], []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        self._append(self.train_loss, metrics.get("train/loss_epoch"))
        self._append(self.train_iou,  metrics.get("train/iou"))

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return  # Ignore the sanity check step
        metrics = trainer.callback_metrics
        self._append(self.val_loss, metrics.get("valid/loss"))
        self._append(self.val_iou,  metrics.get("valid/iou"))

    def on_train_end(self, trainer, pl_module):
        if not self.train_loss:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        train_epochs = range(1, len(self.train_loss) + 1)
        val_epochs = range(1, len(self.val_loss) + 1)

        # Loss
        ax1.plot(train_epochs, self.train_loss, label="Train Loss")
        if self.val_loss:
            ax1.plot(val_epochs, self.val_loss, label="Val Loss")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # IoU
        if self.train_iou:
            ax2.plot(train_epochs, self.train_iou, label="Train IoU")
        if self.val_iou:
            ax2.plot(val_epochs, self.val_iou, label="Val IoU")
        ax2.set_title("IoU (Jaccard Index)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("IoU")
        ax2.legend()
        ax2.grid(True)

        fig.suptitle("Training Curves — DeepLabV3Plus ResNet-50", fontsize=14)
        plt.tight_layout()
        plt.savefig(self.save_path, dpi=150)
        plt.close(fig)
        print(f"\nTraining curves saved to: {self.save_path}")

    @staticmethod
    def _append(lst, value):
        if value is not None:
            lst.append(float(value))


# ---------------------------------------------------------------------------
# PyTorch Profiler Helper
# ---------------------------------------------------------------------------

def profile_model(model, dataloader, num_batches=5, device="cuda",
                  profile_dir="profiler_logs", wait=1, warmup=1, active=3):
    """
    Profile model inference using PyTorch Profiler.
    Saves Chrome trace and text report.
    
    Args:
        model: PyTorch Lightning model
        dataloader: DataLoader to get batches from
        num_batches: Number of batches to profile
        device: Device to profile on
        profile_dir: Directory to save profiler results
        wait: Steps to skip before profiling
        warmup: GPU warmup steps
        active: Steps to actively profile
    """
    os.makedirs(profile_dir, exist_ok=True)
    model = model.to(device)
    model.eval()

    print(f"\n{'='*60}")
    print(f"PROFILING MODEL ON {num_batches} BATCHES")
    print(f"{'='*60}")
    print(f"wait={wait}, warmup={warmup}, active={active}")

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        schedule=torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=1
        ),
        on_trace_ready=lambda p: p.export_chrome_trace(
            os.path.join(profile_dir, "trace.json")
        )
    ) as prof:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break

                images = batch[0].to(device)

                with record_function("model_forward"):
                    _ = model(images)

                if (batch_idx + 1) % max(1, num_batches // 3) == 0:
                    print(f"  Profiled {batch_idx + 1}/{num_batches} batches...")

                prof.step()

    # Print profiler summary
    print("\nProfiler Summary (Top 15 operations by CPU time):")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    if device == "cuda":
        print("\nProfiler Summary (Top 15 operations by CUDA time):")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    # Save detailed report
    report_path = os.path.join(profile_dir, "profiler_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PROFILER REPORT — DeepLabV3Plus ResNet-50\n")
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
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--tune_trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--profile", action="store_true", help="Profile the best model")
    parser.add_argument("--profile_wait", type=int, default=1)
    parser.add_argument("--profile_warmup", type=int, default=1)
    parser.add_argument("--profile_active", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def run_training(config, train_loader, val_loader, val_dataset, trial=None):
    model = ArcheoModel(
        config["arch"],
        encoder_name=config["encoder"],
        encoder_weights=config["weights"],
        in_channels=config['in_channels'], 
        out_classes=1,
        config=config
    )

    callbacks_list = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["checkpoint_path"],
        filename=f"deeplab-resnet50-trial{trial.number if trial else '0'}-{{epoch:02d}}-{{valid/iou:.4f}}",
        monitor="valid/iou",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )
    callbacks_list.append(checkpoint_callback)

    early_stopping = EarlyStopping(
        monitor="valid/iou",
        mode="max",
        patience=config["patience"],
        verbose=True,
    )
    callbacks_list.append(early_stopping)

    if trial is None:
        plot_callback = TrainingPlotCallback(
            save_path=os.path.join(config["checkpoint_path"], "training_curves.png")
        )
        callbacks_list.append(plot_callback)
    else:
        from optuna.integration import PyTorchLightningPruningCallback
        callbacks_list.append(PyTorchLightningPruningCallback(trial, monitor="valid/iou"))

    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        precision=config["precision"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=pl_loggers.TensorBoardLogger(config["checkpoint_path"]),
        log_every_n_steps=1,
        enable_progress_bar=True,
        callbacks=callbacks_list,
        deterministic=True,
    )

    cfg_text = "\n".join([f"{key}: {config[key]}" for key in config])
    if trial is None:
        print("\nTraining Configuration:")
        print(cfg_text)
        trainer.logger.experiment.add_text(tag="config", text_string=cfg_text)

    trainer.fit(
        model,
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )

    best_iou = trainer.callback_metrics.get("valid/iou")
    best_iou_val = float(best_iou) if best_iou is not None else 0.0

    if config["profile"] and trial is None:
        print("\n[Profile] Profiling best model checkpoint...")
        try:
            best_ckpt = checkpoint_callback.best_model_path
            if best_ckpt:
                profiled_model = ArcheoModel.load_from_checkpoint(
                    best_ckpt,
                    arch=config["arch"],
                    encoder_name=config["encoder"],
                    in_channels=config["in_channels"],
                    out_classes=1,
                    config=config,
                )
            else:
                print("[Profile] No best checkpoint found, using current model...")
                profiled_model = model

            profile_dir = os.path.join(config["checkpoint_path"], "profiler_results")
            
            profile_loader = DataLoader(
                val_dataset, batch_size=config["batch_size"],
                shuffle=False, drop_last=False, num_workers=4, pin_memory=True
            )

            profile_model(
                profiled_model, profile_loader,
                num_batches=config["profile_wait"] + config["profile_warmup"] + config["profile_active"] + 2,
                device="cuda" if torch.cuda.is_available() else "cpu",
                profile_dir=profile_dir,
                wait=config["profile_wait"],
                warmup=config["profile_warmup"],
                active=config["profile_active"],
            )
        except Exception as e:
            print(f"[Profile] Warning: Could not profile model: {e}")
            import traceback
            traceback.print_exc()

    return best_iou_val, checkpoint_callback.best_model_path


def main():
    args = parse_args()
    
    PROJECT_ROOT = os.path.expanduser("~/Thesis")
    PATH_LOG = os.path.join(PROJECT_ROOT, "checkpoints_deeplab")
    PATH_DATASETS = PROJECT_ROOT

    base_config = {
        "timestamp": datetime.now().strftime("%d-%m-%Y_%H%M%S"),
        "dataset_path": os.path.join(PATH_DATASETS, "bing_1k"),
        "checkpoint_path": PATH_LOG,
        "random_seed": 1234,
        "arch": "DeepLabV3Plus",
        "encoder": "resnet50",
        "weights": "imagenet",
        "loss": "focal",
        "learning_rate": 0.0001,
        "precision": 32,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "in_channels": 3,
        "patience": 15,
        "profile": args.profile,
        "profile_wait": args.profile_wait,
        "profile_warmup": args.profile_warmup,
        "profile_active": args.profile_active,
    }

    os.makedirs(base_config["checkpoint_path"], exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "deeplab", "logs"), exist_ok=True)

    random.seed(base_config["random_seed"])
    np.random.seed(base_config["random_seed"])
    torch.manual_seed(base_config["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_config["random_seed"])

    originals_count = len(sorted(os.listdir(
        os.path.join(base_config["dataset_path"], "train/originals/sites"))))
    negs_count = len(sorted(os.listdir(
        os.path.join(base_config["dataset_path"], "train/negs/sites"))))
    total_count = originals_count + negs_count

    indices = np.arange(0, total_count)

    (originals_images_dir, originals_masks_dir, negs_images_dir, negs_masks_dir, 
     train_data, val_data, test_data) = load_dataset(
        base_config["dataset_path"], 
        base_config["random_seed"],
        indices
    )

    train_transform, val_transform = get_transforms()

    train_dataset = ArcheoDataset(train_data, originals_images_dir, originals_masks_dir, 
                                   negs_images_dir, negs_masks_dir, transform=train_transform)
    val_dataset = ArcheoDataset(val_data, originals_images_dir, originals_masks_dir, 
                                negs_images_dir, negs_masks_dir, transform=val_transform)
    test_dataset = ArcheoDataset(test_data, originals_images_dir, originals_masks_dir,
                                 negs_images_dir, negs_masks_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=base_config["batch_size"], 
                              shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=base_config["batch_size"], 
                            shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    if args.tune:
        def objective(trial):
            config = base_config.copy()
            config["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            config["checkpoint_path"] = os.path.join(PATH_LOG, f"trial_{trial.number}")
            os.makedirs(config["checkpoint_path"], exist_ok=True)
            
            iou, _ = run_training(config, train_loader, val_loader, val_dataset, trial=trial)
            return iou

        study = optuna.create_study(direction="maximize", study_name="deeplab_tuning")
        study.optimize(objective, n_trials=args.tune_trials)

        print("\n" + "="*60)
        print("OPTUNA TUNING RESULTS")
        print("="*60)
        print(f"Best Trial ID: {study.best_trial.number}")
        print(f"Best Validation IoU: {study.best_value:.4f}")
        print(f"Best Hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print("="*60)

        if args.profile:
            print("\nRunning profiling with best hyperparameters...")
            best_config = base_config.copy()
            best_config.update(study.best_params)
            best_config["checkpoint_path"] = os.path.join(PATH_LOG, "best_trial_profiling")
            os.makedirs(best_config["checkpoint_path"], exist_ok=True)
            run_training(best_config, train_loader, val_loader, val_dataset, trial=None)

    else:
        print("Running standard training...")
        run_training(base_config, train_loader, val_loader, val_dataset, trial=None)


if __name__ == "__main__":
    main()
