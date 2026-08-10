import argparse
import os
import random
import numpy as np
import optuna
import torch
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from datetime import datetime
import webdataset as wds

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for HPC
import matplotlib.pyplot as plt

from dataset import get_webdataset, get_transforms
from model import ArcheoModel
from gpu_monitor import GPUUtilMonitor
from fsop_monitor import FSOpCounter

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
    pass


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

        fig.suptitle("Training Curves — MA-Net WebDataset", fontsize=14)
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
        f.write("PROFILER REPORT — MA-Net WebDataset\n")
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
# RQ2 fs-op profiling (Table~\ref{tab:rq2-fsops})
# ---------------------------------------------------------------------------

def profile_fsops(dataset, batch_size=32, num_batches=12, profile_dir="profiler_logs"):
    """Count open()/stat() calls and bytes read while pulling batches out of
    the WebDataset pipeline, for the same RQ2 comparison PADS/*/train.py's
    profile_data_pipeline() runs against loose/shard/stage.

    num_workers=0 is required, not just a default: FSOpCounter patches
    builtins.open/os.stat in this process only, and a DataLoader worker
    subprocess would silently under-count (see fsop_monitor.py).
    """
    import json

    os.makedirs(profile_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    iterator = iter(loader)

    batches_seen = 0
    with FSOpCounter() as fsop_counter:
        for _ in range(num_batches):
            try:
                next(iterator)
            except StopIteration:
                break
            batches_seen += 1

    summary = fsop_counter.summary(batches=batches_seen)
    summary["batches_profiled"] = batches_seen
    summary_path = os.path.join(profile_dir, "data_policy_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print("\n[FSOpCounter] WebDataset fs-op summary")
    print(f"  Batches profiled: {batches_seen}")
    if batches_seen:
        print(f"  open() calls: {summary['open_calls']} ({summary['open_calls_per_batch']:.1f}/batch)")
        print(f"  stat() calls: {summary['stat_calls']} ({summary['stat_calls_per_batch']:.1f}/batch)")
        print(f"  bytes read: {summary['bytes_read']} ({summary['bytes_read_per_batch']:.0f}/batch)")
    print(f"  Summary saved to: {summary_path}")
    return summary


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
    parser.add_argument("--profile_only", action="store_true",
                        help="Run the RQ2 fs-op profiling pass and exit before training.")
    parser.add_argument("--profile_batches", type=int, default=12,
                        help="Number of batches to use for RQ2 fs-op profiling.")
    return parser.parse_args()


def run_training(config, train_loader, val_loader, val_dataset_func, trial=None):
    model = ArcheoModel(
        config["arch"],
        encoder_name=config["encoder"],
        in_channels=config['in_channels'], 
        out_classes=1,
        config=config
    )

    callbacks_list = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["checkpoint_path"],
        filename=f"manet-effb3-trial{trial.number if trial else '0'}-{{epoch:02d}}-{{valid/iou:.4f}}",
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
        from optuna_integration import PyTorchLightningPruningCallback
        callbacks_list.append(PyTorchLightningPruningCallback(trial, monitor="valid/iou"))

    # Since WebDataset repeats infinitely during training, we must specify limit_train_batches
    limit_train_batches = config["train_size"] // config["batch_size"]
    # Limit val batches to match the exact validation set size (optional, but clean)
    limit_val_batches = config["val_size"] // config["batch_size"]

    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        precision=config["precision"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=pl_loggers.TensorBoardLogger(config["checkpoint_path"]),
        log_every_n_steps=1,
        enable_progress_bar=True,
        callbacks=callbacks_list,
        deterministic="warn",
    )

    cfg_text = "\n".join([f"{key}: {config[key]}" for key in config])
    if trial is None:
        print("\nTraining Configuration:")
        print(cfg_text)
        trainer.logger.experiment.add_text(tag="config", text_string=cfg_text)

    gpu_monitor = GPUUtilMonitor(device_index=0)
    with gpu_monitor:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )

    gpu_util_summary = gpu_monitor.summary()
    gpu_util_path = os.path.join(config["checkpoint_path"], "gpu_util_summary.json")
    import json
    with open(gpu_util_path, "w") as handle:
        json.dump(gpu_util_summary, handle, indent=2)
    if gpu_util_summary["gpu_util_mean_pct"] is not None:
        print(f"\n[GPUUtilMonitor] Mean GPU utilisation over training: "
              f"{gpu_util_summary['gpu_util_mean_pct']:.1f}% "
              f"({gpu_util_summary['gpu_util_n_samples']} samples)")

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
            
            # Use a separate dataloader with 0 workers for profiling stability, and do not repeat
            profile_dataset = val_dataset_func(repeat=False)
            profile_loader = DataLoader(
                profile_dataset, batch_size=config["batch_size"],
                shuffle=False, num_workers=0
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
    PATH_LOG = os.path.join(PROJECT_ROOT, "checkpoints_baseline_wds")
    SHARDS_ROOT = os.path.join(PROJECT_ROOT, "webDataset/shards")

    base_config = {
        "timestamp": datetime.now().strftime("%d-%m-%Y_%H%M%S"),
        "shards_root": SHARDS_ROOT,
        "checkpoint_path": PATH_LOG,
        "random_seed": 1234,
        "arch": "MAnet",
        "encoder": "efficientnet-b3",
        "loss": "focal",
        "learning_rate": 0.0001,
        "precision": 32,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "in_channels": 3,
        "patience": 15,
        "train_size": 4712,  # Match exact splits in create_shards.py
        "val_size": 588,
        "test_size": 589,
        "profile": args.profile,
        "profile_wait": args.profile_wait,
        "profile_warmup": args.profile_warmup,
        "profile_active": args.profile_active,
    }

    os.makedirs(base_config["checkpoint_path"], exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "webDataset", "manet", "logs"), exist_ok=True)

    random.seed(base_config["random_seed"])
    np.random.seed(base_config["random_seed"])
    torch.manual_seed(base_config["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_config["random_seed"])

    # Path matching patterns for WebDataset shards
    train_shards = os.path.join(SHARDS_ROOT, "train", "shard-{000000..000047}.tar")
    val_shards = os.path.join(SHARDS_ROOT, "val", "shard-{000000..000005}.tar")

    train_transform, val_transform = get_transforms()

    # Define WebDataset datasets
    # Note: training dataset uses repeat=True (implied by get_webdataset)
    train_dataset = get_webdataset(train_shards, train_transform, is_training=True)
    
    def val_dataset_func(repeat=False):
        # Allow creating val dataset without repeating for profiling/testing if needed
        dataset = get_webdataset(val_shards, val_transform, is_training=False)
        return dataset

    val_dataset = val_dataset_func(repeat=False)

    if args.profile_only:
        profile_fsops(
            train_dataset,
            batch_size=base_config["batch_size"],
            num_batches=args.profile_batches,
            profile_dir=os.path.join(base_config["checkpoint_path"], "profiler_logs"),
        )
        print("[FSOpCounter] --profile_only: fs-op count taken; exiting before training.")
        return

    # Dataloaders - WebDataset recommends batching via DataLoader or wds pipelines
    train_loader = DataLoader(train_dataset, batch_size=base_config["batch_size"],
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=base_config["batch_size"], 
                            num_workers=4, pin_memory=True)

    if args.tune:
        def objective(trial):
            config = base_config.copy()
            config["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            config["checkpoint_path"] = os.path.join(PATH_LOG, f"trial_{trial.number}")
            os.makedirs(config["checkpoint_path"], exist_ok=True)
            
            iou, _ = run_training(config, train_loader, val_loader, val_dataset_func, trial=trial)
            return iou

        study = optuna.create_study(direction="maximize", study_name="manet_wds_tuning")
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
            run_training(best_config, train_loader, val_loader, val_dataset_func, trial=None)

    else:
        print("Running standard training...")
        run_training(base_config, train_loader, val_loader, val_dataset_func, trial=None)


if __name__ == "__main__":
    main()
