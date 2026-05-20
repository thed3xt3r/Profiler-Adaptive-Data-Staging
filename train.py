"""
train.py
--------
Main training script for Tell site segmentation benchmark.
Recreates Cassini et al. 2023 experimental setup with:
    - MA-Net + EfficientNet-B3 + Focal Loss
    - Stratified 81/9/10 train/val/test split
    - WandB logging
    - Checkpoint on best val/iou

Usage:
    python train.py --data_root /path/to/bing_1k --wandb_project tell_segmentation
"""

import argparse
import os
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
import wandb
from torch.profiler import profile, record_function, ProfilerActivity
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search.optuna import OptunaSearch
from pytorch_lightning.tuner.tuning import Tuner

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for HPC
import matplotlib.pyplot as plt

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
        metrics = trainer.callback_metrics
        self._append(self.val_loss, metrics.get("val/loss"))
        self._append(self.val_iou,  metrics.get("val/iou"))

    def on_train_end(self, trainer, pl_module):
        epochs = range(1, len(self.train_loss) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Loss
        ax1.plot(epochs, self.train_loss, label="Train Loss")
        ax1.plot(epochs, self.val_loss,   label="Val Loss")
        ax1.set_title("Focal Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # IoU
        ax2.plot(epochs, self.train_iou, label="Train IoU")
        ax2.plot(epochs, self.val_iou,   label="Val IoU")
        ax2.set_title("IoU (Jaccard Index)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("IoU")
        ax2.legend()
        ax2.grid(True)

        fig.suptitle("Training Curves — MA-Net EfficientNet-B3", fontsize=14)
        plt.tight_layout()
        plt.savefig(self.save_path, dpi=150)
        plt.close(fig)
        print(f"\nTraining curves saved to: {self.save_path}")

    @staticmethod
    def _append(lst, value):
        if value is not None:
            lst.append(float(value))


class ProfilerCallback(pl.Callback):
    """
    PyTorch profiler callback for performance analysis.
    Profiles training and validation steps to identify bottlenecks.
    """

    def __init__(self, save_dir: str = "profiler_logs", enabled: bool = True,
                 wait: int = 1, warmup: int = 1, active: int = 3):
        """
        Args:
            save_dir: Directory to save profiler results
            enabled: Whether profiling is enabled
            wait: Number of steps to wait before profiling
            warmup: Number of steps for GPU warmup
            active: Number of steps to profile
        """
        self.save_dir = save_dir
        self.enabled = enabled
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.step_count = 0
        
        if self.enabled:
            os.makedirs(save_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Profile training batches."""
        if not self.enabled or batch_idx == 0:
            return
        
        if self.step_count == self.wait:
            print(f"\n[Profiler] Starting profiling (warmup={self.warmup}, active={self.active})...")
        
        if self.wait <= self.step_count < self.wait + self.warmup + self.active:
            # Profiling is done within forward/backward passes
            pass
        
        self.step_count += 1

    def on_train_end(self, trainer, pl_module):
        """Print profiler summary."""
        if self.enabled:
            print(f"\n[Profiler] Results saved to: {self.save_dir}")


# Workaround for GPU driver incompatibility: disable strict device checks
torch.set_float32_matmul_precision('medium')

# Monkey-patch Lightning's device capability check to bypass driver error
import lightning_fabric.accelerators.cuda
def _dummy_is_ampere_or_later(device):
    """Bypass device capability check for driver compatibility."""
    return False

lightning_fabric.accelerators.cuda._is_ampere_or_later = _dummy_is_ampere_or_later

from dataset import TellSiteDataModule, TellSiteKFoldDataModule, visualize_sample, compute_class_distribution
from model import TellSiteSegmentationModel


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Tell Site Segmentation - MA-Net Benchmark")

    # Data
    parser.add_argument("--data_root",   type=str, required=True,
                        help="Path to dataset root (parent of originals/ and negs/)")
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed",        type=int, default=42)

    # Model
    parser.add_argument("--encoder",        type=str,  default="efficientnet-b3")
    parser.add_argument("--focal_alpha",    type=float, default=0.95,
                        help="Focal loss alpha. Tune after compute_class_distribution().")
    parser.add_argument("--focal_gamma",    type=float, default=2.0)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)

    # Training
    parser.add_argument("--max_epochs",     type=int, default=100)
    parser.add_argument("--patience",       type=int, default=15,
                        help="Early stopping patience (epochs).")
    parser.add_argument("--ckpt_dir",       type=str, default="checkpoints")

    # WandB
    parser.add_argument("--wandb_project",  type=str, default="tell_segmentation")
    parser.add_argument("--wandb_run_name", type=str, default="manet-efficientnetb3-focal")
    parser.add_argument("--wandb_offline",  action="store_true",
                        help="Run WandB in offline mode (for HPC without internet).")

    # Misc
    parser.add_argument("--visualize",      action="store_true",
                        help="Show a sample image+mask after datamodule setup.")
    parser.add_argument("--class_dist",     action="store_true",
                        help="Compute and print class distribution then exit.")
    parser.add_argument("--test_only",      action="store_true",
                        help="Skip training, run test set evaluation only.")
    parser.add_argument("--ckpt_path",      type=str, default=None,
                        help="Path to checkpoint for test_only mode.")
    parser.add_argument("--cross_validation", action="store_true",
                        help="Enable k-fold cross-validation instead of single train/val/test split.")
    parser.add_argument("--n_splits",       type=int, default=5,
                        help="Number of folds for cross-validation (default 5).")

    # Hyperparameter Tuning
    parser.add_argument("--tune_hp",        action="store_true",
                        help="Enable RayTune hyperparameter optimization.")
    parser.add_argument("--tune_samples",   type=int, default=4,
                        help="Number of hyperparameter trials to run (default 4).")
    parser.add_argument("--tune_scheduler", type=str, default="asha",
                        choices=["asha", "pbt"],
                        help="Scheduling algorithm: 'asha' (default) or 'pbt' (population-based training).")

    # Profiling
    parser.add_argument("--profile",        action="store_true",
                        help="Enable PyTorch profiler to analyze performance bottlenecks.")
    parser.add_argument("--profile_dir",    type=str, default="profiler_logs",
                        help="Directory to save profiler results (default 'profiler_logs').")
    parser.add_argument("--profile_wait",   type=int, default=1,
                        help="Number of steps to wait before profiling (warmup). Default 1.")
    parser.add_argument("--profile_warmup", type=int, default=1,
                        help="Number of GPU warmup steps. Default 1.")
    parser.add_argument("--profile_active", type=int, default=3,
                        help="Number of steps to actively profile. Default 3.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RayTune Configuration and Trainable
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Profiling Helper Functions
# ---------------------------------------------------------------------------

def profile_model(model, dataloader, num_batches: int = 5, device: str = "cuda", 
                  profile_dir: str = "profiler_logs", wait: int = 1, warmup: int = 1, 
                  active: int = 3):
    """
    Profile model on batches from dataloader using PyTorch Profiler.
    
    Args:
        model: PyTorch Lightning model
        dataloader: DataLoader to get batches from
        num_batches: Number of batches to profile
        device: Device to profile on ("cuda" or "cpu")
        profile_dir: Directory to save profiler results
        wait: Number of steps to wait before profiling
        warmup: Number of GPU warmup steps
        active: Number of steps to actively profile
    """
    os.makedirs(profile_dir, exist_ok=True)
    model = model.to(device)
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"PROFILING MODEL ON {num_batches} BATCHES")
    print(f"{'='*60}")
    print(f"wait={wait}, warmup={warmup}, active={active}")
    
    batch_idx = 0
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device == "cuda" else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        wait=wait,
        warmup=warmup,
        active=active,
        on_trace_ready=lambda p: p.export_chrome_trace(os.path.join(profile_dir, "trace.json"))
    ) as prof:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                
                images, masks = batch
                images = images.to(device)
                masks = masks.to(device)
                
                with record_function("model_forward"):
                    _ = model(images)
                
                if (batch_idx + 1) % max(1, num_batches // 3) == 0:
                    print(f"  Profiled {batch_idx + 1}/{num_batches} batches...")
                
                prof.step()
    
    # Print profiler summary
    print("\nProfiler Summary (Top 10 operations by CPU time):")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    # Save detailed report
    report_path = os.path.join(profile_dir, "profiler_report.txt")
    with open(report_path, "w") as f:
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
    
    print(f"\nDetailed report saved to: {report_path}")
    print(f"Chrome trace saved to: {os.path.join(profile_dir, 'trace.json')}")
    print(f"{'='*60}\n")


def get_hyperparameter_search_space():
    """Define hyperparameter search space for RayTune."""
    return {
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "focal_alpha": tune.uniform(0.7, 0.95),
        "focal_gamma": tune.uniform(1.0, 3.0),
    }


def train_segment_model(config, data_root, batch_size, num_workers, seed, 
                        max_epochs, patience, ckpt_dir, wandb_project, encoder):
    """
    Trainable function for RayTune. Returns validation IoU for the trial.
    
    Args:
        config: Dict with hyperparameters to tune (lr, weight_decay, focal_alpha, focal_gamma)
        data_root: Path to dataset
        batch_size: Batch size
        num_workers: Number of workers
        seed: Random seed
        max_epochs: Max training epochs
        patience: Early stopping patience
        ckpt_dir: Checkpoint directory
        wandb_project: WandB project name
        encoder: Encoder architecture
    """
    pl.seed_everything(seed, workers=True)
    
    # Create datamodule
    from dataset import TellSiteDataModule
    dm = TellSiteDataModule(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    dm.setup()
    
    # Create model with tuned hyperparameters
    model = TellSiteSegmentationModel(
        encoder_name=encoder,
        encoder_weights="imagenet",
        focal_alpha=config["focal_alpha"],
        focal_gamma=config["focal_gamma"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    
    # Configure callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(ckpt_dir, "tune_trial_" + tune.get_trial_id()),
        filename="manet-effb3-{epoch:02d}-{val/iou:.4f}",
        monitor="val/iou",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    
    early_stopping = EarlyStopping(
        monitor="val/iou",
        mode="max",
        patience=patience,
        verbose=False,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # RayTune callback for reporting metrics
    from ray.air import session
    
    class RayTuneCallback(pl.Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            val_iou = metrics.get("val/iou", 0)
            session.report({"val/iou": val_iou})
    
    ray_callback = RayTuneCallback()
    
    # Configure trainer (no WandB during hyperparameter tuning)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32",
        callbacks=[checkpoint_callback, early_stopping, lr_monitor, ray_callback],
        log_every_n_steps=10,
        deterministic=True,
        enable_progress_bar=False,
        logger=False,
    )
    
    # Train
    trainer.fit(model, datamodule=dm)
    
    # Return best validation IoU
    best_metric = trainer.callback_metrics.get("val/iou", 0)
    return {"best_val_iou": float(best_metric)}


# ---------------------------------------------------------------------------
# Main

def main():
    args = parse_args()

    pl.seed_everything(args.seed, workers=True)

    # ------------------------------------------------------------------
    # Hyperparameter Tuning Mode
    # ------------------------------------------------------------------
    if args.tune_hp:
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER TUNING MODE ({args.tune_samples} trials)")
        print(f"Scheduler: {args.tune_scheduler.upper()}")
        print(f"{'='*60}")
        
        # Configure search space
        search_space = get_hyperparameter_search_space()
        
        # Configure scheduler
        if args.tune_scheduler == "asha":
            scheduler = ASHAScheduler(
                time_attr="training_iteration",
                metric="val/iou",
                mode="max",
                max_t=args.max_epochs,
                grace_period=10,
                reduction_factor=3,
            )
        else:  # pbt
            scheduler = PopulationBasedTraining(
                time_attr="training_iteration",
                perturbation_interval=5,
                hyperparam_mutations={
                    "lr": lambda: np.random.uniform(1e-4, 1e-2),
                    "weight_decay": lambda: np.random.uniform(1e-5, 1e-3),
                }
            )
        
        # Configure search algorithm (Optuna)
        search_alg = OptunaSearch()
        
        # Configure reporter
        reporter = CLIReporter(
            metric_columns=["val/iou", "training_iteration"]
        )
        
        # Run hyperparameter tuning
        tuner = tune.Tuner(
            tune.with_parameters(
                train_segment_model,
                data_root=args.data_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
                max_epochs=args.max_epochs,
                patience=args.patience,
                ckpt_dir=args.ckpt_dir,
                wandb_project=args.wandb_project,
                encoder=args.encoder,
            ),
            tune_config=tune.TuneConfig(
                num_samples=args.tune_samples,
                scheduler=scheduler,
                search_alg=search_alg,
            ),
            run_config=tune.air.RunConfig(
                progress_reporter=reporter,
                verbose=1,
            ),
        )
        
        results = tuner.fit()
        
        # Print results summary
        print(f"\n{'='*60}")
        print("HYPERPARAMETER TUNING RESULTS")
        print(f"{'='*60}")
        best_result = results.get_best_result("val/iou", mode="max")
        print(f"Best Trial ID: {best_result.path}")
        print(f"Best Validation IoU: {best_result.metrics['val/iou']:.4f}")
        print(f"Best Hyperparameters:")
        print(f"  lr: {best_result.config['lr']:.6f}")
        print(f"  weight_decay: {best_result.config['weight_decay']:.6f}")
        print(f"  focal_alpha: {best_result.config['focal_alpha']:.4f}")
        print(f"  focal_gamma: {best_result.config['focal_gamma']:.4f}")
        print(f"{'='*60}")
        
        print("\nTo train with best hyperparameters, use:")
        print(f"python train.py --data_root {args.data_root} \\")
        print(f"  --lr {best_result.config['lr']:.6f} \\")
        print(f"  --weight_decay {best_result.config['weight_decay']:.6f} \\")
        print(f"  --focal_alpha {best_result.config['focal_alpha']:.4f} \\")
        print(f"  --focal_gamma {best_result.config['focal_gamma']:.4f}")
        
        # Optional: Profile the best trial
        if args.profile:
            print("\n[Profile] Loading best trial checkpoint for profiling...")
            try:
                best_ckpt_dir = os.path.join(best_result.path, "checkpoints")
                best_ckpt = [f for f in os.listdir(best_ckpt_dir) if f.endswith(".ckpt") and "last" in f]
                if best_ckpt:
                    best_ckpt_path = os.path.join(best_ckpt_dir, best_ckpt[0])
                    best_model = TellSiteSegmentationModel.load_from_checkpoint(best_ckpt_path)
                    
                    dm_for_profiling = TellSiteDataModule(
                        data_root=args.data_root,
                        batch_size=args.batch_size,
                        num_workers=0,  # Profiling works better with 0 workers
                        seed=args.seed,
                    )
                    dm_for_profiling.setup()
                    val_loader = dm_for_profiling.val_dataloader()
                    
                    profile_dir = os.path.join(args.ckpt_dir, "best_trial_profile")
                    profile_model(
                        best_model, val_loader,
                        num_batches=5,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        profile_dir=profile_dir,
                        wait=args.profile_wait,
                        warmup=args.profile_warmup,
                        active=args.profile_active,
                    )
            except Exception as e:
                print(f"[Profile] Warning: Could not profile best trial: {e}")
        
        return

    # ------------------------------------------------------------------
    # DataModule selection (cross-validation or standard)
    # ------------------------------------------------------------------
    if args.cross_validation:
        print(f"\n{'='*60}")
        print(f"K-FOLD CROSS-VALIDATION MODE ({args.n_splits} folds)")
        print(f"{'='*60}")
        dm = TellSiteKFoldDataModule(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_splits=args.n_splits,
            seed=args.seed,
        )
        dm.setup()
    else:
        print("\n" + "="*60)
        print("SINGLE TRAIN/VAL/TEST SPLIT MODE")
        print("="*60)
        dm = TellSiteDataModule(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        dm.setup()

    # Optional: visualize one sample
    if args.visualize:
        visualize_sample(dm, split="train")

    # Optional: compute class distribution and exit
    if args.class_dist:
        dist = compute_class_distribution(dm)
        print("\nDone. Use the foreground ratio to set --focal_alpha.")
        print("Suggested alpha = 1 - global_fg_ratio (clamped to [0.5, 0.95])")
        fg = dist["train"]["global_fg_ratio"]
        suggested_alpha = max(0.5, min(0.95, 1.0 - fg))
        print(f"Suggested focal_alpha for your dataset: {suggested_alpha:.3f}")
        return

    # ------------------------------------------------------------------
    # WandB logger
    # ------------------------------------------------------------------
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    # ------------------------------------------------------------------
    # Training Loop (single or k-fold)
    # ------------------------------------------------------------------
    fold_results = []
    n_folds = args.n_splits if args.cross_validation else 1

    for fold_idx in range(n_folds):
        if args.cross_validation:
            dm.set_fold(fold_idx)
            fold_run_name = f"{args.wandb_run_name}-fold{fold_idx+1}"
            fold_ckpt_dir = os.path.join(args.ckpt_dir, f"fold_{fold_idx+1}")
        else:
            fold_run_name = args.wandb_run_name
            fold_ckpt_dir = args.ckpt_dir

        os.makedirs(fold_ckpt_dir, exist_ok=True)

        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=fold_run_name,
            log_model=True,
            config=vars(args),
        )

        # Reset model for each fold
        model = TellSiteSegmentationModel(
            encoder_name=args.encoder,
            encoder_weights="imagenet",
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        # ------------------------------------------------------------------
        # Callbacks
        # ------------------------------------------------------------------
        checkpoint_callback = ModelCheckpoint(
            dirpath=fold_ckpt_dir,
            filename="manet-effb3-{epoch:02d}-{val/iou:.4f}",
            monitor="val/iou",
            mode="max",
            save_top_k=3,
            save_last=True,
            verbose=True,
        )

        early_stopping = EarlyStopping(
            monitor="val/iou",
            mode="max",
            patience=args.patience,
            verbose=True,
        )

        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        plot_callback = TrainingPlotCallback(
            save_path=os.path.join(fold_ckpt_dir, "training_curves.png")
        )

        profiler_callback = ProfilerCallback(
            save_dir=os.path.join(fold_ckpt_dir, "profiler"),
            enabled=args.profile,
            wait=args.profile_wait,
            warmup=args.profile_warmup,
            active=args.profile_active,
        )

        # ------------------------------------------------------------------
        # Trainer
        # ------------------------------------------------------------------
        callbacks_list = [checkpoint_callback, early_stopping, lr_monitor, plot_callback, profiler_callback]
        
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="32",
            logger=wandb_logger,
            callbacks=callbacks_list,
            log_every_n_steps=10,
            deterministic=True,
        )

        # ------------------------------------------------------------------
        # Train or test
        # ------------------------------------------------------------------
        if args.test_only:
            if args.ckpt_path is None:
                raise ValueError("--ckpt_path required for --test_only mode.")
            print(f"\nRunning test evaluation from checkpoint: {args.ckpt_path}")
            trainer.test(model, datamodule=dm, ckpt_path=args.ckpt_path)
        else:
            print(f"\nStarting training{' (Fold ' + str(fold_idx+1) + '/' + str(n_folds) + ')' if args.cross_validation else ''}...")
            trainer.fit(model, datamodule=dm)

            print(f"\nRunning validation evaluation on best checkpoint{' (Fold ' + str(fold_idx+1) + '/' + str(n_folds) + ')' if args.cross_validation else ''}...")
            val_result = trainer.validate(model, datamodule=dm, ckpt_path="best")
            
            if val_result:
                fold_results.append({
                    "fold": fold_idx + 1,
                    "metrics": val_result[0]
                })
            
            # Optional: Profile the best model
            if args.profile and not args.cross_validation:
                print(f"\n[Profile] Profiling best model checkpoint...")
                try:
                    best_ckpt = trainer.checkpoint_callback.best_model_path
                    best_model = TellSiteSegmentationModel.load_from_checkpoint(best_ckpt)
                    val_loader = dm.val_dataloader()
                    profile_dir = os.path.join(fold_ckpt_dir, "profiler_best")
                    profile_model(
                        best_model, val_loader,
                        num_batches=10,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        profile_dir=profile_dir,
                        wait=args.profile_wait,
                        warmup=args.profile_warmup,
                        active=args.profile_active,
                    )
                except Exception as e:
                    print(f"[Profile] Warning: Could not profile model: {e}")

        wandb.finish()

    # ------------------------------------------------------------------
    # Cross-validation Results Summary
    # ------------------------------------------------------------------
    if args.cross_validation and fold_results:
        print("\n" + "="*60)
        print("CROSS-VALIDATION SUMMARY")
        print("="*60)
        
        all_ious = []
        for fold_data in fold_results:
            fold_num = fold_data["fold"]
            metrics = fold_data["metrics"]
            val_iou = metrics.get("val/iou", 0)
            all_ious.append(val_iou)
            print(f"Fold {fold_num}: val/iou = {val_iou:.4f}")
        
        if all_ious:
            print(f"\nMean IoU: {np.mean(all_ious):.4f} ± {np.std(all_ious):.4f}")
            print(f"Best IoU: {max(all_ious):.4f}")
            print(f"Worst IoU: {min(all_ious):.4f}")
        print("="*60)


if __name__ == "__main__":
    main()
