import argparse
import json
import os
import random
import time
import numpy as np
import optuna
import torch
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for HPC
import matplotlib.pyplot as plt

from dataset import load_dataset, get_transforms, ArcheoDataset, TarArcheoDataset, StagingArcheoDataset
from model import ArcheoModel

# ---------------------------------------------------------------------------
# GPU setup
# ---------------------------------------------------------------------------
# Let fp32 matmuls use tensor cores where the card has them (Ampere and later,
# e.g. the RTX 30xx series). Ignored on older hardware.
torch.set_float32_matmul_precision('medium')

# NOTE: the HPC build carried a monkey-patch that forced Lightning's
# _is_ampere_or_later() to return False, to work around a driver mismatch on
# the cluster nodes. It is deliberately gone: on a local Ampere card it only
# suppresses the fast bf16/tensor-core paths we actually want.


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
# On the HPC every path hung off ~/Thesis. Locally the code, the dataset and the
# run outputs sit in three different places, so both roots are resolved from the
# environment (or --project_root / --data_root) with a filesystem probe as the
# fallback. PADS_PROJECT_ROOT receives checkpoints, logs and scratch;
# PADS_DATA_ROOT is the directory that *contains* the bing_1k folder.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_NAME = "bing_1k"


def resolve_project_root(cli_value=None):
    """Directory that receives checkpoints, logs and scratch.

    Defaults to <repo>/runs, NOT the repo root. The HPC layout puts outputs one
    level above the code (~/Thesis/segformer/logs for ~/Thesis/PADS/segformer/train.py),
    but this repo root already holds the baseline `deeplab/` and `segformer/`
    source directories, so that convention would write PADS logs straight into
    them and make the two indistinguishable. The structure *below* the root is
    unchanged, so PADS_PROJECT_ROOT=~/Thesis restores the exact cluster layout.
    """
    value = cli_value or os.environ.get("PADS_PROJECT_ROOT") or os.path.join(REPO_ROOT, "runs")
    return os.path.abspath(os.path.expanduser(value))


def resolve_data_root(cli_value=None):
    """Directory that contains the bing_1k dataset folder."""
    value = cli_value or os.environ.get("PADS_DATA_ROOT")
    if value:
        return os.path.abspath(os.path.expanduser(value))

    # No override: take the first candidate that actually holds the dataset, so
    # the same file works locally and on the cluster.
    candidates = (
        os.path.join(REPO_ROOT, "data"),
        os.path.join(os.path.dirname(os.path.dirname(REPO_ROOT)), "Thesis", "source"),
        os.path.expanduser("~/Thesis"),
    )
    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, DATASET_NAME, "train")):
            return os.path.abspath(candidate)
    return os.path.abspath(candidates[0])


def resolve_scratch_dir(project_root):
    """Node-local staging area used by the PADS 'stage' policy."""
    value = os.environ.get("PADS_SCRATCH_DIR") or os.path.join(project_root, "scratch")
    scratch_dir = os.path.abspath(os.path.expanduser(value))
    # StagingArcheoDataset swallows copy errors, so a missing directory would
    # silently downgrade the stage policy to a plain read. Create it up front.
    os.makedirs(scratch_dir, exist_ok=True)
    return scratch_dir


def resolve_tar_paths(data_root):
    """Return (originals_tar, negs_tar) for the 'shard' policy, or (None, None).

    Uncompressed .tar is preferred over .tar.gz: the payload is already-compressed
    JPEG/PNG, so gzip costs random-access performance for no size benefit.
    Build them with PADS/build_tars.py -- nothing else in the repo produces
    archives this dataset can read.
    """
    shard_dir = os.environ.get("PADS_SHARD_DIR") or os.path.join(data_root, f"{DATASET_NAME}_tars")
    for ext in (".tar", ".tar.gz"):
        originals_tar = os.path.join(shard_dir, f"originals{ext}")
        negs_tar = os.path.join(shard_dir, f"negs{ext}")
        if os.path.exists(originals_tar) and os.path.exists(negs_tar):
            return originals_tar, negs_tar
    return None, None


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

        fig.suptitle("Training Curves — SegFormer B0", fontsize=14)
        plt.tight_layout()
        plt.savefig(self.save_path, dpi=150)
        plt.close(fig)
        print(f"\nTraining curves saved to: {self.save_path}")

    @staticmethod
    def _append(lst, value):
        if value is not None:
            lst.append(float(value))


class NonFiniteLossGuard(pl.Callback):
    """Stop training as soon as the epoch's mean training loss is not finite.

    An fp16 overflow is a silent disaster otherwise: the loss goes NaN, every
    metric collapses to 0, and EarlyStopping still waits out its full patience
    on a model that is already dead. On this project that cost 15 wasted epochs
    (~2 h) before anyone noticed the "best" score was just the last value before
    the blowup.

    The check is at epoch level on purpose. Individual batches may legitimately
    produce inf/NaN under AMP -- GradScaler detects those, skips the step and
    lowers the scale -- so aborting on the first bad batch would false-positive.
    A non-finite *epoch mean* means the weights themselves are gone.
    """

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train/loss_epoch")
        if loss is None or torch.isfinite(torch.as_tensor(loss)).all():
            return
        raise RuntimeError(
            f"\nTraining loss became non-finite ({loss}) at epoch "
            f"{trainer.current_epoch}; the model will not recover.\n"
            f"This is almost always fp16 overflow. EfficientNet encoders are "
            f"especially prone to it (SiLU + squeeze-excite).\n"
            f"Re-run with --precision bf16-mixed (same exponent range as fp32, "
            f"native on Ampere and later) or --precision 32."
        )


# ---------------------------------------------------------------------------
# PADS profiler helpers
# ---------------------------------------------------------------------------

def classify_bottleneck(metrics, scratch_available=False):
    """Classify the dominant bottleneck from profiled timings."""
    t_data = float(metrics.get("t_data", 0.0))
    t_decode = float(metrics.get("t_decode", 0.0))
    t_h2d = float(metrics.get("t_h2d", 0.0))
    t_gpu = float(metrics.get("t_gpu", 0.0))

    if scratch_available and t_h2d > max(t_decode, t_gpu, 0.05) * 1.25:
        return "stage"
    if t_decode > max(t_h2d, t_gpu) * 1.5:
        return "shard"
    if t_data > max(t_decode, t_gpu) * 1.5:
        return "shard"
    return "loose"


def select_policy(metrics, scratch_available=False):
    """Select a PADS policy from collected profiler metrics."""
    if scratch_available and float(metrics.get("t_h2d", 0.0)) > 0.08:
        return "stage"
    return classify_bottleneck(metrics, scratch_available=scratch_available)


def build_dataloader(dataset, batch_size, shuffle, drop_last, policy="loose", num_workers=4):
    """Create a DataLoader with policy-based worker settings."""
    if policy == "stage":
        workers = min(max(num_workers, 4), 8)
        prefetch_factor = 2
    elif policy == "shard":
        workers = min(max(num_workers, 2), 8)
        prefetch_factor = 2
    else:
        workers = max(1, num_workers - 1)
        prefetch_factor = 1

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "pin_memory": torch.cuda.is_available(),
        "num_workers": workers,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)


def create_dataset(data_items, originals_images_dir, originals_masks_dir,
                   negs_images_dir, negs_masks_dir, transform, policy="loose",
                   originals_tar_path=None, negs_tar_path=None, scratch_dir=None):
    """Factory function to create the appropriate dataset based on policy."""
    if policy == "shard":
        if originals_tar_path and negs_tar_path:
            return TarArcheoDataset(data_items, originals_tar_path, negs_tar_path, transform=transform)
        else:
            print(f"[Warning] shard policy requested but tar files not found, falling back to loose")
            return ArcheoDataset(data_items, originals_images_dir, originals_masks_dir,
                                negs_images_dir, negs_masks_dir, transform=transform)

    elif policy == "stage":
        return StagingArcheoDataset(data_items, originals_images_dir, originals_masks_dir,
                                   negs_images_dir, negs_masks_dir, transform=transform,
                                   scratch_dir=scratch_dir)

    else:  # loose
        return ArcheoDataset(data_items, originals_images_dir, originals_masks_dir,
                            negs_images_dir, negs_masks_dir, transform=transform)


def profile_data_pipeline(model, dataset, batch_size=32, device="cuda",
                          num_batches=12, scratch_available=True,
                          profile_dir="profiler_logs", num_workers=0):
    """Profile batch acquisition, decode and H2D timings and select a PADS policy."""
    os.makedirs(profile_dir, exist_ok=True)
    dataset.reset_profile_stats()
    dataset.profile_enabled = True

    model = model.to(device)
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    iterator = iter(dataloader)
    metrics = {"t_data": 0.0, "t_decode": 0.0, "t_h2d": 0.0, "t_gpu": 0.0}
    batches_seen = 0

    for _ in range(num_batches):
        try:
            batch_start = time.perf_counter()
            batch = next(iterator)
            t_data = time.perf_counter() - batch_start
        except StopIteration:
            break

        sample_count = batch[0].shape[0] if isinstance(batch[0], torch.Tensor) else len(batch[0])
        decode_values = dataset.profile_stats[-sample_count:] if dataset.profile_stats else []
        t_decode = float(np.mean(decode_values)) if decode_values else 0.0
        dataset.reset_profile_stats()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        h2d_start = time.perf_counter()
        images = batch[0].to(device, non_blocking=True).float()
        t_h2d = time.perf_counter() - h2d_start

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gpu_start = time.perf_counter()
        with torch.no_grad():
            _ = model(images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_gpu = time.perf_counter() - gpu_start

        metrics["t_data"] += t_data
        metrics["t_decode"] += t_decode
        metrics["t_h2d"] += t_h2d
        metrics["t_gpu"] += t_gpu
        batches_seen += 1

    if batches_seen == 0:
        metrics = {"t_data": 0.0, "t_decode": 0.0, "t_h2d": 0.0, "t_gpu": 0.0}
        policy = "loose"
    else:
        for key in metrics:
            metrics[key] /= batches_seen
        policy = select_policy(metrics, scratch_available=scratch_available)

    dataset.profile_enabled = False

    summary = {
        "policy": policy,
        "scratch_available": scratch_available,
        "metrics": metrics,
        "batches_profiled": batches_seen,
    }
    summary_path = os.path.join(profile_dir, "data_policy_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print("\n[PADS Profiler] Warm-up summary")
    print(f"  Policy: {policy}")
    print(f"  t_data: {metrics['t_data']:.4f}s")
    print(f"  t_decode: {metrics['t_decode']:.4f}s")
    print(f"  t_h2d: {metrics['t_h2d']:.4f}s")
    print(f"  t_gpu: {metrics['t_gpu']:.4f}s")
    print(f"  Summary saved to: {summary_path}")

    return summary


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

                images = batch[0].to(device).float()

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
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--tune_trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--profile", action="store_true", help="Profile the best model")
    parser.add_argument("--profile_policy", action="store_true", help="Profile the data pipeline and choose a PADS policy")
    parser.add_argument("--profile_batches", type=int, default=8, help="Number of batches to use for PADS policy profiling")
    parser.add_argument("--profile_wait", type=int, default=1)
    parser.add_argument("--profile_warmup", type=int, default=1)
    parser.add_argument("--profile_active", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    # Training is at 512x512 now (the Resize(256) was dropped to match the
    # reference implementation), which is 4x the pixels of the old default.
    # SegFormer-B0 is the lightest of the three; measured peak on a 4 GB card at
    # fp16: batch 8 -> 2.24 GB, batch 16 -> 4.41 GB (over capacity).
    # 8 x 4 keeps the effective batch at 32, matching the HPC runs.
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-step batch size (limited by VRAM)")
    parser.add_argument("--accumulate", type=int, default=2,
                        help="Gradient accumulation steps; effective batch = batch_size * accumulate")
    parser.add_argument("--precision", type=str, default="16-mixed",
                        help="Lightning precision, e.g. 16-mixed, bf16-mixed, 32")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (Windows spawns processes, keep modest)")
    parser.add_argument("--data_policy", type=str, default=None,
                        choices=["loose", "shard", "stage"],
                        help="Force a PADS data policy instead of letting the profiler "
                             "choose (or defaulting to loose). Needed to A/B the policies.")
    parser.add_argument("--profile_only", action="store_true",
                        help="Run the PADS policy profiler, report the decision, and exit "
                             "without training. Implies --profile_policy.")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore any existing last.ckpt and train from epoch 0")
    parser.add_argument("--project_root", type=str, default=None,
                        help="Output root for checkpoints/logs/scratch (env: PADS_PROJECT_ROOT)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Directory containing the bing_1k folder (env: PADS_DATA_ROOT)")
    return parser.parse_args()


def _build_pruning_callback(trial):
    if trial is None:
        return None

    # optuna >= 4.9 moved the Lightning integration into the standalone
    # optuna-integration package; the optuna.integration alias still resolves but
    # emits a FutureWarning and goes away in v6. Try the new path first, then the
    # old one, and degrade to "no pruning" rather than taking down the study.
    try:
        from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
    except ImportError:
        try:
            from optuna.integration import PyTorchLightningPruningCallback
        except Exception as exc:
            print(f"[Optuna] Warning: pruning callback unavailable ({exc}); continuing without it.")
            return None

    try:
        return PyTorchLightningPruningCallback(trial, monitor="valid/iou")
    except Exception as exc:
        print(f"[Optuna] Warning: pruning callback unavailable ({exc}); continuing without it.")
        return None


def run_training(config, train_loader, val_loader, val_dataset, trial=None):
    model = ArcheoModel(
        encoder_name=config["encoder"],
        in_channels=config["in_channels"],
        out_classes=1,
        config=config,
    )

    callbacks_list = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["checkpoint_path"],
        # The monitored metric is "valid/iou", and on Windows that slash is a
        # path separator: with auto_insert_metric_name the name is interpolated
        # verbatim and Lightning ends up creating a "...-valid\" DIRECTORY
        # holding "iou=0.3287.ckpt". Spell the labels out and substitute only
        # the values so the checkpoint stays a single flat file.
        filename=f"segformer-b0-trial{trial.number if trial else '0'}-epoch{{epoch:02d}}-iou{{valid/iou:.4f}}",
        auto_insert_metric_name=False,
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
    callbacks_list.append(NonFiniteLossGuard())

    if trial is None:
        plot_callback = TrainingPlotCallback(
            save_path=os.path.join(config["checkpoint_path"], "training_curves.png")
        )
        callbacks_list.append(plot_callback)
    else:
        pruning_callback = _build_pruning_callback(trial)
        if pruning_callback is not None:
            callbacks_list.append(pruning_callback)

    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        precision=config["precision"],
        accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
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

    # Resume from this run's own last checkpoint if one already exists (e.g.
    # a previous attempt at this exact trial/config got preempted mid-training)
    # instead of silently restarting from epoch 0 and losing that progress.
    # --fresh opts out: without it, re-running a config whose last.ckpt already
    # reached max_epochs looks like a no-op ("max_epochs reached", nothing saved).
    resume_ckpt_path = os.path.join(config["checkpoint_path"], "last.ckpt")
    resume_ckpt_path = resume_ckpt_path if os.path.exists(resume_ckpt_path) else None
    if config.get("fresh") and resume_ckpt_path:
        print(f"\n[Resume] --fresh given; ignoring {resume_ckpt_path} and training from epoch 0.")
        resume_ckpt_path = None
    elif resume_ckpt_path:
        print(f"\n[Resume] Found existing checkpoint at {resume_ckpt_path}; resuming training from there.")
        print("         Pass --fresh (or -Fresh) to ignore it and start from epoch 0.")

    try:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=resume_ckpt_path,
        )
    except Exception as exc:
        # Resuming a checkpoint that already ran past --epochs is a hard
        # MisconfigurationException in Lightning. Turn that stack trace into
        # something actionable, since both ways out are one flag away.
        message = str(exc)
        if resume_ckpt_path and "max_epochs" in message and "current_epoch" in message:
            # print() not SystemExit(msg): the latter writes to stderr, which the
            # launcher redirects to a .err file, so the guidance would be invisible.
            print(
                f"\n[Resume] {message}\n"
                f"         {resume_ckpt_path}\n"
                f"         has already trained past --epochs {config['epochs']}. Either raise\n"
                f"         --epochs to continue it, or pass --fresh (-Fresh) to start over.",
                flush=True,
            )
            raise SystemExit(1) from exc
        raise

    best_iou = trainer.callback_metrics.get("valid/iou")
    best_iou_val = float(best_iou) if best_iou is not None else 0.0

    if config["profile"] and trial is None:
        print("\n[Profile] Profiling best model checkpoint...")
        try:
            best_ckpt = checkpoint_callback.best_model_path
            if best_ckpt:
                profiled_model = ArcheoModel.load_from_checkpoint(
                    best_ckpt,
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
                shuffle=False, drop_last=False,
                num_workers=config.get("num_workers", 4),
                pin_memory=torch.cuda.is_available(),
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

    PROJECT_ROOT = resolve_project_root(args.project_root)
    PATH_DATASETS = resolve_data_root(args.data_root)
    PATH_LOG = os.path.join(PROJECT_ROOT, "checkpoints_segformer")
    SCRATCH_DIR = resolve_scratch_dir(PROJECT_ROOT)
    ORIGINALS_TAR, NEGS_TAR = resolve_tar_paths(PATH_DATASETS)

    dataset_path = os.path.join(PATH_DATASETS, DATASET_NAME)
    if not os.path.isdir(os.path.join(dataset_path, "train")):
        raise SystemExit(
            f"Dataset not found: {os.path.join(dataset_path, 'train')}\n"
            f"Point --data_root (or PADS_DATA_ROOT) at the directory containing "
            f"the '{DATASET_NAME}' folder."
        )

    print("=" * 60)
    print("SegFormer - SegFormer B0")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Dataset:      {dataset_path}")
    print(f"  Scratch:      {SCRATCH_DIR}")
    if torch.cuda.is_available():
        gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  Device:       {torch.cuda.get_device_name(0)} ({gpu_gb:.1f} GB)")
    else:
        print("  Device:       CPU (no CUDA device visible - training will be very slow)")
    print("=" * 60)

    base_config = {
        "timestamp": datetime.now().strftime("%d-%m-%Y_%H%M%S"),
        "dataset_path": dataset_path,
        "checkpoint_path": PATH_LOG,
        "random_seed": 1234,
        "encoder": "b0",
        "loss": "focal",
        "learning_rate": 0.0001,
        "precision": args.precision,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accumulate_grad_batches": args.accumulate,
        "num_workers": args.num_workers,
        "forced_data_policy": args.data_policy,
        "fresh": args.fresh,
        "in_channels": 3,
        "patience": 15,
        "profile": args.profile,
        "profile_policy": args.profile_policy or args.profile_only,
        "profile_batches": args.profile_batches,
        "profile_wait": args.profile_wait,
        "profile_warmup": args.profile_warmup,
        "profile_active": args.profile_active,
    }

    os.makedirs(base_config["checkpoint_path"], exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "segformer", "logs"), exist_ok=True)

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

    dataset_kwargs = dict(
        scratch_dir=SCRATCH_DIR,
        originals_tar_path=ORIGINALS_TAR,
        negs_tar_path=NEGS_TAR,
    )

    # ------------------------------------------------------------------
    # Decide the policy BEFORE building the datasets it selects.
    # ------------------------------------------------------------------
    # This block used to sit *after* dataset construction, so the datasets were
    # always built with the "loose" default and never rebuilt once the profiler
    # had spoken. The selected policy reached only build_dataloader(), whose
    # entire use of it is picking num_workers/prefetch_factor -- meaning
    # TarArcheoDataset and StagingArcheoDataset were never instantiated and
    # every run, whatever policy it "selected", read loose files.
    if base_config.get("forced_data_policy"):
        base_config["data_policy"] = base_config["forced_data_policy"]
        print(f"\n[PADS] Data policy forced to '{base_config['data_policy']}' via --data_policy "
              f"(profiler not consulted).")
    elif base_config.get("profile_policy"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("\n[PADS] Profiling the data pipeline to choose a policy...")
        # The profiler needs something to measure, so it probes the loose
        # reader; the real datasets below are built with whatever it picks.
        probe_dataset = create_dataset(
            train_data, originals_images_dir, originals_masks_dir,
            negs_images_dir, negs_masks_dir, transform=train_transform,
            policy="loose", **dataset_kwargs
        )
        probe_model = ArcheoModel(
            encoder_name=base_config["encoder"],
            in_channels=base_config["in_channels"],
            out_classes=1,
            config=base_config,
        )
        policy_summary = profile_data_pipeline(
            probe_model,
            probe_dataset,
            batch_size=base_config["batch_size"],
            device=device,
            num_batches=base_config["profile_batches"],
            scratch_available=True,
            profile_dir=os.path.join(PROJECT_ROOT, "segformer", "logs"),
        )
        base_config["data_policy"] = policy_summary["policy"]
    else:
        base_config["data_policy"] = "loose"

    # Now build the datasets the policy actually calls for.
    policy = base_config["data_policy"]
    if policy == "shard" and not (ORIGINALS_TAR and NEGS_TAR):
        print("[PADS] Warning: 'shard' selected but no tar archives found; "
              "create_dataset will fall back to loose. Build them with "
              "PADS/build_tars.py.")
    print(f"[PADS] Building datasets with policy: {policy}")

    train_dataset = create_dataset(
        train_data, originals_images_dir, originals_masks_dir,
        negs_images_dir, negs_masks_dir, transform=train_transform,
        policy=policy, **dataset_kwargs
    )
    val_dataset = create_dataset(
        val_data, originals_images_dir, originals_masks_dir,
        negs_images_dir, negs_masks_dir, transform=val_transform,
        policy=policy, **dataset_kwargs
    )
    test_dataset = create_dataset(
        test_data, originals_images_dir, originals_masks_dir,
        negs_images_dir, negs_masks_dir, transform=val_transform,
        policy=policy, **dataset_kwargs
    )
    print(f"[PADS] train dataset class: {type(train_dataset).__name__}")

    if args.profile_only:
        print("[PADS] --profile_only: policy decided and datasets built; exiting before training.")
        return

    train_loader = build_dataloader(
        train_dataset,
        batch_size=base_config["batch_size"],
        shuffle=True,
        drop_last=True,
        policy=base_config["data_policy"],
        num_workers=base_config["num_workers"],
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=base_config["batch_size"],
        shuffle=False,
        drop_last=False,
        policy=base_config["data_policy"],
        num_workers=base_config["num_workers"],
    )

    if args.tune:
        def objective(trial):
            config = base_config.copy()
            config["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            config["checkpoint_path"] = os.path.join(PATH_LOG, f"trial_{trial.number}")
            os.makedirs(config["checkpoint_path"], exist_ok=True)

            iou, _ = run_training(config, train_loader, val_loader, val_dataset, trial=trial)
            return iou

        # Persist the study to disk so a preempted/resubmitted job resumes
        # instead of losing every previously-completed trial and restarting
        # trial numbering (and therefore checkpoint directories) from zero.
        optuna_storage = f"sqlite:///{os.path.join(PATH_LOG, 'optuna_study.db')}"
        study = optuna.create_study(
            direction="maximize",
            study_name="segformer_tuning",
            storage=optuna_storage,
            load_if_exists=True,
        )
        completed_trials = sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        remaining_trials = max(0, args.tune_trials - completed_trials)
        if remaining_trials > 0:
            print(f"\n[Optuna] {completed_trials} trial(s) already completed; running {remaining_trials} more "
                  f"toward the target of {args.tune_trials}.")
            study.optimize(objective, n_trials=remaining_trials)
        else:
            print(f"\n[Optuna] Target of {args.tune_trials} completed trials already reached; skipping optimize().")

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
