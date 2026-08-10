import argparse
import contextlib
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

from dataset import (load_dataset, get_transforms, ArcheoDataset, TarArcheoDataset,
                     StagingArcheoDataset, compute_adaptive_stage_depth)
from model import ArcheoModel
from gpu_monitor import GPUUtilMonitor
from fsop_monitor import FSOpCounter

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
    level above the code (~/Thesis/segformer/logs for ~/Thesis/3-pads/segformer/train.py),
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
    """Node-local staging area used by the PADS 'stage' policy.

    Deliberately NOT derived from project_root: giving every isolated
    project_root (one per RQ/ablation config, for checkpoint isolation) its
    own scratch subdirectory meant every stage-policy job independently
    copied its own multi-GB snapshot of the dataset -- 69 such copies (~91GB)
    accumulated in one session and exhausted the disk quota. A real
    node-local scratch mount is shared per-node regardless of which
    experiment is using it; this restores that by defaulting to one fixed
    location under the repo root unless PADS_SCRATCH_DIR overrides it.
    """
    value = os.environ.get("PADS_SCRATCH_DIR") or os.path.join(REPO_ROOT, "runs", "scratch")
    scratch_dir = os.path.abspath(os.path.expanduser(value))
    # StagingArcheoDataset swallows copy errors, so a missing directory would
    # silently downgrade the stage policy to a plain read. Create it up front.
    os.makedirs(scratch_dir, exist_ok=True)
    return scratch_dir


def resolve_tar_paths(data_root):
    """Return (originals_tar, negs_tar) for the 'shard' policy, or (None, None).

    Uncompressed .tar is preferred over .tar.gz: the payload is already-compressed
    JPEG/PNG, so gzip costs random-access performance for no size benefit.
    Build them with 3-pads/build_tars.py -- nothing else in the repo produces
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

def classify_bottleneck(metrics, scratch_available=False, gamma_s=1.25, gamma_d=1.5,
                         epsilon=0.05):
    """Classify the dominant bottleneck from profiled timings.

    gamma_s, gamma_d and epsilon are the thresholds from the methodology's
    classification rule (Section~\\ref{ch:methodology}): thresholds are set
    above unity so a policy is adopted only when the corresponding cost
    dominates, not merely when it is present. Defaults (1.25, 1.5) match the
    values used throughout the thesis; overriding them is how Chapter 6's
    classifier-threshold-sensitivity ablation is run.
    """
    t_data = float(metrics.get("t_data", 0.0))
    t_decode = float(metrics.get("t_decode", 0.0))
    t_h2d = float(metrics.get("t_h2d", 0.0))
    t_gpu = float(metrics.get("t_gpu", 0.0))

    if scratch_available and t_h2d > max(t_decode, t_gpu, epsilon) * gamma_s:
        return "stage"
    if t_decode > max(t_h2d, t_gpu) * gamma_d:
        return "shard"
    if t_data > max(t_decode, t_gpu) * gamma_d:
        return "shard"
    return "loose"


def select_policy(metrics, scratch_available=False, gamma_s=1.25, gamma_d=1.5,
                   epsilon=0.05, stage_h2d_threshold=0.08):
    """Select a PADS policy from collected profiler metrics."""
    if scratch_available and float(metrics.get("t_h2d", 0.0)) > stage_h2d_threshold:
        return "stage"
    return classify_bottleneck(metrics, scratch_available=scratch_available,
                                gamma_s=gamma_s, gamma_d=gamma_d, epsilon=epsilon)


def build_dataloader(dataset, batch_size, shuffle, drop_last, policy="loose", num_workers=4,
                      prefetch_depth=None):
    """Create a DataLoader with policy-based worker settings.

    prefetch_depth, if given, overrides the policy-derived prefetch_factor
    below -- needed for RQ4's fixed-depth sweep (1/2/4/8) against PADS's
    auto-tuned depth, and to make method 2 ("tuned DataLoader") an explorable
    knob rather than one hardcoded guess.
    """
    if policy == "stage":
        workers = min(max(num_workers, 4), 8)
        prefetch_factor = 2
    elif policy == "shard":
        workers = min(max(num_workers, 2), 8)
        prefetch_factor = 2
    else:
        workers = max(1, num_workers - 1)
        prefetch_factor = 1

    if prefetch_depth is not None:
        prefetch_factor = prefetch_depth

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
                   originals_tar_path=None, negs_tar_path=None, scratch_dir=None,
                   full_prestage=False, scratch_capacity_pct=None, stage_depth=None):
    """Factory function to create the appropriate dataset based on policy."""
    if policy == "shard":
        if originals_tar_path and negs_tar_path:
            return TarArcheoDataset(data_items, originals_tar_path, negs_tar_path, transform=transform)
        else:
            print(f"[Warning] shard policy requested but tar files not found, falling back to loose")
            return ArcheoDataset(data_items, originals_images_dir, originals_masks_dir,
                                negs_images_dir, negs_masks_dir, transform=transform)

    elif policy == "stage":
        stage_kwargs = dict(scratch_dir=scratch_dir, full_prestage=full_prestage,
                             scratch_capacity_pct=scratch_capacity_pct)
        if stage_depth is not None:
            stage_kwargs["stage_depth"] = stage_depth
        return StagingArcheoDataset(data_items, originals_images_dir, originals_masks_dir,
                                   negs_images_dir, negs_masks_dir, transform=transform,
                                   **stage_kwargs)

    else:  # loose
        return ArcheoDataset(data_items, originals_images_dir, originals_masks_dir,
                            negs_images_dir, negs_masks_dir, transform=transform)


def profile_data_pipeline(model, dataset, batch_size=32, device="cuda",
                          num_batches=12, scratch_available=True,
                          profile_dir="profiler_logs", num_workers=0,
                          gamma_s=1.25, gamma_d=1.5, epsilon=0.05,
                          stage_h2d_threshold=0.08, discard_batches=0,
                          checkpoint_batches=None):
    """Profile batch acquisition, decode and H2D timings and select a PADS policy.

    discard_batches: run this many batches unmeasured before the timed loop
    starts, to burn in one-time CUDA/cuDNN warmup cost (kernel autotuning,
    allocator warmup) instead of letting it dominate a short average -- see
    the profiler-bias diagnostic (8-batch vs. 147-batch full-epoch probe).
    checkpoint_batches: optional list of post-discard batch counts at which
    to snapshot the running metrics/ratio, for sweeping window size in one
    pass instead of re-running the whole probe per candidate N.
    """
    warmup_start = time.perf_counter()
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
    checkpoints = {}
    remaining_checkpoints = sorted(checkpoint_batches) if checkpoint_batches else []

    # RQ2 (Table~\ref{tab:rq2-fsops}): counting open()/stat() calls only works
    # if __getitem__ runs in THIS process -- a DataLoader worker subprocess
    # does not share the patched builtins.open/os.stat below, so with
    # num_workers > 0 the counts would be silently wrong (under-counted), not
    # just imprecise. Only enable counting when num_workers == 0.
    fsop_counter = FSOpCounter() if num_workers == 0 else None
    if num_workers != 0:
        print(f"[PADS Profiler] num_workers={num_workers} != 0: filesystem "
              f"operation counts will not be recorded for this run (workers "
              f"run in separate processes that don't share the patched "
              f"builtins). Pass num_workers=0 to measure RQ2.")

    for _ in range(discard_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        images = batch[0].to(device, non_blocking=True).float()
        with torch.no_grad():
            _ = model(images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    dataset.reset_profile_stats()

    with (fsop_counter if fsop_counter is not None else contextlib.nullcontext()):
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

            while remaining_checkpoints and batches_seen == remaining_checkpoints[0]:
                n = remaining_checkpoints.pop(0)
                checkpoints[n] = {
                    "t_data": metrics["t_data"] / n, "t_decode": metrics["t_decode"] / n,
                    "t_h2d": metrics["t_h2d"] / n, "t_gpu": metrics["t_gpu"] / n,
                    "ratio_data_gpu": (metrics["t_data"] / n) / max(metrics["t_gpu"] / n, 1e-9),
                }

    if batches_seen == 0:
        metrics = {"t_data": 0.0, "t_decode": 0.0, "t_h2d": 0.0, "t_gpu": 0.0}
        policy = "loose"
    else:
        for key in metrics:
            metrics[key] /= batches_seen
        policy = select_policy(metrics, scratch_available=scratch_available,
                                gamma_s=gamma_s, gamma_d=gamma_d, epsilon=epsilon,
                                stage_h2d_threshold=stage_h2d_threshold)

    dataset.profile_enabled = False
    warmup_wallclock_s = time.perf_counter() - warmup_start

    summary = {
        "policy": policy,
        "scratch_available": scratch_available,
        "metrics": metrics,
        "batches_profiled": batches_seen,
        "thresholds": {
            "gamma_s": gamma_s, "gamma_d": gamma_d, "epsilon": epsilon,
            "stage_h2d_threshold": stage_h2d_threshold,
        },
        # Wall-clock cost of the whole warm-up window (dataset/dataloader
        # construction through the last profiled batch) -- distinct from the
        # per-batch t_data/t_decode/t_h2d/t_gpu averages above, and what
        # Table~\ref{tab:overhead} (profiling overhead) actually needs: the
        # one-off cost paid before PADS starts training, not a per-batch rate.
        "warmup_wallclock_s": warmup_wallclock_s,
        # RQ2 (Table~\ref{tab:rq2-fsops}): None if num_workers != 0 -- see the
        # warning printed above the profiling loop for why.
        "fsops": (fsop_counter.summary(batches=batches_seen)
                  if fsop_counter is not None else None),
        "discard_batches": discard_batches,
        "checkpoints": checkpoints,
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
    print(f"  Warm-up wall-clock: {warmup_wallclock_s:.2f}s")
    if summary["fsops"] is not None:
        print(f"  open() calls: {summary['fsops']['open_calls']} "
              f"({summary['fsops']['open_calls_per_batch']:.1f}/batch)")
        print(f"  stat() calls: {summary['fsops']['stat_calls']} "
              f"({summary['fsops']['stat_calls_per_batch']:.1f}/batch)")
        print(f"  bytes read: {summary['fsops']['bytes_read']}")
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
    parser.add_argument("--sweep_discard", type=int, default=0,
                         help="Batches to run unmeasured before profiling starts, to burn in CUDA/cuDNN warmup cost (see profiler-bias diagnostic).")
    parser.add_argument("--sweep_checkpoints", type=str, default="",
                         help="Comma-separated post-discard batch counts to snapshot t_data/t_gpu/ratio at, e.g. '8,16,32,64'.")
    parser.add_argument("--profile_wait", type=int, default=1)
    parser.add_argument("--profile_warmup", type=int, default=1)
    parser.add_argument("--profile_active", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    # Training is at 512x512 now (the Resize(256) was dropped to match the
    # reference implementation), which is 4x the pixels of the old default.
    # SegFormer-B0 is the lightest of the three; measured peak on a 4 GB card at
    # fp16: batch 8 -> 2.24 GB, batch 16 -> 4.41 GB (over capacity).
    # 8 x 4 keeps the effective batch at 32, matching the HPC runs.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Per-step batch size (limited by VRAM)")
    parser.add_argument("--accumulate", type=int, default=1,
                        help="Gradient accumulation steps; effective batch = batch_size * accumulate")
    parser.add_argument("--precision", type=str, default="32",
                        help="Lightning precision, e.g. 16-mixed, bf16-mixed, 32")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (Windows spawns processes, keep modest)")
    parser.add_argument("--data_policy", type=str, default=None,
                        choices=["loose", "shard", "stage"],
                        help="Force a PADS data policy instead of letting the profiler "
                             "choose (or defaulting to loose). Needed to A/B the policies.")
    parser.add_argument("--full_prestage", action="store_true",
                        help="Evaluation method 4: copy the entire dataset to node-local "
                             "scratch before training starts (implies --data_policy stage; "
                             "no adaptive/rolling staging). This is the static ceiling that "
                             "PADS's adaptive staging is compared against, not a PADS mode.")
    parser.add_argument("--scratch_capacity_pct", type=float, default=None,
                        help="Chapter 6 scratch-capacity ablation: cap the 'stage' policy's "
                             "scratch usage at this fraction (0.0-1.0) of the corpus, "
                             "synthetically simulating scratch too small to hold everything. "
                             "Unset = unconstrained (real scratch capacity).")
    parser.add_argument("--prefetch_depth", type=int, default=None,
                        help="Override the policy-derived DataLoader prefetch_factor with a "
                             "fixed value (e.g. 1/2/4/8). Needed for RQ4's fixed-depth sweep "
                             "against PADS's auto-tuned depth. Unset = policy default.")
    parser.add_argument("--gamma_s", type=float, default=1.25,
                        help="Stage-policy threshold (Section~ch:methodology). Default 1.25. "
                             "Needed for Chapter 6's classifier-threshold-sensitivity ablation.")
    parser.add_argument("--gamma_d", type=float, default=1.5,
                        help="Shard-policy threshold (Section~ch:methodology). Default 1.5. "
                             "Needed for Chapter 6's classifier-threshold-sensitivity ablation.")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Floor preventing division by near-zero timings on fast local "
                             "storage (Section~ch:methodology). Default 0.05.")
    parser.add_argument("--stage_h2d_threshold", type=float, default=0.08,
                        help="Absolute t_h2d threshold (seconds) select_policy() checks before "
                             "falling back to the gamma-based classify_bottleneck() rule.")
    parser.add_argument("--profile_only", action="store_true",
                        help="Run the PADS policy profiler, report the decision, and exit "
                             "without training. Implies --profile_policy.")
    parser.add_argument("--scale_input", action="store_true",
                        help="Divide inputs by 255 before ImageNet standardisation. "
                             "DEVIATION from Casini et al., which standardises raw 0-255.")
    parser.add_argument("--focal_alpha", type=float, default=None,
                        help="Focal loss alpha. Reference uses None (unweighted).")
    parser.add_argument("--val_crop", type=str, default="random", choices=["random", "center"],
                        help="Validation crop. Reference random-crops and averages 10 runs.")
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

    gpu_monitor = GPUUtilMonitor(device_index=0)
    try:
        with gpu_monitor:
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

    gpu_util_summary = gpu_monitor.summary()
    gpu_util_path = os.path.join(config["checkpoint_path"], "gpu_util_summary.json")
    with open(gpu_util_path, "w") as handle:
        json.dump(gpu_util_summary, handle, indent=2)
    if gpu_util_summary["gpu_util_mean_pct"] is not None:
        print(f"\n[GPUUtilMonitor] Mean GPU utilisation over training: "
              f"{gpu_util_summary['gpu_util_mean_pct']:.1f}% "
              f"({gpu_util_summary['gpu_util_n_samples']} samples)")
    else:
        print("\n[GPUUtilMonitor] No GPU utilisation samples recorded "
              "(pynvml unavailable or nvmlInit failed) -- see "
              f"{gpu_util_path}")

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
        "scale_input": args.scale_input,
        "focal_alpha": args.focal_alpha,
        "val_crop": args.val_crop,
        "forced_data_policy": args.data_policy,
        "full_prestage": args.full_prestage,
        "scratch_capacity_pct": args.scratch_capacity_pct,
        "prefetch_depth": args.prefetch_depth,
        "gamma_s": args.gamma_s,
        "gamma_d": args.gamma_d,
        "epsilon": args.epsilon,
        "stage_h2d_threshold": args.stage_h2d_threshold,
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

    train_transform, val_transform = get_transforms(val_crop=base_config["val_crop"])

    dataset_kwargs = dict(
        scratch_dir=SCRATCH_DIR,
        originals_tar_path=ORIGINALS_TAR,
        negs_tar_path=NEGS_TAR,
        full_prestage=base_config.get("full_prestage", False),
        scratch_capacity_pct=base_config.get("scratch_capacity_pct"),
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
    if base_config.get("full_prestage"):
        # Evaluation method 4: the full pre-stage ceiling is a "stage" dataset
        # by construction (only StagingArcheoDataset knows how to write to
        # scratch), just with full_prestage=True instead of PADS's adaptive
        # rolling window. --data_policy/--profile_policy are not consulted.
        base_config["data_policy"] = "stage"
        print("\n[PADS] --full_prestage given: forcing data_policy='stage' with "
              "the entire dataset staged up front (evaluation method 4).")
    elif base_config.get("forced_data_policy"):
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
            gamma_s=base_config["gamma_s"],
            gamma_d=base_config["gamma_d"],
            epsilon=base_config["epsilon"],
            stage_h2d_threshold=base_config["stage_h2d_threshold"],
            discard_batches=args.sweep_discard,
            checkpoint_batches=([int(x) for x in args.sweep_checkpoints.split(",") if x.strip()]
                                 if args.sweep_checkpoints else None),
        )
        base_config["data_policy"] = policy_summary["policy"]
    else:
        base_config["data_policy"] = "loose"

    # Now build the datasets the policy actually calls for.
    policy = base_config["data_policy"]
    if policy == "shard" and not (ORIGINALS_TAR and NEGS_TAR):
        print("[PADS] Warning: 'shard' selected but no tar archives found; "
              "create_dataset will fall back to loose. Build them with "
              "3-pads/build_tars.py.")
    print(f"[PADS] Building datasets with policy: {policy}")

    # RQ4 (Table~\ref{tab:rq4-depth}): resolve the predictive-staging
    # prefetch depth d before building datasets, per Section~\ref{ch:methodology}
    # Stage 4. --prefetch_depth fixes it directly (the RQ4.1-4.4 sweep);
    # otherwise it's derived from measured timings via the thesis's own
    # formula, d = min(ceil(t_stage/t_consume) + 1, d_max) -- this is what
    # makes RQ4.5 ("auto-tuned") an actual measurement instead of a relabeled
    # copy of whatever the hardcoded default used to be.
    if policy == "stage" and not base_config.get("full_prestage"):
        if base_config.get("prefetch_depth") is not None:
            dataset_kwargs["stage_depth"] = base_config["prefetch_depth"]
            print(f"[PADS] Stage prefetch depth fixed via --prefetch_depth: "
                  f"d={dataset_kwargs['stage_depth']}")
        else:
            print("\n[PADS] Measuring t_stage/t_consume to auto-tune the "
                  "predictive-staging prefetch depth...")
            stage_depth_info = compute_adaptive_stage_depth(
                train_data, originals_images_dir, originals_masks_dir,
                negs_images_dir, negs_masks_dir, SCRATCH_DIR,
            )
            dataset_kwargs["stage_depth"] = stage_depth_info["depth"]
            print(f"[PADS] Auto-tuned prefetch depth: d={stage_depth_info['depth']} "
                  f"(t_stage={stage_depth_info['t_stage']*1000:.1f}ms, "
                  f"t_consume={stage_depth_info['t_consume']*1000:.1f}ms, "
                  f"d_max={stage_depth_info['d_max']})")
            depth_dir = os.path.join(PROJECT_ROOT, "segformer", "logs")
            os.makedirs(depth_dir, exist_ok=True)
            depth_summary_path = os.path.join(depth_dir, "stage_depth_summary.json")
            with open(depth_summary_path, "w") as handle:
                json.dump(stage_depth_info, handle, indent=2)
            print(f"[PADS] Stage-depth summary saved to: {depth_summary_path}")

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
        # RQ2 (Table~\ref{tab:rq2-fsops}): profile_data_pipeline() above only
        # ever profiles a loose probe (it needs *something* to measure before
        # a policy is even chosen) -- when --data_policy forces loose/shard/
        # stage directly, that branch is skipped entirely and no fsop count
        # is produced at all. Count real fs ops against the policy actually
        # in effect here (train_dataset, whatever create_dataset() built
        # above) so forced and profiler-selected policies both get a number.
        fsop_dir = os.path.join(PROJECT_ROOT, "segformer", "logs")
        os.makedirs(fsop_dir, exist_ok=True)
        fsop_loader = DataLoader(train_dataset, batch_size=base_config["batch_size"],
                                  shuffle=False, num_workers=0)
        fsop_iterator = iter(fsop_loader)
        batches_seen = 0
        with FSOpCounter() as fsop_counter:
            for _ in range(base_config["profile_batches"]):
                try:
                    next(fsop_iterator)
                except StopIteration:
                    break
                batches_seen += 1
        fsops = fsop_counter.summary(batches=batches_seen)
        fsops["policy"] = policy
        fsops["batches_profiled"] = batches_seen
        fsops_path = os.path.join(fsop_dir, f"fsops_policy_{policy}.json")
        with open(fsops_path, "w") as handle:
            json.dump(fsops, handle, indent=2)
        print(f"\n[FSOpCounter] fs-op summary for policy={policy}")
        if batches_seen:
            print(f"  open() calls: {fsops['open_calls']} ({fsops['open_calls_per_batch']:.1f}/batch)")
            print(f"  stat() calls: {fsops['stat_calls']} ({fsops['stat_calls_per_batch']:.1f}/batch)")
            print(f"  bytes read: {fsops['bytes_read']} ({fsops['bytes_read_per_batch']:.0f}/batch)")
        print(f"  Summary saved to: {fsops_path}")

        print("[PADS] --profile_only: policy decided and datasets built; exiting before training.")
        return

    train_loader = build_dataloader(
        train_dataset,
        batch_size=base_config["batch_size"],
        shuffle=True,
        drop_last=True,
        policy=base_config["data_policy"],
        num_workers=base_config["num_workers"],
        prefetch_depth=base_config.get("prefetch_depth"),
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=base_config["batch_size"],
        shuffle=False,
        drop_last=False,
        policy=base_config["data_policy"],
        num_workers=base_config["num_workers"],
        prefetch_depth=base_config.get("prefetch_depth"),
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
