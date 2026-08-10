# Profiler-Adaptive Data Staging for Large-Scale Satellite-Image Segmentation

Master's thesis code. The numbered directories are the experimental stages, in
the order the thesis presents them. Each stage runs all three architectures --
DeepLabV3+, MA-Net and SegFormer.

| Directory | Stage | What it establishes |
|---|---|---|
| `0-reproduction/` | Reproduction of Casini et al. | The accuracy floor. A faithful replica at the reference's own protocol (batch 32, 20 epochs, 10 test passes), reaching 72.8-73.5% IoU. Appendix A. |
| `1-baseline/` | Baseline data paths | Evaluation methods 1, 2 and 4: default loader, tuned loader, and the full pre-stage ceiling. The reference points everything else is measured against. |
| `2-webdataset/` | Fixed tar-shard streaming | Evaluation method 3. Isolates the benefit of *sharding* from the benefit of *adaptation*. |
| `3-pads/` | PADS-Loader | Evaluation method 5, the contribution: profile the data path, classify the bottleneck, select a staging policy, derive prefetch depth from measurement. |

Supporting directories:

| Directory | Contents |
|---|---|
| `thesis/` | LaTeX source, figures, bibliography, and the scripts that generate figures from run outputs |
| `ops/` | One-off environment repair jobs (`fix_numpy_abi.slurm` and friends) |
| `runs/` | Job outputs: logs, checkpoints, profiler JSON |
| `bing_1k/`, `bing_1k_tars/` | Dataset, loose and tar-shard forms |

## A note on `1-baseline/`

The baseline stage has no trainer of its own, by design. It runs the same
trainer as `3-pads/` with the adaptive layer switched off, so model, dataset,
hyperparameters and hardware are held identical and the only thing varying
across the five methods is the data path. A separate copy of `train.py` would
be free to drift, and any drift would silently invalidate the comparison.

## Running

Build the tar shards once before any `shard`-policy run (the reader falls back
to loose silently if they are missing):

```
sbatch 3-pads/build_tars.slurm
```

Then, per stage:

```
sbatch 1-baseline/manet/run_baseline_manet.slurm 1      # method 1, default loader
sbatch 1-baseline/manet/run_baseline_manet.slurm 2      # method 2, tuned loader
sbatch 1-baseline/manet/run_baseline_manet.slurm 4      # method 4, pre-stage ceiling
sbatch 3-pads/manet/PAD_manet_hpc.slurm standard 20 shard
```

If `.pylibs` is ever reported broken, repair it **once** with
`sbatch ops/fix_numpy_abi.slurm` -- never per-job, since a live
`pip install --target` can race a sibling job that is already importing numpy.
