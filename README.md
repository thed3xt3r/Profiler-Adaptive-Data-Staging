# Profiler-Adaptive Data Staging for Large-Scale Satellite-Image Segmentation

Code for a master's thesis on the data path of deep-learning training on shared
HPC clusters, applied to segmentation of mounded settlement sites (*tells*) in
satellite imagery. PADS-Loader is a drop-in replacement for the PyTorch
`DataLoader` that profiles itself, picks a staging policy, and derives its
prefetch depth from those measurements.

The manuscript is maintained outside this repository.

## Layout

| Directory | What it is |
|---|---|
| `0-reproduction/` | Replica of the reference training protocol. Produces the checkpoints used for inference. |
| `1-baseline/` | Baseline loader configurations: default, tuned, and full pre-stage. |
| `2-webdataset/` | Fixed tar-shard streaming. |
| `3-pads/` | PADS-Loader itself, plus `discovery/`, the large-area inference pipeline. |
| `4-inference/` | Benchmarking, per-tile scoring, and figure generation. |
| `ops/` | Slurm job scripts. |
| `qgis/` | Vector layers defining the survey area and the excluded known sites. |
| `images/` | Discovery tiles (untracked) and the README screenshots. |

Stages `0`, `2` and `3` each run three architectures: DeepLabV3+ (ResNet-50),
MA-Net (EfficientNet-B3) and SegFormer-B0. Within those stages the per-model
directories all have the same shape: `model.py` defines the network,
`dataset.py` the loading pipeline, `train.py` the entry point, and the
`.slurm` files submit it. In `3-pads/` the interesting file is `dataset.py`,
which is where profiling, policy selection and staging live.

Untracked, because they are data or build products: `bing_1k/`, `images/*.jpg`,
`discovery_tiles/`, `runs/`, `checkpoints_*/`, `.pylibs/`, `.geolibs/` and
`tell_seg.sif`.

## Setup

Everything runs inside the Apptainer image `tell_seg.sif`, built from
`tell_seg.def`. Two Python dependency trees are bound in at runtime rather than
baked into the image.

`.pylibs/` holds the training dependencies. If the numpy ABI breaks, repair it
once, never per job:

```bash
sbatch ops/fix_numpy_abi.slurm
```

`.geolibs/` holds fiona, shapely and rasterio, which the container does not
ship. Create or rebuild it with:

```bash
sbatch ops/discovery/probe_geo.slurm
```

Do not install numpy into `.geolibs/`. `PYTHONPATH` takes precedence over the
container's site-packages, so it would shadow the numpy that torch was built
against.

## Running the code

Submit every job from the repository root. The `#SBATCH --output` paths are
relative to the submission directory, not to the script, so submitting from
elsewhere fails to write logs.

### Reproduction

There is no Slurm wrapper for this stage; `train.py` is invoked directly,
inside the container, once per architecture:

```bash
python 0-reproduction/train.py --arch deeplab --epochs 20 --test_passes 10
```

`--arch` takes `deeplab`, `manet` or `segformer`. Checkpoints and results land
in `0-reproduction/runs/<arch>_full/`, and those checkpoints are what the
discovery pipeline uses.

### Baselines

```bash
sbatch 1-baseline/deeplab/run_baseline_deeplab.slurm
```

One script per architecture. Edit the flags inside to select the configuration:
the default loader, the manually tuned loader, or `--full_prestage` for the
full pre-stage ceiling.

### Tar-shard streaming

```bash
python 2-webdataset/create_shards.py
sbatch 2-webdataset/manet/run_manet_wds.slurm
```

Build the shards first; they are written to `2-webdataset/shards/`. Only MA-Net
has a Slurm wrapper here; the other two are run by calling their `train.py`
directly.

### PADS

```bash
sbatch 3-pads/build_tars.slurm
sbatch 3-pads/deeplab/PAD_deeplab_hpc.slurm
```

Build the tar archives once, then train. Each architecture has three Slurm
variants: plain, `_hpc` and `_l40s`, differing only in partition and resource
request. `train.py` decides whether the policy is fixed or chosen adaptively.

### Discovery inference

Run these in order:

```bash
sbatch ops/discovery/probe_geo.slurm             # install the geo stack
sbatch ops/discovery/build_manifest.slurm        # rebuild tiles.json from qgis/
sbatch ops/discovery/check_discovery_ckpt.slurm  # confirm checkpoints load
sbatch ops/discovery/preflight_discovery.slurm   # confirm everything else
sbatch ops/discovery/run_discovery.slurm         # inference, then stitching
sbatch ops/discovery/build_overviews.slurm       # make the output usable in QGIS
```

Run the preflight. `run_discovery.py` has no dry-run mode and falls straight
into hours of inference, whereas the preflight makes the same checks in
seconds.

Paths, checkpoints and inference parameters are set in
`3-pads/discovery/config.hpc.json`.

### Figures and scoring

```bash
sbatch ops/thesis/score_tiles.slurm
sbatch ops/thesis/make_pred_fig.slurm
```

These drive the scripts in `4-inference/`, which write to
`4-inference/figures/`.

## Discovery output

The run writes three single-band float32 probability rasters, one per
architecture, plus a four-band RGBA composite of the thresholded results and a
JSON legend, all in EPSG:32638.

In the composite each model owns one colour channel, so overlapping detections
combine additively:

| Colour | Models above threshold |
|---|---|
| Red | DeepLabV3+ |
| Green | MA-Net |
| Blue | SegFormer |
| Yellow | DeepLabV3+ and MA-Net |
| Magenta | DeepLabV3+ and SegFormer |
| Cyan | MA-Net and SegFormer |
| White | All three |
| Dark grey | None |
| Transparent | Outside the survey area |

Build overviews before opening these in QGIS, or rendering will be unusably
slow. Use nearest-neighbour resampling, which `build_overviews.slurm` already
does, so that averaging does not blend the class colours into shades that
correspond to no class.

The composite drawn over the imagery, with the excluded known sites in orange:

![Agreement layer over irrigated farmland](images/agreement-irrigated-farmland.png)

![Agreement layer over field systems and a watercourse](images/agreement-field-systems.png)

![Agreement layer over an arid margin](images/agreement-arid-margin.png)

## Storage

Home replicates every file twice, so a file costs double its apparent size
against quota. The discovery run writes to scratch instead; `output_dir` in
`config.hpc.json` points there. Scratch is neither backed up nor permanent, so
copy the rasters off it once a run finishes.
