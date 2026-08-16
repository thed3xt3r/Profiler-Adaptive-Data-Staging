# QGIS tiles to four discovery layers

This directory provides one HPC command for the QGIS tile export. It reads
`tiles.json` or `tiles.csv`, runs DeepLabV3+, MA-Net and SegFormer sequentially,
and writes exactly four GeoTIFF layers for QGIS:

1. `deeplab_probability.tif`
2. `manet_probability.tif`
3. `segformer_probability.tif`
4. `model_agreement_provenance_rgba.tif`

The probability layers are float32 (`0..1`, nodata `-9999`). The overlay is
RGBA and uses additive model colours: DeepLab red, MA-Net green, SegFormer blue,
DeepLab+MA-Net yellow, DeepLab+SegFormer magenta, MA-Net+SegFormer cyan, and
all three white. Dark grey means no model exceeds the configured threshold;
transparent pixels are gaps/excluded tiles. The same mapping is recorded in
`model_agreement_provenance_legend.json`.

## Prepare once

Copy the complete QGIS export without flattening its layout:

```text
discovery_tiles/
  tiles.json                 # or tiles.csv
  images/tile_000000.jpg
  images/tile_000001.jpg
  ...
```

Copy `config.hpc.example.json` to `config.hpc.json` and replace the three
checkpoint placeholders with the best validation-IoU checkpoints. Expected
names are:

- `deeplab-resnet50-trial0-epochNN-iouX.ckpt`
- `manet-effb3-trial0-epochNN-iouX.ckpt`
- `segformer-b0-trial0-epochNN-iouX.ckpt`

Use these best checkpoints rather than `last.ckpt`. Standard training runs put
them below `runs/checkpoints_{deeplab,manet,segformer}`. Policy-isolated runs put
them below `runs/policy_<policy>[/alpha_<alpha>][/tag_<tag>]/checkpoints_<model>`.

The container must already contain the packages in
`requirements-inference.txt`. SegFormer-B0 must be in the same HuggingFace cache
used for training. The runner forces offline lookup and never downloads model
files. Its preflight checks every dependency, model file, checkpoint, manifest
image, CUDA availability, and the SegFormer cache before inference begins.

## One command on HPC

From the project root (`$HOME/Thesis` in the supplied Slurm scripts):

```bash
apptainer exec --nv \
  --bind "$HOME/Thesis:/workspace" \
  --env PYTHONPATH=/workspace/.pylibs \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HUB_OFFLINE=1 \
  "$HOME/Thesis/tell_seg.sif" \
  python -u /workspace/3-pads/discovery/run_discovery.py \
  /workspace/3-pads/discovery/config.hpc.json
```

The model saw 512-pixel crops resized to 256 during training. The runner
therefore covers each 1024-pixel QGIS tile with deterministic 512-pixel windows,
runs the model at 256 pixels, maps probabilities back to the window, and blends
overlaps when `stride_pixels < window_pixels`. Default stride 512 reproduces a
non-overlapping four-window survey of each tile.

Stitching uses manifest bounds rather than filename order. GeoTIFFs are tiled,
compressed, BigTIFF-safe and sparse, so excluded grid gaps do not allocate dense
arrays in RAM or materialise unnecessary raster blocks. Overlapping geographic
tiles are rejected; the QGIS renderer's default 1 km grid is non-overlapping.
