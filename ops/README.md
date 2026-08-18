# ops/

Slurm job scripts. All are submitted from the repository root, e.g.

```bash
sbatch ops/discovery/run_discovery.slurm
```

`#SBATCH --output` paths are relative to the submission directory, not to the
script, so submitting from anywhere else will fail to write logs.

## Layout

| path | contents |
|---|---|
| `discovery/` | the landscape-scale inference campaign (Section 7 of the thesis) |
| `thesis/` | figure and table generation jobs, driving `4-inference/` |
| `fix_*.slurm` | one-time environment repairs for `.pylibs` |

The `fix_*.slurm` scripts stay at the top level because the training scripts
under `1-baseline/` and `3-pads/` refer to them by that path in their error
messages; moving them would silently invalidate about ten such references.

## discovery/

Ordered as they would be run:

| script | purpose |
|---|---|
| `probe_geo.slurm` | installs fiona/shapely/rasterio into `.geolibs` (the container ships no geo stack) |
| `build_manifest.slurm` | reconstructs `discovery_tiles/tiles.json` from the two QGIS vector layers |
| `check_discovery_ckpt.slurm` | verifies the reproduction checkpoints load with `strict=True` |
| `preflight_discovery.slurm` | replicates every check `run_discovery.py` makes, before committing to a long job |
| `bench_discovery.slurm` | measures per-tile inference cost to project the campaign runtime |
| `run_discovery.slurm` | the campaign: three models over 55,008 tiles, then mosaicking |
| `stitch_only.slurm` | mosaicking alone, when predictions already exist on scratch |
| `build_overviews.slurm` | overview pyramid on the agreement raster, so QGIS can render it |
| `count_classes.slurm` | exact per-class pixel areas over the full-resolution raster |
| `test_float16_roundtrip.slurm` | checks half-precision predictions survive inference and mosaicking |
| `quota_check.slurm` | reports home and scratch quota |

Note that `.geolibs` deliberately contains no numpy of its own, so fiona and
rasterio bind to the container's numpy 1.24.3 — the version torch 2.0.1 was
built against. Reinstalling numpy into it will break inference.
