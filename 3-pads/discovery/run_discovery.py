#!/usr/bin/env python3
"""One-command runner: infer all three models, then create four QGIS layers."""
import argparse, importlib, json, sys
from pathlib import Path

REQUIRED=("numpy","PIL","torch","lightning","segmentation_models_pytorch","transformers","rasterio")
NAMES=("deeplab","manet","segformer")

def _path(value,base):
    path=Path(value).expanduser()
    return (base/path).resolve() if not path.is_absolute() else path.resolve()

def run(config_path):
    config_path=Path(config_path).resolve(); base=config_path.parent
    cfg=json.loads(config_path.read_text(encoding="utf-8"))
    missing=[]
    for module in REQUIRED:
        try: importlib.import_module(module)
        except Exception as exc: missing.append("{} ({})".format(module,exc))
    if missing: raise RuntimeError("missing dependencies: "+"; ".join(missing))
    from infer_tiles import run as infer
    from stitch_tile_predictions import load_manifest, stitch
    manifest=_path(cfg["manifest"],base); tiles_root=_path(cfg["tiles_root"],base)
    output=_path(cfg["output_dir"],base)
    project_root=_path(cfg.get("project_root",Path(__file__).resolve().parents[2]),base)
    checkpoints={name:_path(cfg["checkpoints"][name],base) for name in NAMES}
    errors=[]
    if not manifest.is_file(): errors.append("manifest not found: "+str(manifest))
    if not tiles_root.is_dir(): errors.append("tiles_root not found: "+str(tiles_root))
    if not project_root.is_dir(): errors.append("project_root not found: "+str(project_root))
    for name,path in checkpoints.items():
        if not path.is_file(): errors.append("{} checkpoint not found: {}".format(name,path))
    for name in NAMES:
        model_file=project_root/"3-pads"/name/"model.py"
        if not model_file.is_file(): errors.append("{} model definition not found: {}".format(name,model_file))
    if errors: raise RuntimeError("preflight failed:\n  - "+"\n  - ".join(errors))
    tiles,_=load_manifest(manifest)
    for tile in tiles:
        if not (tiles_root/tile.filename).is_file():
            raise RuntimeError("preflight failed: manifest image not found: "+str(tiles_root/tile.filename))
    inference=cfg.get("inference",{}); device=inference.get("device","cuda")
    if device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("preflight failed: CUDA requested but unavailable")
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512",local_files_only=True)
    except Exception as exc:
        raise RuntimeError("preflight failed: SegFormer-B0 is absent from the offline HuggingFace cache") from exc
    pred_root=output/"tile_probabilities"; dirs={name:pred_root/name for name in NAMES}
    print("Preflight OK: {} tiles; running models sequentially".format(len(tiles)),flush=True)
    for name in NAMES:
        infer(name,manifest,tiles_root,checkpoints[name],dirs[name],project_root,device,
              int(inference.get("window_pixels",512)),int(inference.get("stride_pixels",512)),
              int(inference.get("model_input_pixels",256)),int(inference.get("batch_size",4)))
    outputs=stitch(manifest,dirs,output/"qgis_layers",float(cfg.get("threshold",.5)))
    print("SUCCESS - add these four layers to QGIS:\n  "+"\n  ".join(map(str,outputs)))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("config",type=Path)
    args=parser.parse_args()
    try: run(args.config)
    except Exception as exc:
        print("ERROR: {}".format(exc),file=sys.stderr); raise SystemExit(2)
