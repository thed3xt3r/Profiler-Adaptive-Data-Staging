#!/usr/bin/env python3
"""Manifest-driven, model-specific full-tile sliding inference."""
import hashlib, importlib.util, json, os
from pathlib import Path
import numpy as np
from PIL import Image
from stitch_tile_predictions import load_manifest

SPECS={"deeplab":("deeplab","DeepLabV3Plus","resnet50"),
       "manet":("manet","MAnet","efficientnet-b3"),
       "segformer":("segformer",None,"b0")}

def _class(root,name):
    path=Path(root)/"3-pads"/SPECS[name][0]/"model.py"
    spec=importlib.util.spec_from_file_location("pads_"+name,path); module=importlib.util.module_from_spec(spec)
    if spec.loader is None: raise ImportError("cannot load "+str(path))
    spec.loader.exec_module(module); return module.ArcheoModel

def _load(root,name,checkpoint,device):
    import torch
    checkpoint=Path(checkpoint)
    if not checkpoint.is_file(): raise FileNotFoundError("checkpoint: "+str(checkpoint))
    cfg={"loss":"focal","focal_alpha":None,"focal_gamma":2.0,"learning_rate":1e-4,
         "batch_size":1,"scale_input":False,"encoder":SPECS[name][2],"in_channels":3}
    cls=_class(root,name)
    if name=="segformer":
        os.environ.setdefault("TRANSFORMERS_OFFLINE","1"); os.environ.setdefault("HF_HUB_OFFLINE","1")
        try: model=cls(encoder_name="b0",in_channels=3,out_classes=1,config=cfg)
        except Exception as exc: raise RuntimeError("SegFormer-B0 base files are absent from the offline HF cache") from exc
    else: model=cls(SPECS[name][1],encoder_name=SPECS[name][2],encoder_weights=None,in_channels=3,out_classes=1,config=cfg)
    raw=torch.load(checkpoint,map_location="cpu"); state=raw.get("state_dict",raw) if isinstance(raw,dict) else raw
    model.load_state_dict(state,strict=True); return model.eval().to(device)

def _starts(size,window,stride):
    if size<window: raise ValueError("tile smaller than inference window")
    values=list(range(0,size-window+1,stride))
    if values[-1]!=size-window: values.append(size-window)
    return values

def _predict(model,name,path,device,window,stride,input_pixels,batch_size):
    import torch
    import torch.nn.functional as F
    image=np.asarray(Image.open(path).convert("RGB")); h,w=image.shape[:2]
    positions=[(y,x) for y in _starts(h,window,stride) for x in _starts(w,window,stride)]
    total=np.zeros((h,w),"float32"); count=np.zeros((h,w),"float32")
    axis=np.ones(window,"float32") if stride>=window else np.maximum(np.hanning(window),.05).astype("float32")
    weight=np.outer(axis,axis)
    for offset in range(0,len(positions),batch_size):
        pos=positions[offset:offset+batch_size]
        patches=np.stack([image[y:y+window,x:x+window] for y,x in pos])
        tensor=torch.from_numpy(patches).permute(0,3,1,2).float()
        if name=="segformer": tensor/=255.0
        tensor=F.interpolate(tensor,size=(input_pixels,input_pixels),mode="bilinear",align_corners=False).to(device)
        with torch.inference_mode():
            pred=torch.sigmoid(model(tensor)); pred=F.interpolate(pred,size=(window,window),mode="bilinear",align_corners=False)
        for (y,x),a in zip(pos,pred[:,0].cpu().numpy()): total[y:y+window,x:x+window]+=a*weight; count[y:y+window,x:x+window]+=weight
    if np.any(count==0): raise RuntimeError("uncovered pixels in "+str(path))
    return total/count

def run(name,manifest,tiles_root,checkpoint,output,project_root,device="cuda",window=512,stride=512,input_pixels=256,batch_size=4):
    import torch
    if name not in SPECS: raise ValueError("unknown model: "+name)
    if device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    tiles,meta=load_manifest(manifest); tiles_root=Path(tiles_root)
    images=[(tiles_root/t.filename).resolve() for t in tiles]
    missing=[p for p in images if not p.is_file()]
    if missing: raise FileNotFoundError("manifest image missing: "+str(missing[0]))
    model=_load(project_root,name,checkpoint,device); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    for i,(tile,path) in enumerate(zip(tiles,images),1):
        a=_predict(model,name,path,device,window,stride,input_pixels,batch_size)
        if a.shape!=(tile.output_pixels,tile.output_pixels): raise ValueError("image/manifest pixel mismatch for "+tile.tile_id)
        # float16 halves the intermediate footprint (231 GB -> 116 GB per model;
        # 692 GB -> 346 GB for all three), which matters because stitch() needs
        # every tile from all three models present at once to build the agreement
        # overlay. _probability() casts back to float32 on load, so the stitch is
        # unaffected, and half precision resolves ~3 decimal digits over [0,1] --
        # far finer than a 0.5 threshold requires.
        np.save(output/(tile.tile_id+".npy"),a.astype("float16"),allow_pickle=False)
        if i==1 or i%100==0 or i==len(tiles): print("{} {}/{}".format(name,i,len(tiles)),flush=True)
    hasher=hashlib.sha256()
    with Path(checkpoint).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): hasher.update(chunk)
    digest=hasher.hexdigest()
    (output/"inference_metadata.json").write_text(json.dumps({"model":name,"checkpoint":str(Path(checkpoint).resolve()),
        "checkpoint_sha256":digest,"tile_count":len(tiles),"window":window,"stride":stride,
        "model_input_pixels":input_pixels,"probability_dtype":"float16",
        "manifest_metadata":meta},indent=2),encoding="utf-8")
    return output
