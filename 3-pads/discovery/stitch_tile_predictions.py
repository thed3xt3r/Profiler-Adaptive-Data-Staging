#!/usr/bin/env python3
"""Write three sparse probability mosaics and one additive RGBA overlay."""
import csv, json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

NAMES = ("deeplab", "manet", "segformer")
BITS = {"deeplab": 1, "manet": 2, "segformer": 4}
COLORS = {0:(32,32,32), 1:(255,0,0), 2:(0,255,0), 3:(255,255,0),
          4:(0,0,255), 5:(255,0,255), 6:(0,255,255), 7:(255,255,255)}
LABELS = {0:"No detection", 1:"DeepLab", 2:"MANet", 3:"DeepLab + MANet",
          4:"SegFormer", 5:"DeepLab + SegFormer", 6:"MANet + SegFormer", 7:"All three"}

@dataclass(frozen=True)
class Tile:
    tile_id: str; filename: str
    xmin: float; ymin: float; xmax: float; ymax: float
    crs_authid: str; crs_wkt: str; output_pixels: int

def load_manifest(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        meta = json.loads(path.read_text(encoding="utf-8")); rows = meta.get("tiles", [])
    elif path.suffix.lower() == ".csv":
        meta = {}
        with path.open(newline="", encoding="utf-8-sig") as stream: rows = list(csv.DictReader(stream))
    else: raise ValueError("manifest must be tiles.json or tiles.csv")
    if not rows: raise ValueError("manifest contains no tiles")
    tiles=[]; seen=set()
    for row in rows:
        tid=str(row.get("tile_id", "")).strip()
        if not tid or tid in seen: raise ValueError("missing/duplicate tile_id: {!r}".format(tid))
        seen.add(tid)
        tile=Tile(tid, str(row["filename"]), *[float(row[k]) for k in ("xmin","ymin","xmax","ymax")],
                  str(row.get("crs_authid") or meta.get("crs_authid") or ""),
                  str(row.get("crs_wkt") or meta.get("crs_wkt") or ""),
                  int(float(row.get("output_pixels") or meta.get("output_pixels") or 0)))
        if tile.xmax<=tile.xmin or tile.ymax<=tile.ymin or tile.output_pixels<=0:
            raise ValueError("invalid extent/output_pixels for "+tid)
        tiles.append(tile)
    if len({(t.crs_authid,t.crs_wkt) for t in tiles}) != 1: raise ValueError("mixed CRS in manifest")
    if len({t.output_pixels for t in tiles}) != 1: raise ValueError("mixed output_pixels in manifest")
    return tiles, meta

def _mapping(tiles, directory):
    directory=Path(directory)
    if not directory.is_dir(): raise FileNotFoundError("prediction directory: "+str(directory))
    found={p.stem:p for p in directory.glob("*.npy")}
    result={}
    for t in tiles:
        hits={found[k] for k in (t.tile_id,Path(t.filename).stem) if k in found}
        if len(hits)!=1: raise ValueError("{} maps to {} predictions in {}".format(t.tile_id,len(hits),directory))
        result[t.tile_id]=hits.pop()
    extras=set(found.values())-set(result.values())
    if extras: raise ValueError("unmapped predictions: "+", ".join(map(str,sorted(extras)[:10])))
    return result

def _geometry(tiles):
    p=tiles[0].output_pixels
    rs=[(t.xmax-t.xmin)/p for t in tiles]+[(t.ymax-t.ymin)/p for t in tiles]
    r=float(np.median(rs))
    if any(abs(x-r)>r*1e-5 for x in rs): raise ValueError("nonuniform pixel resolution")
    xmin,ymin=min(t.xmin for t in tiles),min(t.ymin for t in tiles)
    xmax,ymax=max(t.xmax for t in tiles),max(t.ymax for t in tiles)
    return xmin,ymin,xmax,ymax,r,round((xmax-xmin)/r),round((ymax-ymin)/r)

def _probability(path, pixels):
    a=np.asarray(np.load(path,allow_pickle=False)).squeeze().astype("float32")
    if a.shape!=(pixels,pixels): raise ValueError("{} shape {}, expected {}".format(path,a.shape,(pixels,pixels)))
    if not np.isfinite(a).all() or a.min()<0 or a.max()>1: raise ValueError("invalid probabilities: "+str(path))
    return a

def stitch(manifest, prediction_dirs, output_dir, threshold=.5):
    import rasterio
    from rasterio.enums import ColorInterp
    from rasterio.transform import from_origin
    from rasterio.windows import Window
    tiles,meta=load_manifest(manifest); maps={n:_mapping(tiles,prediction_dirs[n]) for n in NAMES}
    xmin,_,_,ymax,res,width,height=_geometry(tiles); p=tiles[0].output_pixels
    base=dict(driver="GTiff",height=height,width=width,crs=tiles[0].crs_wkt or tiles[0].crs_authid,
              transform=from_origin(xmin,ymax,res,res),tiled=True,compress="deflate",
              BIGTIFF="IF_SAFER",SPARSE_OK="TRUE")
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    probs={n:out/(n+"_probability.tif") for n in NAMES}; overlay_path=out/"model_agreement_provenance_rgba.tif"
    dst={n:rasterio.open(path,"w",count=1,dtype="float32",nodata=-9999.0,predictor=3,**base) for n,path in probs.items()}
    overlay=rasterio.open(overlay_path,"w",count=4,dtype="uint8",**base)
    overlay.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.alpha)
    occupied=set()
    try:
        for i,t in enumerate(tiles,1):
            cf=(t.xmin-xmin)/res; rf=(ymax-t.ymax)/res; c,r=round(cf),round(rf)
            if abs(cf-c)>1e-5 or abs(rf-r)>1e-5: raise ValueError("off-grid tile: "+t.tile_id)
            if (r,c) in occupied: raise ValueError("overlapping grid tile: "+t.tile_id)
            occupied.add((r,c)); window=Window(c,r,p,p)
            arrays={n:_probability(maps[n][t.tile_id],p) for n in NAMES}
            for n,a in arrays.items(): dst[n].write(a,1,window=window)
            code=sum(BITS[n]*(a>=threshold).astype("uint8") for n,a in arrays.items())
            rgba=np.empty((4,p,p),dtype="uint8")
            for value,color in COLORS.items():
                mask=code==value; rgba[0,mask],rgba[1,mask],rgba[2,mask]=color
            rgba[3]=255; overlay.write(rgba,window=window)
            if i==1 or i%100==0 or i==len(tiles): print("stitched {}/{}".format(i,len(tiles)),flush=True)
    finally:
        for d in dst.values(): d.close()
        overlay.close()
    (out/"model_agreement_provenance_legend.json").write_text(json.dumps({
        "threshold":threshold,"bit_mapping":BITS,"classes":[{"value":i,"label":LABELS[i],"rgb":COLORS[i]} for i in range(8)],
        "manifest":str(Path(manifest).resolve()),"tile_count":len(tiles),"manifest_metadata":meta},indent=2),encoding="utf-8")
    return [probs[n] for n in NAMES]+[overlay_path]
