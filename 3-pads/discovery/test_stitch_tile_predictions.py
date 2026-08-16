import json
from pathlib import Path
import numpy as np
from stitch_tile_predictions import COLORS, _geometry, _mapping, _probability, load_manifest, stitch

def test_manifest_mapping_and_geometry_without_gis(tmp_path: Path):
    manifest={"crs_authid":"EPSG:3857","output_pixels":4,"tiles":[
        {"tile_id":"a","filename":"images/a.jpg","xmin":0,"ymin":0,"xmax":4,"ymax":4},
        {"tile_id":"b","filename":"images/b.jpg","xmin":8,"ymin":0,"xmax":12,"ymax":4}]}
    path=tmp_path/"tiles.json"; path.write_text(json.dumps(manifest),encoding="utf-8")
    tiles,_=load_manifest(path); predictions=tmp_path/"predictions"; predictions.mkdir()
    for tile in tiles: np.save(predictions/(tile.tile_id+".npy"),np.full((4,4),.25,dtype="float32"))
    mapped=_mapping(tiles,predictions)
    assert set(mapped)=={"a","b"}
    assert _geometry(tiles)==(0.0,0.0,12.0,4.0,1.0,12,4)
    assert np.allclose(_probability(mapped["a"],4),.25)

def test_sparse_layers_and_additive_colours(tmp_path: Path):
    manifest={"crs_authid":"EPSG:3857","output_pixels":4,"tiles":[
        {"tile_id":"a","filename":"images/a.jpg","xmin":0,"ymin":0,"xmax":4,"ymax":4},
        {"tile_id":"b","filename":"images/b.jpg","xmin":8,"ymin":0,"xmax":12,"ymax":4}]}
    path=tmp_path/"tiles.json"; path.write_text(json.dumps(manifest),encoding="utf-8")
    dirs={}
    values={"deeplab":.9,"manet":.9,"segformer":.1}
    for name,value in values.items():
        d=tmp_path/name; d.mkdir(); dirs[name]=d
        np.save(d/"a.npy",np.full((4,4),value,dtype="float32"))
        np.save(d/"b.npy",np.full((4,4),.1,dtype="float32"))
    outputs=stitch(path,dirs,tmp_path/"out",.5)
    assert len(outputs)==4 and all(p.is_file() for p in outputs)
    import rasterio
    with rasterio.open(outputs[3]) as src:
        assert src.count==4 and src.width==12 and src.height==4
        rgba=src.read(window=rasterio.windows.Window(0,0,4,4))
        assert tuple(rgba[:3,0,0])==COLORS[3] and rgba[3,0,0]==255
        assert np.all(src.read(4,window=rasterio.windows.Window(4,0,4,4))==0)
