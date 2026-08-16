#!/usr/bin/env python3
"""Reconstruct a stopped QGIS Bing-render manifest from authoritative inputs."""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import fiona
from shapely.geometry import box, shape
from shapely.ops import unary_union


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_union(path):
    layers = fiona.listlayers(path)
    if len(layers) != 1:
        raise ValueError("Expected one layer in {}; found {}: {}".format(path, len(layers), layers))
    with fiona.open(path, layer=layers[0]) as source:
        geometries = [shape(feature["geometry"]) for feature in source if feature["geometry"]]
        if not geometries:
            raise ValueError("No usable geometry in {}".format(path))
        return unary_union(geometries), source.crs, source.crs_wkt, layers[0]


def arguments():
    parser = argparse.ArgumentParser(
        description="Replay render_bing_discovery_tiles_qgis.py ordering and write tiles.json")
    parser.add_argument("--project-area", required=True, type=Path)
    parser.add_argument("--known-site-exclusion", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--take-first", required=True, type=int,
                        help="Explicit count of consecutive tile_NNNNNN images")
    parser.add_argument("--tile-size-metres", type=float, default=1000.0)
    parser.add_argument("--output-pixels", type=int, default=1024)
    parser.add_argument("--minimum-area-fraction", type=float, default=0.01)
    parser.add_argument("--extension", default="jpg", choices=("jpg", "jpeg", "png"))
    return parser.parse_args()


def main():
    args = arguments()
    if args.take_first <= 0 or args.tile_size_metres <= 0 or args.output_pixels <= 0:
        raise ValueError("take-first, tile size, and output pixels must be positive")
    for path in (args.project_area, args.known_site_exclusion):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.images_dir.is_dir():
        raise NotADirectoryError(args.images_dir)
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite: {}".format(args.output))

    area, area_crs, area_wkt, area_layer = read_union(args.project_area)
    exclusion, exclusion_crs, _, exclusion_layer = read_union(args.known_site_exclusion)
    if area_crs != exclusion_crs:
        raise ValueError("CRS mismatch: area {} versus exclusion {}".format(area_crs, exclusion_crs))

    xmin0 = math.floor(area.bounds[0] / args.tile_size_metres) * args.tile_size_metres
    ymin0 = math.floor(area.bounds[1] / args.tile_size_metres) * args.tile_size_metres
    columns = int(math.ceil((area.bounds[2] - xmin0) / args.tile_size_metres))
    rows = int(math.ceil((area.bounds[3] - ymin0) / args.tile_size_metres))
    square_area = args.tile_size_metres ** 2
    tiles, eligible_total = [], 0

    # Exact renderer order: row outer, column inner; increment sequence only
    # after both area and buffered-site eligibility tests pass.
    for row in range(rows):
        y0 = ymin0 + row * args.tile_size_metres
        for column in range(columns):
            x0 = xmin0 + column * args.tile_size_metres
            square = box(x0, y0, x0 + args.tile_size_metres, y0 + args.tile_size_metres)
            if square.intersection(area).area / square_area < args.minimum_area_fraction:
                continue
            if not exclusion.is_empty and square.intersects(exclusion):
                continue
            eligible_total += 1
            if eligible_total <= args.take_first:
                tile_id = "tile_{:06d}".format(eligible_total)
                filename = tile_id + "." + args.extension
                image = args.images_dir / filename
                if not image.is_file():
                    raise FileNotFoundError("Expected consecutive image: {}".format(image))
                tiles.append({
                    "tile_id": tile_id, "filename": filename,
                    "row": row, "column": column,
                    "xmin": x0, "ymin": y0,
                    "xmax": x0 + args.tile_size_metres,
                    "ymax": y0 + args.tile_size_metres,
                    "cx": x0 + args.tile_size_metres / 2,
                    "cy": y0 + args.tile_size_metres / 2,
                    "crs_authid": area_crs.to_string(),
                    "tile_size_metres": args.tile_size_metres,
                    "output_pixels": args.output_pixels,
                })

    if eligible_total < args.take_first:
        raise ValueError("Requested first {}, but renderer yields only {} eligible tiles".format(
            args.take_first, eligible_total))
    expected = {"tile_{:06d}.{}".format(i, args.extension)
                for i in range(1, args.take_first + 1)}
    actual = {path.name for path in args.images_dir.glob("*." + args.extension)}
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        raise ValueError("Inventory mismatch: {} missing, {} extra; examples {} {}".format(
            len(missing), len(extra), missing[:3], extra[:3]))

    manifest = {
        "schema_version": 1,
        "reconstruction": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "algorithm": "QGIS Bing renderer row-major eligibility replay",
            "take_first": args.take_first, "eligible_total": eligible_total,
            "grid_origin": [xmin0, ymin0], "grid_rows": rows, "grid_columns": columns,
            "project_area": {"path": str(args.project_area.resolve()), "layer": area_layer,
                             "sha256": file_hash(args.project_area)},
            "known_site_exclusion": {
                "path": str(args.known_site_exclusion.resolve()), "layer": exclusion_layer,
                "sha256": file_hash(args.known_site_exclusion)},
        },
        "crs_authid": area_crs.to_string(), "crs_wkt": area_wkt,
        "tile_size_metres": args.tile_size_metres,
        "output_pixels": args.output_pixels,
        "pixel_size_metres": args.tile_size_metres / args.output_pixels,
        "tile_count": len(tiles), "tiles": tiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Wrote {} mappings to {}".format(len(tiles), args.output))
    print("Renderer-eligible total: {}; grid origin: ({}, {})".format(
        eligible_total, xmin0, ymin0))
    print("Area SHA-256: {}".format(manifest["reconstruction"]["project_area"]["sha256"]))
    print("Exclusion SHA-256: {}".format(
        manifest["reconstruction"]["known_site_exclusion"]["sha256"]))


if __name__ == "__main__":
    main()
