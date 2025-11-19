import json
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

# 1. Load whole Vietnam GeoJSON
with open("vn.json", "r", encoding="utf-8") as f:
    vn = json.load(f)

# 2. Filter Gia Lai (VN30) + Binh Dinh (VN31)
target_ids = {"VN30", "VN31"}
features = [
    feat for feat in vn["features"]
    if feat["properties"].get("id") in target_ids
]

if len(features) != 2:
    raise RuntimeError(f"Expected 2 features for VN30 + VN31, got {len(features)}")

# 3. Merge geometries
geoms = [shape(feat["geometry"]) for feat in features]
merged_geom = unary_union(geoms)

# 4. Compute bounding box (lon/lat order)
minx, miny, maxx, maxy = merged_geom.bounds
print("Bounds (lon/lat):", minx, miny, maxx, maxy)

# 5. Build a new GeoJSON with a single merged feature
out_fc = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "id": "VN30_VN31",
                "name": "Gia Lai + Binh Dinh"
            },
            "geometry": mapping(merged_geom),
        }
    ],
}

# 6. Save to gia_lai_boundary.geojson
with open("gia_lai_boundary.geojson", "w", encoding="utf-8") as f:
    json.dump(out_fc, f, ensure_ascii=False)

print("Saved gia_lai_boundary.geojson")