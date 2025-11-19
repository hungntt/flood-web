import os
import json
from datetime import datetime

import numpy as np
import xarray as xr
import cdsapi
import pytz
from dotenv import load_dotenv
from flask import Flask, jsonify, send_from_directory
from shapely.geometry import shape, Point

# ==========================================================
# Configuration
# ==========================================================
load_dotenv()

CDS_URL = os.getenv("CDS_URL")
CDS_KEY = os.getenv("CDS_KEY")

if not CDS_URL or not CDS_KEY:
    raise ValueError("CDS_URL and CDS_KEY must be set in .env file")

# Vietnam local timezone
VN_TZ = pytz.timezone("Asia/Ho_Chi_Minh")

# Bounding box for Gia Lai region
# Format [N, W, S, E]
GIA_LAI_BBOX = [14.9, 107.0, 12.8, 109.5]

GRIB_TEMPLATE = "glofas_forecast_gialai_{date}.grib"

# Optional Gia Lai boundary for clipping
GIA_LAI_BOUNDARY_PATH = "gia_lai_boundary.geojson"
GIA_LAI_POLYGON = None

if os.path.exists(GIA_LAI_BOUNDARY_PATH):
    with open(GIA_LAI_BOUNDARY_PATH, "r", encoding="utf8") as f:
        gj = json.load(f)
    geom = gj["features"][0]["geometry"]
    GIA_LAI_POLYGON = shape(geom)
    print("[INFO] Loaded Gia Lai boundary polygon")
else:
    print("[WARN] Gia Lai boundary file not found, using bounding box only")


# ==========================================================
# Utility download GloFAS forecast
# ==========================================================
def download_glofas_forecast():
    """
    Download the daily GloFAS GRIB file for Gia Lai
    if it does not already exist for today.
    """

    now_vn = datetime.now(VN_TZ)
    date_key = now_vn.strftime("%Y%m%d")
    # date_key = 20251119

    grib_path = GRIB_TEMPLATE.format(date=date_key)

    if os.path.exists(grib_path):
        print(f"[SKIP] GRIB already exists for {date_key}: {grib_path}")
        return grib_path

    print(f"[DOWNLOAD] Fetching GloFAS forecast for Gia Lai date {date_key}")

    client = cdsapi.Client(url=CDS_URL, key=CDS_KEY)

    request_params = {
        "system_version": "operational",
        "hydrological_model": "lisflood",
        "product_type": "control_forecast",
        "variable": "river_discharge_in_the_last_24_hours",
        "year": now_vn.strftime("%Y"),
        "month": now_vn.strftime("%m"),
        "day": now_vn.strftime("%d"),
        # Only 24 hour lead time
        "leadtime_hour": ["24"],
        # Gia Lai bounding box only
        "area": GIA_LAI_BBOX,
        "format": "grib",
    }

    client.retrieve("cems-glofas-forecast", request_params, grib_path)

    print(f"[DONE] File downloaded: {grib_path}")
    return grib_path


# ==========================================================
# Utility pick a usable variable from the dataset
# ==========================================================
def choose_variable(ds):
    candidates = [
        "river_discharge_in_the_last_24_hours",
        "dis24",
        "dis",
    ]

    for name in candidates:
        if name in ds.data_vars:
            print(f"[INFO] Using variable: {name}")
            return ds[name]

    fallback = list(ds.data_vars)[0]
    print(f"[WARN] Using fallback variable: {fallback}")
    return ds[fallback]


# ==========================================================
# Risk classification for legend
# ==========================================================
def compute_thresholds(values):
    """
    Compute quantile based thresholds to map discharge to
    proxy return periods 2, 5, 20 years.
    """

    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return 1.0, 5.0, 10.0

    q2 = float(np.quantile(positive, 0.55))
    q5 = float(np.quantile(positive, 0.85))
    q20 = float(np.quantile(positive, 0.98))

    return q2, q5, q20


def classify_point(discharge, q2, q5, q20):
    """
    Map one discharge value to risk_level and risk_color
    matching the legend in index.html.
    """

    if not np.isfinite(discharge) or discharge <= 0:
        return None, None

    if discharge >= q20:
        return "20", "purple"
    elif discharge >= q5:
        return "5", "red"
    elif discharge >= q2:
        return "2", "orange"
    else:
        return "1", "gray"


# ==========================================================
# Core convert GRIB to GeoJSON polygons for frontend
# ==========================================================
def grib_to_geojson(grib_path):
    """
    Open the GRIB file and return a GeoJSON FeatureCollection
    with Polygon cells per grid for Leaflet overlay.

    Each feature has
      geometry Polygon (grid cell)
      properties discharge, risk_level, risk_color
    """

    print(f"[PROCESS] Opening GRIB {grib_path}")

    ds = xr.open_dataset(grib_path, engine="cfgrib")

    var = choose_variable(ds)

    # Use only the 24 hour lead time instead of 7 day maximum
    if "step" in var.dims:
        step_coord = var.coords.get("step", None)
        if step_coord is not None:
            try:
                data_24h = var.sel(step=24)
            except Exception:
                data_24h = var.isel(step=0)
        else:
            data_24h = var.isel(step=0)
    else:
        data_24h = var

    data_max = data_24h

    for dim in list(data_max.dims):
        if (
            dim not in ("latitude", "lat", "longitude", "lon")
            and data_max.sizes[dim] == 1
        ):
            data_max = data_max.isel({dim: 0})

    if "latitude" in data_max.dims:
        lat_name = "latitude"
    else:
        lat_name = "lat"

    if "longitude" in data_max.dims:
        lon_name = "longitude"
    else:
        lon_name = "lon"

    lat = ds[lat_name].values
    lon = ds[lon_name].values
    values = data_max.values

    if values.ndim != 2:
        raise RuntimeError(
            f"Expected 2D field after reduction, got shape {values.shape}"
        )

    q2, q5, q20 = compute_thresholds(values)
    print(f"[INFO] Thresholds q2={q2} q5={q5} q20={q20}")

    features = []

    n_lat = len(lat)
    n_lon = len(lon)

    for i in range(n_lat - 1):
        for j in range(n_lon - 1):
            discharge = float(values[i, j])

            risk_level, risk_color = classify_point(discharge, q2, q5, q20)
            if risk_level is None:
                continue

            lon_left = float(lon[j])
            lon_right = float(lon[j + 1])
            lat_top = float(lat[i])
            lat_bottom = float(lat[i + 1])

            # Optional clip to Gia Lai polygon based on cell centroid
            if GIA_LAI_POLYGON is not None:
                centroid_lon = 0.5 * (lon_left + lon_right)
                centroid_lat = 0.5 * (lat_top + lat_bottom)
                if not GIA_LAI_POLYGON.contains(Point(centroid_lon, centroid_lat)):
                    continue

            polygon = [
                [lon_left, lat_top],
                [lon_right, lat_top],
                [lon_right, lat_bottom],
                [lon_left, lat_bottom],
                [lon_left, lat_top],
            ]

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon],
                },
                "properties": {
                    "discharge": round(discharge, 2),
                    "risk_level": risk_level,
                    "risk_color": risk_color,
                },
            }
            features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    print(f"[DONE] GeoJSON built with {len(features)} polygons")
    return geojson
