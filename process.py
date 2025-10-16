import os
import io
import zipfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union  # kept, but we won't use it for envelope

# Try to import a robust make_valid; provide safe fallback
try:
    from shapely.validation import make_valid as _make_valid
except Exception:
    try:
        from shapely import (
            make_valid as _make_valid,
        )  # shapely>=2 sometimes exposes here
    except Exception:
        _make_valid = None

# --------------------
# Config (via env)
# --------------------
PARQUET_PATH = os.getenv("PARQUET_PATH", "geojson_assets.parquet")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "geojsons"))

START_DATE = datetime.strptime(os.getenv("START_DATE", "2025-09-01"), "%Y-%m-%d").date()
END_DATE = datetime.strptime(os.getenv("END_DATE", "2025-09-30"), "%Y-%m-%d").date()

# Public URL prefix used to populate hrefs in the parquet (does not need to be reachable for testing)
ASSET_BASE_URL_GEOJSON = os.getenv(
    "ASSET_BASE_URL_GEOJSON", "http://127.0.0.1:9091/geojsons"
)

STYLE_URL = os.getenv(
    "STYLE_URL",
    "https://raw.githubusercontent.com/gtif-cerulean/assets/refs/heads/main/styles/dmi-ice-charts.json",
)

# USNIC prd prefix (e.g., "30")
USNIC_PREFIX = os.getenv("USNIC_PREFIX", "30")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------
# Helpers
# --------------------
def date_iter(d0, d1):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def usnic_zip_url(d):
    # prd = <prefix><mmddYYYY> e.g., 30 + 10032025 -> 3010032025
    return f"https://usicecenter.gov/File/DownloadArchive?prd={USNIC_PREFIX}{d.strftime('%m%d%Y')}"


def download_zip_bytes(url: str) -> bytes:
    import requests

    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def extract_first_shp(zip_bytes: bytes, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(target_dir)
    shp_files = list(target_dir.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError("No .shp found inside archive")
    return shp_files[0]


def load_existing_parquet(path: Path) -> gpd.GeoDataFrame:
    if path.exists():
        return gpd.read_parquet(path)
    # Simple, flat schema to avoid Arrow struct issues
    return gpd.GeoDataFrame(
        columns=["id", "datetime", "href", "geometry"],
        geometry="geometry",
        crs="EPSG:4326",
    )


def to_geojson_ll(gdf_src: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure CRS is known, then reproject to EPSG:4326 for GeoJSON writing."""
    if gdf_src.crs is None:
        # Your sample WKT is a South Pole stereographic; EPSG:3031 is a sensible default.
        gdf_src = gdf_src.set_crs(3031, allow_override=True)
    gdf_ll = gdf_src.to_crs(4326)
    # Repair geometries to avoid topology errors
    if _make_valid:
        gdf_ll["geometry"] = gdf_ll.geometry.apply(_make_valid)
    # buffer(0) as extra cleanup for self-intersections, only where needed
    gdf_ll["geometry"] = gdf_ll.geometry.apply(
        lambda geom: geom if geom is None or geom.is_valid else geom.buffer(0)
    )
    # Drop empties just in case
    gdf_ll = gdf_ll[~gdf_ll.geometry.is_empty & gdf_ll.geometry.notna()]
    return gdf_ll


def bbox_envelope(gdf_ll: gpd.GeoDataFrame):
    """Create a safe envelope box from total bounds (no topology ops)."""
    minx, miny, maxx, maxy = gdf_ll.total_bounds
    return box(minx, miny, maxx, maxy)


# --------------------
# Main
# --------------------
def main():
    print(
        f"Fetching USNIC shapefiles from {START_DATE} to {END_DATE} (prefix={USNIC_PREFIX})"
    )
    tmp_root = OUTPUT_DIR / "_tmp_usnic"
    tmp_root.mkdir(exist_ok=True)

    existing = load_existing_parquet(Path(PARQUET_PATH))
    existing_ids = set(existing["id"].astype(str)) if not existing.empty else set()

    new_rows = []

    for d in date_iter(START_DATE, END_DATE):
        item_id = f"USNIC_{d.strftime('%Y%m%d')}"
        if item_id in existing_ids:
            print(f"Skip {item_id}: already present")
            continue

        url = usnic_zip_url(d)
        try:
            print(f"→ Download {url}")
            zbytes = download_zip_bytes(url)
            shp_path = extract_first_shp(zbytes, tmp_root / item_id)
        except Exception as e:
            print(f"✖️  Failed to get shapefile for {d}: {e}")
            continue

        try:
            gdf_src = gpd.read_file(shp_path)
            if gdf_src.empty or gdf_src.geometry.isna().all():
                print(f"✖️  Shapefile has no valid geometry for {d}")
                continue

            # Reproject + repair to WGS84
            gdf_ll = to_geojson_ll(gdf_src)
            if gdf_ll.empty:
                print(f"✖️  All geometries became empty/invalid after repair for {d}")
                continue

            # Write actual GeoJSON (lon/lat, valid)
            out_geojson = OUTPUT_DIR / f"{item_id}.geojson"
            gdf_ll.to_file(out_geojson, driver="GeoJSON")

            # Safe envelope from bbox
            geom_env = bbox_envelope(gdf_ll)

            new_rows.append(
                {
                    "id": item_id,
                    "datetime": pd.to_datetime(d),
                    "href": f"{ASSET_BASE_URL_GEOJSON}/{item_id}.geojson",
                    "geometry": geom_env,
                }
            )

        except Exception as e:
            print(f"✖️  Error processing {d}: {e}")
            continue

    # Save parquet if anything new
    if new_rows:
        df_new = gpd.GeoDataFrame(new_rows, geometry="geometry", crs="EPSG:4326")

        if existing.empty:
            df_out = df_new
        else:
            # Align columns and order
            for c in existing.columns:
                if c not in df_new.columns:
                    df_new[c] = pd.NA
            df_new = df_new[existing.columns]
            df_out = pd.concat([existing, df_new], ignore_index=True)

        df_out.to_parquet(PARQUET_PATH)
        print(f"✅ Saved {len(new_rows)} new items to {PARQUET_PATH}")
    else:
        print("No new valid items to save.")

    # Copy parquet to output dir for convenience
    if Path(PARQUET_PATH).exists():
        print(f"Copying {PARQUET_PATH} to {OUTPUT_DIR}")
        shutil.copy(PARQUET_PATH, OUTPUT_DIR)


if __name__ == "__main__":
    main()
