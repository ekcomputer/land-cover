#!/usr/bin/env python
# coding: utf-8

"""
Summary
For biomass change within polygons representing outwards buffers from lakes.
Can optionally join in all attributes from original lake dataset.
With optimizations like pre-loading raster datasets, memory management, and using a coarsened raster
for large catchments: runs at 60 it/sec for a 39-band AGB raster.

Write three outputs:
1. All input variables for lakes that were matched to land cover
2. All input variables for all lakes
3. Selected input variables for all lakes

TODO:
* test multiple buffers
"""

import fcntl
import gc
import os
from multiprocessing import get_context
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall
import rasterio as rio
import seaborn as sns
from pyproj import CRS
from rasterio.mask import mask as rio_mask
from scipy.stats.mstats import theilslopes
from shapely import wkb as _wkb
from tqdm import tqdm

## Params
FLOAT_FORMAT_LONG = "%.3f"  # csv digits (to save storage space) for time series features
FLOAT_FORMAT_SHORT = "%.3f"  # csv digits for normalized features
SCALE_FACTOR = 100

## Function
def extractBufferZonalStats(
    poly: gpd.GeoDataFrame,
    buffer_lengths: list,
    raster_dataset,
    years: list[int],
    nodata: int = -999,
    all_touched: bool = False,
    join_index: str = "join_idx",
    large_lake_threshold: float = 30e6,
    raster_dataset_coarse=None,
):
    """
    Compute mean and standard deviation of continuous raster values in buffers.

    Rasterizes buffers once per band and computes mean and std dev of raster
    values in each buffer zone. Supports dual-raster system for processing
    large lakes with coarsened resolution.

    Parameters
    ----------
    poly : geopandas.GeoDataFrame
        Lake geometry with at minimum 'Area_m2', 'Perim_m2' columns.
    buffer_lengths : list
        Buffer distances from lake edge (meters). Include 0 for lake-only analysis.
    raster_dataset : rasterio.DatasetReader
        Open raster dataset with continuous values (e.g., biomass) as bands.
    years : list[int]
        Year corresponding to each raster band.
    nodata : int, optional
        Raster nodata value. Default is -999.
    all_touched : bool, optional
        If True, include all cells touched by buffer geometries. Default is False.
    join_index : str, optional
        Column name for join index. Default is 'join_idx'.
    large_lake_threshold : float, optional
        Lake area threshold (m²) for using coarse raster. Default is 30e6.
    raster_dataset_coarse : rasterio.DatasetReader, optional
        Coarsened raster for large lakes. Default is None.

    Returns
    -------
    pandas.DataFrame or None
        Mean and standard deviation per buffer per year. Returns None if geometry
        does not overlap raster.
    """
    assert len(buffer_lengths) == 1, "Only one buffer length supported so far"
    assert len(poly) >= 1, "poly must contain at least one geometry"
    lake_geom = poly.geometry.iloc[0] if len(poly) == 1 else poly.unary_union

    # Select appropriate raster based on lake size
    if raster_dataset_coarse is not None and lake_geom.area > large_lake_threshold:
        src = raster_dataset_coarse
    else:
        src = raster_dataset

    # Enhanced CRS validation for projected coordinates
    if not src.crs.is_projected:
        print(
            f"Warning: Raster CRS {src.crs} is not projected - area calculations may be inaccurate"
        )
    assert src.crs == poly.crs, f"CRS mismatch: raster {src.crs} vs poly {poly.crs}"

    # Sort buffers so last one is the outermost ROI for cropping
    buffer_lengths = list(buffer_lengths)
    order = np.argsort(buffer_lengths)
    buffer_lengths_sorted = [buffer_lengths[i] for i in order]

    # Handle zero-buffer case (lake-only analysis)
    buf_geoms = [
        lake_geom if length == 0 else lake_geom.buffer(length) for length in buffer_lengths_sorted
    ]

    # Crop to outermost buffer; everything outside is filled with nodata
    try:
        data, tr = rio_mask(src, [buf_geoms[-1]], crop=True, filled=True, nodata=nodata)
    except ValueError as e:
        if "do not overlap raster" in str(e).lower():
            # Return None to indicate no overlap
            return None
        raise

    n_bands, H, W = data.shape
    n_buffers = len(buf_geoms)

    # Rasterize buffer IDs once
    labels = rio.features.rasterize(
        [(g, i + 1) for i, g in enumerate(buf_geoms)],
        out_shape=(H, W),
        transform=tr,
        all_touched=all_touched,
        dtype="uint8",
    )

    # Compute mean and std per buffer per band
    means = np.empty((n_bands, n_buffers), dtype=np.float32)
    stds = np.empty((n_bands, n_buffers), dtype=np.float32)

    for buf_id in range(1, n_buffers + 1):
        mask = (labels == buf_id) & (data != nodata)
        if mask.any():
            buf_vals = data[mask].astype(np.float32)
            means[:, buf_id - 1] = np.nanmean(buf_vals.reshape(n_bands, -1), axis=1)
            stds[:, buf_id - 1] = np.nanstd(buf_vals.reshape(n_bands, -1), axis=1)
        else:
            means[:, buf_id - 1] = np.nan
            stds[:, buf_id - 1] = np.nan

    means /= SCALE_FACTOR
    stds /= SCALE_FACTOR
    if np.all(np.isnan(means)) and np.all(np.isnan(stds)):
        return None
    # Assemble dataframe
    # TODO: test multiple buffers
    df = pd.DataFrame(
        {
            "Year": np.repeat(years[:n_bands], n_buffers),
            "Buffer_m": np.tile(buffer_lengths, n_bands),
            "mean": means.flatten(),
            "std": stds.flatten(),
        }
    )
    df[join_index] = poly[join_index].iloc[0] if join_index in poly.columns else None
    df["Area_m2"] = poly["Area_m2"].iloc[0] if "Area_m2" in poly.columns else None
    df["Perim_m2"] = poly["Perim_m2"].iloc[0] if "Perim_m2" in poly.columns else None

    return df


# multiprocessing + POSIX file lock (fcntl) version with resume + append
# ---------- worker globals (set by initializer) ----------
_CSV_PATH = None
_RASTER_CRS = None
_BUFFER_LENGTHS = None
_PTH_LC_IN = None
_YEARS = None
_JOIN_INDEX = None
# New globals for enhanced functionality
_RASTER_DATASET = None
_RASTER_DATASET_COARSE = None
_LARGE_LAKE_THRESHOLD = None


def _create_nan_dataframe_biomass(
    poly: gpd.GeoDataFrame,
    buffer_lengths: list,
    years: list[int],
    join_index: str,
) -> pd.DataFrame:
    """Create NaN-filled DataFrame for geometries outside of raster coverage."""
    n_bands = len(years)
    n_buffers = len(buffer_lengths)
    n_rows = n_bands * n_buffers

    # Create DataFrame with mean and std columns filled with NaN
    df = pd.DataFrame(
        {
            "Year": np.repeat(years[:n_bands], n_buffers),
            "Buffer_m": np.tile(sorted(buffer_lengths), n_bands),
            "mean": np.full(n_rows, np.nan),
            "std": np.full(n_rows, np.nan),
        }
    )

    # Add metadata columns
    df[join_index] = poly[join_index].iloc[0] if join_index in poly else None
    df["Area_m2"] = poly["Area_m2"].iloc[0] if "Area_m2" in poly else None
    df["Perim_m2"] = poly["Perim_m2"].iloc[0] if "Perim_m2" in poly else None

    return df


def _init_worker(
    csv_path: str,
    raster_crs_wkt: str,
    buffer_lengths: list,
    pth_lc_in: str,
    years: list[int],
    join_index: str,
    pth_lc_in_coarse: str | None = None,
    large_lake_threshold: float = 30e6,
) -> None:
    """Initialize worker process with global raster and configuration state."""
    global _CSV_PATH, _RASTER_CRS, _BUFFER_LENGTHS, _PTH_LC_IN, _YEARS, _JOIN_INDEX
    global _RASTER_DATASET, _RASTER_DATASET_COARSE, _LARGE_LAKE_THRESHOLD

    _CSV_PATH = csv_path
    _RASTER_CRS = CRS.from_wkt(raster_crs_wkt)
    _BUFFER_LENGTHS = tuple(buffer_lengths)
    _PTH_LC_IN = pth_lc_in
    _YEARS = list(years)
    _JOIN_INDEX = join_index
    _LARGE_LAKE_THRESHOLD = large_lake_threshold

    # Open raster datasets once per worker process (more efficient)
    _RASTER_DATASET = rio.open(pth_lc_in)

    # Open coarse raster if provided for large lakes
    if pth_lc_in_coarse is not None:
        _RASTER_DATASET_COARSE = rio.open(pth_lc_in_coarse)
    else:
        _RASTER_DATASET_COARSE = None


def _gdf_from_payload(payload: dict, crs) -> "gpd.GeoDataFrame":
    """Reconstruct GeoDataFrame from picklable payload dictionary."""
    geom = _wkb.loads(payload["geometry_wkb"])
    join_index = _JOIN_INDEX
    # Note column order matters
    gdf = gpd.GeoDataFrame(
        {
            join_index: [payload[join_index]],
            "Area_m2": [payload["Area_m2"]],
            "Perim_m2": [payload["Perim_m2"]],
        },
        geometry=[geom],
        crs=crs,
    )
    return gdf


def _append_df_csv_locked(df: pd.DataFrame, csv_path: str) -> None:
    """Atomically append DataFrame to CSV with POSIX file locking."""
    with open(csv_path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        need_header = f.tell() == 0  # at end because "a+" opens and seeks to end
        df.to_csv(f, index=False, header=need_header, float_format=FLOAT_FORMAT_LONG)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def _worker(payload: dict) -> tuple:
    """Process single lake: extract buffer zonal stats and write to CSV."""
    join_index = _JOIN_INDEX

    try:
        poly = _gdf_from_payload(payload, _RASTER_CRS)
        df = extractBufferZonalStats(
            poly,
            _BUFFER_LENGTHS,
            _RASTER_DATASET,
            years=_YEARS,
            join_index=join_index,
            large_lake_threshold=_LARGE_LAKE_THRESHOLD,
            raster_dataset_coarse=_RASTER_DATASET_COARSE,
        )

        # Always write to CSV, even if df is None (will be NaN-filled)
        return_value = 1
        if df is None:
            df = _create_nan_dataframe_biomass(poly, _BUFFER_LENGTHS, _YEARS, join_index)
            return_value = 0
        _append_df_csv_locked(df, _CSV_PATH)
        del df
        gc.collect()
        return payload[join_index], return_value
    except Exception as e:
        return payload[join_index], f"ERROR: {e}"


def _prime_header(csv_path: Path, years_cols: tuple) -> None:
    """Initialize CSV file with column headers if missing or empty."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        cols = list(years_cols)
        empty = pd.DataFrame(columns=cols)
        _append_df_csv_locked(empty, str(csv_path))


def extractTimeSeriesForLakes(
    pth_shp_in: str | Path,
    buffer_lengths: list,
    csv_out_pth: str | Path,
    pth_lc_in: str | Path,
    years: list[int],
    envelope_pth: str | Path | None = None,
    join_index: str | None = None,
    n_workers: int = 8,
    pth_lc_in_coarse: str | Path | None = None,
    large_lake_threshold: float = 30e6,
) -> None:
    """
    Extract continuous raster time series (e.g., biomass) for buffers around lakes.

    Loads lake geometries, computes concentric buffers, and extracts zonal
    statistics (mean, std) from raster data. Uses multiprocessing with POSIX
    file locking for CSV writes and supports resumption from incomplete runs.

    Parameters
    ----------
    pth_shp_in : str | Path
        Path to input lake polygon shapefile or GeoPackage.
    buffer_lengths : list
        Buffer distances from lake edge (meters). Include 0 for lake-only.
    csv_out_pth : str | Path
        Output CSV path for continuous raster time series.
    pth_lc_in : str | Path
        Path to high-resolution raster with continuous values (e.g., biomass).
    years : list[int]
        Year per raster band.
    envelope_pth : str | Path, optional
        Path to polygon geometry for spatial filtering. Default is None.
    join_index : str, optional
        Column name for unique lake identifier. Auto-created if None.
    n_workers : int, optional
        Number of parallel processes. Use 1 for serial debugging. Default is 8.
    pth_lc_in_coarse : str | Path, optional
        Path to coarsened raster for large lakes. Default is None.
    large_lake_threshold : float, optional
        Lake area threshold (m2) for using coarse raster. Default is 30e6.

    Raises
    ------
    AssertionError
        If envelope CRS does not match polygon CRS.

    Notes
    -----
    Automatically resumes from incomplete runs by checking existing CSV.
    Returns silently if all polygons already processed.
    """
    print("Paths:")
    print(csv_out_pth)
    print(f"\nLarge lake threshold: {large_lake_threshold/1e6:.1f} km²")

    # Load polygons and project to raster CRS
    polys = gpd.read_file(pth_shp_in)
    with rio.open(pth_lc_in) as src:
        raster_crs = src.crs
    if polys.crs != raster_crs:
        print(f"reprojecting polygons to {raster_crs}")
        polys = polys.to_crs(raster_crs)

    # Attributes needed downstream
    if join_index is None:
        join_index = "join_idx"
        polys[join_index] = polys.index
    polys["Area_m2"] = polys.area
    polys["Perim_m2"] = polys.length

    # Optional envelope filter
    if envelope_pth is not None:
        envelope = gpd.read_file(envelope_pth)
        assert envelope.crs == polys.crs, "CRS mismatch: envelope vs. polygons"
        polys = polys[polys.geometry.intersects(envelope.union_all(), align=False)]
        print(f"Filtered polygons by envelope: {len(polys)} features")

    out_path = Path(csv_out_pth)

    # Resume support: read already processed join_index
    done_idx = set()  # e.g. {}
    try:
        if out_path.exists() and out_path.stat().st_size > 0:
            done_col = pd.read_csv(out_path, usecols=[join_index])[join_index]
            done_idx = set(done_col.dropna().unique().tolist())
            print(f"Resuming: found {len(done_idx)} completed lakes in existing CSV.")
        else:
            _prime_header(
                out_path,
                ("Year", "Buffer_m", "agb_mean", "agb_std", join_index, "Area_m2", "Perim_m2"),
            )
    except Exception as e:
        print(f"Resume read failed ({e}); proceeding without resume filtering.")

    pending = polys[~polys[join_index].isin(done_idx)]
    total = len(pending)
    if total == 0:
        print("Nothing to do. All polygons already processed.")
        return

    # Prepare small, picklable payloads
    payloads = []
    for _, row in pending.iterrows():
        payloads.append(
            {
                join_index: row[join_index],
                "Area_m2": float(row["Area_m2"]),
                "Perim_m2": float(row["Perim_m2"]),
                "geometry_wkb": row.geometry.wkb,
            }
        )

    # Multiprocessing (fork on Unix/macOS)
    ctx = get_context("fork")
    chunksize = max(1, len(payloads) // (n_workers * 8) or 1)

    initargs = (
        str(out_path),
        raster_crs.to_wkt(),
        list(buffer_lengths),
        str(pth_lc_in),
        list(years),
        join_index,
        str(pth_lc_in_coarse) if pth_lc_in_coarse else None,
        large_lake_threshold,
    )

    # Serial vs. multiprocessing execution
    if n_workers == 1:
        _init_worker(
            str(out_path),
            raster_crs.to_wkt(),
            list(buffer_lengths),
            str(pth_lc_in),
            list(years),
            join_index,
            str(pth_lc_in_coarse) if pth_lc_in_coarse else None,
            large_lake_threshold,
        )
        iterator = map(_worker, payloads)
        iterator = tqdm(iterator, total=len(payloads), desc="Lakes")
        completed = 0
        errors = 0
        empty = 0
        for join_idx, status in iterator:
            if status == 1:
                completed += 1
            elif status == 0:
                empty += 1
            else:
                errors += 1
                print(f"[Join_idx={join_idx}] {status}")
    else:
        with ctx.Pool(processes=n_workers, initializer=_init_worker, initargs=initargs) as pool:
            iterator = pool.imap_unordered(_worker, payloads, chunksize=chunksize)
            iterator = tqdm(iterator, total=len(payloads), desc="Lakes")
            completed = 0
            errors = 0
            empty = 0
            for join_idx, status in iterator:
                if status == 1:
                    completed += 1
                elif status == 0:
                    # no lc data for lake
                    empty += 1
                else:
                    errors += 1
                    print(f"[Join_idx={join_idx}] {status}")

    print(f"done. wrote {completed} lakes, {empty} empty results, {errors} errors.")
