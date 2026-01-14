#!/usr/bin/env python
# coding: utf-8

"""
Summary
For calculating land cover and its change within polygons representing outwards buffers from lakes.
Can optionally join in all attributes from original lake dataset.
With optimizations like pre-loading raster datasets, memory management, and using a coarsened raster
for large catchments: runs at 60 it/sec for a one-band CEC raster.

Write three outputs:
1. All input variables for lakes that were matched to land cover
2. All input variables for all lakes
3. Selected input variables for all lakes

TODO:
* Check that water normalization only refers to largest/central lake within buffer.
* IMPORTANT: Find a way to automatically include Lat/Long and any note columns in final spreadsheet (perhaps join in?) Right now, I'm just using a quick fix in Excel.
* Fix functionality so that multiple buffers can be run at once.
* Remove "Lake_name" field

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
FLOAT_FORMAT_LONG = "%.5f" # csv digits (to save storage space) for time series features
FLOAT_FORMAT_SHORT = "%.3f" # csv digits for normalized features


## Function
def extractBufferZonalHist(
    poly: gpd.GeoDataFrame,
    buffer_lengths: list,
    raster_dataset,
    classes: list[str],
    years: list[int],
    nodata: int = 255,
    all_touched: bool = False,
    join_index: str = "join_idx",
    large_lake_threshold: float = 30e6,
    raster_dataset_coarse=None,
):
    """
    Compute zonal histogram of land cover classes for concentric lake buffers.

    Rasterizes buffers once per band for efficiency and computes area-weighted
    counts of each land cover class. Supports dual-raster system for processing
    large lakes with coarsened resolution.

    Parameters
    ----------
    poly : geopandas.GeoDataFrame
        Lake geometry with at minimum 'Area_m2', 'Perim_m2' columns.
    buffer_lengths : list
        Buffer distances from lake edge (meters). Include 0 for lake-only analysis.
    raster_dataset : rasterio.DatasetReader
        Open raster dataset with land cover classes as bands.
    classes : list[str]
        Ordered land cover class names (length must match raster value range).
    years : list[int]
        Year corresponding to each raster band.
    nodata : int, optional
        Raster nodata value. Default is 255.
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
        Area (hectares) per class per buffer per year. Returns None if geometry
        does not overlap raster.
    """
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
            # Return NaN-filled DataFrame instead of None for consistent output
            return _create_nan_dataframe(poly, buffer_lengths, classes, years, join_index)
        raise

    n_bands, H, W = data.shape
    n_buffers = len(buf_geoms)
    nclasses = len(classes)

    # --- multi-buffer path: rasterize buffer IDs once, then bin per band ---
    labels = rio.features.rasterize(
        [(g, i + 1) for i, g in enumerate(buf_geoms)],
        out_shape=(H, W),
        transform=tr,
        all_touched=all_touched,
        dtype="uint8",
    )
    counts = np.empty((n_bands, n_buffers, nclasses), dtype=np.float32)
    valid_vals = (data >= 1) & (data <= nclasses)

    for bi in range(n_bands):
        vals = data[bi]
        m = (labels > 0) & valid_vals[bi] & (vals != nodata)
        if m.any():
            keys = (labels[m] - 1) * nclasses + (vals[m] - 1)  # (buffer_id, class_id) -> 1D key
            bc = np.bincount(keys, minlength=n_buffers * nclasses).reshape(n_buffers, nclasses)
        else:
            # Return None when no valid data
            return None
        counts[bi] = bc

    # Scale to area (hectares)
    pix_area_ha = abs(src.res[0] * src.res[1]) / 10000.0
    areas = counts.reshape(n_bands * n_buffers, nclasses) * pix_area_ha

    # --- assemble dataframe efficiently ---
    df = pd.DataFrame(areas, columns=classes)
    df["Year"] = np.repeat(years[:n_bands], n_buffers)
    df["Buffer_m"] = np.tile(buffer_lengths_sorted, n_bands)
    df["Lake_name"] = poly.index[0]
    df[join_index] = poly[join_index].iloc[0] if join_index in poly else None
    df["Area_m2"] = poly["Area_m2"].iloc[0] if "Area_m2" in poly else None
    df["Perim_m2"] = poly["Perim_m2"].iloc[0] if "Perim_m2" in poly else None
    return df


# multiprocessing + POSIX file lock (fcntl) version with resume + append
# ---------- worker globals (set by initializer) ----------
_CSV_PATH = None
_RASTER_CRS = None
_BUFFER_LENGTHS = None
_PTH_LC_IN = None
_CLASSES = None
_YEARS = None
_JOIN_INDEX = None
# New globals for enhanced functionality
_RASTER_DATASET = None
_RASTER_DATASET_COARSE = None
_LARGE_LAKE_THRESHOLD = None


def _create_nan_dataframe(
    poly: gpd.GeoDataFrame,
    buffer_lengths: list,
    classes: list[str],
    years: list[int],
    join_index: str,
) -> pd.DataFrame:
    """Create NaN-filled DataFrame for geometries outside of raster coverage."""
    n_bands = len(years)
    n_buffers = len(buffer_lengths)
    n_rows = n_bands * n_buffers

    # Create DataFrame filled with NaN
    df = pd.DataFrame(np.full((n_rows, len(classes)), np.nan), columns=classes)

    # Add metadata columns
    df["Year"] = np.repeat(years[:n_bands], n_buffers)
    df["Buffer_m"] = np.tile(sorted(buffer_lengths), n_buffers)
    df["Lake_name"] = poly.index[0]
    df[join_index] = poly[join_index].iloc[0] if join_index in poly else None
    df["Area_m2"] = poly["Area_m2"].iloc[0] if "Area_m2" in poly else None
    df["Perim_m2"] = poly["Perim_m2"].iloc[0] if "Perim_m2" in poly else None

    return df


def _init_worker(
    csv_path: str,
    raster_crs_wkt: str,
    buffer_lengths: list,
    pth_lc_in: str,
    classes: list[str],
    years: list[int],
    join_index: str,
    pth_lc_in_coarse: str | None = None,
    large_lake_threshold: float = 30e6,
) -> None:
    """Initialize worker process with global raster and configuration state."""
    global _CSV_PATH, _RASTER_CRS, _BUFFER_LENGTHS, _PTH_LC_IN, _CLASSES, _YEARS, _JOIN_INDEX
    global _RASTER_DATASET, _RASTER_DATASET_COARSE, _LARGE_LAKE_THRESHOLD

    _CSV_PATH = csv_path
    _RASTER_CRS = CRS.from_wkt(raster_crs_wkt)
    _BUFFER_LENGTHS = tuple(buffer_lengths)
    _PTH_LC_IN = pth_lc_in
    _CLASSES = list(classes)
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
            "Lake_name": [payload["Lake_name"]],
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
        df = extractBufferZonalHist(
            poly,
            _BUFFER_LENGTHS,
            _RASTER_DATASET,
            classes=_CLASSES,
            years=_YEARS,
            join_index=join_index,
            large_lake_threshold=_LARGE_LAKE_THRESHOLD,
            raster_dataset_coarse=_RASTER_DATASET_COARSE,
        )

        # Always write to CSV, even if df is None (will be NaN-filled)
        return_value = 1
        if df is None:
            df = _create_nan_dataframe(poly, _BUFFER_LENGTHS, _CLASSES, _YEARS, join_index)
            return_value = 0
        _append_df_csv_locked(df, _CSV_PATH)
        del df
        gc.collect()
        return payload[join_index], return_value
    except Exception as e:
        return payload[join_index], f"ERROR: {e}"


def _prime_header(csv_path: Path, classes: list[str], years_cols: tuple) -> None:
    """Initialize CSV file with column headers if missing or empty."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        cols = list(classes) + list(years_cols)
        empty = pd.DataFrame(columns=cols)
        _append_df_csv_locked(empty, str(csv_path))


def extractTimeSeriesForLakes(
    pth_shp_in: str | Path,
    buffer_lengths: list,
    csv_out_pth: str | Path,
    pth_lc_in: str | Path,
    use_simplified_classes: bool,
    classes: list[str],
    years: list[int],
    envelope_pth: str | Path | None = None,
    join_index: str | None = None,
    n_workers: int = 8,
    pth_lc_in_coarse: str | Path | None = None,
    large_lake_threshold: float = 30e6,
) -> None:
    """
    Extract land cover time series for buffers around lake/catchment polygons.

    Loads lake geometries, computes concentric buffers, and extracts zonal
    statistics from raster data. Uses multiprocessing with POSIX file locking
    for CSV writes and supports resumption from incomplete runs.

    Parameters
    ----------
    pth_shp_in : str | Path
        Path to input lake polygon shapefile or GeoPackage.
    buffer_lengths : list
        Buffer distances from lake edge (meters). Include 0 for lake-only.
    csv_out_pth : str | Path
        Output CSV path for land cover time series.
    pth_lc_in : str | Path
        Path to high-resolution land cover raster.
    use_simplified_classes : bool
        If True, use simplified class set (unused in current version).
    classes : list[str]
        Land cover class names matching raster value ordering.
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
    print(f"\nUse simplified classes: {use_simplified_classes}")
    print(f"Large lake threshold: {large_lake_threshold/1e6:.1f} km²")

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
    polys["Lake_name"] = polys.index

    # Optional envelope filter
    if envelope_pth is not None:
        envelope = gpd.read_file(envelope_pth)
        assert envelope.crs == polys.crs, "CRS mismatch: envelope vs. polygons"
        polys = polys[polys.geometry.intersects(envelope.union_all(), align=False)]
        print(f"Filtered polygons by envelope: {len(polys)} features")

    out_path = Path(csv_out_pth)

    # Resume support: read already processed join_index
    done_idx = set() # e.g. {}
    try:
        if out_path.exists() and out_path.stat().st_size > 0:
            done_col = pd.read_csv(out_path, usecols=[join_index])[join_index]
            done_idx = set(done_col.dropna().unique().tolist())
            print(f"Resuming: found {len(done_idx)} completed lakes in existing CSV.")
        else:
            _prime_header(
                out_path,
                classes,
                ("Year", "Buffer_m", "Lake_name", join_index, "Area_m2", "Perim_m2"),
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
                "Lake_name": int(row["Lake_name"]),
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
        list(classes),
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
            list(classes),
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


def normalizeTimeSeries(
    out_pth: str | Path,
    out_norm_pth: str,
    classes_wet: list[str],
    classes_dry: list[str],
    classes_dry_rn: list[str],
    use_simplified_classes: bool = False,
    wetland_class: str = "Shallows/littoral",
) -> None:
    """
    Normalize land cover areas to percentages of water and land.

    Computes water and land class percentages separately and creates derived
    metrics like inundation percentage and vegetation group aggregates.

    Parameters
    ----------
    out_pth : str
        Path to input CSV with raw land cover areas (hectares).
    out_norm_pth : str
        Path to output CSV with normalized percentages.
    classes_wet : list[str]
        Water class names (typically 'Water', 'Wetland').
    classes_dry : list[str]
        Terrestrial land cover class names.
    classes_dry_rn : list[str]
        Renamed terrestrial class names (spaces/slashes removed).
    use_simplified_classes : bool, optional
        Must be False (full 14 classes required). Default is False.
    wetland_class : str, optional
        Name of littoral/shallow water class. Default is 'Shallows/littoral'.

    Raises
    ------
    AssertionError
        If use_simplified_classes is True.
    """
    assert use_simplified_classes==False, "Must run on full 14 (not-simplified) classes."

    ## Load
    print('Normalizing land cover...')
    df = pd.read_csv(out_pth)

    ## find littoral percent of water areas (TODO: ensure it only comes from largest/central water body within buffer)
    df['Littorals_pct'] = df[wetland_class] / df.loc[:,classes_wet].sum(axis=1)*100

    ## Find wetland percent, like michela does, by taking: (L+B+F)/(L+B+F+W)*100
    df['Littoral_wetland_pct'] = (df[wetland_class] + df.Bog + df.Fen)/ (df.loc[:,classes_wet].sum(axis=1) + df.Bog + df.Fen)*100

    ## Find class percent of dry areas
    # normDry = lambda var: df[var] / df.loc[:,classes_dry].sum(axis=1)*100 # just keeping lambda function for practice
    for var in classes_dry:
        df[var + '_pct'] = df[var] / df.loc[:,classes_dry].sum(axis=1)*100

    ## Rename cols to remove spaces
    mapper = {var: var.replace(' ','_').replace('/','_') for var in df.columns}
    df.rename(columns=mapper, inplace=True) # rename cols

    ## Lump into groups
    df['Total_inun'] = df.Water + df.Shallows_littoral
    df['Trees'] = df.Evergreen_Forest + df.Deciduous_Forest + df.Mixed_Forest + df.Woodland
    df['Shrubs'] = df.Low_Shrub + df.Tall_Shrub + df.Open_Shrubs
    df['Wetlands'] = df.Fen + df.Bog # + df.Shallows_littoral
    df['Graminoid'] = df.Herbaceous + df.Tussock_Tundra
    df['Sparse'] = df.Barren + df.Sparsely_Vegetated

    ## Find class percent of lumped dry areas
    lumped_classes = ['Trees', 'Shrubs', 'Wetlands', 'Graminoid', 'Sparse']
    for var in lumped_classes:
        df[var + '_pct'] = df[var] / df.loc[:,classes_dry_rn].sum(axis=1)*100

    ## Write out
    df.to_csv(out_norm_pth, float_format=FLOAT_FORMAT_SHORT)
    print(f'Wrote normalized output table: {out_norm_pth}')


def normalizeTimeSeries_above_boreal(
    out_pth: str | Path,
    out_norm_pth: str | Path,
    classes_wet: list[str],
    classes_dry: list[str],
    wetland_class: str = "Wetland",
    index_class: str = "Lake_id_glakes",
) -> None:
    """
    Normalize ABoVE land cover classes to percentages (Hu et al. 2025).

    Computes water and land class percentages and creates aggregated vegetation
    groups (Trees, Sparse, Total_inun). Supports both CSV and Parquet output.

    Parameters
    ----------
    out_pth : str
        Path to input CSV/Parquet with raw land cover areas (hectares).
    out_norm_pth : str
        Path to output CSV/Parquet with normalized percentages.
    classes_wet : list[str]
        Water class names.
    classes_dry : list[str]
        Terrestrial land cover class names.
    wetland_class : str, optional
        Name of wetland class. Default is 'Wetland'.
    index_class : str, optional
        Column name for lake unique identifier. Default is 'Lake_id_glakes'.
    """
    classes = np.unique(classes_dry + classes_wet).tolist()
    ## Load
    print("Normalizing land cover...")
    usecols = classes + ["Year", index_class, "Area_m2", "Perim_m2"]
    df = pd.read_csv(out_pth, usecols=usecols) #, nrows=1000) # all but Lake_name and Buffer_m

    ## Find wetland percent, like michela does, by taking: (L+B+F)/(L+B+F+W)*100 (TODO: ensure it only comes from largest/central water body within buffer)
    df["Littoral_wetland_pct"] = df[wetland_class] / df.loc[:, classes_wet].sum(axis=1) * 100

    ## Find class percent of all vars, with denominator of total non-water buffer
    class_sum = df.loc[:, classes].sum(axis=1)
    for var in classes_dry:
        df[var + "_pct"] = df[var] / class_sum * 100

    ## Rename cols to remove spaces
    mapper = {var: var.replace(" ", "_").replace("/", "_") for var in df.columns}
    df.rename(columns=mapper, inplace=True)  # rename cols

    ## Lump into groups
    df["Total_inun"] = df.Water + df.Wetland
    df["Trees"] = df.Evergreen_Forest + df.Deciduous_Forest + df.Mixed_Forest
    df["Sparse"] = df.Bare_Sparsely_vegetated + df.Ice_Snow

    ## Find class percent of lumped areas, with denominator of total non-water buffer
    lumped_classes = ["Trees", "Sparse", "Total_inun"]
    for var in lumped_classes:
        df[var + "_pct"] = df[var] / class_sum * 100

    ## Write out
    # write with FLOAT_FORMAT_SHORT decimal places
    if out_norm_pth.endswith('.parquet'):
        df.to_parquet(out_norm_pth, index=False)
    else:
        df.to_csv(out_norm_pth, float_format=FLOAT_FORMAT_SHORT)
    print(f"Wrote normalized output table: {out_norm_pth}")


def plotTimeSeries(
    buffer_lengths: list,
    out_norm_pth: str | Path,
    plot_dir: str,
    index_col: str | None = None,
    index: list | None = None,
    combined: bool = False,
) -> None:
    """
    Create faceted time-series plots of land cover for each lake.

    Generates FacetGrid plots with separate panels per land cover class and
    hue for different buffer sizes. Supports single-lake and aggregated views.

    Parameters
    ----------
    buffer_lengths : list
        Buffer distances used (typically includes smallest buffer).
    out_norm_pth : str | Path
        Path to normalized land cover CSV/Parquet.
    plot_dir : str
        Output directory for saving plots.
    index_col : str, optional
        Lake identifier column name. Default is None.
    index : list, optional
        List of lake identifiers to plot. If None, plots all. Default is None.
    combined : bool, optional
        If True, creates single aggregated plot with area-weighted means.
        Default is False.

    Notes
    -----
    Plots are saved as PNG files with format 'ts-facets-{index_col}-{lake}.png'.
    Requires columns: Year, Buffer_m, Area_m2, and land cover class columns.
    """

    ## vars
    buf_len = buffer_lengths[0] # use the smallest (90 m) buffer for plotting

    ## Load
    print('Plotting land cover...')
    if out_norm_pth.endswith(".parquet"):
        df = pd.read_parquet(out_norm_pth)
    elif out_norm_pth.endswith(".csv"):
        df = pd.read_csv(out_norm_pth, index_col=0)
    else:
        raise ValueError("Unsupported file format for out_norm_pth")

    if index_col is not None:
        assert index_col in df.columns, f"Column '{index_col}' not found in DataFrame. Available columns: {df.columns.tolist()}"
        assert set(index).issubset(set(df[index_col].unique())), f"Some values in index are not found in {index_col} column"

    if "Buffer_m" not in df.columns:
        df["Buffer_m"] = buffer_lengths[0]  # assume single buffer if missing

    if index_col is not None and index is not None:
        dfg = df[df[index_col].isin(index)].groupby(index_col)
    else:
        dfg = df.groupby(index_col)

    if combined:
        # Group by Year and calculate area-weighted means across all lakes
        df_combined = (
            df.groupby('Year')
            .apply(lambda group: pd.Series({
                col: np.average(group[col], weights=group['Area_m2']) 
                if col not in ['Year', 'Buffer_m', 'Lake_name', index_col, 'Area_m2', 'Perim_m2'] and group[col].notna().any()
                else group[col].iloc[0] if col in ['Year', 'Buffer_m'] 
                else group[col].sum() if col in ['Area_m2', 'Perim_m2']
                else 'Aggregated'
                for col in df.columns
            }))
            .reset_index(drop=True)
        )

        # Set aggregation identifier
        df_combined['agg_group'] = 'Aggregated'

        # Create single-group iterator for plotting
        dfg = df_combined.groupby("agg_group")
    else:
        pass

    ## Plot for all lakes!
    ## Note: OG slices were 1-16, 21-36, 42-47

    plot_types = { # dict for plotting params
        'Ha': {'slice': slice(1,16), 'col_wrap': 4, 'subdir':'time-series-by-lake'},
        'Normalized area (%)': {'slice': slice(21,36), 'col_wrap': 4, 'subdir':'time-series-by-lake-norm'},
        'Norm. land group area (%)': {'slice': slice(13,22), 'col_wrap': 2, 'subdir':'time-series-by-lake-grouped-norm'} # NOTE: columns 34-40 are non-normalized groups
        }
    for type in ['Norm. land group area (%)']: #plot_types: # HERE: Switch to modify which type of plot to use or use full dict
        os.makedirs(os.path.join(
                    plot_dir,
                    plot_types[type]["subdir"],
                ), exist_ok=True)
        for lake in dfg.groups:
            group = dfg.get_group(lake)
            dfl = pd.melt(group, id_vars=['Year', 'Buffer_m'], value_vars=df.columns[plot_types[type]['slice']], var_name = 'Class', value_name=type)# data frame long format
            g = sns.FacetGrid(dfl, col="Class", hue="Buffer_m", col_wrap=plot_types[type]['col_wrap'])
            g.map(sns.lineplot, 'Year', type)
            g.add_legend(title="Buffer (m)")
            g.fig.subplots_adjust(top=0.93) # adjust the Figure to add super title
            g.fig.suptitle(f'{lake} ({group.Area_m2.mode()[0]/1e6:.2f} $km^2$)') # used mode, but mean, first, med would give same answer
            # plt.show()
            plt.close()
            g.savefig(
                os.path.join(
                    plot_dir,
                    plot_types[type]["subdir"],
                    f"ts-facets-{index_col}-{lake}.png",
                ).replace(" ", "-")
            )
            print(lake)
    print('Done plotting.')


def extractTimeSeriesFeatures(
    out_norm_pth: str | Path,
    years: list[int],
    classes_dry_rn: list[str],
    pth_shp_in: str,
    ds_specific_vars: list[str],
    csv_out_time_series_features_pth: str,
    important_vars: list[str],
    csv_out_time_series_features_core_pth: str,
    shp_out_time_series_features_core_pth: str,
    join_index: str = "join_idx",
) -> None:
    """
    Extract time series summary metrics from normalized land cover data.

    Computes statistics per lake including median values, temporal trends,
    dominant vegetation, and shape metrics (SDF, perimeter/area ratio).
    Outputs full feature set, core subset, and GeoPackage with geometries.

    Parameters
    ----------
    out_norm_pth : str | Path
        Path to normalized land cover CSV with all years/buffers.
    years : list[int]
        Years corresponding to land cover time series.
    classes_dry_rn : list[str]
        Terrestrial land cover class names (space/slash-free).
    pth_shp_in : str
        Path to input lake polygon shapefile for geometry/attributes.
    ds_specific_vars : list[str]
        Dataset-specific columns to include in output (metadata).
    csv_out_time_series_features_pth : str
        Output CSV/Parquet with all computed features.
    important_vars : list[str]
        Subset of features for core output.
    csv_out_time_series_features_core_pth : str
        Output CSV/Parquet with important_vars subset.
    shp_out_time_series_features_core_pth : str
        Output GeoPackage with geometries and important_vars.
    join_index : str, optional
        Lake identifier column name. Default is 'join_idx'.

    Notes
    -----
    Computes Mann-Kendall trends and Theil-Sen slopes for temporal changes.
    Uses 2014 as reference year (assumes specific year ordering in input).
    """

    ## Load
    print('Calculating time series features...')
    df = pd.read_csv(out_norm_pth, index_col=0)

    ## Filter by buffer length
    df.query('Buffer_m == @buffer_lengths[0]', inplace=True)

    ## Group by lake
    dfg = df.groupby('Lake_name')

    ## Take last (year 2014) value as initial features for output df
    stats_last = dfg.last()

    ## Remove unnecessary columns
    stats_last.drop('Year', axis=1, inplace=True)

    ## Compute median vals
    meta_columns = ['Buffer_m', join_index, 'Area_m2', 'Perim_m2'] # metadata
    stats_median = dfg.median().drop(columns=meta_columns)

    ## Rename stats vars for 2014
    mapper = {var: (var + '_2014') for var in stats_last.drop(meta_columns, axis=1).columns}
    stats_last.rename(columns=mapper, inplace=True) # rename cols

    ## Rename stats vars for median
    mapper = {var: (var + '_med') for var in stats_median.columns}
    stats_median.rename(columns=mapper, inplace=True) # rename cols

    ## Insert median stats into stats df
    stats = pd.concat((stats_last, stats_median), axis='columns')

    ## Reorder to put meta vars first
    [stats.insert(0, col, stats.pop(col)) for col in meta_columns[-1::-1]] # re-order cols

    ## Compute more features
    # dropna?
    # 1 Dynamism, 1.5 RSD of water, 2 trend in water and shrubs, 3 trend in dom veg

    grouped_classes = ['Trees', 'Shrubs', 'Wetlands', 'Graminoid', 'Sparse']

    stats['Total_inun_RSD'] = dfg.Total_inun.std()/dfg.Total_inun.mean()
    stats['Total_inun_dyn_pct'] = (dfg.Total_inun.max() - dfg.Total_inun.min()) / dfg.Total_inun.max() * 100
    stats['Hi_water_yr'] = dfg.Total_inun.apply(lambda group: years[np.argmax(group)]) # Cool! Use GroupBy.apply to apply a lambda function over all groups!
    stats['Lo_water_yr'] = dfg.Total_inun.apply(lambda group: years[np.argmin(group)]) # Cool! Use GroupBy.apply to apply a lambda function over all groups!
    stats['Dominant_veg_2014'] = dfg.last().loc[:, classes_dry_rn].apply(lambda lake: classes_dry_rn[np.argmax(lake)], axis='columns')
    stats['Dominant_veg_group_2014'] = dfg.last().loc[:, grouped_classes].apply(lambda lake: grouped_classes[np.argmax(lake)], axis='columns')
    stats['SDF'] = stats.Perim_m2 / (2 * np.sqrt(np.pi * stats.Area_m2))
    stats['Perim_area_ratio'] = stats.Perim_m2 / stats.Area_m2
    # dfg['Year', 'Total_inun'].apply(lambda group: theilslopes(group.Year, group.Total_inun))
    for lcClass in ['Total_inun'] + grouped_classes:
        stats[lcClass + '_change'] = dfg[lcClass].apply(lambda group: theilslopes(group)[0]) # Using method from Kuhn et and Butman 2021, PNAS
        stats[lcClass + '_trend'] = dfg[lcClass].apply(lambda group: pymannkendall.original_test(group)[0])

    ## Join in lat/long and location from og-mod csv: load files
    gdf_og_data = gpd.read_file(pth_shp_in)
    geoms = gdf_og_data.geometry
    crs = gdf_og_data.crs

    ## rm unnecessary cols
    joined_cols = ds_specific_vars
    gdf_og_data = gdf_og_data[joined_cols]

    ## join and rename index
    stats = stats.merge(gdf_og_data, left_on='Lake_name', right_index=True, how='inner', validate='1:1') # TODO: make more flexible for when I actually have lake name
    stats.index.rename('Lake', inplace=True) # Note the 'lake' corresponds to index in pth_shp_in (arbitrary index after concatennating PLD and WBD lakes)

    ## Reorder to put meta vars first
    [stats.insert(0, col, stats.pop(col)) for col in joined_cols[-1::-1]] # re-order cols # TODO import load.sortColumns

    ## Write out
    if csv_out_time_series_features_pth.endswith(".parquet"):
        stats.to_parquet(csv_out_time_series_features_pth, index=False)
    else:
        stats.to_csv(csv_out_time_series_features_pth, float_format=FLOAT_FORMAT_LONG)
    print(f'Wrote time series output table: {csv_out_time_series_features_pth}')

    ## Save and write out most important stats
    if csv_out_time_series_features_core_pth.endswith(".parquet"):
        stats.loc[:, ds_specific_vars + important_vars].to_parquet(
            csv_out_time_series_features_core_pth, index=False
        )
    else:
        stats.loc[:, ds_specific_vars + important_vars].to_csv(
            csv_out_time_series_features_core_pth, float_format=FLOAT_FORMAT_LONG
        )
    print(f'Wrote time series output table (greatest hits): {csv_out_time_series_features_core_pth}')

    ## Save shapefile
    gdf_stats = gpd.GeoDataFrame(stats, geometry=geoms, crs=crs)
    gdf_stats.loc[:, ds_specific_vars + important_vars + ['geometry']].to_file(shp_out_time_series_features_core_pth)

    pass


def extractTimeSeriesFeatures_above_boreal(
    out_norm_pth: str | Path,
    years: list[int],
    classes_dry_rn: list[str],
    pth_shp_in: str | Path,
    ds_specific_vars: list[str],
    csv_out_time_series_features_pth: str,
    important_vars: list[str],
    csv_out_time_series_features_core_pth: str,
    csv_out_time_series_features_short_pth: str,
    join_index: str = "Lake_id_glakes",
    grouped_classes: list[str] = ["Trees", "Shrub", "Wetland", "Herb", "Sparse"],
) -> None:
    """
    Extract time series metrics from ABoVE land cover data (Hu et al. 2025).

    Computes lake-level statistics including dominant vegetation, inundation
    dynamics, and renormalized class transitions. Outputs three variants: full
    features, core subset, and short version before joining with spatial data.

    Parameters
    ----------
    out_norm_pth : str
        Path to normalized ABoVE land cover CSV with all years/buffers.
    years : list[int]
        Years corresponding to time series (e.g., 1986-2014).
    classes_dry_rn : list[str]
        All land cover class names (space/slash-free).
    pth_shp_in : str | Path
        Path to input lake polygon shapefile for attributes.
    ds_specific_vars : list[str]
        Dataset-specific metadata columns to include.
    csv_out_time_series_features_pth : str
        Output CSV/Parquet with full features joined to spatial data.
    important_vars : list[str]
        Subset of features for core output.
    csv_out_time_series_features_core_pth : str
        Output CSV/Parquet with important_vars subset.
    csv_out_time_series_features_short_pth : str
        Output CSV with features before spatial join (diagnostic).
    join_index : str, optional
        Lake identifier column. Default is 'Lake_id_glakes'.
    grouped_classes : list[str], optional
        Aggregated vegetation group names. Default is standard ABoVE grouping.

    Notes
    -----
    Uses nth(28) to extract 2014 values (assumes 29-year series with annual steps).
    Computes class transitions via renormalized changes from first to last year.
    """

    def _renorm_changes_above_boreal(
        g: "pd.DataFrame",
        forest: set[str] = {"Deciduous_Forest", "Evergreen_Forest", "Mixed_Forest"},
    ) -> "pd.Series":
        """Compute renormalized land cover class transitions between first/last year."""
        first = g.iloc[0]
        last = g.iloc[-1]

        first_dry = first[classes_dry_rn]
        last_dry = last[classes_dry_rn]

        s1 = float(first_dry.sum())
        s2 = float(last_dry.sum())

        # renormalize to sum(classes_dry) == 100
        first_norm = (first_dry / s1 * 100.0) if s1 > 0 else first_dry * 0.0
        last_norm = (last_dry / s2 * 100.0) if s2 > 0 else last_dry * 0.0
        delta_norm = last_norm - first_norm

        EF_change = float(delta_norm.get("Evergreen_Forest", 0.0))
        DF_change = float(delta_norm.get("Deciduous_Forest", 0.0))
        Shrub_change = float(delta_norm.get("Shrub", 0.0))
        F_change = float(delta_norm[[c for c in classes_dry_rn if c in forest]].sum())

        # Non-renormalized change for Water in native percent units (if available)
        if "Total_inun_pct" in g.columns:
            Inun_change_raw = float(last.get("Total_inun_pct", np.nan) - first.get("Total_inun_pct", np.nan))
        else:
            Inun_change_raw = np.nan

        return pd.Series(
            {
                "EF_diff": EF_change,
                "DF_diff": DF_change,
                "Shrub_diff": Shrub_change,
                "F_diff": F_change,
                "Inun_diff": Inun_change_raw,
            }
        )

    ## Load
    print("Calculating time series features...")
    df = pd.read_csv(out_norm_pth) #, nrows=3500) #, index_col=0)
    if "Unnamed: 0" in df.columns:
        df.drop(columns="Unnamed: 0", inplace=True)

    ## Filter by buffer length
    # df.query("Buffer_m == @buffer_lengths[0]", inplace=True)

    ## Group by lake
    dfg = df.groupby(join_index)
    ## Take a specific year (year 2014) value as initial features for output df
    stats_last = dfg.nth(28).set_index(join_index)

    ## Remove unnecessary columns
    # stats_last.drop(columns=["Year"], inplace=True)

    ## Compute median vals
    meta_columns = ["Area_m2", "Perim_m2"]  # metadata # "Buffer_m"
    stats_median = dfg.median().drop(columns=meta_columns)

    ## Rename stats vars for 2014
    mapper = {var: (var + "_2014") for var in stats_last.drop(meta_columns, axis=1).columns}
    stats_last = stats_last.rename(columns=mapper, inplace=False)  # rename cols

    ## Rename stats vars for median
    mapper = {var: (var + "_med") for var in stats_median.columns}
    stats_median = stats_median.rename(columns=mapper, inplace=False)  # rename cols

    ## Insert median stats into stats df
    stats = pd.concat((stats_last, stats_median), axis="columns")  # .drop(columns=["Year"])

    ## Reorder to put meta vars first
    [stats.insert(0, col, stats.pop(col)) for col in meta_columns[-1::-1]]  # re-order cols

    ## Compute more features
    # dropna?
    # 1 Dynamism, 1.5 RSD of water, 2 trend in water and shrubs, 3 trend in dom veg

    stats["Total_inun_RSD"] = dfg.Total_inun.std() / dfg.Total_inun.mean()
    stats["Total_inun_dyn_pct"] = (
        (dfg.Total_inun.max() - dfg.Total_inun.min()) / dfg.Total_inun.max() * 100
    )
    stats["Hi_water_yr"] = dfg.Total_inun.apply(
        lambda group: years[np.argmax(group)]
    )  # Cool! Use GroupBy.apply to apply a lambda function over all groups!
    stats["Lo_water_yr"] = dfg.Total_inun.apply(
        lambda group: years[np.argmin(group)]
    )  # Cool! Use GroupBy.apply to apply a lambda function over all groups!
    stats["Dominant_veg_1986"] = (
        dfg.first()
        .loc[:, classes_dry_rn]
        .apply(lambda lake: classes_dry_rn[np.argmax(lake)], axis="columns")
    )
    stats["Dominant_veg_group_1986"] = (
        dfg.first()
        .loc[:, grouped_classes]
        .apply(lambda lake: grouped_classes[np.argmax(lake)], axis="columns")
    )
    stats["Dominant_veg_2014"] = (
        dfg.nth(28).set_index(join_index)
        .loc[:, classes_dry_rn]
        .apply(lambda lake: classes_dry_rn[np.argmax(lake)], axis="columns")
    )
    stats["Dominant_veg_group_2014"] = (
        dfg.nth(28).set_index(join_index)
        .loc[:, grouped_classes]
        .apply(lambda lake: grouped_classes[np.argmax(lake)], axis="columns")
    )
    stats["SDF"] = stats.Perim_m2 / (2 * np.sqrt(np.pi * stats.Area_m2))
    stats["Perim_area_ratio"] = stats.Perim_m2 / stats.Area_m2
    # dfg['Year', 'Total_inun'].apply(lambda group: theilslopes(group.Year, group.Total_inun))
    for lcClass in ["Total_inun"] + grouped_classes:
        stats[lcClass + "_change"] = dfg[lcClass].apply(
            lambda group: theilslopes(group)[0]
        )  # Using method from Kuhn et and Butman 2021, PNAS
        stats[lcClass + "_trend"] = dfg[lcClass].apply(
            lambda group: pymannkendall.original_test(group)[0]
        )

    # Class transitions
    transitions = dfg.apply(_renorm_changes_above_boreal)
    stats = stats.join(transitions, validate="1:1")

    ## Save short version # not joined with rest of dataset
    stats.to_csv(
        csv_out_time_series_features_short_pth,
        float_format=FLOAT_FORMAT_LONG,
    )
    print(f"Wrote short time series output table: {csv_out_time_series_features_short_pth}")

    ## Join in lat/long and location from og-mod csv: load files
    gdf_og_data = gpd.read_file(pth_shp_in)
    geoms = gdf_og_data.geometry
    crs = gdf_og_data.crs

    ## rm unnecessary cols
    joined_cols = ds_specific_vars
    gdf_og_data = gdf_og_data[joined_cols]

    ## join and rename index
    stats = stats.merge(
        gdf_og_data, left_index=True, right_on=join_index, how="right", # validate="1:1"
    )  # TODO: make more flexible for when I actually have lake name
    # stats.index.rename(
    #     "Lake", inplace=True
    # )  # Note the 'lake' corresponds to index in pth_shp_in (arbitrary index after concatennating PLD and WBD lakes)

    ## Reorder to put meta vars first
    [
        stats.insert(0, col, stats.pop(col)) for col in joined_cols[-1::-1]
    ]  # re-order cols # TODO import load.sortColumns

    ## Write out
    # Write full output in appropriate format
    if csv_out_time_series_features_pth.endswith(".parquet"):
        stats.to_parquet(csv_out_time_series_features_pth, index=False)
    else:
        stats.to_csv(csv_out_time_series_features_pth, float_format=FLOAT_FORMAT_LONG, index=False)
    print(f"Wrote time series output table: {csv_out_time_series_features_pth}")

    ## Save and write out most important stats
    if csv_out_time_series_features_core_pth.endswith(".parquet"):
        stats.loc[:, important_vars].to_parquet(csv_out_time_series_features_core_pth, index=False)
    else:
        stats.loc[:, important_vars].to_csv(
            csv_out_time_series_features_core_pth, float_format=FLOAT_FORMAT_LONG, index=False
        )
    print(
        f"Wrote time series output table (greatest hits): {csv_out_time_series_features_core_pth}"
    )

    ## Save shapefile
    # gdf_stats = gpd.GeoDataFrame(stats, geometry=geoms, crs=crs)
    # gdf_stats.loc[:, ds_specific_vars + important_vars + ["geometry"]].to_file(
    #     shp_out_time_series_features_core_pth
    # )

    pass
