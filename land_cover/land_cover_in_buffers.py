#!/usr/bin/env python
"""
Extract land cover time series for lake buffer zones using parallel processing.

This module provides functions to calculate land cover composition within buffer zones
around lakes across multiple years. It uses multiprocessing for efficiency and supports
incremental writing with resume capability.
"""

import fcntl
import gc
import os
from multiprocessing import get_context
from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.features
import rasterio.mask
import rasterio.windows
from pyproj import CRS
from shapely import wkb as _wkb
from tqdm import tqdm

# Worker process globals (set by initializer)
_RASTER_DATASET = None
_RASTER_DATASET_COARSE = None
_RASTER_CRS = None
_BUFFER_LENGTHS = None
_CLASSES = None
_YEARS = None
_JOIN_INDEX = None
_PARQUET_PATH = None
_LARGE_LAKE_THRESHOLD = None


def _init_worker(
    raster_path: str,
    buffer_lengths: tuple,
    classes: list,
    years: list,
    join_index: str,
    parquet_path: str,
    large_lake_threshold: float = 30e6,
    raster_path_coarse: Optional[str] = None,
):
    """
    Initialize worker process with shared configuration.

    Opens the raster dataset once per worker to avoid repeated I/O.

    Parameters
    ----------
    raster_path : str
        Path to land cover raster (VRT or single file)
    buffer_lengths : tuple
        Buffer distances in meters
    classes : list
        Land cover class names
    years : list
        Years corresponding to raster bands
    join_index : str
        Column name for unique lake identifier
    parquet_path : str
        Output parquet file path
    large_lake_threshold : float, default=30e6
        Area threshold (m2) above which to use coarse raster if provided
    raster_path_coarse : str, optional
        Path to coarse-resolution raster for large lakes
    """
    global _RASTER_DATASET, _RASTER_DATASET_COARSE, _RASTER_CRS, _BUFFER_LENGTHS, _CLASSES, _YEARS, _JOIN_INDEX, _PARQUET_PATH, _LARGE_LAKE_THRESHOLD

    _RASTER_DATASET = rio.open(raster_path)
    _RASTER_CRS = _RASTER_DATASET.crs
    _BUFFER_LENGTHS = tuple(buffer_lengths)
    _CLASSES = list(classes)
    _YEARS = list(years)
    _JOIN_INDEX = join_index
    _PARQUET_PATH = parquet_path
    _LARGE_LAKE_THRESHOLD = large_lake_threshold

    # Open coarse raster if provided
    if raster_path_coarse is not None:
        _RASTER_DATASET_COARSE = rio.open(raster_path_coarse)
    else:
        _RASTER_DATASET_COARSE = None


def _reconstruct_geodataframe(payload: dict, crs: CRS) -> gpd.GeoDataFrame:
    """
    Reconstruct GeoDataFrame from serialized payload.

    Parameters
    ----------
    payload : dict
        Dictionary containing geometry WKB and attributes
    crs : CRS
        Coordinate reference system

    Returns
    -------
    gpd.GeoDataFrame
        Single-row GeoDataFrame with lake geometry and attributes
    """
    geom = _wkb.loads(payload["geometry_wkb"])

    return gpd.GeoDataFrame(
        {
            "Lake_name": [payload["Lake_name"]],
            _JOIN_INDEX: [payload[_JOIN_INDEX]],
            "Area_m2": [payload["Area_m2"]],
            "Perim_m2": [payload["Perim_m2"]],
        },
        geometry=[geom],
        crs=crs,
    )


def extract_buffer_zonal_histogram(
    lake_gdf: gpd.GeoDataFrame,
    raster_dataset: rio.DatasetReader,
    buffer_lengths: tuple,
    classes: list,
    years: list,
    join_index: str,
    nodata: int = 255,
    all_touched: bool = False,
    max_area_m2: float = 50e6,
) -> Optional[pd.DataFrame]:
    """
    Calculate land cover histogram for buffer zones around a lake.

    Efficiently computes land cover composition for multiple buffers and years
    by rasterizing buffer zones once and binning pixel values per band.

    Parameters
    ----------
    lake_gdf : gpd.GeoDataFrame
        Single-row GeoDataFrame with lake geometry and attributes
    raster_dataset : rio.DatasetReader
        Open rasterio dataset (multi-band land cover raster)
    buffer_lengths : tuple
        Buffer distances in meters (sorted small to large recommended)
    classes : list
        Land cover class names (length must match number of classes in raster)
    years : list
        Years corresponding to raster bands
    join_index : str
        Column name for unique lake identifier
    nodata : int, default=255
        NoData value in raster
    all_touched : bool, default=False
        Whether to include all pixels touched by buffer geometry
    max_area_m2 : float, default=50e6
        Maximum lake area to process (larger lakes are skipped)
    use_overview : bool, default=False
        Whether to use raster overview for coarser resolution
    overview_level : int, default=2
        Overview level to use (1 = first overview, typically 2x coarser)

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns for each class, plus Year, Buffer_m, Lake_name,
        join_index, Area_m2, Perim_m2. Returns None if lake is too large or
        doesn't overlap raster.
    """
    assert len(lake_gdf) >= 1, "lake_gdf must contain at least one geometry"

    # Extract lake geometry (union if multiple)
    lake_geom = lake_gdf.geometry.iloc[0] if len(lake_gdf) == 1 else lake_gdf.unary_union

    # Skip excessively large lakes
    if lake_geom.area > max_area_m2:
        return None

    # Ensure CRS matches
    assert raster_dataset.crs == lake_gdf.crs, "CRS mismatch between lake and raster"

    # Ensure CRS is projected for buffering/area calculations
    assert raster_dataset.crs is not None, "Raster CRS is undefined"
    proj_crs = CRS(raster_dataset.crs)
    assert proj_crs.is_projected, f"CRS must be projected for buffering/area ops (got {proj_crs})"

    # Sort buffers so outermost is last (used for cropping)
    buffer_lengths = list(buffer_lengths)
    sorted_indices = np.argsort(buffer_lengths)
    buffer_lengths_sorted = [buffer_lengths[i] for i in sorted_indices]
    buffer_geoms = [lake_geom.buffer(length) for length in buffer_lengths_sorted]

    # Crop raster to outermost buffer extent
    try:
        # For now, skip overview usage as older rasterio versions don't support out_shape
        # and reading overviews via indexes doesn't work correctly with mask.mask()
        data, transform = rasterio.mask.mask(
            raster_dataset,
            [buffer_geoms[-1]],
            crop=True,
            filled=True,
            nodata=nodata,
        )
    except ValueError as e:
        if "do not overlap raster" in str(e).lower():
            return None
        raise

    n_bands, height, width = data.shape
    n_buffers = len(buffer_geoms)
    n_classes = len(classes)

    # Rasterize buffer zones with unique labels (1, 2, 3, ...)
    buffer_labels = rasterio.features.rasterize(
        [(geom, idx + 1) for idx, geom in enumerate(buffer_geoms)],
        out_shape=(height, width),
        transform=transform,
        all_touched=all_touched,
        dtype="uint16",
    )

    # Count pixels for each buffer × class × year
    counts = np.empty((n_bands, n_buffers, n_classes), dtype=np.uint32)
    valid_mask = (data >= 1) & (data <= n_classes)

    for band_idx in range(n_bands):
        band_data = data[band_idx]
        mask = (buffer_labels > 0) & valid_mask[band_idx] & (band_data != nodata)

        if mask.any():
            # Create 1D keys combining buffer_id and class_id
            flat_keys = (buffer_labels[mask] - 1) * n_classes + (band_data[mask] - 1)
            bin_counts = np.bincount(flat_keys, minlength=n_buffers * n_classes)
            counts[band_idx] = bin_counts.reshape(n_buffers, n_classes)
        else:
            # No valid data in this band (lake outside raster coverage)
            return None

    # Convert pixel counts to area (hectares)
    pixel_area_ha = abs(raster_dataset.res[0] * raster_dataset.res[1]) / 10000.0
    areas_ha = counts.reshape(n_bands * n_buffers, n_classes).astype("float64") * pixel_area_ha

    # Assemble output DataFrame
    df = pd.DataFrame(areas_ha, columns=classes)
    df["Year"] = np.repeat(years[:n_bands], n_buffers)
    df["Buffer_m"] = np.tile(buffer_lengths_sorted, n_bands)
    df["Lake_name"] = lake_gdf.index[0]
    df[join_index] = lake_gdf[join_index].iloc[0] if join_index in lake_gdf else None
    df["Area_m2"] = lake_gdf["Area_m2"].iloc[0] if "Area_m2" in lake_gdf else None
    df["Perim_m2"] = lake_gdf["Perim_m2"].iloc[0] if "Perim_m2" in lake_gdf else None

    return df


def _append_to_parquet_locked(df: pd.DataFrame, parquet_path: str):
    """
    Append DataFrame to Parquet file with file locking.

    Uses POSIX advisory locks and fastparquet engine for true append mode
    without reading existing data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to append
    parquet_path : str
        Path to output Parquet file
    """
    # Ensure all column names are strings (fastparquet requirement)
    df.columns = df.columns.astype(str)

    lock_path = parquet_path + ".lock"

    # Acquire lock
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)

        try:
            # Check if file exists
            file_exists = os.path.exists(parquet_path) and os.path.getsize(parquet_path) > 0

            # Use fastparquet for true append without reading existing data
            df.to_parquet(
                parquet_path,
                engine="fastparquet",
                append=file_exists,
                index=False,
            )

        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _process_lake(payload: dict) -> tuple:
    """
    Worker function to process a single lake.

    Parameters
    ----------
    payload : dict
        Serialized lake data (geometry WKB and attributes)

    Returns
    -------
    tuple
        (join_index_value, status) where status is 1 for success, 0 for skipped,
        or error message string for failures
    """
    try:
        # Reconstruct GeoDataFrame
        lake_gdf = _reconstruct_geodataframe(payload, _RASTER_CRS)

        # Choose raster based on lake size
        lake_area = payload["Area_m2"]
        if _RASTER_DATASET_COARSE is not None and lake_area > _LARGE_LAKE_THRESHOLD:
            raster_to_use = _RASTER_DATASET_COARSE
        else:
            raster_to_use = _RASTER_DATASET

        # Extract land cover data
        result_df = extract_buffer_zonal_histogram(
            lake_gdf=lake_gdf,
            raster_dataset=raster_to_use,
            buffer_lengths=_BUFFER_LENGTHS,
            classes=_CLASSES,
            years=_YEARS,
            join_index=_JOIN_INDEX,
        )

        if result_df is None or result_df.empty:
            return payload[_JOIN_INDEX], 0

        # Append to output file
        _append_to_parquet_locked(result_df, _PARQUET_PATH)

        # Clean up
        del result_df
        gc.collect()

        return payload[_JOIN_INDEX], 1

    except Exception as e:
        return payload[_JOIN_INDEX], f"ERROR: {e}"


def _get_completed_indices(parquet_path: Path, join_index: str) -> set:
    """
    Read completed lake indices from existing Parquet file.

    Parameters
    ----------
    parquet_path : Path
        Path to output Parquet file
    join_index : str
        Column name for unique lake identifier

    Returns
    -------
    set
        Set of already-processed lake identifiers
    """
    if not parquet_path.exists() or parquet_path.stat().st_size == 0:
        return set()

    try:
        # Read only the join_index column for efficiency
        df = pd.read_parquet(parquet_path, columns=[join_index], engine="fastparquet")

        # Handle both int and string types
        completed = set(df[join_index].dropna().unique())
        print(f"Resuming: found {len(completed)} completed lakes in existing file.")
        return completed

    except Exception as e:
        print(f"Resume read failed ({e}); proceeding without resume filtering.")
        return set()


def extract_time_series_for_lakes(
    lake_shapefile: str,
    raster_path: str,
    output_parquet: str,
    buffer_lengths: List[float],
    classes: List[str],
    years: List[int],
    join_index: Optional[str] = None,
    envelope_shapefile: Optional[str] = None,
    raster_path_coarse: Optional[str] = None,
    large_lake_threshold: float = 30e6,
    n_workers: int = 8,
) -> None:
    """
    Extract land cover time series for multiple lakes in parallel.

    Processes lake polygons to calculate land cover composition in buffer zones
    across multiple years. Supports incremental writing and resume capability.
    For large lakes (>large_lake_threshold), uses a coarse-resolution raster
    if provided for better performance.

    Parameters
    ----------
    lake_shapefile : str
        Path to shapefile or geopackage with lake polygons
    raster_path : str
        Path to multi-band land cover raster (VRT or single file)
    output_parquet : str
        Path to output Parquet file
    buffer_lengths : list of float
        Buffer distances in meters
    classes : list of str
        Land cover class names (must match raster class count)
    years : list of int
        Years corresponding to raster bands
    join_index : str, optional
        Column name for unique lake identifier. If None, creates "join_idx"
        from DataFrame index. Can be int or string type.
    envelope_shapefile : str, optional
        Path to envelope polygon for spatial filtering
    raster_path_coarse : str, optional
        Path to coarse-resolution raster for large lakes (e.g., resampled
        using majority resampling). Improves performance for large polygons.
    large_lake_threshold : float, default=30e6
        Area threshold (m2) above which to use coarse raster if provided
        (30e6 = 30 km²)
    n_workers : int, default=8
        Number of parallel worker processes
    """
    print(f"Output path: {output_parquet}")
    print(f"Processing {len(buffer_lengths)} buffer(s): {buffer_lengths} m")
    if raster_path_coarse is not None:
        print(
            f"Using coarse raster for lakes > {large_lake_threshold/1e6:.1f} km²: {raster_path_coarse}"
        )

    # Load lake polygons
    print(f"Loading lake polygons from {lake_shapefile}...")
    lakes_gdf = gpd.read_file(lake_shapefile)
    print(f"Loaded {len(lakes_gdf)} lakes")

    # Load raster CRS and reproject if needed
    with rio.open(raster_path) as src:
        raster_crs = src.crs

    if lakes_gdf.crs != raster_crs:
        print(f"Reprojecting lakes from {lakes_gdf.crs} to {raster_crs}")
        lakes_gdf = lakes_gdf.to_crs(raster_crs)

    # Set up join index
    if join_index is None:
        join_index = "join_idx"
        lakes_gdf[join_index] = lakes_gdf.index

    # Calculate geometric attributes
    lakes_gdf["Area_m2"] = lakes_gdf.area
    lakes_gdf["Perim_m2"] = lakes_gdf.length
    lakes_gdf["Lake_name"] = lakes_gdf.index

    # Apply spatial envelope filter if provided
    if envelope_shapefile is not None:
        print(f"Applying envelope filter from {envelope_shapefile}...")
        envelope = gpd.read_file(envelope_shapefile)

        if envelope.crs != lakes_gdf.crs:
            envelope = envelope.to_crs(lakes_gdf.crs)

        envelope_union = envelope.union_all()
        lakes_gdf = lakes_gdf[lakes_gdf.geometry.intersects(envelope_union, align=False)]
        print(f"Filtered to {len(lakes_gdf)} lakes within envelope")

    # Resume support: identify already-processed lakes
    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_indices = _get_completed_indices(output_path, join_index)
    pending_lakes = lakes_gdf[~lakes_gdf[join_index].isin(completed_indices)]

    if len(pending_lakes) == 0:
        print("All lakes already processed. Nothing to do.")
        return

    print(f"Processing {len(pending_lakes)} pending lakes...")

    # Serialize lake data for multiprocessing
    payloads = []
    for _, row in pending_lakes.iterrows():
        payloads.append(
            {
                "Lake_name": int(row["Lake_name"]),
                join_index: row[join_index],  # Keep original type (int or str)
                "Area_m2": float(row["Area_m2"]),
                "Perim_m2": float(row["Perim_m2"]),
                "geometry_wkb": row.geometry.wkb,
            }
        )

    # Set up multiprocessing
    ctx = get_context("fork")
    chunksize = max(1, len(payloads) // (n_workers * 8))

    init_args = (
        raster_path,
        tuple(buffer_lengths),
        list(classes),
        list(years),
        join_index,
        str(output_path),
        large_lake_threshold,
        raster_path_coarse,
    )

    # Process lakes in parallel
    with ctx.Pool(processes=n_workers, initializer=_init_worker, initargs=init_args) as pool:
        iterator = pool.imap_unordered(_process_lake, payloads, chunksize=chunksize)

        # Track progress
        completed = 0
        skipped = 0
        errors = 0

        for lake_id, status in tqdm(iterator, total=len(payloads), desc="Processing lakes"):
            if status == 1:
                completed += 1
            elif status == 0:
                skipped += 1
            else:
                errors += 1
                print(f"[{join_index}={lake_id}] {status}")

    print(f"\nCompleted: {completed} lakes")
    print(f"Skipped: {skipped} lakes (outside raster or too large)")
    print(f"Errors: {errors} lakes")
    print(f"Output written to: {output_parquet}")
