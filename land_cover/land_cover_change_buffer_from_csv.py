#!/usr/bin/env python
# coding: utf-8

'''
Summary
For calculating land cover and its change within polygons representing outwards buffers from lakes.
Can optionally join in all attributes from original lake dataset.

Write three outputs:
1. All input variables for lakes that were matched to land cover
2. All input variables for all lakes
3. Selected input variables for all lakes

TODO: 
* Check that water normalization only refers to largest/central lake within buffer.
* Add watershed buffer x
* IMPORTANT: Find a way to automatically include Lat/Long and any note columns in final spreadsheet (perhaps join in?) Right now, I'm just using a quick fix in Excel.
* Remove shortcut hack to skip large lakes -> eventually resample coarsen landcover to use for large lakes
* Remove hacks for numeric types if running on GLAKES again

2025 problems:
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.plot import reshape_as_image
# from scipy.stats import binned_statistic
from rasterstats import zonal_stats
from scipy.stats.mstats import theilslopes
import pymannkendall
from tqdm import tqdm
import gc
import fcntl
from pathlib import Path
from multiprocessing import get_context
from shapely import wkb as _wkb
from pyproj import CRS

## Params
checkpoint_frequency = 1000
FLOAT_FORMAT_LONG = "%.5f" # csv digits (to save storage space) for time series features
FLOAT_FORMAT_SHORT = "%.3f" # csv digits for normalized features

## Function
def extractBufferZonalHist(
    poly, buffer_lengths, pth_lc_in, classes, years, nodata=255, all_touched=False, join_index="join_idx"
):
    """
    Zonal histogram for many buffers x all bands (years) in looped pass.
    Rasterizing once per iteration speeds up operations
    """
    assert len(poly) >= 1, "poly must contain at least one geometry"
    lake_geom = poly.geometry.iloc[0] if len(poly) == 1 else poly.unary_union

    # Hack: skip large datasets:
    if lake_geom.area > 50e6:
        return None
    with rio.open(pth_lc_in) as src:
        assert src.crs == poly.crs, "CRS mismatch"

        # sort buffers so last one is the outermost ROI for cropping
        buffer_lengths = list(buffer_lengths)
        order = np.argsort(buffer_lengths)
        buffer_lengths_sorted = [buffer_lengths[i] for i in order]
        buf_geoms = [lake_geom.buffer(L) for L in buffer_lengths_sorted]

        # crop to outermost buffer; everything outside is filled with nodata
        try:
            data, tr = rio.mask.mask(src, [buf_geoms[-1]], crop=True, filled=True, nodata=nodata)
        except ValueError as e:
            if "do not overlap raster" in str(e).lower():
                return None
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
            dtype="uint16",
        )
        counts = np.empty((n_bands, n_buffers, nclasses), dtype=np.uint32)
        valid_vals = (data >= 1) & (data <= nclasses)
        for bi in range(n_bands):
            vals = data[bi]
            m = (labels > 0) & valid_vals[bi] & (vals != nodata)
            if m.any():
                keys = (labels[m] - 1) * nclasses + (
                    vals[m] - 1
                )  # (buffer_id, class_id) -> 1D key
                bc = np.bincount(keys, minlength=n_buffers * nclasses).reshape(
                    n_buffers, nclasses
                )
            else:
                # bc = np.zeros((n_buffers, nclasses), dtype=np.uint32)
                # These would be lakes inside of my crude envelope but outside of the data area of the rasters
                return None
            counts[bi] = bc

        # scale to area (hectares), consistent with your previous division by 1e4
        pix_area_ha = abs(src.res[0] * src.res[1]) / 10000.0
        areas = counts.reshape(n_bands * n_buffers, nclasses).astype("float64") * pix_area_ha

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


def _init_worker(csv_path, raster_crs_wkt, buffer_lengths, pth_lc_in, classes, years, join_index):
    global _CSV_PATH, _RASTER_CRS, _BUFFER_LENGTHS, _PTH_LC_IN, _CLASSES, _YEARS, _JOIN_INDEX
    _CSV_PATH = csv_path
    _RASTER_CRS = CRS.from_wkt(raster_crs_wkt)
    _BUFFER_LENGTHS = tuple(buffer_lengths)
    _PTH_LC_IN = pth_lc_in
    _CLASSES = list(classes)
    _YEARS = list(years)
    _JOIN_INDEX = join_index


def _gdf_from_payload(payload, crs):
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


def _append_df_csv_locked(df: pd.DataFrame, csv_path: str):
    # POSIX advisory lock on the file while writing
    # Ensures header is written once and rows append atomically
    with open(csv_path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        need_header = f.tell() == 0  # at end because "a+" opens and seeks to end
        df.to_csv(f, index=False, header=need_header, float_format=FLOAT_FORMAT_LONG)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def _worker(payload):
    join_index = _JOIN_INDEX
    try:
        poly = _gdf_from_payload(payload, _RASTER_CRS)
        df = extractBufferZonalHist(
            poly, _BUFFER_LENGTHS, _PTH_LC_IN, classes=_CLASSES, years=_YEARS, join_index=join_index
        )
        if df is None or df.empty:
            return payload[join_index], 0
        _append_df_csv_locked(df, _CSV_PATH)
        del df
        gc.collect()
        return payload[join_index], 1
    except Exception as e:
        return payload[join_index], f"ERROR: {e}"


def _prime_header(
    csv_path: Path,
    classes,
    years_cols,
):
    # Create file with header if missing/empty to guarantee consistent column order
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        cols = list(classes) + list(years_cols)
        empty = pd.DataFrame(columns=cols)
        _append_df_csv_locked(empty, str(csv_path))


def extractTimeSeriesForLakes(
    pth_shp_in,
    buffer_lengths,
    csv_out_pth,
    pth_lc_in,
    use_simplified_classes,
    classes,
    years,
    envelope_pth=None,
    join_index=None,
    n_workers=8,
    pth_lc_in_coarse=None,
):
    print("Paths:")
    print(csv_out_pth)
    print(f"\nUse simplified classes: {use_simplified_classes}")

    # Load polygons and project to raster CRS
    polys = gpd.read_file(pth_shp_in) #, rows=slice(0, 12000))  # slice(90000,90090)) #
    with rio.open(pth_lc_in) as src:
        raster_crs = src.crs
    if polys.crs != raster_crs:
        print(f"reprojecting polygons to {raster_crs}")
        polys = polys.to_crs(raster_crs)

    # Attributes needed downstream
    if join_index is None:
        join_index = "join_idx"
        polys[join_index] =  polys.index
    polys["Area_m2"] = polys.area
    polys["Perim_m2"] = polys.length
    polys["Lake_name"] = polys.index  # keep stable name

    # Optional envelope filter
    if envelope_pth is not None:
        envelope = gpd.read_file(envelope_pth)
        assert envelope.crs == polys.crs, "CRS mismatch: envelope vs. polygons"
        polys = polys[
            polys.geometry.intersects(envelope.union_all(), align=False)#  & (polys.Area_m2 < 10e6)
        ]
        print(f"Filtered polygons by envelope: {len(polys)} features")

    out_path = Path(csv_out_pth)
    _prime_header(
        out_path, classes, ("Year", "Buffer_m", "Lake_name", join_index, "Area_m2", "Perim_m2")
    )

    # Resume support: read already processed Join_idx
    done_idx = set()
    try:
        if out_path.exists() and out_path.stat().st_size > 0:
            # Only load Join_idx column for speed/memory
            done_col = pd.read_csv(out_path, usecols=[join_index], dtype={join_index: "Int64"})[
                join_index
            ]
            done_idx = set(done_col.dropna().astype(int).unique().tolist())
            print(f"Resuming: found {len(done_idx)} completed lakes in existing CSV.")
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
                join_index: row[join_index], # HERE HACK not to use int
                "Area_m2": float(row["Area_m2"]),
                "Perim_m2": float(row["Perim_m2"]),
                "geometry_wkb": row.geometry.wkb,
            }
        )

    # Multiprocessing (fork on Unix/macOS)
    ctx = get_context("fork")
    chunksize = max(1, len(payloads) // (n_workers * 8) or 1)

    # Progress
    initargs = (
        str(out_path),
        raster_crs.to_wkt(),
        list(buffer_lengths),
        pth_lc_in,
        list(classes),
        list(years),
        join_index,
    )

    with ctx.Pool(processes=n_workers, initializer=_init_worker, initargs=initargs) as pool:
        iterator = pool.imap_unordered(_worker, payloads, chunksize=chunksize)
        iterator = tqdm(iterator, total=len(payloads), desc="Lakes")
        completed = 0
        errors = 0
        for join_idx, status in iterator:
            if status == 1:
                completed += 1
            elif status == 0:
                # no data for lake (skipped)
                pass
            else:
                errors += 1
                print(f"[Join_idx={join_idx}] {status}")

    print(
        f"done. wrote {completed} lakes, {errors} errors, {len(payloads)-completed-errors} outside of raster area."
    )


def normalizeTimeSeries(
    xlsx_out_pth,
    xlsx_out_norm_pth,
    classes_wet,
    classes_dry,
    classes_dry_rn,
    use_simplified_classes=False,
    wetland_class='Shallows/littoral',
):
    '''
    Loads 'xlsx_out_pth', normalizes water classes by total water and land classes by total land (in buffer). Outputs data to 'xlsx_out_norm_pth'.
    '''
    assert use_simplified_classes==False, "Must run on full 14 (not-simplified) classes."

    ## Load
    print('Normalizing land cover...')
    df = pd.read_csv(xlsx_out_pth)

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
    df.to_csv(xlsx_out_norm_pth, float_format=FLOAT_FORMAT_SHORT)
    print(f'Wrote normalized output table: {xlsx_out_norm_pth}')


def normalizeTimeSeries_above_boreal(
    xlsx_out_pth,
    xlsx_out_norm_pth,
    classes_wet,
    classes_dry,
    wetland_class="Wetland",
    index_class="Lake_id_glakes",
):
    """
    As above, but modified for the new land cover classes in Hu et al. 2025.
    """
    classes = np.unique(classes_dry + classes_wet).tolist()
    ## Load
    print("Normalizing land cover...")
    usecols = classes + ["Year", index_class, "Area_m2", "Perim_m2"]
    df = pd.read_csv(xlsx_out_pth, usecols=usecols) #, nrows=1000) # all but Lake_name and Buffer_m

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
    # write with 3 decimal places
    if xlsx_out_norm_pth.endswith('.parquet'):
        df.to_parquet(xlsx_out_norm_pth, index=False)
    else:
        df.to_csv(xlsx_out_norm_pth, float_format=FLOAT_FORMAT_SHORT)
    print(f"Wrote normalized output table: {xlsx_out_norm_pth}")


def plotTimeSeries(buffer_lengths, xlsx_out_norm_pth, plot_dir, index_col=None, index=None, combined=False):
    """
    Loads 'xlsx_out_norm_pth', manipulates data, and creates a multi-facted time-series plot for each lake from the ABoVE landcover dataset, plotting in ha, not normalized, by default. Saves plots to 'plot_dir'
    If index_col is provided, only plots indexes in `index`
    If combined == True, plots all lakes in one figure, using area-weighted means across lakes.
    """
    ## vars
    buf_len = buffer_lengths[0] # use the smallest (90 m) buffer for plotting

    ## Load
    print('Plotting land cover...')
    if xlsx_out_norm_pth.endswith(".parquet"):
        df = pd.read_parquet(xlsx_out_norm_pth)
    elif xlsx_out_norm_pth.endswith(".csv"):
        df = pd.read_csv(xlsx_out_norm_pth, index_col=0)
    else:
        raise ValueError("Unsupported file format for xlsx_out_norm_pth")

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
    # group = dfg.get_group('Balloon lake') # formerly ('Balloon lake', buf_len)

    ## Plot with mpl
    # fig, ax = plt.subplots()
    # group.plot(x='Year', y='Littorals_pct', ax=ax)
    # plt.savefig(os.path.join(plot_dir, 'time-series-1.png'))

    ## Try facet grid in seaborn
    # dfl = pd.melt(group, id_vars=['Year', 'Buffer_m'], value_vars=df.columns[1:16], var_name = 'Class', value_name=value_name)# data frame long format # use df.columns[-14:] for normalized vals
    # g = sns.FacetGrid(dfl, col="Class", hue="Buffer_m", col_wrap=4)
    # g.map(sns.lineplot, 'Year', value_name)
    # g.add_legend(title="Buffer (m)")
    # plt.show()
    # g.savefig(os.path.join(plot_dir, 'time-series-facets-1.png'))

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
    xlsx_out_norm_pth,
    years,
    classes_dry_rn,
    pth_shp_in,
    ds_specific_vars,
    csv_out_time_series_features_pth,
    important_vars,
    csv_out_time_series_features_core_pth,
    shp_out_time_series_features_core_pth,
    join_index="join_idx",
):
    '''
    Loads data from 'xlsx_out_norm_pth' and reduces each time series for the specified buffer (probably smallest buffer) to a series of features/metrics.
    Outputs data to 'xlsx_out_time_series_features_pth'.
    '''

    ## Load
    print('Calculating time series features...')
    df = pd.read_csv(xlsx_out_norm_pth, index_col=0)

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
    stats.to_csv(csv_out_time_series_features_pth, float_format=FLOAT_FORMAT_LONG)
    print(f'Wrote time series output table: {csv_out_time_series_features_pth}')

    ## Save and write out most important stats
    stats.loc[:, ds_specific_vars + important_vars].to_csv(csv_out_time_series_features_core_pth, float_format=FLOAT_FORMAT_LONG)
    print(f'Wrote time series output table (greatest hits): {csv_out_time_series_features_core_pth}')

    ## Save shapefile
    gdf_stats = gpd.GeoDataFrame(stats, geometry=geoms, crs=crs)
    gdf_stats.loc[:, ds_specific_vars + important_vars + ['geometry']].to_file(shp_out_time_series_features_core_pth)

    pass


def extractTimeSeriesFeatures_above_boreal(
    xlsx_out_norm_pth,
    years,
    classes_dry_rn,
    pth_shp_in,
    ds_specific_vars,
    csv_out_time_series_features_pth,
    important_vars,
    csv_out_time_series_features_core_pth,
    csv_out_time_series_features_short_pth,
    join_index="Lake_id_glakes",
    grouped_classes=["Trees", "Shrub", "Wetland", "Herb", "Sparse"],
):
    """
    Loads data from 'xlsx_out_norm_pth' and reduces each time series for the specified buffer (probably smallest buffer) to a series of features/metrics.
    Outputs data to 'xlsx_out_time_series_features_pth'.
    """

    def _renorm_changes_above_boreal(
        g: pd.DataFrame, forest={"Deciduous_Forest", "Evergreen_Forest", "Mixed_Forest"}
    ) -> pd.Series:
        # g = g.sort_values("Year")
        first = g.iloc[0] #.astype(float) # HACK
        last = g.iloc[-1] # .astype(float)

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
    df = pd.read_csv(xlsx_out_norm_pth) #, nrows=3500) #, index_col=0)
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
    stats.to_parquet(csv_out_time_series_features_pth.replace(".csv", ".parquet"), index=False)
    print(f"Wrote time series output table: {csv_out_time_series_features_pth}")

    ## Save and write out most important stats
    stats.loc[:, important_vars].to_parquet(
        csv_out_time_series_features_core_pth.replace(".csv", ".parquet"), index=False
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
