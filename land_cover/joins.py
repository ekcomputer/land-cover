import geopandas as gpd
import numpy as np

def merge_left_one_one(
    left_gdf,
    right_gdf,
    left_key="Lake_id",
    left_area_col="Area_PW",
    right_area_col="area_km2_gswl",
    rel_thresh=None,
    predicate="contains",
    sort_by=None,
    match_col="match_gswl",
    float_fields=[]
):
    """
    Left spatial join (left_gdf ⟕ right_gdf) and collapse many-to-one matches by taking the last()
    row within each left_key group (after sorting). Then flag 1:1 matches by relative area diff.

    rel_diff = |sqrt(right_area) - sqrt(left_area)| / sqrt(left_area) <= rel_thresh

    Still not working properly- operation hangs for > 10 min. See notebooks/joins/join_greennessx2.ipynb
    for working implementation.
    """
    if sort_by is None:
        sort_by = left_area_col

    # spatial join
    joined = left_gdf.sjoin(right_gdf, how="left", predicate=predicate)
    print('joined')

    # count how many right-side hits each left feature received
    grouped = joined.sort_values(sort_by).groupby(left_key, dropna=False)
    join_counts = grouped[match_col].count()

    # reduce many-to-one: keep last row per left_key (after the sort above)
    reduced = grouped.last()

    if len(float_fields) > 0:
        reduced[float_fields] = grouped[float_fields].mean()

    # attach the counts
    reduced["right_count"] = join_counts

    if rel_thresh is not None:
        # initialize match flag
        reduced[match_col] = np.nan
        reduced.loc[reduced[right_area_col].ge(0), match_col] = (
            0  # default "not matched" for any non-negative area
        )

        # compute relative difference and flag matches only for unique joins
        valid = reduced[right_area_col].notna() & (reduced[left_area_col] > 0)
        rel_diff = (np.sqrt(reduced[right_area_col]) - np.sqrt(reduced[left_area_col])).abs() / np.sqrt(
            reduced[left_area_col]
        )
        reduced.loc[valid & (rel_diff <= rel_thresh) & (reduced["right_count"] == 1), match_col] = 1

    # pretty print summary
    non_nan = (~reduced[match_col].isna()).sum()
    matched = np.nansum(reduced[match_col].to_numpy(dtype=float))
    total = len(reduced)

    print(
        f"Found {total:,} features; matches = {int(matched):,} / {int(non_nan):,}, length of left: {len(left_gdf):,}"
    )

    # keep GeoDataFrame-ness
    reduced = gpd.GeoDataFrame(reduced, geometry="geometry", crs=left_gdf.crs)
    return reduced  # .drop(columns="index_right")
