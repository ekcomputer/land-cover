import os

import geopandas as gpd
import numpy as np

from land_cover.load import (
    cec_30m_lc_pth,
    cec_30m_lc_rs250_pth,
    doc_jn_catchment_pth,
    load_ecoregions,
    plot_dir,
    time_series_features_cec_doc_parquet_pth,
)


def join_ecoregions(gdf):
    """
    Join ecoregion data to a GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame to join ecoregions to.

    Returns
    -------
    GeoDataFrame
        Updated GeoDataFrame with ecoregion field added as 'ecoregion'.
    """
    eco = load_ecoregions()
    eco.to_crs(gdf.crs, inplace=True)
    eco = eco.rename(columns={"NA_L1NAME": "ecoregion"})

    # Use spatial join (point-in-polygon) if lakes fall within ecoregions
    # This is much faster than sjoin_nearest for containment
    gdf_joined = gpd.sjoin(gdf, eco[["geometry", "ecoregion"]], how="left", predicate="within")
    gdf_joined = gdf_joined.groupby(level=0).first()
    gdf_joined = gdf_joined.drop(columns=["index_right"])

    return gdf_joined
