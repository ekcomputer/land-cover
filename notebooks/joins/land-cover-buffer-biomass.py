"""Version of lake buffers script for harmonized DOC dataset joined to lake catchments.

Modified to use CEC land cover dataset instead of above boreal landcover.
Uses re-write script called land_cover.land_cover_in_buffers.extract_time_series_for_lakes

Un-rewrite so that it uses my current library functions in land_cover/land_cover_change_buffer_from_csv.py
TODO:
* Fix auto-loading if running on large dataset?
"""

import os
from pathlib import Path

import geopandas as gpd
import numpy as np

from land_cover.biomass_change import extractTimeSeriesForLakes
from land_cover.load import (
    biomass_30m_pth,
    biomass_300m_pth,
    doc_jn_catchment_pth,
    topocat_biomass_test_subset_pth,
    test_dir,
    topocat_subset_aea_pth,
    plot_dir,
    time_series_features_agb_doc_csv_pth,
)

# I/O
years = list(range(1986, 2025))  # 39 bands

## in: for greennessx2 and Land Cover v2

## out
plot_dir = os.path.join(plot_dir, "doc_harm_cec")

## buffers, in order small -> large
buffer_lengths = [0]  # (90, 180) # in m # 90, 990 # 1350

## classes for land cover (CEC North America 30m)

ds_specific_vars = [
    "sample_idx",
    "lat",
    "lon",
    "sample_id",
    "area_km2",
    "doc",
    "dic",
    "source",
    "Outlet_id_tpcat",
    "lake_id_tpcat",
    "D_out_id_tpcat",
    "D_lake_id_tpcat",
    "Cat_area_tpcat",
    "Cat_type_tpcat",
    "Basin_id_tpcat",
    "Shape_Length_tpcat",
    "Shape_Area_tpcat",
]

# variables from both datasets I want in the core output for easy viewing
# TODO
important_vars = [
    "sample_idx",
    "lat",
    "lon",
    "sample_id",
    "area_km2",
    "doc",
    "dic",
    "source",
    "Outlet_id_tpcat",
    "lake_id_tpcat",
    "D_out_id_tpcat",
    "D_lake_id_tpcat",
    "Cat_area_tpcat",
    "Cat_type_tpcat",
    "Basin_id_tpcat",
    "Shape_Area_tpcat",
    "Bare_Sparsely_vegetated_2014",
    "Deciduous_Forest_2014",
    "Evergreen_Forest_2014",
    "Mixed_Forest_2014",
    "Herb_2014",
    "Shrub_2014",
    "Water_2014",
    "Wetland_2014",
    "Sparse_2014",
    "Total_inun_RSD",
    "Total_inun_dyn_pct",
    "Hi_water_yr",
    "Lo_water_yr",
    "Dominant_veg_2014",
    "Dominant_veg_group_2014",
    "Dominant_veg_1986",
    "Dominant_veg_group_1986",
    "SDF",
    "Perim_area_ratio",
    "Total_inun_change",
    "Total_inun_trend",
    "EF_diff",
    "DF_diff",
    "Shrub_diff",
    "F_diff",
    # "Water_diff_raw",
]

# Input path:
# Can run mosaic_vrt.sh to generate virtual raster mosaic dataset

# Output paths
base_path = Path(time_series_features_agb_doc_csv_pth)
stem = base_path.stem

parquet_out_norm_pth = base_path.parent / f"{stem}_norm.parquet"
parquet_out_time_series_features_pth = base_path.parent / f"{stem}_tsFeatures.parquet"
parquet_out_time_series_features_core_pth = base_path.parent / f"{stem}_core_tsFeatures.parquet"
parquet_out_time_series_features_short_pth = base_path.parent / f"{stem}_short_tsFeatures.parquet"
shp_out_time_series_features_core_pth = base_path.parent / f"{stem}_core_tsFeatures.gpkg"

## RUN for preliminary harmonized DOC with no buffers
# extractTimeSeriesForLakes(
#     pth_shp_in=doc_jn_catchment_pth,
#     buffer_lengths=buffer_lengths,
#     csv_out_pth=time_series_features_agb_doc_csv_pth,
#     pth_lc_in=biomass_30m_pth,
#     pth_lc_in_coarse=biomass_300m_pth,
#     years=years,
#     n_workers=8,
#     join_index="sample_idx",
# )

## TEST for TopoCAT dataset test subset
extractTimeSeriesForLakes(
    pth_shp_in=topocat_biomass_test_subset_pth,
    buffer_lengths=buffer_lengths,
    csv_out_pth=test_dir / "out" / "tcat_biomass_test_subset_agbBuffers.csv",
    pth_lc_in=biomass_30m_pth,
    pth_lc_in_coarse=biomass_300m_pth,
    years=years,
    n_workers=8,
    join_index="Outlet_id",
)

## RUN for entire TopoCAT dataset
# extractTimeSeriesForLakes(
#     pth_shp_in=topocat_subset_aea_pth,
#     buffer_lengths=buffer_lengths,
#     csv_out_pth="",
#     pth_lc_in=biomass_30m_pth,
#     pth_lc_in_coarse=biomass_300m_pth,
#     years=years,
#     n_workers=8,
#     join_index="sample_idx",
# )


print("DONE.")
