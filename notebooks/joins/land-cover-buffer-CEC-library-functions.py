"""Version of lake buffers script for harmonized DOC dataset joined to lake catchments.

Modified to use CEC land cover dataset instead of above boreal landcover.
Uses re-write script called land_cover.land_cover_in_buffers.extract_time_series_for_lakes

Un-rewrite so that it uses my current library functions in land_cover/land_cover_change_buffer_from_csv.py
TODO:
* Fix auto-loading if running on large dataset?
"""

import os

import geopandas as gpd
import numpy as np

from land_cover.land_cover_change_buffer_from_csv import extractTimeSeriesForLakes
from pathlib import Path
from land_cover.load import (cec_30m_lc_pth, cec_30m_lc_rs250_pth,
                             doc_jn_catchment_pth, plot_dir,
                             time_series_features_cec_doc_csv_pth)

# I/O

## in: for greennessx2 and Land Cover v2

## out
plot_dir = os.path.join(plot_dir, "doc_harm_cec")

## buffers, in order small -> large
buffer_lengths = [0]  # (90, 180) # in m # 90, 990 # 1350

## classes for land cover (CEC North America 30m)
"""
Original:
land_cover_types = {
    1: "Temperate or Subpolar Needleaf Forest",
    2: "Subpolar Taiga Needleleaf Forest",
    3: "Tropical or Subtropical Broadleaf Evergreen Forest",
    4: "Tropical or Subtropical Broadleaf Deciduous Forest",
    5: "Temperate or Subpolar Broadleaf Deciduous Forest",
    6: "Mixed Forest",
    7: "Tropical or Subtropical Shrubland",
    8: "Temperate or Subpolar Shrubland",
    9: "Tropical or Subtropical Grassland",
    10: "Temperate or Subpolar Grassland",
    11: "Subpolar or Polar Shrubland-Lichen-Moss",
    12: "Subpolar or Polar Grassland-Lichen-Moss",
    13: "Subpolar or Polar Barren-Lichen-Moss",
    14: "Wetland",
    15: "Cropland",
    16: "Barren Land",
    17: "Urban and Built-up",
    18: "Water",
    19: "Snow and Ice",
}
"""
land_cover_types = {
    1: "Needleaf Forest",
    2: "Taiga Needleleaf Forest",
    3: "T/S Broadleaf Evergreen Forest",
    4: "T/S Broadleaf Deciduous Forest",
    5: "Broadleaf Deciduous Forest",
    6: "Mixed Forest",
    7: "T/S Shrubland",
    8: "Shrubland",
    9: "T/S Grassland",
    10: "Grassland",
    11: "Shrubland-Lichen-Moss",
    12: "Grassland-Lichen-Moss",
    13: "Barren-Lichen-Moss",
    14: "Wetland",
    15: "Cropland",
    16: "Barren",
    17: "Urban",
    18: "Water",
    19: "Snow/Ice",
}
classes = list(land_cover_types.values())
classes_dry = [name for name in land_cover_types.values() if name != "Water"]
classes_wet = ["Wetland", "Water"]
years = [2010]  # CEC land cover is single year 2010

# variables from vector ds I want in the output

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

## dynamic values
classes_dry_rn = [item.replace(" ", "_").replace("/", "_") for item in classes_dry]

base_path = Path(time_series_features_cec_doc_csv_pth)
stem = base_path.stem

parquet_out_norm_pth = base_path.parent / f"{stem}_norm.parquet"
parquet_out_time_series_features_pth = base_path.parent / f"{stem}_tsFeatures.parquet"
parquet_out_time_series_features_core_pth = base_path.parent / f"{stem}_core_tsFeatures.parquet"
parquet_out_time_series_features_short_pth = base_path.parent / f"{stem}_short_tsFeatures.parquet"
shp_out_time_series_features_core_pth = base_path.parent / f"{stem}_core_tsFeatures.gpkg"

## RUN for preliminary harmonized DOC with no buffers

extractTimeSeriesForLakes(
    doc_jn_catchment_pth,
    buffer_lengths,
    time_series_features_cec_doc_csv_pth,
    cec_30m_lc_pth,
    use_simplified_classes=False,
    classes=classes,
    years=years,
    # envelope_pth=above_lc_boreal_envelope_pth,
    join_index="sample_idx",
    n_workers=8,
    pth_lc_in_coarse=cec_30m_lc_rs250_pth,
)

print("DONE.")
