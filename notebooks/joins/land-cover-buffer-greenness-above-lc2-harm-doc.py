"""Run as Step L4b: after join_gswl_lakes.ipynb.
Independent of notebooks/joins/regrid_to_bawld_and_plot.ipynb

TODO:
* Fix auto-loading if running on large dataset?
"""

import os

import geopandas as gpd
import numpy as np

from land_cover.land_cover_change_buffer_from_csv import (
    extractTimeSeriesFeatures_above_boreal, extractTimeSeriesForLakes,
    normalizeTimeSeries_above_boreal, plotTimeSeries)
from land_cover.load import (
    above_lc_boreal_envelope_pth,
    above_lc_boreal_pth,
    doc_jn_catchment_pth,
    plot_dir,
    time_series_features_doc_parquet_pth,
)

# I/O

## in: for greennessx2 and Land Cover v2
pth_shp_in = doc_jn_catchment_pth  # lake polygons
pth_lc_in = above_lc_boreal_pth
# pth_lc_in_simp = "/Volumes/thebe/Wang-above-land-cover/ABoVE_LandCover_simplified.vrt"  # simplified 10-class landcover

## out
parquet_out_pth = time_series_features_doc_parquet_pth
plot_dir = os.path.join(plot_dir, "doc_harm_abovelc25")

## buffers, in order small -> large
buffer_lengths = [0]  # (90, 180) # in m # 90, 990 # 1350

use_simplified_classes = False
## classes for land cover (2025 dataset)
land_cover_types = {
    1: "Bare/Sparsely vegetated",
    2: "Deciduous Forest",
    3: "Evergreen Forest",
    4: "Mixed Forest",
    5: "Herb",
    6: "Shrub",
    7: "Water",
    8: "Wetland",
    9: "Ice/Snow",
}

classes = [
    "Bare/Sparsely vegetated",
    "Deciduous Forest",
    "Evergreen Forest",
    "Mixed Forest",
    "Herb",
    "Shrub",
    "Water",
    "Wetland",
    "Ice/Snow",
]
classes_dry = [
    "Bare/Sparsely vegetated",
    "Deciduous Forest",
    "Evergreen Forest",
    "Mixed Forest",
    "Herb",
    "Shrub",
    "Ice/Snow",
    "Wetland", # Note, I've included in both dry and wet
]
classes_wet = ["Wetland", "Water"]
years = np.arange(1986, 2020 + 1)

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

parquet_out_norm_pth = parquet_out_pth.replace(
    ".parquet", "_norm.parquet"
)  # e.g. /Volumes/thebe/ABoVE2021/Mapping/out/xlsx/ABOVE_coordinates_for_Ethan_10-19-21_jn_PADLakesVis_landCoverBuffers_norm.csv
parquet_out_time_series_features_pth = parquet_out_pth.replace(
    ".parquet", "_tsFeatures.parquet"
)  # e.g. /Volumes/thebe/ABoVE2021/Mapping/out/xlsx/ABOVE_coordinates_for_Ethan_10-19-21_jn_PADLakesVis_landCoverBuffers_tsFeatures.csv
parquet_out_time_series_features_core_pth = parquet_out_pth.replace(".parquet", "_core_tsFeatures.parquet")
parquet_out_time_series_features_short_pth = parquet_out_pth.replace(
    ".parquet", "_short_tsFeatures.parquet"
)
shp_out_time_series_features_core_pth = parquet_out_time_series_features_core_pth.replace(
    ".parquet", ".gpkg"
)

## RUN for preliminary harmonized DOC with no buffers

extractTimeSeriesForLakes(
    pth_shp_in,
    buffer_lengths,
    parquet_out_pth,
    pth_lc_in,
    classes=classes,
    years=years,
    envelope_pth=above_lc_boreal_envelope_pth,
    join_index="sample_idx",
    n_workers=4,
)
normalizeTimeSeries_above_boreal(
    parquet_out_pth, parquet_out_norm_pth, classes_wet, classes_dry, wetland_class="Wetland", index_class="sample_idx"
)

# plotTimeSeries(buffer_lengths, parquet_out_norm_pth, plot_dir)

extractTimeSeriesFeatures_above_boreal(
    parquet_out_norm_pth,
    years,
    classes_dry_rn,
    pth_shp_in,
    ds_specific_vars,
    parquet_out_time_series_features_pth,
    important_vars,
    parquet_out_time_series_features_core_pth,
    parquet_out_time_series_features_short_pth,
    join_index="sample_idx",
    grouped_classes=["Trees", "Shrub", "Wetland", "Herb", "Sparse"],
)
print("DONE.")
