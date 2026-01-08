"""Run as Step L4b: after join_gswl_lakes.ipynb.
Untested archive version that uses I/O for Wang above dataset v1 (core domain)
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from land_cover.land_cover_change_buffer_from_csv import (
    extractTimeSeriesFeatures,
    extractTimeSeriesForLakes,
    normalizeTimeSeries,
    plotTimeSeries,
)
from land_cover.load import (
    GLAKES_gswl_pth,
    above_lc_boreal_envelope_pth,
    above_lc_boreal_pth,
    bawld_join_gswl_abz_filtered_pth,
    greennessx2_albers_pth,
    loadBAWLD,
    loadGLAKES_GSWL,
    loadGreenness,
    time_series_features_csv_pth,
)
from land_cover.plotting import plot_choro_and_hist, reg_hexplot

# I/O

## I/O

## Switches (uncomment)
# use_simplified_classes=True
use_simplified_classes = False

## in: for Bogard and ABove landcover v1
pth_shp_in = "/Volumes/metis/ABOVE3/Tom/Selected_PLD_Lakes_2024-10-21/added_PLD/Efflux_Bogard_PLD_WBD.shp"  # lake polygons
pth_lc_in = "/Volumes/thebe/Wang-above-land-cover/ABoVE_LandCover_5km_buffer.vrt"
pth_lc_in_simp = "/Volumes/thebe/Wang-above-land-cover/ABoVE_LandCover_simplified.vrt"  # simplified 10-class landcover
pth_csv_in = "/Volumes/thebe/ABoVE2021/Mapping/ABOVE_coordinates_for_Ethan_10-19-21_mod.csv"  # native (edited) data format from Martin. Used to join in at end

## out
csv_out_pth = (
    "/Volumes/metis/ABOVE3/land_cover_joins/out/xlsx/"
    + os.path.basename(pth_shp_in)[:-4]
    + "_landCoverBuffers.csv"
)  # e.g. /Volumes/thebe/ABoVE2021/Mapping/out/xlsx/ABOVE_coordinates_for_Ethan_10-19-21_jn_PADLakesVis_landCoverBuffers.csv
# shp_projected_out_pth = pth_shp_in.replace('_geom.shp', '_albers_geom.shp')
plot_dir = "/Volumes/metis/ABOVE3/land_cover_joins/plots"

## buffers, in order small -> large
buffer_lengths = [180]  # (90, 180) # in m # 90, 990 # 1350

# classes for land cover
classes = [
    "Evergreen Forest",
    "Deciduous Forest",
    "Mixed Forest",
    "Woodland",
    "Low Shrub",
    "Tall Shrub",
    "Open Shrubs",
    "Herbaceous",
    "Tussock Tundra",
    "Sparsely Vegetated",
    "Fen",
    "Bog",
    "Shallows/littoral",
    "Barren",
    "Water",
]
classes_dry = [
    "Evergreen Forest",
    "Deciduous Forest",
    "Mixed Forest",
    "Woodland",
    "Low Shrub",
    "Tall Shrub",
    "Open Shrubs",
    "Herbaceous",
    "Tussock Tundra",
    "Sparsely Vegetated",
    "Fen",
    "Bog",
    "Barren",
]
classes_dry_rn = [
    item.replace(" ", "_").replace("/", "_") for item in classes_dry
]  # rename var too
classes_wet = ["Shallows/littoral", "Water"]
classes_simp = [
    "Evergreen Forest",
    "Deciduous Forest",
    "Shrubland",
    "Herbaceous",
    "Sparsely Vegetated",
    "Barren",
    "Fen",
    "Bog",
    "Shallows/littoral",
    "Water",
]
years = np.arange(1984, 2014 + 1)

ds_specific_vars = [  # For Martin OG dataset
    "latitude",
    "longitude",
    "Location",
]

# ds_specific_vars = [ # For Efflux lakes
#     'Lat_DD',
#     'Lon_DD',
#     'AvgOfTempC',
#     'AvgOfpH',
#     'AvgOfALKum',
#     'AvgOfpCO2',
#     'StDevOfpCO',
# ]

ds_specific_vars = [  # For Efflux and Bogard lakes
    "Lat_DD",
    "Lon_DD",
    "AvgOfTempC",
    "AvgOfpH",
    "AvgOfALKum",
    "AvgOfpCO2",
    "StDevOfpCO",
    "Name",
    "Reference",
    # 'SIMILAR',
    "mean_bound",  # mean_bound_dist
    "max_bound_",
]

important_vars = [  # for
    "Area_m2",
    "Perim_m2",
    "Total_inun_2014",
    "Trees_pct_2014",
    "Shrubs_pct_2014",
    "Wetlands_pct_2014",
    "Graminoid_pct_2014",
    "Sparse_pct_2014",
    "Littorals_pct_2014",
    "Littoral_wetland_pct_2014",
    "Total_inun_RSD",
    "Total_inun_dyn_pct",
    "Hi_water_yr",
    "Lo_water_yr",
    "Dominant_veg_2014",
    "Dominant_veg_group_2014",
    "SDF",
    "Perim_area_ratio",
    "Total_inun_change",
    "Total_inun_trend",
]

## dynamic values
if use_simplified_classes:
    pth_lc_in = pth_lc_in_simp
    classes = classes_simp
    csv_out_pth = csv_out_pth.replace(".csv", "_simpl_classes.csv")

xlsx_out_norm_pth = csv_out_pth.replace(
    ".csv", "_norm.csv"
)  # e.g. /Volumes/thebe/ABoVE2021/Mapping/out/xlsx/ABOVE_coordinates_for_Ethan_10-19-21_jn_PADLakesVis_landCoverBuffers_norm.csv
xlsx_out_time_series_features_pth = csv_out_pth.replace(
    ".csv", "_tsFeatures.csv"
)  # e.g. /Volumes/thebe/ABoVE2021/Mapping/out/xlsx/ABOVE_coordinates_for_Ethan_10-19-21_jn_PADLakesVis_landCoverBuffers_tsFeatures.csv
xlsx_out_time_series_features_core_pth = xlsx_out_time_series_features_pth.replace(
    "_tsFeatures.csv", "_core_tsFeatures.csv"
)
csv_out_time_series_features_short_pth = xlsx_out_time_series_features_pth.replace(
    "_tsFeatures", "_short_tsFeatures"
)
shp_out_time_series_features_core_pth = xlsx_out_time_series_features_core_pth.replace(
    "xlsx", "shp"
)

## dynamic values
classes_dry_rn = [item.replace(" ", "_").replace("/", "_") for item in classes_dry]
if use_simplified_classes:
    pth_lc_in = pth_lc_in_simp
    classes = classes_simp
    csv_out_pth = csv_out_pth.replace(".csv", "_simpl_classes.csv")

csv_out_norm_pth = csv_out_pth.replace(
    ".csv", "_norm.csv"
)  # e.g. /Volumes/thebe/ABoVE2021/Mapping/out/xlsx/ABOVE_coordinates_for_Ethan_10-19-21_jn_PADLakesVis_landCoverBuffers_norm.csv
csv_out_time_series_features_pth = csv_out_pth.replace(
    ".csv", "_tsFeatures.csv"
)  # e.g. /Volumes/thebe/ABoVE2021/Mapping/out/xlsx/ABOVE_coordinates_for_Ethan_10-19-21_jn_PADLakesVis_landCoverBuffers_tsFeatures.csv
csv_out_time_series_features_core_pth = csv_out_pth.replace(".csv", "_core_tsFeatures.csv")
csv_out_time_series_features_short_pth = csv_out_pth.replace(".csv", "_short_tsFeatures.csv")
shp_out_time_series_features_core_pth = csv_out_time_series_features_core_pth.replace("csv", "gpkg")

## RUN

extractTimeSeriesForLakes(
    pth_shp_in,
    buffer_lengths,
    csv_out_pth,
    pth_lc_in,
    classes=classes,
    years=years,
    join_index="Lake_id_glakes",
    n_workers=7,
)
normalizeTimeSeries(
    csv_out_pth, csv_out_norm_pth, classes_wet, classes_dry, wetland_class="Wetland"
)

plotTimeSeries(buffer_lengths, csv_out_norm_pth, plot_dir)

extractTimeSeriesFeatures(
    csv_out_norm_pth,
    years,
    classes_dry_rn,
    pth_shp_in,
    ds_specific_vars,
    csv_out_time_series_features_pth,
    important_vars,
    csv_out_time_series_features_core_pth,
    csv_out_time_series_features_short_pth,
    shp_out_time_series_features_core_pth,
    join_index="Lake_id_glakes",
    grouped_classes=["Trees", "Shrub", "Wetland", "Herb", "Sparse"],
)
print("DONE.")
