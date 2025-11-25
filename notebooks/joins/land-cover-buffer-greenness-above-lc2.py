import os

import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from land_cover.land_cover_change_buffer_from_csv import (
    extractTimeSeriesForLakes,
    extractTimeSeriesFeatures_above_boreal,
    normalizeTimeSeries_above_boreal,
    plotTimeSeries,
)
from land_cover.load import (
    bawld_join_gswl_abz_filtered_pth,
    greennessx2_albers_pth,
    loadBAWLD,
    loadGLAKES_GSWL,
    loadGreenness,
    plot_dir,
    above_lc_boreal_pth,
    above_lc_boreal_envelope_pth,
)
from land_cover.plotting import plot_choro_and_hist, reg_hexplot

# I/O

## in: for greennessx2 and Land Cover v2
pth_shp_in = greennessx2_albers_pth  # lake polygons
pth_lc_in = above_lc_boreal_pth
# pth_lc_in_simp = "/Volumes/thebe/Wang-above-land-cover/ABoVE_LandCover_simplified.vrt"  # simplified 10-class landcover

## out
csv_out_pth = (
    "/Volumes/metis/ABOVE3/land_cover_joins/out/glakes_green_abovelc25/xlsx/"
    + os.path.basename(pth_shp_in).split(".")[0]
    + "_landCoverBuffers.csv"
)  # e.g. /Volumes/thebe/ABoVE2021/Mapping/out/xlsx/ABOVE_coordinates_for_Ethan_10-19-21_jn_PADLakesVis_landCoverBuffers.csv
# shp_projected_out_pth = pth_shp_in.replace('_geom.shp', '_albers_geom.shp')
plot_dir = os.path.join(plot_dir, "glakes_green_abovelc25")

## buffers, in order small -> large
buffer_lengths = [180]  # (90, 180) # in m # 90, 990 # 1350

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
    "Lake_id_glakes",
    "Area_bound_glakes",
    "Area_PW_glakes",
    "Continent_glakes",
    "Lat_glakes",
    "Lon_glakes",
    "GFed_flag_glakes",
    "PFed_flag_glakes",
    "Endo_flag_glakes",
    "Rser_flag_glakes",
    "Shape_Leng_glakes",
    "Shape_Area_glakes",
    "areaP1",
    "areaP2",
    "areaP3",
    "NDVI8499",
    "NDVI0010",
    "NDVI1121",
    "vegeP1",
    "validP1",
    "vegeP2",
    "validP2",
    "vegeP3",
    "validP3",
    "occP1",
    "occP2",
    "occP3",
    "area_1984_1999_wm",
    "area_2000_2009_wm",
    "area_2010_2019_wm",
    "LEV_p1",
    "LEV_p2",
    "LEV_p3",
    "LEV_p13ain",
    "LEV_p13rin",
    "LEV_p23ain",
    "LEV_p23rin",
    "LEV_p13occ_ain",
    "LEV_p23occ_ain",
    "LEV_p13in",
    "max_bound_dist",
    "mean_bound_dist",
    "Hylak_id",
    "Lake_name_hylak",
    "Country_hylak",
    "Continent_hylak",
    "Poly_src_hylak",
    "Lake_type_hylak",
    "Grand_id_hylak",
    "Lake_area_hylak",
    "Shore_len_hylak",
    "Shore_dev_hylak",
    "Vol_total_hylak",
    "Vol_res_hylak",
    "Vol_src_hylak",
    "Depth_avg_hylak",
    "Dis_avg_hylak",
    "Res_time_hylak",
    "Elevation_hylak",
    "Slope_100_hylak",
    "Wshd_area_hylak",
    "Pour_long_hylak",
    "Pour_lat_hylak",
    "sen_slope",
    "mann_kendall_trend",
    "trend_significance",
    "b2_mean",
    "b2_stddev",
    "hylak_count",
    "Dmax_est_PAVEW_m",
    "Dmax_use_m",
    "dyn_ratio_glak_hylak",
    "depth_ratio_globath",
]

# variables from both datasets I want in the core output for easy viewing
important_vars = [
    "Lake_id_glakes",
    "Area_PW_glakes",
    "areaP1",
    "areaP2",
    "areaP3",
    "NDVI8499",
    "NDVI0010",
    "NDVI1121",
    "vegeP1",
    "validP1",
    "vegeP2",
    "validP2",
    "vegeP3",
    "validP3",
    "occP1",
    "occP2",
    "occP3",
    "area_1984_1999_wm",
    "area_2000_2009_wm",
    "area_2010_2019_wm",
    "LEV_p1",
    "LEV_p2",
    "LEV_p3",
    "LEV_p13ain",
    "LEV_p13rin",
    "LEV_p23ain",
    "LEV_p23rin",
    "LEV_p13occ_ain",
    "LEV_p23occ_ain",
    "LEV_p13in",
    "max_bound_dist",
    "mean_bound_dist",
    "Lake_area_hylak",
    "Shore_len_hylak",
    "Shore_dev_hylak",
    "Vol_total_hylak",
    "Depth_avg_hylak",
    "Elevation_hylak",
    "Slope_100_hylak",
    "Wshd_area_hylak",
    "sen_slope",
    "mann_kendall_trend",
    "trend_significance",
    "b2_mean",
    "b2_stddev",
    "hylak_count",
    "Dmax_est_PAVEW_m",
    "Dmax_use_m",
    "dyn_ratio_glak_hylak",
    "depth_ratio_globath",
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
]

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
csv_out_time_series_features_core_pth = csv_out_time_series_features_pth.replace(
    "_tsFeatures.csv", "_core_tsFeatures.csv"
)
shp_out_time_series_features_core_pth = csv_out_time_series_features_core_pth.replace("csv", "gpkg")

## RUN

# extractTimeSeriesForLakes(
#     pth_shp_in,
#     buffer_lengths,
#     csv_out_pth,
#     pth_lc_in,
#     classes=classes,
#     years=years,
#     envelope_pth=above_lc_boreal_envelope_pth,
#     join_index="Lake_id_glakes",
#     n_workers=7,
# )
# normalizeTimeSeries_above_boreal(
#     csv_out_pth, csv_out_norm_pth, classes_wet, classes_dry, wetland_class="Wetland"
# )

# plotTimeSeries(buffer_lengths, csv_out_norm_pth, plot_dir)

extractTimeSeriesFeatures_above_boreal(
    csv_out_norm_pth,
    years,
    classes_dry_rn,
    pth_shp_in,
    ds_specific_vars,
    csv_out_time_series_features_pth,
    important_vars,
    csv_out_time_series_features_core_pth,
    join_index="Lake_id_glakes",
    grouped_classes=["Trees", "Shrub", "Wetland", "Herb", "Sparse"],
)
print("DONE.")
