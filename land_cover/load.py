""" Loads RS and GIS datasets"""
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

gee_table_pth = "/Volumes/metis/ABOVE3/Tom/gee_input/gee_cleaned_sample_data_2025-03-06.csv" # old
stolpmann_indexed_pth = (
    "/Volumes/metis/ABOVE3/Stolpmann21-DOC-permafrost/edk_out/Stolpmann21_idx.gpkg"
)
bogard_output_path_raw = (
    "/Volumes/metis/ABOVE3/Digitizing/gee_asset_download/merged_asset_tables_20250812.shp"
)
dranga_shoreline_pth = "/Volumes/metis/Datasets/Dranga-2017/edk_out/shp/dranga17_shorelines.gpkg"
efflux_bogard_dict = {
    'AvgOfpCO2':'pco2uatm',
    'Lat_DD': 'lat',
    'Lon_DD': 'long'}
first_columns = ['AvgOfpCO2', 'Lat_DD', 'Lon_DD', 'Area_m2', 'Perim_m2', 'mean_bound',
       'max_bound_', 'Perim_area_ratio', 'SDF', 'AvgOfpH', 'AvgOfALKum', 'AvgOfTempC']
cols_to_drop = ['Lake', 'Lat_DD', 'Lon_DD', 'Total_inun_trend', 'Name', 'Reference', 'Dominant_veg_2014',
       'Dominant_veg_group_2014', 'StDevOfpCO', 'Total_inun_2014']
plot_dir = "/Volumes/metis/ABOVE3/fig"
kurek_bounds = [-156.8973100000000045, 58.3921899999999994, -111.0319899999999933, 71.2416300000000007]

# Current paths
GLAKES_filtered_fix_pth = Path("/Volumes/metis/Datasets/GLAKES/out/GLAKES_filtered_fix.shp")
GLAKES_MA_pth = Path("/Volumes/metis/Datasets/Liu_aq_veg/figshare/MA.csv")
GLAKES_NDVI_pth = Path("/Volumes/metis/Datasets/Liu_aq_veg/figshare/NDVI.csv")
GLAKES_VO_pth = Path("/Volumes/metis/Datasets/Liu_aq_veg/figshare/VO.csv")
GLAKES_filtered_fix_aqveg_pth = Path(
    "/Volumes/metis/Datasets/Liu_aq_veg/figshare/v4/25012091/edk_out/GLAKES_filtered_fix_aqveg.gpkg"
)

GLAKES_filtered_fix_aqveg_dist_pth = Path(
    "/Volumes/metis/Datasets/Liu_aq_veg/figshare/v4/25012091/edk_out/GLAKES_filtered_fix_aqveg_dist.gpkg"
)
greennessx2_albers_pth = Path(
    "/Volumes/metis/Datasets/Liu_aq_veg/figshare/v4/25012091/edk_out/join_hl_greenness/greennessx2.gpkg"
)
greennessx2_albers_pth = Path(
    "/Volumes/metis/Datasets/Liu_aq_veg/figshare/v4/25012091/edk_out/join_hl_greenness/greennessx2_albers.gpkg"
)

# greennessx2_pth + Gudasz GSWL morphometry
GLAKES_gswl_pth = Path(
    "/Volumes/metis/Datasets/Liu_aq_veg/figshare/v4/25012091/edk_out/GLAKES_gswl.gpkg"
)
# clipped to na_abz
GLAKES_gswl_abz_pth = Path(
    "/Volumes/metis/Datasets/Liu_aq_veg/figshare/v4/25012091/edk_out/GLAKES_gswl_na_abz.gpkg"
)
GLAKES_tpcat_pth = Path(
    "/Volumes/metis/Datasets/Liu_aq_veg/figshare/v4/25012091/edk_out/GLAKES_tcat.gpkg"
)
# land cover
above_lc_boreal_pth = "/Volumes/metis/Datasets/Hu_Wang_ABOVE_landcover_2025/Boreal_LandCoverClasses_AK_CA/data/ABoVE_LandCover_boreal.vrt"
above_lc_boreal_envelope_pth = "/Volumes/metis/Datasets/Hu_Wang_ABOVE_landcover_2025/Boreal_LandCoverClasses_AK_CA/data/edk_out/above_lc_boreal_envelope.shp"

## Working paths
# filtered based on some criteria
GLAKES_gswl_abz_filtered_pth = Path(
    "/Volumes/metis/Datasets/Liu_aq_veg/figshare/v4/25012091/edk_out/GLAKES_gswl_na_abz_filtered.gpkg"
)
bawld_join_gswl_abz_filtered_pth = Path(
    "/Volumes/metis/ABOVE3/other_outputs/BAWLD_GLAKES_gswl_filtered.gpkg"
)
doc_jn_catchment_pth = Path("/Volumes/metis/ABOVE3/Digitizing/catchments/doc_jn_catchments.gpkg")
time_series_features_doc_parquet_pth = (
    "/Volumes/metis/ABOVE3/Digitizing/catchments/land-cover/"
    + os.path.basename(doc_jn_catchment_pth).split(".")[0]
    + "_landCoverBuffers.parquet"
)

# Land cover time features
# Variation: _norm, _core, _tsFeatures, _short_tsFeatures
# _landCoverBuffers and *_norm have data for each year.
# _short_tsFeatures has just time-series features
# *_tsFeatures adds lake database data from gswl, etc.

time_series_features_csv_pth = (
    "/Volumes/metis/ABOVE3/land_cover_joins/out/glakes_green_abovelc25/xlsx/"
    + os.path.basename(greennessx2_albers_pth).split(".")[0]
    + "_landCoverBuffers.csv"
)  # e.g. /Volumes/thebe/ABoVE2021/Mapping/out/xlsx/ABOVE_coordinates_for_Ethan_10-19-21_jn_PADLakesVis_landCoverBuffers.csv
# shp_projected_out_pth = pth_shp_in.replace('_geom.shp', '_albers_geom.shp')


## Archived paths
old_GLAKES_filtered_fix_aqveg_dist_pth = Path(
    "/Volumes/metis/Datasets/Liu_aq_veg/figshare/original-private-repo/edk_out/GLAKES_filtered_fix_aqveg_dist.gpkg"
)

## Outputs
aleb_landcover_greenness_spatial = "/Volumes/metis/ABOVE3/land_cover_joins/out/shp/Efflux_Bogard_PLD_WBD_landCoverBuffers_core_tsFeatures_greenx2.gpkg"

def loadEfflux():
    return gpd.read_file('/Volumes/metis/ABOVE3/LAKESHAPE/effluxlakes.shp')


def loadEffluxShp():
    gdf_jn_PLD = gpd.read_file('/Volumes/metis/ABOVE3/Tom/Selected_PLD_Lakes_2024-10-21/EffluxLakes_selected_PLDLakes_2024-10-11.shp')
    df = pd.read_excel('/Volumes/metis/ABOVE3/Tom/PrelimLakeMatchupData_2024-10-21.xlsx', sheet_name='Measurements').query("Name == 'EffluxLakes'")
    gdf = gdf_jn_PLD.merge(df, on='lake_id', how='inner') #, validate='1:1')
    gdf = gdf.groupby('lake_id').first().reset_index() # hot fix to remove dups
    gdf.crs = gdf_jn_PLD.crs
    return gdf


def loadBogardMapShp(ABOVE_region=True, region=None):
    '''Loads all lakes with matchup, even if in Europe'''
    if ABOVE_region:
        bbox = (-170, 51, -127 , 72) # W NA
    else:
        if region=="WH":
            bbox = (-180, 1, -13, 90)  # W NA
        else:
            bbox = None
    gdf_jn_PLD = gpd.read_file('/Volumes/metis/ABOVE3/Tom/Selected_PLD_Lakes_2024-10-21/BogardMapLakes_selected_PLDLakes_2024-10-11.shp', bbox=bbox)
    # gdf_jn_PLD.rename(columns={v: k for k, v in efflux_bogard_dict.items()}, inplace=True)
    df = pd.read_excel('/Volumes/metis/ABOVE3/Tom/PrelimLakeMatchupData_2024-10-21.xlsx', sheet_name='Measurements').query("Name == 'BogardMapLakes'")
    gdf = gdf_jn_PLD.merge(df, on='lake_id', how='inner') #, validate='1:1')
    for key in efflux_bogard_dict.keys():
        gdf[key] = gdf[key].fillna(gdf[efflux_bogard_dict[key]])
    gdf = gdf.groupby('lake_id').first().reset_index() # hot fix to remove dups
    gdf.crs = gdf_jn_PLD.crs
    return gdf    


def loadBogardSuppl():
    """Western hemisphere only, data direct from paper SI"""
    wd = Path("/Volumes/metis/ABOVE3/Bogard_suppl_data")
    filename = "Bogard19_ESM_alldata_wh"
    out_dir = wd / "edk_out"
    csv_in_pth = out_dir / f"{filename}.csv"
    df = pd.read_csv(csv_in_pth)
    return df, out_dir, filename


def _parse_google_earth_text(df:pd.DataFrame):
    "Modified in place"
    # extract Sample No, Latitude, Longitude from PopupInfo
    df["Sample No"] = df["PopupInfo"].str.extract(r"Sample No:\s*([A-Za-z0-9.-]+)").astype(float)
    df["Latitude"]  = df["PopupInfo"].str.extract(r"Latitude:\s*([0-9.+-]+)").astype(float)
    df["Longitude"] = df["PopupInfo"].str.extract(r"Longitude:\s*([0-9.+-]+)").astype(float)
    df.drop(columns="PopupInfo", inplace=True)


def load_prelim_matchup_data():
    # has lat/lon, TopoCat lake_id
    df_matchup_data = pd.read_excel(
        "/Volumes/metis/ABOVE3/Tom/PrelimLakeMatchupData_2024-10-21.xlsx",
        sheet_name="Measurements",
        usecols=["PopupInfo", "Name", "lake_id"],
    ).query("Name == 'LitReviewLakes'")
    _parse_google_earth_text(df_matchup_data)
    df_matchup_data.dropna(subset="Longitude", inplace=True)  # keeps only Dranga et al. paper
    return df_matchup_data


def load_lit_review_lakes_jn_pld(bbox=None):
    "Partial dataset"
    return gpd.read_file(
        "/Volumes/metis/ABOVE3/Tom/Selected_PLD_Lakes_2024-10-21/LitReviewLakes_selected_PLDLakes_2024-10-11.shp",
        bbox=bbox,
    ).dropna(subset="lake_id")[["lake_id", "geometry"]]


def loadKurek():
    dataset_path = (
        "/Volumes/metis/ABOVE3/Kurek_GBC22_data/out/Kurek_ABoVE Lakes DOM_GBC_2023_Table S1.csv"
    )
    shorelines_path = "/Volumes/metis/ABOVE3/Kurek_GBC22_data/out/shorelines/ABOVE_coordinates_for_Ethan_10-19-21_geom.shp"

    df_csv = pd.read_csv(dataset_path)
    df_csv.columns = df_csv.columns.str.strip()
    gdf_shp = gpd.read_file(shorelines_path)
    merged = df_csv.merge(
        gdf_shp, left_on="Match_name", right_on="Sample_nam", how="left", indicator=False
    )
    # merged.rename(columns={"Note": "Digitizing note"})
    merged.rename(
        columns=dict(
            zip(gdf_shp.columns, ["dig_" + col for col in gdf_shp.columns if col != "geometry"])
        ),
        inplace=True,
    )
    merged = gpd.GeoDataFrame(merged, crs=gdf_shp.crs)
    merged["lake_area_km2"] = merged.area / 1e6
    return merged


def loadDranga17():
    """Note: companion csv with min detection level and og excel with full field descriptions"""
    wd = Path("/Volumes/metis/Datasets/Dranga-2017")
    filename = "as-2017-0039suppla"
    out_dir = wd / "edk_out"
    csv_in_pth = out_dir / f"{filename}.csv"
    df = pd.read_csv(csv_in_pth)
    df["DOC"] = df["DOC"].replace(' ', np.nan).astype(float)
    return df, out_dir, filename


def loadStolpmann21(region=None):
    gdf = gpd.read_file(
        "file:///Volumes/metis/ABOVE3/Stolpmann21-DOC-permafrost/Stolpmann-etal_2021_shapefile.zip"
    )
    if region=="na":
        # Longitude < 0
        gdf = gdf[gdf.Longitude < 0]
    return gdf


def loadShahabinia25():
    return pd.read_csv(
        "/Volumes/metis/ABOVE3/Shahabinia25/Fulldataset_LakePulse_DOM_PARAFAC_FTMS.csv",
        # index_col="sample.no",
    )


def loadWBD():
    '''Note: bbox is for AK'''
    return gpd.read_file('/Volumes/thebe/Other/Feng-High-res-inland-surface-water-tundra-boreal-NA/edk_out/fixed_geoms/WBD.shp', engine='pyogrio', bbox = (-170, 51, -125 , 72)) # bbox for NA


def load_gee_input(
    path="/Volumes/metis/ABOVE3/Tom/gee_input/updated/gee_cleaned_sample_data_2025-07-09_suppl.csv",
    source="LitReviewLakes",
    western_hem=True,
):
    """ this is the input file Tom used for GEE digitizing, and it indicates whether a lake has
    been matched to PLD"""
    usecols = ["CurrentlyM", "Latitude", "Longitude", "SampleUID", "Source", "WesternHem"]

    df = pd.read_csv(path, usecols=usecols)

    # df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    # df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    df = df.dropna(subset=["Latitude", "Longitude"])
    if source:
        df = df[df["Source"] == source]
    if western_hem:
        df = df[df["WesternHem"] == True]
    # gdf = gpd.GeoDataFrame(
    #     df,
    #     geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
    #     crs="EPSG:4326",
    # )

    return df

def clean_gee_results(gdf:gpd.GeoDataFrame):
    """remove duplicates by lake index `SampleUID` based on order of priority of the type of match
    "savetype": "New polygon" is preferred over "New polygon - manual classification" over 
    "MatchedManuallyDrawnPolygon" over "MatchedManuallyDrawnPolygonThisSession" over MatchedPLD
    """
    savetype_priority = {
        "New polygon": 1,
        "New polygon - manual classification": 2,
        "MatchedManuallyDrawnPolygon": 3,
        "MatchedManuallyDrawnPolygonThisSession": 4,
        "MatchedPLD": 5,
    }
    gdf = gdf.copy()
    gdf["savetype_priority"] = gdf["savetype"].map(savetype_priority)
    gdf = gdf.sort_values("savetype_priority")
    gdf = gdf.drop_duplicates(subset="SampleUID", keep="first")
    gdf = gdf.drop(columns="savetype_priority")
    return gdf


def load_gee_digitized(
    path="/Volumes/metis/ABOVE3/Digitizing/gee_asset_download/Tom/merged_asset_tables_20251118.shp",
    prefix="LRL",
):
    """ tom and Andy digitized these lakes in GEE"""
    gdf = gpd.read_file(path)
    gdf = gdf[gdf["savetype"] != "Mismatch"]
    gdf = gdf[gdf["sampleUID"].str.startswith(prefix)]
    gdf[gdf.pld_match == -99999.31415] = np.nan
    return clean_gee_results(gdf.rename(columns={"sampleUID": "SampleUID"}))


def loadLandCoverJoined():
    return pd.read_excel('/Volumes/metis/ABOVE3/land_cover_joins/out/xlsx/Efflux_Bogard_PLD_WBD_landCoverBuffers_core_tsFeatures.xlsx')


def loadLandCoverJoinedShp():
    df = pd.read_excel(
        "/Volumes/metis/ABOVE3/land_cover_joins/out/xlsx/Efflux_Bogard_PLD_WBD_landCoverBuffers_core_tsFeatures.xlsx"
    )
    gdf = gpd.read_file(
        "/Volumes/metis/ABOVE3/land_cover_joins/out/shp/Efflux_Bogard_PLD_WBD_landCoverBuffers_core_tsFeatures.shp"
    )
    return gpd.GeoDataFrame(df, geometry=gdf.geometry, crs=gdf.crs)


def sortColumns(df, order=first_columns):
    for col in order[::-1]:
        if col in df.columns:
            df.insert(0, col, df.pop(col))
    return


def dropColumns(df, cols=cols_to_drop):
    return df[[col for col in df.columns if col not in cols_to_drop]]


def loadGLAKES():
    gdf = gpd.read_file(GLAKES_filtered_fix_pth)
    old_names = gdf.columns.drop("geometry")
    new_names = [name + "_glakes" for name in old_names]
    gdf = gdf.rename(columns=dict(zip(old_names, new_names)))
    return gdf


def loadLiu():
    gdf = gpd.read_file(GLAKES_filtered_fix_aqveg_pth)
    return gdf


def loadKuhnGreenness():
    hydrolakes_shp = (
        "/Volumes/thebe/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp"
    )
    kuhn_txt = "/Volumes/metis/Datasets/Kuhn-lake-greenness/ABoVE_GrowingSeason_Lake_Color_1866/data/trends_1984_2019_landsat_ABoVE_lake_greenness.txt"

    gdf_hl = gpd.read_file(hydrolakes_shp)
    old_names = gdf_hl.columns.drop("geometry")
    new_names = [name + "_hylak" for name in old_names]
    gdf_hl = gdf_hl.rename(columns=dict(zip(old_names, new_names)))
    gdf_hl = gdf_hl.rename(columns={"Hylak_id_hylak": "Hylak_id"})
    df_kuhn = pd.read_csv(kuhn_txt).drop(columns=["longitude", "latitude"])

    df_kuhn = df_kuhn.rename(columns={"hylak_id": "Hylak_id"})
    merged = gdf_hl.merge(df_kuhn, on="Hylak_id", how="inner")
    return merged


def loadGreenness(bounds=None):
    """Loads working file with Liu and Khun greenness to a custom spatial domain and field names

    One domain could be NA and Scandinavia N of 45 degrees

    Kuhn fields: continent,country,hylak_id,latitude,longitude,sen_slope,mann_kendall_trend,trend_significance,b2_mean,b2_stddev
    """
    engine=None
    if bounds=='kurek':
        bounds = kurek_bounds
        engine='fiona'
    else:
        bounds=None
    gdf = gpd.read_file(
        greennessx2_albers_pth,
        bounds=bounds,
        engine=engine,
    )
    gdf.rename(
        columns=dict(
            zip(
                [
                    "trends_198",
                    "trends_1_1",
                    "trends_1_2",
                    "trends_1_3",
                    "trends_1_4",
                    "Liu MA_are",
                    "Liu MA_a_1",
                    "Liu MA_a_2",
                ],
                [
                    "green_sen_slope",
                    "green_mann_kendall_trend",
                    "green_trend_significance",
                    "green_b2_mean",
                    "green_b2_stddev",
                    "MA_p1",
                    "MA_p2",
                    "MA_p3",
                ],
            )
        ),
        inplace=True,
    )
    return gdf


def loadGlobathy():
    df = pd.read_csv(
        "/Volumes/thebe/Other/Khazaei-GLOBathy/GLOBathy_basic_parameters/GLOBathy_basic_parameters(ALL_LAKES).csv"
    ).drop(
        columns=[
            "Lake_name",
            "Country",
            "Pour_long",
            "Pour_lat",
            "HYBAS_ID_LVL1",
            "HYBAS_ID_LVL2",
            "HYBAS_ID_LVL3",
            "Dmax_box_m",
            "Dmax_cone_m",
            "Dmax_prism_m",
            "Dmax_ellip_m",
            "Dmax_est_PA_m",
        ]
    )
    return df


# gdf = loadEffluxShp()
# gdf = loadWBD()
# loadBogardMapShp()
# pass


def loadGSWL(ABOVE_region=False):
    gdf = gpd.read_file(
        "/Volumes/metis/Datasets/Gudasz-2025/Data/lake_morphometry/shp/GSWL.gpkg",
    )
    gdf["area_km2"] = gdf["area"] / 1e6
    gdf["dyn_ratio"] = np.sqrt(gdf.area_km2) / gdf.Zmean
    gdf["depth_ratio"] = gdf.Zmean / gdf.Zmax

    # compute area in km^2 from gswl and matching criterion (within 8% of Area_PW)
    gdf.drop(columns="area", inplace=True)

    # rename namespace vars by replacing columns `old_names` with `new_names`
    old_names = gdf.columns.drop("geometry")
    new_names = [name + "_gswl" for name in old_names]
    gdf = gdf.rename(columns=dict(zip(old_names, new_names)))
    return gdf


def loadOlson(region="na_abz"):
    """Olson terrestrial ecosystems of the world"""
    gdf = gpd.read_file(
        "/Volumes/thebe/Other/Olson2001TerrEcosWorld/TerrestrialEcos.zip",
    )
    if region == "na_abz":
        gdf = gdf[(np.isin(gdf.BIOME, [6, 11])) & (gdf.REALM == "NA")]
    return gdf


# def loadGLAKES_GSWL(region="na_abz", filter_matches=False):
# """ Slow because uses fiona"""
#     if region=="na_abz":
#         mask = loadOlson()
#     else:
#         mask = None
#     gdf = gpd.read_file(
#         GLAKES_gswl_pth,
#         mask=mask
#     )
#     if filter_matches:
#         gdf = gdf[gdf.match_gswl == 1]
#     return gdf


# def loadGLAKES_GSWL(region="na_abz", filter_matches=False):
#     gdf = gpd.read_file(
#         GLAKES_gswl_pth,
#         # mask=mask
#     )
#     if filter_matches:
#         gdf = gdf[gdf.match_gswl == 1]
#     if region == "na_abz":
#         olson = loadOlson()
#         mask = gpd.GeoDataFrame(geometry=olson.union_all(), crs=olson.crs).to_crs(gdf.crs)
#         gdf = gdf[gdf.geometry.within(mask)]
#     return gdf


def loadGLAKES_GSWL(region="na_abz", filter_matches=False, force_reload=True):
    """Use to save GLAKES_gswl_abz_pth for ease of loading (saves 40 sec)"""
    if region=="na_abz" and GLAKES_gswl_abz_pth.exists() and not force_reload:
        print(f"Read existing abz file: {GLAKES_gswl_abz_pth}")
        return gpd.read_file(GLAKES_gswl_abz_pth)

    # Otherwise, load full dataset
    gdf = gpd.read_file(GLAKES_gswl_pth)
    if filter_matches:
        gdf = gdf[gdf.match_gswl == 1]

    if region == "na_abz":            
        mask = loadOlson()
        # ensure GeoDataFrame and matching CRS
        mask_gdf = gpd.GeoDataFrame(geometry=mask.to_crs(gdf.crs).geometry)

        # (optional) fix invalid polygons that can stall predicates
        # mask_gdf["geometry"] = mask_gdf.geometry.buffer(0)

        # fast: spatial index + predicate, avoids expensive union_all
        hits = gpd.sjoin(gdf[["geometry"]], mask_gdf[["geometry"]], predicate="within", how="inner")
        gdf = gdf.loc[hits.index.unique()]
        if not force_reload:
            # Rewrites to speed up loading next time (currently, this will never execute)
            gdf.to_file(GLAKES_gswl_abz_pth)
            print(f"Overwrote abz file: {GLAKES_gswl_abz_pth}")
    if "index_right" in gdf.columns:
        gdf.drop(columns="index_right", inplace=True)
    return gdf


def loadBAWLD():
    """Includes my LEV estimate"""
    gdf = (
        gpd.read_file(
            "/Volumes/thebe/Other/Kuhn-olefeldt-BAWLD/BAWLD/edk_out/joined_lev/BAWLD_V1_LEV_v30.shp"
        )
        .rename(
            columns={
                "LEV_MEAN_k": "LEV_MEAN_km2_k23",
                "LEV_MAX_km": "LEV_MAX_km2_k23",
                "LEV_MIN_km": "LEV_MIN_km2_k23",
                "LEV_MEAN_f": "LEV_MEAN_frac_k23",
                "LEV_MAX_fr": "LEV_MAX_frac_k23",
                "LEV_MIN_fr": "LEV_MIN_frac_k23",
                "LEV_MEAN_g": "LEV_MEAN_grid_frac_k23",
                "LEV_MAX_gr": "LEV_MAX_grid_frac_k23",
                "LEV_MIN_gr": "LEV_MIN_grid_frac_k23",
                "est_mg_m2_": "est_mg_m2_day_k23",
                "est_g_day": "est_g_day_k23",
                "lake_count": "lake_count_k23",
                "Cell_ID": "Cell_ID_bawld",
                "Long": "Long_bawld",
                "Lat": "Lat_bawld",
                "Area_Pct": "Area_Pct_bawld",
                "Shp_Area": "Shp_Area_bawld",
            }
        )
    )
    gdf.drop(columns=["d_counting", "d_counti_1"], inplace=True)
    gdf.drop(
        columns=[c for c in gdf.columns if (c.endswith("_L")) or (c.endswith("_H"))], inplace=True
    )
    gdf[
        [
            "LEV_MEAN_km2_k23",
            "LEV_MAX_km2_k23",
            "LEV_MIN_km2_k23",
            "LEV_MEAN_frac_k23",
            "LEV_MAX_frac_k23",
            "LEV_MIN_frac_k23",
            "LEV_MEAN_grid_frac_k23",
            "LEV_MAX_grid_frac_k23",
            "LEV_MIN_grid_frac_k23",
        ]
    ] *= 100
    return gdf


def load_regrid_BAWLD():
    return gpd.read_file(bawld_join_gswl_abz_filtered_pth)


def load_reconstruct_time_series_above_boreal():
    "All outputs are clipped to 3 decimal places for float csv variables. This function loads in"
    "the underlying lake data set and joins to derived land cover variables, for which 3 decimals is fine"
    "Also, the default time_series_features_csv_pth, in addition to lacking precision, lacks GSWL vars"
    "above_boreal refers to above landcover v2 boreal dataset"
    from land_cover.utils import pct_change

    csv_out_time_series_features_short_pth = time_series_features_csv_pth.replace(
        ".csv", "_short_tsFeatures.csv"
    )
    df = pd.read_csv(csv_out_time_series_features_short_pth)
    gdf = gpd.read_file(GLAKES_tpcat_pth, ignore_geometry=True)
    df = df.merge(gdf, on="Lake_id_glakes", how="left", validate="1:1")
    # Ensure required base columns exist so downstream derived columns can be created safely.
    if "NDVI0010" not in df.columns:
        df["NDVI0010"] = np.nan
    else:
        # Compute NDVI change
        df["NDVI_p1p2_liu"] = df["NDVI0010"] - df["NDVI8499"]
        df["NDVI_p1p3_liu"] = df["NDVI1121"] - df["NDVI8499"]
        df["NDVI_p2p3_liu"] = df["NDVI1121"] - df["NDVI0010"]

    if "area_1984_1999_wm" not in df.columns:
        df["area_1984_1999_wm"] = np.nan
    else:
        # Compute water percent change
        df["water_pchange_p1p3_glakes"] = pct_change(df.area_1984_1999_wm, df.area_2010_2019_wm)
        df["water_pchange_p2p3_glakes"] = pct_change(df.area_2000_2009_wm, df.area_2010_2019_wm)
        df["water_pchange_p1p2_glakes"] = pct_change(df.area_1984_1999_wm, df.area_2000_2009_wm)

    return df


def load_topocat_catchments():
    """Load TopoCat catchments for specified PFAF zones and merge into single GeoDataFrame.

    Loads Catchments_pfaf_## layers for zones: 81, 82, 83, 84, 85, 86, 78, 71, 72, 73, 74
    from the PLD_TopoCat_v1.1 geodatabase.

    # TODO: pick vars

    Returns:
        GeoDataFrame: Merged catchments from all specified PFAF zones with original CRS
    """
    gdb_path = "/Volumes/metis/Datasets/TOPOCAT-PLD/PLD_TopoCat_v1.1.gdb"
    pfaf_zones = [81, 82, 83, 84, 85, 86, 78, 71, 72, 73, 74]

    gdfs = []
    for pfaf in pfaf_zones:
        layer_name = f"Catchments_pfaf_{pfaf:02d}"
        gdf = gpd.read_file(gdb_path, layer=layer_name)
        gdfs.append(gdf)

    # Merge all GeoDataFrames
    result = pd.concat(gdfs, ignore_index=True)
    result = gpd.GeoDataFrame(result, geometry="geometry", crs=gdfs[0].crs)
    # Append '_tpcat' to all non-geometry columns
    result = result.rename(columns={col: f"{col}_tpcat" for col in result.columns if col != "geometry"})

    return result


def load_topocat_pld_lakes():
    """Load TopoCat PLD v106 lakes for specified Pfafstetter level2 basins and merge into single
    GeoDataFrame.

    Loads Lakes_pfaf_## layers for zones: 81, 82, 83, 84, 85, 86, 78, 71, 72, 73, 74
    from the PLD_TopoCat_v1.1 geodatabase.

    Returns:
        GeoDataFrame: Merged lakes from all specified PFAF zones with original CRS
    """
    gdb_path = "/Volumes/metis/Datasets/TOPOCAT-PLD/PLD_TopoCat_v1.1.gdb"
    pfaf_zones = [81, 82, 83, 84, 85, 86, 78, 71, 72, 73, 74]

    gdfs = []
    for pfaf in pfaf_zones:
        layer_name = f"Lakes_pfaf_{pfaf:02d}"
        gdf = gpd.read_file(
            gdb_path,
            layer=layer_name,
            columns=[
                "lake_id",
                "Lake_area",
                "Outlet_n",
                "Cat_a_lake",
                "Lake_type",
                "Lake_order",
                "Laktyp_mhv",
            ],
        )
        # Rename columns by appending _tpcat
        gdf = gdf.rename(columns={col: f"{col}_tpcat" for col in gdf.columns if col != "geometry"})
        gdfs.append(gdf)

    # Merge all GeoDataFrames
    result = pd.concat(gdfs, ignore_index=True)
    result = gpd.GeoDataFrame(result, geometry="geometry", crs=gdfs[0].crs)

    return result


def harmonize_doc_dataset(
    gdf_or_df,
    lat_var,
    lon_var,
    sample_id_var,
    area_var,
    doc_var,
    dic_var,
    sample_idx_prefix="",
):
    """Harmonize a DOC dataset by renaming variables and creating geometry if needed.

    Parameters
    ----------
    gdf_or_df : GeoDataFrame or DataFrame
        Input dataset to harmonize
    lat_var : str
        Name of latitude column
    lon_var : str
        Name of longitude column
    sample_id_var : str
        Name of sample ID column
    area_var : str
        Name of area column (can be empty string for missing data)
    doc_var : str
        Name of DOC column
    dic_var : str
        Name of DIC column (can be empty string for missing data)
    sample_idx_prefix : str
        Two-letter prefix to prepend to dataset index to create unique sample_idx

    Returns
    -------
    GeoDataFrame
        Harmonized dataset with standardized column names (lat, lon, sample_id, area, doc, dic)
        and geometry column created from lat/lon if not already present
    """
    df = gdf_or_df.copy()

    # Drop geometry temporarily if present
    has_geometry = isinstance(df, gpd.GeoDataFrame)
    if has_geometry:
        df = df.to_crs("EPSG:4326")
        df = pd.DataFrame(df)

    # Build rename dictionary for non-empty variable names
    rename_dict = {
        lat_var: "lat",
        lon_var: "lon",
        sample_id_var: "sample_id",
        doc_var: "doc",
    }

    # Add area if provided
    if area_var:
        rename_dict[area_var] = "area_km2"

    # Add DIC if provided
    if dic_var:
        rename_dict[dic_var] = "dic"

    df = df.rename(columns=rename_dict)

    # Create unique sample_idx with prefix if provided
    if sample_idx_prefix:
        df["sample_idx"] = sample_idx_prefix + df.index.astype(str)

    # Normalize longitude to -180 to 180 range

    df["lon"] = df["lon"].apply(
        lambda x: (
            x + 360 if pd.notna(x) and x < -180 else (x - 360 if pd.notna(x) and x > 180 else x)
        )
    )

    # Create geometry from lat/lon if not already present
    if "geometry" not in df.columns:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs="EPSG:4326",
        )
    else:
        if df["geometry"].isna().any():
            df.loc[df["geometry"].isna(), "geometry"] = gpd.points_from_xy(
            df.loc[df["geometry"].isna(), "lon"],
            df.loc[df["geometry"].isna(), "lat"]
            )
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    return gdf


def load_harmonized_doc(NH=True) -> gpd.GeoDataFrame:
    """Load and harmonize all DOC datasets into a single GeoDataFrame.

    Combines data from Kurek, Dranga17, Stolpmann21, and Shahabinia25 with
    standardized column names and geometry.

    sample_id is index from original data set, but it may not be unique
    sample_idx is my unique index

    Returns
    -------
    GeoDataFrame
        Concatenated dataset with columns: lat, lon, sample_id, area (where available),
        doc, dic (where available), source, and geometry
    """
    gdfs = []

    # Kurek
    df_kurek = loadKurek()
    gdf_kurek = harmonize_doc_dataset(
        df_kurek,
        lat_var="Latitude",
        lon_var="Longitude",
        sample_id_var="Sample",
        area_var="lake_area_km2",
        doc_var="DOC (mg L-1)",
        dic_var="",
        sample_idx_prefix="K",
    )
    gdf_kurek["source"] = "Kurek23"
    # gdf_kurek["area_km2"] = gdf_kurek.to_crs("ESRI:102001").area / 1e6
    gdfs.append(gdf_kurek)

    # Dranga17
    # df_dranga, _, _ = loadDranga17()
    # df_dranga = load_prelim_matchup_data()
    dranga_shoreline_chem_pth = dranga_shoreline_pth.replace(".gpkg", "_chem.gpkg")
    df_dranga = gpd.read_file(dranga_shoreline_chem_pth)
    gdf_dranga = harmonize_doc_dataset(
        df_dranga,
        lat_var="Latitude",
        lon_var="Longitude",
        sample_id_var="Sample No",
        # area_var="area",
        area_var="Lake Surface Area",
        doc_var="DOC",
        dic_var="DIC",
        sample_idx_prefix="D",
    )
    gdf_dranga["source"] = "Dranga17"
    gdfs.append(gdf_dranga)

    # Stolpmann21
    gdf_stolpmann = loadStolpmann21()
    gdf_stolpmann = harmonize_doc_dataset(
        gdf_stolpmann,
        lat_var="Latitude",
        lon_var="Longitude",
        sample_id_var="SampleID",
        area_var="",
        doc_var="DOC",
        dic_var="",
        sample_idx_prefix="S",
    )
    gdf_stolpmann["source"] = "Stolpmann21"
    gdfs.append(gdf_stolpmann)

    # Shahabinia25
    df_shahabinia = loadShahabinia25()
    gdf_shahabinia = harmonize_doc_dataset(
        df_shahabinia,
        lat_var="lat_center",
        lon_var="long_center",
        sample_id_var="sample.no",
        area_var="area",
        doc_var="DOC",
        dic_var="DIC",
        sample_idx_prefix="H",
    )
    gdf_shahabinia["source"] = "Shahabinia25"
    gdfs.append(gdf_shahabinia)

    # Concatenate all datasets
    result = pd.concat(gdfs, ignore_index=True)[
        ["lat", "lon", "sample_id", "sample_idx", "area_km2", "doc", "dic", "source", "geometry"]
    ]
    if NH:
        result = result[result["lon"] < 0]
    result = gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:4326")

    # Check if sample_idx is unique
    sample_idx_counts = result.sample_idx.value_counts()
    duplicates = sample_idx_counts[sample_idx_counts > 1]
    assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate sample_idx values"

    return result
