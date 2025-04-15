import pandas as pd
import pandas as pd
import geopandas as gpd

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

def loadEfflux():
    return gpd.read_file('/Volumes/metis/ABOVE3/LAKESHAPE/effluxlakes.shp')


def loadEffluxShp():
    gdf_jn_PLD = gpd.read_file('/Volumes/metis/ABOVE3/Tom/Selected_PLD_Lakes_2024-10-21/EffluxLakes_selected_PLDLakes_2024-10-11.shp')
    df = pd.read_excel('/Volumes/metis/ABOVE3/Tom/PrelimLakeMatchupData_2024-10-21.xlsx', sheet_name='Measurements').query("Name == 'EffluxLakes'")
    gdf = gdf_jn_PLD.merge(df, on='lake_id', how='inner') #, validate='1:1')
    gdf = gdf.groupby('lake_id').first().reset_index() # hot fix to remove dups
    gdf.crs = gdf_jn_PLD.crs
    return gdf


def loadBogardMapShp(ABOVE_region=True):
    '''Loads all lakes with matchup, even if in Europe'''
    if ABOVE_region:
        bbox = (-170, 51, -127 , 72) # W NA
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


def loadWBD():
    '''Note: bbox is for AK'''
    return gpd.read_file('/Volumes/thebe/Other/Feng-High-res-inland-surface-water-tundra-boreal-NA/edk_out/fixed_geoms/WBD.shp', engine='pyogrio', bbox = (-170, 51, -125 , 72)) # bbox for NA


def loadLandCoverJoined():
    return pd.read_excel('/Volumes/metis/ABOVE3/land_cover_joins/out/xlsx/Efflux_Bogard_PLD_WBD_landCoverBuffers_core_tsFeatures.xlsx')


def sortColumns(df, order=first_columns):
    for col in order[::-1]:
        if col in df.columns:
            df.insert(0, col, df.pop(col))
    return


def dropColumns(df, cols=cols_to_drop):
    return df[[col for col in df.columns if col not in cols_to_drop]]


def loadKurek():
    dataset_path = "/Volumes/metis/ABOVE3/Kurek_GBC22_data/out/Kurek_ABoVE Lakes DOM_GBC_2023_Table S1.csv"
    shorelines_path = "/Volumes/metis/ABOVE3/Kurek_GBC22_data/out/shorelines/ABOVE_coordinates_for_Ethan_10-19-21_geom.shp"

    df_csv = pd.read_csv(dataset_path)
    gdf_shp = gpd.read_file(shorelines_path)
    merged = df_csv.merge(
        gdf_shp, left_on="Match_name", right_on="Sample_nam", how="left", indicator=False
    )
    # merged.rename(columns={"Note": "Digitizing note"})
    merged.rename(columns=dict(zip(gdf_shp.columns, ['dig_' + col for col in gdf_shp.columns if col != 'geometry'])), inplace=True)
    merged = gpd.GeoDataFrame(merged, crs=gdf_shp.crs)
    merged['lake_area_km2'] = merged.area / 1e6
    return merged


# def loadLiu():
#     df_csv = pd.read_csv("/Volumes/metis/Datasets/Liu_aq_veg/figshare/MA.csv")

def loadGreenness(bounds=None):
    """working file with Liu and Khun greenness. Needs refreshing."""
    engine=None
    if bounds=='kurek':
        bounds = kurek_bounds
        engine='fiona'
    else:
        bounds=None
    gdf = gpd.read_file(
        "/Volumes/metis/Datasets/Liu_aq_veg/figshare/edk_out/join_hl_greenness/greennessx2.shp",
        bounds=bounds, engine=engine
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
                    "MA_p1",
                    "MA_p1",
                ],
            )
        ),
        inplace=True,
    )
    return gdf

# gdf = loadEffluxShp()
# gdf = loadWBD()
# loadBogardMapShp()
# pass
