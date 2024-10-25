import pandas as pd
import pandas as pd
import geopandas as gpd

efflux_bogard_dict = {
    'AvgOfpCO2':'pco2uatm',
    'Lat_DD': 'lat',
    'Lon_DD': 'long'}

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


# gdf = loadEffluxShp()
# gdf = loadWBD()
# loadBogardMapShp()
# pass