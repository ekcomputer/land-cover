import pandas as pd
import pandas as pd
import geopandas as gpd

def loadEfflux():
    return gpd.read_file('/Volumes/metis/ABOVE3/LAKESHAPE/effluxlakes.shp')


def loadEffluxShp():
    gdf_jn_PLD = gpd.read_file('/Volumes/metis/ABOVE3/Tom/Selected_PLD_Lakes_2024-10-21/EffluxLakes_selected_PLDLakes_2024-10-11.shp')
    df = pd.read_excel('/Volumes/metis/ABOVE3/Tom/PrelimLakeMatchupData_2024-10-21.xlsx', sheet_name='Measurements')
    gdf = gdf_jn_PLD.merge(df, on='lake_id', how='inner') #, validate='1:1')
    gdf = gdf.groupby('lake_id').first().reset_index() # hot fix to remove dups
    gdf.crs = gdf_jn_PLD.crs
    return gdf


def loadWBD():
    '''Note: bbox is for AK'''
    return gpd.read_file('/Volumes/thebe/Other/Feng-High-res-inland-surface-water-tundra-boreal-NA/edk_out/fixed_geoms/WBD.shp', engine='pyogrio', bbox = (-170, 51, -125 , 72))


# gdf = loadEffluxShp()
# gdf = loadWBD()
# pass