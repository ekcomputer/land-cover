import pandas as pd
import geopandas as gpd

def loadEfflux():
    return gpd.read_file('/Volumes/metis/ABOVE3/LAKESHAPE/effluxlakes.shp')