from datetime import datetime

import ee
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import shape

# Initialize the Earth Engine module.
"""
GOOGLE_APPLICATION_CREDENTIALS=/Users/ekyzivat/.config/gcloud/application_default_credentials_ee_ekyzivat.json
earthengine authenticate --scopes https://www.googleapis.com/auth/cloud-platform
cp /Users/ekyzivat/.config/gcloud/application_default_credentials.json /Users/ekyzivat/.config/gcloud/application_default_credentials_ee_ekyzivat_june2025.json

TODO:
Postprocessing: polygon simplification, duplicate check, joins to PLD, make sure no nearly duplicate geometires digitized twice

"""
ee.Authenticate()
ee.Initialize(project="ee-ekyzivat")


def list_table_assets(asset_folder):
    """List all table assets in a given Earth Engine asset folder."""
    assets = ee.data.listAssets({"parent": asset_folder})["assets"]
    table_assets = [a["name"] for a in assets if a["type"] == "TABLE"]
    return table_assets


def ee_table_to_gdf(asset_id):
    """Download an Earth Engine table asset and convert to GeoDataFrame."""
    # Get the FeatureCollection
    fc = ee.FeatureCollection(asset_id)
    # Get all features as GeoJSON
    features = fc.getInfo()["features"]
    # Extract properties and geometry
    rows = []
    for f in features:
        props = f["properties"]
        geom = shape(f["geometry"])
        props["geometry"] = geom
        rows.append(props)
        if props["savetype"] == "MatchedManuallyDrawnPolygonThisSession":
            pass
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    return gdf


def merge_asset_tables_to_gdf(asset_folder):
    """Download all tables in an asset folder, merge, and return as GeoDataFrame."""
    table_assets = list_table_assets(asset_folder)
    gdfs = []
    for asset in table_assets:
        gdf = ee_table_to_gdf(asset)
        gdfs.append(gdf)
    if gdfs:
        merged_gdf = pd.concat(gdfs, ignore_index=True)
        merged_gdf = gpd.GeoDataFrame(merged_gdf, geometry="geometry", crs="EPSG:4326")
        return merged_gdf
    else:
        return gpd.GeoDataFrame()


if __name__ == "__main__":
    # Testing
    asset_folder = "projects/ee-ekyzivat/assets/HABL/working/"
    local_output_dir = "/Volumes/metis/ABOVE3/Digitizing/gee_asset_download"
    merged_gdf = merge_asset_tables_to_gdf(asset_folder)
    print(merged_gdf)
    timestamp = datetime.now().strftime("%Y%m%d")
    merged_gdf.to_file(f"{local_output_dir}/merged_asset_tables_{timestamp}.shp")
