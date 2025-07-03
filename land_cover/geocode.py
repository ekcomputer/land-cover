import time

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import requests
from geopy.distance import geodesic
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from shapely.geometry import Point, shape
from shapely.ops import transform
from tqdm import tqdm

USER_AGENT = "edk"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
GEONAMES_USERNAME = "ekyzivat"  # Replace with your actual username

geolocator = Nominatim(user_agent=USER_AGENT)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)


def query_nominatim(
    place_name, lat_lon: list = None, within=5, country_codes=["US", "CA"], limit=5
):
    location = geocode(place_name, exactly_one=False, limit=limit, country_codes=country_codes)
    return None, None


def polygons_nominatim(name, limit=5) -> list:
    params = {"q": f"{name}", "format": "json", "polygon_geojson": 1, "limit": limit}
    headers = {"User-Agent": USER_AGENT}
    try:
        response = requests.get(NOMINATIM_URL, params=params, headers=headers)
        if response.status_code == 200 and response.json():
            result = response.json()
            # return result.get("geojson"), result.get("display_name")
            return result
    except Exception as e:
        print(f"Error: {e}")
    return None


def verified_polygon(
    name: str, lat_lon=None, service=Nominatim, within=5, limit=5, country_codes=["US", "CA"]
):
    if service != Nominatim:
        raise ValueError("Only Nominatim service is supported.")

    results = polygons_nominatim(name, limit=limit)
    if results is not None:
        for item in results:
            geojson = item.get("geojson") if "geojson" in item else None
            display_name = item.get("display_name")
            lat = float(item.get("lat", np.nan)) if item.get("lat") else np.nan
            lon = float(item.get("lon", np.nan)) if item.get("lon") else np.nan
            place_id = item.get("place_id") if item.get("place_id") else np.nan
            osm_id = item.get("osm_id") if item.get("osm_id") else np.nan
            # Get centroid of geojson polygon
            if geojson and "coordinates" in geojson:
                try:
                    polygon = shape(geojson)
                    if lat_lon is None:
                        # If no lat_lon provided, don't verify coordinates
                        return geojson, display_name, lat, lon, place_id, osm_id
                    dist_km = nearest_distance(lat_lon, polygon)
                    if dist_km <= within:
                        return polygon, display_name, lat, lon, place_id, osm_id
                except Exception:
                    continue
            else:
                # if only points are returned (need to verify this is possible)
                if lat_lon is None:
                    return None, display_name, lat, lon, place_id, osm_id
                dist_km = geodesic((lat_lon[0], lat_lon[1]), (item["lat"], item["lon"])).km
                if dist_km <= within:
                    return geojson, display_name, lat, lon, place_id, osm_id
    return None, np.nan, np.nan, np.nan, np.nan, np.nan


# TODO: if no polygon, return location
# todo distance from geojson bounds

def geonames_search(lake_name):
    url = "http://api.geonames.org/searchJSON"
    params = {
        "q": lake_name,
        "maxRows": 10,
        "featureClass": "H",  # Hydrographic features
        "username": GEONAMES_USERNAME,
    }
    response = requests.get(url, params=params)
    return response.json().get("geonames", [])


def nearest_distance(lat_lon, polygon):
    """
    Find the nearest distance between appoints and polygon, assumed to both be in WGS84
    coordinates, using a local conformal projection.
    """
    # Calculate minimum distance from lat_lon to polygon bounds
    point = Point(lat_lon[1], lat_lon[0])  # shapely uses (lon, lat)
    # Project to a local Azimuthal equidistan conformal projection centered on lat_lon
    proj_str = f"+proj=aeqd +lat_0={lat_lon[0]} +lon_0={lat_lon[1]} +datum=WGS84 +units=m +no_defs"
    projection_transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", proj_str, always_xy=True
    ).transform

    polygon_proj = transform(projection_transformer, polygon)
    point_proj = transform(projection_transformer, point)
    return polygon_proj.boundary.distance(point_proj) / 1000  # meters to km


if __name__ == "__main__":
    df = pd.read_csv("/Volumes/metis/ABOVE3/Bogard_suppl_data/edk_out/Bogard19_ESM_alldata_wh.csv")
    # df = df[94:100]  # temp
    geojson_list, names, lats, lons, place_ids, osm_ids = [], [], [], [], [], []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        lake_name = row["lake name provided"]
        lat_lon = [row["lat (decimal)"], row["long (decimal)"]]
        if lake_name is not np.nan:
            geojson, display_name, lat, lon, place_id, osm_id = verified_polygon(
                lake_name, lat_lon, within=5, limit=10
            )
            geojson_list.append(geojson)
            names.append(display_name)
            lats.append(lat)
            lons.append(lon)
            place_ids.append(place_id)
            osm_ids.append(osm_id)
            time.sleep(1)
        else:
            names.append(np.nan)
            geojson_list.append(None)
            lats.append(np.nan)
            lons.append(np.nan)
            place_ids.append(np.nan)
            osm_ids.append(np.nan)

    df["geometry"] = geojson_list
    df["geocode_full_name"] = names
    df["geocode_lat"] = lats
    df["geocode_lon"] = lons
    df["geocode_place_id"] = place_ids
    df["geocode_osm_id"] = osm_ids
    df_filtered = df[df["geometry"].notnull()]
    # Create GeoDataFrame from filtered DataFrame
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")

    # # Define Lambert azimuthal equal-area projection centered at given lat/lon
    # laea_proj = (
    #     f"+proj=laea +lat_0=64.95 +lon_0=-89.65 +datum=WGS84 +units=m +no_defs"
    # )

    gdf = gdf.to_crs("ESRI:102001")  # Canada Albers Equal Area Conic projection

    # Write to shapefile
    gdf.to_file(
        "/Volumes/metis/ABOVE3/Bogard_suppl_data/edk_out/shp/Bogard19_ESM_alldata_wh_geocoded.gpkg"
    )
    # df_filtered.to_csv(
    #     "/Volumes/metis/ABOVE3/Bogard_suppl_data/edk_out/Bogard19_ESM_alldata_wh_geocoded.csv", index=False
    # )
