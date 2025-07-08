import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import requests
from geopy.distance import geodesic
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from geopy.geocoders.google import GoogleV3
from shapely.geometry import Point, shape
from shapely.ops import transform
from tqdm import tqdm

wd = Path("/Volumes/metis/ABOVE3/Bogard_suppl_data")
out_dir = wd / "edk_out"
bogard_esm_pth = out_dir / "Bogard19_ESM_alldata_wh.csv"
out_gdb_pth = out_dir / "shp" / "Bogard19_ESM_alldata_wh_geocoded.gpkg"
out_shp_pth = out_dir / "shp" / "Bogard19_ESM_alldata_wh_geocoded.shp"

USER_AGENT = "edk"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
GEONAMES_USERNAME = "ekyzivat"  # Replace with your actual username

geolocator = Nominatim(user_agent=USER_AGENT)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)


# def query_nominatim(
#     place_name, lat_lon: list = None, within=5, country_codes=["US", "CA"], limit=5
# ):
#     """Using python API, which can't return geojson"""
#     location = geocode(place_name, exactly_one=False, limit=limit, country_codes=country_codes)
#     return None, None


def polygons_nominatim(name, limit=5, polygon_geojson=1) -> list:
    params = {"q": f"{name}", "format": "json", "polygon_geojson": polygon_geojson, "limit": limit}
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
    """
    Goes through different geocoding APIs to find a polygon (or at least a point) for a lake name
    within a certain distance from the provided lat_lon coordinates.

    Order:
    1. Nominatim (OpenStreetMap)
    2. Geonames
    3. Google
    """
    if service != Nominatim:
        raise ValueError("Only Nominatim service is supported.")

    results = polygons_nominatim(name, limit=limit, polygon_geojson=1)
    """
    Example:
    {'place_id': 348738704, 'licence': 'Data © OpenStreetMap contributors, ODbL 1.0. http://osm.org/copyright',
    'osm_type': 'node', 'osm_id': 2038281680, 'lat': '54.8505431', 'lon': '-67.8700287', 'class': 'natural',
    'type': 'water', 'place_rank': 22, 'importance': 0.1067125375753414, 'addresstype': 'water',
    'name': 'Lac Chaigneau', 'display_name':
    'Lac Chaigneau, Caniapiscau, Caniapiscau (MRC), Côte-Nord, Québec, Canada',
    'boundingbox': ['54.8504931', '54.8505931', '-67.8700787', '-67.8699787'],
    'geojson': {'type': 'Point', 'coordinates': [-67.8700287, 54.8505431]}}
    """
    if results is not None:
        for item in results:
            geojson = item.get("geojson") if "geojson" in item else None
            # Get distane to geojson polygon
            if geojson and geojson["type"] == "Polygon":
                try:
                    polygon = shape(geojson)
                    if lat_lon is None:
                        # If no lat_lon provided, don't verify coordinates
                        return item, polygon
                    dist_km = nearest_distance(lat_lon, polygon)
                    if dist_km <= within:
                        return item, polygon
                except Exception:
                    continue
            else:
                # if only points are returned (need to verify this is possible and that it would give different
                # or more complete results than if geojson is just a point)
                results = polygons_nominatim(name, limit=limit, polygon_geojson=0)
                if lat_lon is None:
                    return item, np.nan
                dist_km = geodesic((lat_lon[0], lat_lon[1]), (item["lat"], item["lon"])).km
                if dist_km <= within:
                    return item, np.nan

    #########################################
    ## OSM/Nominatim failed, move to Google
    #########################################

    #########################################
    ## Google failed, move to Geonames
    #########################################
    results_gn = geonames_search(name, limit=10)
    """
    Example:
    {'adminCode1': '10', 'lng': '-67.84333', 'geonameId': 6003831, 'toponymName': 'Lac Chaigneau',
    'countryId': '6251999', 'fcl': 'H', 'population': 0, 'countryCode': 'CA', 'name': 'Lac Chaigneau',
    'fclName': 'stream, lake, ...', 'adminCodes1': {'ISO3166_2': 'QC'}, 'countryName': 'Canada',
    'fcodeName': 'lake', 'adminName1': 'Quebec', 'lat': '54.81972', 'fcode': 'LK'}
    """
    if len(results_gn) > 0:
        for item in results_gn:
            try:
                lat = float(item["lat"])
                lon = float(item["lng"])
                polygon = Point(lon, lat)
                if lat_lon is None:
                    # If no lat_lon provided, don't verify coordinates
                    return item, polygon
                dist_km = nearest_distance(lat_lon, polygon)
                if dist_km <= within:
                    return item, np.nan
            except Exception:
                continue
    return {}, np.nan  # or None?


def geonames_search(lake_name, limit=5):
    url = "http://api.geonames.org/searchJSON"
    params = {
        "q": lake_name,
        "maxRows": limit,
        "featureClass": "H",  # Hydrographic features
        "username": GEONAMES_USERNAME,
    }
    response = requests.get(url, params=params)
    return response.json().get("geonames", [])


def nearest_distance(lat_lon, geometry):
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

    geometry_proj = transform(projection_transformer, geometry)
    point_proj = transform(projection_transformer, point)
    if geometry_proj.geom_type == "Point":
        distance_km = geometry_proj.distance(point_proj) / 1000  # meters to km
    elif geometry_proj.geom_type == "Polygon":
        distance_km = geometry_proj.boundary.distance(point_proj) / 1000  # meters to km
    else:
        raise ValueError("Geometry was not a Point or Polygon.")
    return distance_km


if __name__ == "__main__":
    df = pd.read_csv(bogard_esm_pth)
    df = df[94:200]  # temp, starts at geojosn matches
    # df = df[500:]  # temp
    (
        polygon_list,
        names,
        lats,
        lons,
        place_ids,
        osm_ids,
        item_classes,
        types,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        lake_name = row["lake name provided"]
        lat_lon = [row["lat (decimal)"], row["long (decimal)"]]
        if lake_name is not np.nan:
            item, polygon = verified_polygon(lake_name, lat_lon, within=5, limit=10)
            display_name = item.get("display_name")
            lat = float(item.get("lat", np.nan)) if item.get("lat") else np.nan
            if "lon" in item:
                lon = float(item.get("lon", np.nan)) if item.get("lon") else np.nan
            elif "lng" in item:
                lon = float(item.get("lng", np.nan)) if item.get("lng") else np.nan
            else:
                lon = np.nan
            place_id = item.get("place_id") if item.get("place_id") else np.nan
            osm_id = item.get("osm_id") if item.get("osm_id") else np.nan
            item_class = item.get("class") if item.get("class") else np.nan  # e.g. natural
            item_type = item.get("type") if item.get("type") else np.nan  # e.g. water

            polygon_list.append(polygon)
            names.append(display_name)
            lats.append(lat)
            lons.append(lon)
            place_ids.append(place_id)
            osm_ids.append(osm_id)
            item_classes.append(item_class)
            types.append(item_type)
            # No other geonames fields saved for now TODO

            time.sleep(1)
        else:
            names.append(np.nan)
            polygon_list.append(None)
            lats.append(np.nan)
            lons.append(np.nan)
            place_ids.append(np.nan)
            osm_ids.append(np.nan)
            item_classes.append(np.nan)
            types.append(np.nan)

    # OSM fields
    df["geometry"] = polygon_list
    df["osm_full_name"] = names
    df["osm_lat"] = lats
    df["osm_lon"] = lons
    df["osm_place_id"] = place_ids
    df["osm_id"] = osm_ids
    df["osm_item_class"] = item_classes
    df["osm_item_type"] = types

    # Geonames fields

    df_filtered = df[df["geometry"].notnull()]
    # Create GeoDataFrame from filtered DataFrame
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")

    # # Define Lambert azimuthal equal-area projection centered at given lat/lon
    # laea_proj = (
    #     f"+proj=laea +lat_0=64.95 +lon_0=-89.65 +datum=WGS84 +units=m +no_defs"
    # )

    gdf = gdf.to_crs("ESRI:102001")  # Canada Albers Equal Area Conic projection

    # Write to geoPackage and CSV
    gdf.to_file(
        "/Volumes/metis/ABOVE3/Bogard_suppl_data/edk_out/shp/Bogard19_ESM_alldata_wh_geocoded.gpkg"
    )
    gdf.to_file(
            "/Volumes/metis/ABOVE3/Bogard_suppl_data/edk_out/shp/Bogard19_ESM_alldata_wh_geocoded.shp"
        )
    df.drop(columns="geometry").to_csv(
        "/Volumes/metis/ABOVE3/Bogard_suppl_data/edk_out/Bogard19_ESM_alldata_wh_geocoded.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Geocoded {df_filtered.shape[0]} out of {df.shape[0]} features.")
    pass
