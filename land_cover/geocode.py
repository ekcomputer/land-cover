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
from shapely.geometry import Point, Polygon, shape
from shapely.ops import transform
from tqdm import tqdm
import os

wd = Path("/Volumes/metis/ABOVE3/Bogard_suppl_data")
out_dir = wd / "edk_out"
bogard_esm_pth = out_dir / "Bogard19_ESM_alldata_wh.csv"
out_gdb_pth = out_dir / "shp" / "Bogard19_ESM_alldata_wh_geocoded.gpkg"
out_shp_pth = out_dir / "shp" / "Bogard19_ESM_alldata_wh_geocoded.shp"

USER_AGENT = "edk"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
GEONAMES_USERNAME = "ekyzivat"  # Replace with your actual username
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
VERSION = "v2"
# list of prefixes for lake names to not geolocate
BAD_PREFIXES = ["JR", "WB", "DV", "E5", "HW", "s1", "F1", "F2", "E5", "RDC"]

# nominatim_geolocator = Nominatim(user_agent=USER_AGENT)
# nominatim_geocode = RateLimiter(nominatim_geolocator.geocode, min_delay_seconds=1)
google_geolocator = GoogleV3(api_key=GOOGLE_MAPS_API_KEY)

# def query_nominatim_api(
#     place_name, lat_lon: list = None, within=5, country_codes=["US", "CA"], limit=5
# ):
#     """Using python API, which can't return geojson"""
#     location = geocode(place_name, exactly_one=False, limit=limit, country_codes=country_codes)
#     return None, None


def query_nominatim(name, limit=5, polygon_geojson=1) -> list:
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

query_nominatim = RateLimiter(query_nominatim, min_delay_seconds=1)

def query_google_api(place_name, country_codes=["US", "CA"], limit=5):
    """
    Using Google API, which can't return geojson. No rate limiting implemented.

    Free usage: 10k / month, then $5 / 1000 requests
    Limit argument is only used to determine whether to call "exactly_one" or not.
    """
    components = [("country", code) for code in country_codes]
    try:
        result = google_geolocator.geocode(
            place_name, exactly_one=False if limit > 1 else True, components=components
        )
        return result
    except Exception as e:
        print(f"Google geocode error: {e}")
        return None


def geonames_search(lake_name, limit=5):
    """ """
    url = "http://api.geonames.org/searchJSON"
    params = {
        "q": lake_name,
        "maxRows": limit,
        "featureClass": "H",  # Hydrographic features
        "username": GEONAMES_USERNAME,
    }

    response = requests.get(url, params=params)
    return response.json().get("geonames", [])

geonames_search = RateLimiter(geonames_search, min_delay_seconds=1)

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


def verified_polygon(name: str, lat_lon=None, within=5, limit=5, country_codes=["US", "CA"]):
    """
    Goes through different geocoding APIs to find a polygon (or at least a point) for a lake name
    within a certain distance from the provided lat_lon coordinates.

    Args:
        within: kilomter buffer around feature geometry to filter matches. Doubled if only
            lat_lon is returned.

    Order:
    1. Nominatim (OpenStreetMap)
    2. Geonames
    3. Google
    """

    results = query_nominatim(name, limit=limit, polygon_geojson=1)
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
        first_time = True
        for item in results:
            geojson = item.get("geojson") if "geojson" in item else None
            item["geocoder"] = "nominatim"
            item["geocode_id"] = item.get("osm_id")
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
                results = query_nominatim(name, limit=limit, polygon_geojson=0)
                if lat_lon is None:
                    return item, np.nan
                dist_km = geodesic((lat_lon[0], lat_lon[1]), (item["lat"], item["lon"])).km
                if dist_km <= within * 2:
                    return item, np.nan

    #########################################
    ## OSM/Nominatim failed, move to Geonames
    #########################################

    results_gn = geonames_search(name, limit=limit)
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
                # format for consistency
                item["lon"] = lon
                item["geocoder"] = "geonames"
                item["display_name"] = item.get("name")
                item["geocode_id"] = item.get("geonameId")
                item["type"] = item.get("fcodeName")
                item["class"] = item.get("fclName")
                polygon = Point(lon, lat)
                if lat_lon is None:
                    # If no lat_lon provided, don't verify coordinates
                    return item, polygon
                dist_km = nearest_distance(lat_lon, polygon)
                if dist_km <= within * 2:
                    return item, np.nan
            except Exception:
                continue

    #########################################
    ## Geonames failed, move to Google
    #########################################

    results_google = query_google_api(name, country_codes=country_codes, limit=limit)
    """
    Example:
    {'address': 'United States', 'altitude': 0.0, 'latitude': 38.7945952, 'longitude': -106.5348379,
    'point': Point(38.7945952, -106.5348379, 0.0), 'raw':{'address_components': [{'long_name':
    'United States', 'short_name': 'US', 'types': ['country', 'political']}], 'formatted_address':
    'United States', 'geometry': {'bounds': {'northeast': {'lat': 74.071038, 'lng': -66.885417},
    'southwest': {'lat': 18.7763, 'lng': 166.9999999}}, 'location': {'lat': 38.7945952,
    'lng': -106.5348379}, 'location_type': 'APPROXIMATE', 'viewport': {'northeast':
    {'lat': 72.7087158, 'lng': -66.3193754}, 'southwest': {'lat': 15.7760139, 'lng': -173.2992296}}},
    'partial_match': True, 'place_id': 'ChIJCzYy5IS16lQRQrfeQ5K5Oxw', 'types': ['country', 'political']}}
    """
    if results_google is not None and len(results_google) > 0:
        for location in results_google:
            try:
                lat = float(location.latitude)
                lon = float(location.longitude)
                item = location.raw.copy()
                # format for consistency
                item["lat"] = lat
                item["lon"] = lon
                item["geocoder"] = "google"
                item["display_name"] = (
                    location.raw["formatted_address"] if location.raw else location.address
                )
                item["geocode_id"] = location.raw["place_id"]
                item["class"] = location.raw["types"][1] if len(location.raw["types"]) > 0 else None
                item["type"] = location.raw["geometry"]["location_type"]  # e.g. APPROXIMATE
                polygon = Point(lon, lat)
                if lat_lon is None:
                    # If no lat_lon provided, don't verify coordinates
                    return item, polygon
                dist_km = nearest_distance(lat_lon, polygon)
                if dist_km <= within * 2:
                    return item, np.nan
            except Exception:
                continue

    # No results found in any of the services
    return {}, np.nan  # or None?


if __name__ == "__main__":
    df = pd.read_csv(bogard_esm_pth)
    # df = df[94:200]  # temp, starts at geojosn matches
    # df = df[500:510]  # temp
    (
        polygon_list,
        names,
        lats,
        lons,
        place_ids,
        geocode_ids,
        item_classes,
        types,
        geocoders,
        has_polygons,
    ) = (
        [],
        [],
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
        use_geocoder = isinstance(lake_name, str) and (
            (lake_name[:1].isalpha()) and (not np.isin(lake_name[:2], BAD_PREFIXES))
        )
        if use_geocoder:
            item, polygon = verified_polygon(lake_name, lat_lon, within=10, limit=10)
            has_polygon = False if polygon is np.nan else True
            display_name = item.get("display_name") if item.get("display_name") else np.nan
            lat = float(item.get("lat", np.nan)) if item.get("lat") else np.nan
            lon = float(item.get("lon", np.nan)) if item.get("lon") else np.nan
            place_id = item.get("place_id") if item.get("place_id") else np.nan
            geocode_id = item.get("geocode_id") if item.get("geocode_id") else np.nan
            item_class = item.get("class") if item.get("class") else np.nan  # e.g. natural
            item_type = item.get("type") if item.get("type") else np.nan  # e.g. water
            geocoder = item.get("geocoder") if item.get("geocoder") else np.nan

            polygon_list.append(polygon)
            names.append(display_name)
            lats.append(lat)
            lons.append(lon)
            place_ids.append(place_id)
            geocode_ids.append(geocode_id)
            item_classes.append(item_class)
            types.append(item_type)
            geocoders.append(geocoder)
            has_polygons.append(has_polygon)
        else:
            names.append(np.nan)
            polygon_list.append(None)
            lats.append(np.nan)
            lons.append(np.nan)
            place_ids.append(np.nan)
            geocode_ids.append(np.nan)
            item_classes.append(np.nan)
            types.append(np.nan)
            geocoders.append(np.nan)
            has_polygons.append(False)
            print("Skipping geocoding for: ", lake_name)

    # OSM fields
    df["geometry"] = polygon_list
    df["geocode_full_name"] = names
    df["geocode_lat"] = lats
    df["geocode_lon"] = lons
    df["geocode_id"] = place_ids
    df["geocode_id"] = geocode_ids
    df["geocode_item_class"] = item_classes
    df["geocode_item_type"] = types
    df["geocode_geom"] = has_polygons
    df["geocoder"] = geocoders
    # Geonames fields

    df_filtered = df[df["geocode_lat"].notnull()]
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")

    # # Define Lambert azimuthal equal-area projection centered at given lat/lon
    # laea_proj = (
    #     f"+proj=laea +lat_0=64.95 +lon_0=-89.65 +datum=WGS84 +units=m +no_defs"
    # )

    # gdf = gdf.to_crs("ESRI:102001")  # Canada Albers Equal Area Conic projection

    # Write to geoPackage and CSV
    gdf.to_file(
        f"/Volumes/metis/ABOVE3/Bogard_suppl_data/edk_out/shp/Bogard19_ESM_alldata_wh_geocoded_{VERSION}.gpkg"
    )
    gdf.to_file(
        f"/Volumes/metis/ABOVE3/Bogard_suppl_data/edk_out/shp/Bogard19_ESM_alldata_wh_geocoded_{VERSION}.shp"
    )
    df.drop(columns="geometry").to_csv(
        f"/Volumes/metis/ABOVE3/Bogard_suppl_data/edk_out/Bogard19_ESM_alldata_wh_geocoded_{VERSION}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Geocoded {df_filtered.shape[0]} out of {df.shape[0]} features.")
    pass
