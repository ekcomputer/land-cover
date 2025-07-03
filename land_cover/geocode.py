import time

import numpy as np
import pandas as pd
import requests
from geopy.distance import geodesic
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from shapely.geometry import Point, shape

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

    for item in results:
        geojson = item.get("geojson") if "geojson" in item else None
        display_name = item.get("display_name")
        lat = float(item.get("lat", np.nan)) if item.get("lat") else np.nan
        lon = float(item.get("lon", np.nan)) if item.get("lon") else np.nan
        # Get centroid of geojson polygon
        if geojson and "coordinates" in geojson:
            try:
                # polygon = shape(geojson)
                if lat_lon is None:
                    # If no lat_lon provided, don't verify coordinates
                    return geojson, display_name, lat, lon
                # centroid = polygon.centroid
                dist_km = geodesic((lat_lon[0], lat_lon[1]), (item["lat"], item["lon"])).km
                if dist_km <= within:
                    return geojson, display_name, lat, lon
            except Exception:
                continue
        else:
            # if only points are returned (need to verify this is possible)
            return geojson, display_name, lat, lon
    return [np.nan] * 4


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


def within_10km(row):
    try:
        point = Point(row["long (decimal)"], row["lat (decimal)"])
        polygon = shape(row["lake_geojson"])
        return point.distance(polygon) <= 0.1  # ~10 km in degrees
    except:
        return False


if __name__ == "__main__":
    df = pd.read_csv("/Volumes/metis/ABOVE3/Bogard_suppl_data/edk_out/Bogard19_ESM_alldata_wh.csv")

    geojson_list, names, lats, lons = [], []
    for idx, row in df[94:].iterrows():
        lake_name = row["lake name provided"]
        lat_lon = [row["lat (decimal)"], row["long (decimal)"]]
        if lake_name is not np.nan:
            geojson, display_name, lat, lon = verified_polygon(lake_name, lat_lon, limit=10)
            geojson_list.append(geojson)
            names.append(display_name)
            lats.append(lat)
            lons.append(lon)
            time.sleep(1)
        else:
            names.append(np.nan)
            geojson_list.append(None)
            lats.append(np.nan)
            lons.append(np.nan)

    df["lake_geojson"] = geojson_list
    df["lake_full_name"] = names
    df = df[df["lake_geojson"].notnull()]

    df["within_10km"] = df.apply(within_10km, axis=1)
    df_filtered = df[df["within_10km"]]
    df_filtered.to_csv("lakes_within_10km.csv", index=False)
