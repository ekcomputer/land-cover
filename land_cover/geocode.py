import time

import pandas as pd
import requests
from shapely.geometry import Point, shape

USER_AGENT = "edk"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
GEONAMES_USERNAME = "ekyzivat"  # Replace with your actual username


def query_nominatim(name):
    params = {"q": f"{name}", "format": "json", "polygon_geojson": 1, "limit": 5}
    headers = {"User-Agent": USER_AGENT}
    try:
        response = requests.get(NOMINATIM_URL, params=params, headers=headers)
        if response.status_code == 200 and response.json():
            result = response.json()[0]
            return result.get("geojson"), result.get("display_name")
    except Exception as e:
        print(f"Error: {e}")
    return None, None


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

    geojson_list, names = [], []
    for name in df["lake name provided"]:
        geojson, display_name = query_nominatim(name)
        geojson_list.append(geojson)
        names.append(display_name)
        time.sleep(1)

    df["lake_geojson"] = geojson_list
    df["lake_full_name"] = names
    df = df[df["lake_geojson"].notnull()]

    df["within_10km"] = df.apply(within_10km, axis=1)
    df_filtered = df[df["within_10km"]]
    df_filtered.to_csv("lakes_within_10km.csv", index=False)
