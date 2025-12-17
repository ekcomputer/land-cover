import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# File paths
input_csv = "/Volumes/metis/ABOVE3/abcfluxv2/Arctic_Boreal_CO2_Flux_V2_2448/data/Arctic_Boreal_CO2_Flux_V2.csv"
output_gpkg = "/Volumes/metis/ABOVE3/abcfluxv2/Arctic_Boreal_CO2_Flux_V2_2448/data/edk_out/shp/ABCFlux_V2_doc.gpkg"

# Load CSV with specified NaN values
df = pd.read_csv(input_csv, na_values=["NA", -9999])

# Drop rows where "water_doc" is NaN
df = df.dropna(subset=["water_doc"])

df = df[df.waterbody_type == "Lentic"]

# Create geometry column using latitude and longitude
geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# Save as GeoPackage
gdf.to_file(output_gpkg, driver="GPKG")
