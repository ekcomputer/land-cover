import geopandas as gpd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import img_tiles
from matplotlib import pyplot as plt
from pyproj.crs.crs import CRS
import lonboard
# from lonboard import PolygonLayer, ScatterplotLayer
import warnings
from IPython.display import display
from matplotlib.colors import Normalize
import seaborn as sns
from scipy.stats import pearsonr

# from palettable.colorbrewer.diverging import PuOr_10_r
# from palettable.colorbrewer.sequential import Oranges_9, BuPu_6
# from palettable.colorbrewer.diverging import PuOr_5_r # Earth_3
# from palettable.matplotlib import Magma_13

def _crs2ccrs(crs):
    epsg_code = crs.to_epsg()
    # epsg_code = gdf.crs.to_epsg()  # Extract EPSG code
    if epsg_code:
        ccrs_projection = ccrs.epsg(epsg_code)
    else:
        if crs.to_dict()['proj'] == 'aea':
            ccrs_projection = _cartopy_albers(crs)
        elif crs.name == 'WGS 84':
            ccrs_projection = ccrs.PlateCarree()
        else:
            raise ValueError(f"Mapping crs not supported: {crs.name}")
    return ccrs_projection


def _cartopy_albers(crs):

    # # Example parameters for Albers Equal Area (adjust based on your GeoDataFrame)
    # central_longitude = crs['longitude'] if 'longitude' in crs else -96  # default example
    # central_latitude = crs['latitude'] if 'latitude' in crs else 37.5  # default example
    # std_parallel_1 = crs['lat_1'] if 'lat_1' in crs else 29.5  # example standard parallel 1
    # std_parallel_2 = crs['lat_2'] if 'lat_2' in crs else 45.5  # example standard parallel 2

    # # Create the Cartopy Albers Equal Area CRS
    # ccrs_projection = ccrs.AlbersEqualArea(
    #     central_longitude=central_longitude,
    #     central_latitude=central_latitude,
    #     standard_parallels=(std_parallel_1, std_parallel_2)
    # )

    # Example parameters for Albers Equal Area (adjust based on your GeoDataFrame)
    crs_dict = crs.to_dict()
    central_longitude = crs_dict['lon_0'] 
    central_latitude = crs_dict['lat_0'] 
    std_parallel_1 = crs_dict['lat_1'] 
    std_parallel_2 = crs_dict['lat_2'] 

    # Create the Cartopy Albers Equal Area CRS
    ccrs_projection = ccrs.AlbersEqualArea(
        central_longitude=central_longitude,
        central_latitude=central_latitude,
        standard_parallels=(std_parallel_1, std_parallel_2)
    )
    return ccrs_projection


def plot_basemap(gdf:gpd.GeoDataFrame, crs:CRS=None, color='red', zoom=6, alpha=0.7, **kwargs):
    # set default
    if crs is None:
        crs = CRS.from_authority("ESRI", 102001)

    ccrs_for_map = _crs2ccrs(crs)
    ccrs_of_gdf = _crs2ccrs(gdf.crs)
    
    # Set up the plot with a specific Cartopy CRS for Alaska
    fig, ax = plt.subplots(
        figsize=(7, 10), 
        subplot_kw={
        # 'projection': 'ESRI:102001'})
        'projection': ccrs_for_map}) # TODO: error here. The following works: 'projection': ccrs.AlbersEqualArea(central_longitude=-152, central_latitude=63)}
            # 'projection': ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=40)})

    # Zoom into Alaska
    # ax.set_extent([-170, -130, 54, 72], crs=ccrs.PlateCarree()) # TODO: auto read

    # Add Google Satellite imagery
    ax.add_image(img_tiles.GoogleTiles(style='satellite'), zoom)

    # Add state outlines using Cartopy's features
    ax.add_feature(cfeature.STATES.with_scale('110m'), edgecolor='white')

    # Plot the GeoDataFrames
    gdf.plot(ax=ax, color=color, markersize=5,
                    transform=ccrs_of_gdf, alpha=alpha, **kwargs) #, label='Efflux lakes')

    # ax.legend(title='Legend', loc='upper right')
    plt.show()

# Custom basemap for `lonboard` plots
BASEMAP_URL = "https://api.maptiler.com/maps/1cdadb3b-20d8-4473-ac47-f3267fb12411/style.json?key=c7Pwm48hgeayqir5riN6"
# Enforce custom basemap for `lonboard` plots and try to ignore a reprojection warning.
def viz(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Input being reprojected to EPSG:4326")
        display(
            lonboard.viz(*args, **kwargs, map_kwargs={"basemap_style": BASEMAP_URL})
        )

def Map(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Input being reprojected to EPSG:4326")
        display(
            lonboard.Map(*args, **kwargs, basemap_style=BASEMAP_URL)
        )


def add_corr_line(x, y, **kwargs):
    """use with pairplot g.map"""
    ax = plt.gca()
    sns.regplot(x=x, y=y, scatter=False, ax=ax, color="red")
    r, _ = pearsonr(x, y)
    ax.annotate(f"r = {r:.2f}", xy=(0.05, 0.9), xycoords="axes fraction")


def add_r2(x, y, xy=(0.05, 0.9), **kwargs):
    ax = plt.gca()
    mask = ~np.isnan(x) & ~np.isnan(y)
    r, _ = pearsonr(x[mask], y[mask])
    ax.annotate(f"$r^2 =$ {r**2:.2f}", xy, xycoords="axes fraction", **kwargs)
