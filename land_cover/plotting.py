import geopandas as gpd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import img_tiles
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

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


def add_corr_line(x, y, xy=(0.05, 0.9), add_line=True, **kwargs):
    """use with pairplot g.map"""
    ax = plt.gca()
    if add_line is True:
        sns.regplot(x=x, y=y, scatter=False, ax=ax, color="red")
    mask = ~np.isnan(x) & ~np.isnan(y)
    if sum(mask) > 2:
        r, _ = pearsonr(x[mask], y[mask])
        ax.annotate(f"$r^2 =$ {r**2:.2f}", xy, xycoords="axes fraction", **kwargs)


def add_r2(x, y, xy=(0.05, 0.9), **kwargs):
    ax = plt.gca()
    mask = ~np.isnan(x) & ~np.isnan(y)
    r, _ = pearsonr(x[mask], y[mask])
    ax.annotate(f"$r^2 =$ {r**2:.2f}", xy, xycoords="axes fraction", **kwargs)


def reg_hexplot(gdf, xvar, yvar, gridsize=30, mincnt=40, vmin=None, vmax=None, norm=None, ax=None, **kwargs):
    if norm is not None:
        norm = LogNorm(vmin=1)
    if ax is None:
        fig, ax = plt.subplots()
    hb = ax.hexbin(
        gdf[xvar],
        gdf[yvar],
        gridsize=gridsize,
        mincnt=mincnt,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        **kwargs,
    )
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    ax.set_title(f"{xvar} vs {yvar}")
    plt.colorbar(hb, ax=ax, label="count")
    plt.tight_layout()


def plot_choro_and_hist(
    gdf, var, grid_crs=None, cmap="Greens", bins=None, figsize=(12, 5), hist_stats=False
):
    """
    Plot two adjacent panels:
      - left: choropleth of `var` with vmin/vmax set to the 2nd and 98th percentiles
               (or symmetric bounds with RdBu_r if values include negatives)
      - right: histogram of `var` with `bins` bins and vertical lines for mean/median
    Assumes gdf is a GeoDataFrame. Returns (fig, axs).
    """
    if grid_crs is None:
        grid_crs = getattr(gdf, "crs", None)

    vals = gdf[var].values
    if np.all(~np.isfinite(vals)):
        vmin, vmax = 0.0, 1.0
    else:
        # if any negative values, use a diverging cmap with symmetric bounds
        if np.nanmin(vals) < 0:
            p2, p98 = np.nanpercentile(vals, [2, 98])
            absmax = max(abs(p2), abs(p98))
            vmin, vmax = -absmax, absmax
            cmap = "RdBu_r"
        else:
            vmin, vmax = np.nanpercentile(vals, [2, 98])
            if vmin == vmax:  # fallback if constant
                vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            absmax = None

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # Chloropleth
    gdf.plot(column=var, ax=axs[0], legend=True, edgecolor=None, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].set_title(var)

    # Histogram
    data = gdf[var].dropna()
    if bins is None:
        if absmax is None:
            bins = 40
        else:
            bins = np.linspace(-absmax, absmax, 40)
    else:
        bins = 40
    counts, edges, patches = axs[1].hist(data, bins=bins, color="C0", alpha=0.8)
    axs[1].set_title(f"{var} histogram")
    axs[1].set_xlabel(var)
    axs[1].set_ylabel("count")

    # add vlines for mean and median (if finite)
    if data.size > 0 and np.any(np.isfinite(data)) and (hist_stats is True):
        mean_val = np.nanmean(data)
        median_val = np.nanmedian(data)
        ymax = axs[1].get_ylim()[1]
        if np.isfinite(mean_val):
            axs[1].vlines(
                mean_val, 0, ymax, colors="k", linestyles="--", label=f"Mean = {mean_val:0.2}"
            )
        if np.isfinite(median_val):
            axs[1].vlines(
                median_val, 0, ymax, colors="r", linestyles="-.", label=f"Median = {median_val:0.2}"
            )
        axs[1].legend()

    plt.tight_layout()
    return fig, axs
