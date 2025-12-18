# from lonboard import PolygonLayer, ScatterplotLayer
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import lonboard
import numpy as np
import pandas as pd
import seaborn as sns
from cartopy.io import img_tiles
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from pyproj.crs.crs import CRS
from scipy.stats import f_oneway, linregress, pearsonr
from statannotations.Annotator import Annotator

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


def add_regress(x, y, xy=(0.05, 0.9), **kwargs):
    ax = plt.gca()
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() > 2:
        r, p = pearsonr(x[mask], y[mask])
        slope = linregress(x[mask], y[mask]).slope
        text = f"r²={r**2:.2g}, p={p:.2g}, slope={slope:.2g}"
    else:
        text = "insufficient data"
    ax.annotate(text, xy, xycoords="axes fraction", **kwargs)


def reg_hexplot(
    gdf,
    xvar,
    yvar,
    gridsize=30,
    mincnt=40,
    vmin=None,
    vmax=None,
    norm=None,
    ax=None,
    x_robust=False,
    y_robust=False,
    **kwargs,
):
    if norm is not None:
        norm = LogNorm(vmin=1)
    if ax is None:
        fig, ax = plt.subplots()

    xvals = gdf[xvar].values
    yvals = gdf[yvar].values

    # determine extent for hexbin based on robust percentiles if requested
    xmin, xmax = np.nanmin(xvals), np.nanmax(xvals)
    ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)

    if x_robust:
        try:
            p2, p98 = np.nanpercentile(xvals[~np.isnan(xvals)], [2, 98])
            if p2 < p98:
                xmin, xmax = p2, p98
        except Exception:
            pass

    if y_robust:
        try:
            p2, p98 = np.nanpercentile(yvals[~np.isnan(yvals)], [2, 98])
            if p2 < p98:
                ymin, ymax = p2, p98
        except Exception:
            pass

    extent = (xmin, xmax, ymin, ymax)

    hb = ax.hexbin(
        gdf[xvar],
        gdf[yvar],
        gridsize=gridsize,
        mincnt=mincnt,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        extent=extent,
        **kwargs,
    )
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    ax.set_title(f"{xvar} vs {yvar}")
    plt.colorbar(hb, ax=ax, label="count")
    plt.tight_layout()


def reg_scatterplot(
    gdf,
    xvar,
    yvar,
    norm=None,
    ax=None,
    x_robust=False,
    y_robust=False,
    **kwargs,
):
    if norm is not None:
        norm = LogNorm(vmin=1)
    if ax is None:
        fig, ax = plt.subplots()

    xvals = gdf[xvar].values
    yvals = gdf[yvar].values

    # determine extent for hexbin based on robust percentiles if requested
    xmin, xmax = np.nanmin(xvals), np.nanmax(xvals)
    ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)

    if x_robust:
        try:
            p2, p98 = np.nanpercentile(xvals[~np.isnan(xvals)], [2, 98])
            if p2 < p98:
                xmin, xmax = p2, p98
        except Exception:
            pass

    if y_robust:
        try:
            p2, p98 = np.nanpercentile(yvals[~np.isnan(yvals)], [2, 98])
            if p2 < p98:
                ymin, ymax = p2, p98
        except Exception:
            pass

    sns.scatterplot(
        gdf,
        x=xvar,
        y=yvar,
        ax=ax,
        **kwargs,
    )
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    ax.set_title(f"{xvar} vs {yvar}")

    # set axis lims to extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
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


def plot_neon_analytes_timeseries_pandas(
    csv_path="/Volumes/metis/ABOVE3/NEON/NEON_chem-surfacewater/stackedFiles/swc_externalLabDataByAnalyte.csv",
    analytes=("DOC", "DIC", "UV Absorbance (280 nm)", "TP", "TN"),
    decimate=1,
    hue="siteID",
    site_id_filter=None,
    outlier_iqr=[],
):
    use_cols = ["collectDate", "siteID", hue, "analyte", "analyteConcentration", "analyteUnits"]
    df = pd.read_csv(csv_path, usecols=use_cols, low_memory=False)
    if site_id_filter:
        df.query("siteID == @site_id_filter", inplace=True)
    df["collectDate"] = pd.to_datetime(df["collectDate"], errors="coerce", utc=True)
    df["analyteConcentration"] = pd.to_numeric(df["analyteConcentration"], errors="coerce")
    df = df[df["analyte"].isin(analytes)]
    df = df.dropna(subset=["collectDate", "analyteConcentration"])

    # Remove outliers for analyteConcentration
    title_addition = ""
    if outlier_iqr:
        title_addition = " (inliers)"
        for analyte in analytes:
            analyte_mask = df["analyte"] == analyte
            analyte_values = df.loc[analyte_mask, "analyteConcentration"]
            if len(analyte_values) > 0:
                a, b = outlier_iqr
                q1 = analyte_values.quantile(a)
                q3 = analyte_values.quantile(b)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_mask = (df["analyte"] == analyte) & (
                    (df["analyteConcentration"] < lower_bound)
                    | (df["analyteConcentration"] > upper_bound)
                )
                df = df[~outlier_mask]
    if decimate > 1:
        df = df.iloc[::decimate, :]

    sns.set_style("whitegrid")
    g = sns.relplot(
        data=df,
        x="collectDate",
        y="analyteConcentration",
        hue=hue,
        col="analyte",
        kind="line",
        marker="o",
        col_wrap=3,
        facet_kws={"sharey": False},
    )
    g.set_axis_labels("Date", "Concentration")
    g.set_titles("{col_name}")
    if site_id_filter:
        g.figure.suptitle(site_id_filter + title_addition)
    plt.tight_layout()
    plt.show()


def plot_neon_in_situ_timeseries_pandas(
    csv_path="/Volumes/metis/ABOVE3/NEON/NEON_chem-surfacewater/stackedFiles/swc_externalLabDataByAnalyte.csv",
    use_cols=["startDateTime", "chlorophyll"],
    decimate=1,
    date_var="startDateTime",
    y="chlorophyll",
    hue="siteID",
    date_filter=[],
    resample=None
):
    "No tall df"
    df = pd.read_csv(csv_path, usecols=use_cols + [hue], low_memory=False)
    df[date_var] = pd.to_datetime(df[date_var], errors="coerce", utc=True)
    df[y] = pd.to_numeric(df[y], errors="coerce")
    df = df.dropna(subset=[date_var, y])
    if date_filter:
        start_date, end_date = date_filter
        df = df[(df[date_var] >= pd.to_datetime(start_date)) & (df[date_var] < pd.to_datetime(end_date))]
    if decimate > 1:
        df = df.iloc[::decimate, :]
    # resample to 6-hour averages
    # ensure date_var is datetime index for resampling
    df = df.set_index(date_var)

    if resample and hue and hue in df.columns:
        # group by hue (e.g., siteID) and resample each group to 6H using mean of the value column
        df = df.groupby(hue)[y].resample("6H").mean().reset_index()
    else:
        # global 6H resample
        df = df[[y]].resample(resample).mean().reset_index()
    sns.set_style("whitegrid")
    g = sns.relplot(
        data=df,
        x=date_var,
        y=y,
        hue=hue,
        kind="line",
        marker="o",
        # col_wrap=3,
        # facet_kws={"sharey": True},
        markeredgecolor=None, markeredgewidth=0,
        markersize=4
    )
    g.set_axis_labels("Date", "Concentration")
    g.set_titles("{col_name}")
    plt.tight_layout()
    plt.show()


## for yvar in yvars: Barplot of df, with bars categorized by Total_inun_trend
def boxplots_by_group(
    df,
    yvar,
    group_col="Total_inun_trend",
    hue=None,
    figsize=(6, 4),
    showfliers=False,
    whis=1.5,
    show=True,
    order=["decreasing", "no trend", "increasing"],
    hue_order=None,
    anova=False,
    legend_option="inside",
    palette=None,
):
    """
    Create boxplots for each yvar grouped by `group_col`, with a subsampled stripplot overlay.

    Parameters
    - df: DataFrame
    - yvar: column name to plot on y axis
    - group_col: categorical column to group by (default "Total_inun_trend")
    - figsize: tuple for figure size
    - showfliers: pass to sns.boxplot (default False to hide outliers)
    - whis: whisker length for boxplot
    - show: whether to call plt.show() for each figure
    """
    if anova:
        figsize = (figsize[0]-1, figsize[1]+2)
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x=group_col, y=yvar, showfliers=showfliers, whis=whis, ax=ax, order=order, hue=hue, hue_order=hue_order, palette=palette)

    # Perform ANOVA and add significance annotations
    if anova is True and hue is None:
        groups = [df[df[group_col] == cat][yvar].dropna() for cat in order]
        if len(groups) > 1:
            f_stat, p_value = f_oneway(*groups)
            print(f"ANOVA F-statistic: {f_stat:.2f}, p-value: {p_value:.2e}")

        # Add significance annotations
        # Filter pairs to include only valid group combinations present in the data
        pairs = [(g1, g2) for g1 in order for g2 in order]
        annotator = Annotator(ax, pairs, data=df, x=group_col, y=yvar, order=order, hue=hue)
        annotator.configure(test="t-test_ind", text_format="star", loc="outside")
        annotator.apply_and_annotate()

    # sample sizes
    counts = df[group_col].value_counts()

    for x, cat in enumerate(order):
        n = counts.get(cat, 0)
        ax.text(
            x,  # x-position (category index)
            ax.get_ylim()[0],  # top of plot
            f"n={n:,}",  # label text
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title(f"{yvar} by {group_col} (outliers hidden)")

    if legend_option == "outside" and hue is not None:
        ax.legend(bbox_to_anchor=(0.3, -0.1), loc="upper right", ncol=2, title=hue)
    elif legend_option == "inside" and hue is not None:
        ax.legend(title=hue)
    elif legend_option is None and hue is not None:
        ax.get_legend().remove()
    plt.tight_layout()


# testing
if __name__== "__main__":
    plot_neon_in_situ_timeseries_pandas(
        "/Volumes/metis/ABOVE3/NEON/NEON_water-quality/stackedFiles/waq_instantaneous.csv",
        use_cols=["startDateTime", "siteID", "chlorophyll"],
        decimate=10000,
        date_var="startDateTime",
        y="chlorophyll",
        date_filter=["2017-09-14 00:00:00+00:00", "2018-05-14 00:00:00+00:00"],
    )
