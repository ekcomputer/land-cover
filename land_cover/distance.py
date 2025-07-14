import dask
import geopandas as gpd
import pointpats as pp
from dask import delayed
from shapely.geometry import Point
from tqdm import tqdm
import numpy as np
from dask.diagnostics import ProgressBar

# from dask.distributed import Client

# client = Client(n_workers=8)  # Use 8 cores


def calcMeanBoundDistSerial(gdf, include_max=False, num_samples=1000):
    '''Calculate the mean distance to polygon boundary for an arbitrary number of random points inside.'''
    for i, feature in tqdm(enumerate(gdf.loc[:, 'geometry']), total=len(gdf)): # 0:4
        rand_pts = pp.random.poisson(feature, size=num_samples)
        rand_pts_gs = gpd.GeoSeries([Point(pt) for pt in rand_pts], crs=gdf.crs)
        gdf.loc[i, 'mean_bound_dist'] = rand_pts_gs.distance(feature.boundary).mean()
        if include_max:
            gdf.loc[i, 'max_bound_dist'] = rand_pts_gs.distance(feature.boundary).max()
    return gdf


@delayed
def _calc_bound_dist_for_feature(
    feature,
    crs,
    num_samples=1000,
    include_max=False,
    simplify=None,
    simplify_vertex_threshold=10000,
): 
    # calc num vertexes
    if feature.geom_type == "Polygon":
        num_vertices = len(feature.exterior.coords)
    elif feature.geom_type == "MultiPolygon":
        # Sum vertices from all polygons in the MultiPolygon
        num_vertices = sum(len(poly.exterior.coords) for poly in feature.geoms)
    if simplify is not None and num_vertices > simplify_vertex_threshold:
        feature = feature.simplify(tolerance=simplify, preserve_topology=True)
    rand_pts = pp.random.poisson(feature, size=num_samples)
    rand_pts_gs = gpd.GeoSeries([Point(pt) for pt in rand_pts], crs=crs)

    mean_bound_dist = rand_pts_gs.distance(feature.boundary).mean()
    max_bound_dist = rand_pts_gs.distance(feature.boundary).max() if include_max else None

    return mean_bound_dist, max_bound_dist


# Function to parallelize the boundary distance calculation across features
def calcMeanBoundDist(gdf, include_max=False, num_samples=1000, simplify=None, simplify_vertex_threshold=10000):
    tasks = []
    """
    calcMeanBoundDist runs in parallel

    num_samples: size of simulation
    simplify: in projected units, if not None
    simplify_vertex_threshold: only simplify polygons with more vertices than this threshold

    Returns:
        _description_
    """
    # Loop over each feature, creating delayed tasks for parallel execution
    for i, feature in enumerate(gdf['geometry']):
        task = _calc_bound_dist_for_feature(
            feature,
            gdf.crs,
            num_samples,
            include_max,
            simplify,
            simplify_vertex_threshold,
        )
        tasks.append(task)

    # Compute all tasks in parallel
    with ProgressBar():
        results = dask.compute(*tasks, scheduler="processes")

    # Assign the results back to the original GeoDataFrame
    for i, (mean_dist, max_dist) in enumerate(results):
        gdf.loc[i, 'mean_bound_dist'] = mean_dist
        if include_max and max_dist is not None:
            gdf.loc[i, 'max_bound_dist'] = max_dist

    return gdf

if __name__ == "__main__":
    # Example/ testing
    gdf_hl = gpd.read_file(
        "/Volumes/thebe/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp"
    )
    print("loaded")
    # calcMeanBoundDistSerial(gdf_hl[100000:100100].to_crs("ESRI:102001"))
    calcMeanBoundDistSerial(gdf_hl[:50].to_crs("ESRI:102001"))
    # gdf_hl = calcMeanBoundDist(gdf_hl[100000:100100].to_crs("ESRI:102001").reset_index(drop=True), num_samples=10000)
    # gdf_hl = calcMeanBoundDist(
    #     gdf_hl[:50].to_crs("ESRI:102001").reset_index(drop=True), num_samples=10000, simplify=5000, simplify_vertex_threshold=1000
    # )
    pass
