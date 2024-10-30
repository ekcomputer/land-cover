import geopandas as gpd
import pointpats as pp
from tqdm import tqdm
from shapely.geometry import Point
import dask
from dask import delayed
import dask.dataframe as dd


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
def calc_bound_dist_for_feature(feature, crs, num_samples=1000, include_max=False):
    rand_pts = pp.random.poisson(feature, size=num_samples)
    rand_pts_gs = gpd.GeoSeries([Point(pt) for pt in rand_pts], crs=crs)
    
    mean_bound_dist = rand_pts_gs.distance(feature.boundary).mean()
    max_bound_dist = rand_pts_gs.distance(feature.boundary).max() if include_max else None
    
    return mean_bound_dist, max_bound_dist


# Function to parallelize the boundary distance calculation across features
def calcMeanBoundDist(gdf, include_max=False, num_samples=1000):
    tasks = []
    
    # Loop over each feature, creating delayed tasks for parallel execution
    for i, feature in enumerate(tqdm(gdf['geometry'])):
        task = calc_bound_dist_for_feature(feature, gdf.crs, num_samples, include_max)
        tasks.append(task)

    # Compute all tasks in parallel
    results = dask.compute(*tasks)

    # Assign the results back to the original GeoDataFrame
    for i, (mean_dist, max_dist) in enumerate(results):
        gdf.loc[i, 'mean_bound_dist'] = mean_dist
        if include_max and max_dist is not None:
            gdf.loc[i, 'max_bound_dist'] = max_dist

    return gdf