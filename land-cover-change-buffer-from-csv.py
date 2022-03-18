#!/usr/bin/env python
# coding: utf-8

'''
Summary
For calculating land cover change within polygons. Used for whitefish lake for Karen Wang and Jonathan Wang ABoVE land cover dataset- multitemporal.
EDK 2021.11.19
TODO: 
* Can use zonal statistics on multiple polygons (buffers) at once, instead of in loop.
* Remove unnecessary buffer computation if I'm only saving two of them. 
* OR: actually save data for each buffer- rewrite so dataframes are merged w/i function and record buffer length and pt location.
* Clean up commented out lines
* Don't compute redundant land cover for same lakes at different times
* Vectorize zonal stats?
* Add original csv/shp attributes from join based on index.
* Check that water normalization only refers to largest/central lake within buffer.
* Seaborn default plots
* LBF percent
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.plot import reshape_as_image
import fiona
import rasterio.mask
# from scipy.stats import binned_statistic
from rasterstats import zonal_stats

## I/O

## Switches (uncomment)
# use_simplified_classes=True
use_simplified_classes=False

## in
# pth_shp_in = '/mnt/f/ABoVE2021/Mapping/shp/ABOVE_coordinates_for_Ethan_10-19-21.shp' # points # temp until I get lake geometries for full dataset!
pth_shp_in = '/mnt/f/ABoVE2021/Mapping/shp/polygon_geom/ABOVE_coordinates_for_Ethan_10-19-21_jn_PADLakesVis.shp' # polygons
pth_lc_in = '/mnt/f/Wang-above-land-cover/ABoVE_LandCover_5km_buffer.vrt'
# pth_lc_in_simp = '/mnt/f/Wang-above-land-cover/ABoVE_LandCover_Simplified_Bh04v01.tif' # simplified 10-calss landcover

## out
xlsx_out_pth = '/mnt/f/ABoVE2021/Mapping/out/xlsx/' + os.path.basename(pth_shp_in)[:-4] + '_landCoverBuffers.xlsx'
plot_dir = '/mnt/d/pic/above-land-cover'

## buffers
buffer_lengths = (90, 990) # in m # 90, 990 # 1350

## classes for land cover 
classes =       ['Evergreen Forest','Deciduous Forest',	'Mixed Forest',	'Woodland',	'Low Shrub',	'Tall Shrub',	'Open Shrubs',	'Herbaceous',	'Tussock Tundra',	'Sparsely Vegetated',	'Fen',	'Bog',	'Shallows/littoral',	'Barren',	'Water']
classes_dry =   ['Evergreen Forest','Deciduous Forest',	'Mixed Forest',	'Woodland',	'Low Shrub',	'Tall Shrub',	'Open Shrubs',	'Herbaceous',	'Tussock Tundra',	'Sparsely Vegetated',	'Fen',	'Bog',	'Barren']
classes_wet =   ['Shallows/littoral', 'Water']
classes_simp = ['Evergreen Forest','Deciduous Forest',	'Shrubland', 'Herbaceous',	'Sparsely Vegetated',	'Barren',	'Fen',	'Bog',	'Shallows/littoral', 'Water']
years = np.arange(1984, 2014+1)

## dynamic values
if use_simplified_classes:
    pth_lc_in = pth_lc_in_simp
    classes = classes_simp

nBuffers = len(buffer_lengths)
nclasses = len(classes)
xlsx_out_norm_pth = xlsx_out_pth.replace('.xlsx', '_norm.xlsx')

## Create custom function for zonal stats that better resembles the arc/Q version
def my_hist(lc):
    ''' Gives counts for each integer-valued landcover class, with total number hard-coded in as nclasses.'''
    return np.histogram(np.ndarray.flatten(lc), range=[1,nclasses+1], bins = nclasses)[0] # bin counts

## Function
def extractBufferZonalHist(poly, buffer_lengths):
    ''' Buffer_lengths is in map units (probably m).'''
    ## buffer pts 
    buffers = pd.concat([poly.buffer(length) for length in buffer_lengths])

    ## load raster subset
    with rio.open(pth_lc_in) as src:
        lc, lc_transform = rasterio.mask.mask(src, buffers[-1:], crop=True) # use outermost buffer as mask ROI to avoid loading too much data
        lc_meta = src.meta
        # lc = src.read()
        lc = reshape_as_image(lc)
        src_crs=src.crs
        src_res=src.res
        src_shp=src.shape
    nYears = lc.shape[2]


    ## Loop over all years and most buffer lengths (can be sped up by vectorizing buffers: run zonal_stats on multiple features at once)
    # array=np.full([nYears, nclasses, nBuffers], np.nan, dtype='uint32') # init array for outpu
    
    ## Init
    dfba = pd.DataFrame(columns = classes + ['Year', 'Buffer_m', 'Join_idx']) # 'df buffer append' # classes.extend(['Year', 'Buffer_m'])
    n = 0 # init

    ## Loop
    for j, ring in enumerate(buffers):
        for i, year in enumerate(range(nYears)):
            
            ## Zonal stats. Source: https://automating-gis-processes.github.io/CSC/notebooks/L5/zonal-statistics.html
            stat = zonal_stats(ring, lc[:,:,i], affine=lc_transform, stats='count unique', add_stats = {'histogram':my_hist}, nodata=255) # could use count_unique=True option, but I want zeros in my histograms
            # array[i,:,j] = stat[0]['histogram']
            dfba = dfba.append(pd.DataFrame(stat[0]['histogram'][np.newaxis, :] * np.prod(src_res)/10000, columns=classes), ignore_index=True, verify_integrity=True)
            dfba.loc[n, 'Year'] = years[i]
            dfba.loc[n, 'Buffer_m'] = buffer_lengths[j]
            dfba.loc[n, 'Lake_name'] = poly.index[0]
            dfba.loc[n, 'Join_idx'] = poly.Join_idx[0]
            n += 1
    return dfba

def extractTimeSeriesForLakes():
    ## validate
    print('Paths:')
    print(xlsx_out_pth)
    print(f'\nUse simplified classes: {use_simplified_classes}')

    ## roi for cropping
    # with fiona.open(pth_roi_in, "r") as shapefile:
    #     roi = [feature["geometry"] for feature in shapefile]

    ## load lake polygons
    polys = gpd.read_file(pth_shp_in) # geodataframe of all lake outlines

    ## save orig index to join back in attributes later
    polys['Join_idx'] = polys.index

    ## Create gdf of unique polygons
    polys_g = polys.groupby('Sample_nam')
    polys_u = polys_g.first() # unique lakes

    ## Test if any lakes are collected in multiple locs or need unique names
    # center_diff =polys_g.latitude.max() - polys_g.latitude.min()
    # center_diff.to_csv('Python/Land-cover/center_diff.csv')

    ## Run function in loop
    for i in range(len(polys_u)): #range(4): #range(len(polys_u)): # i, (_, poly) in enumerate(polys[:3].iterrows()): # range(4)
        poly = polys_u.iloc[i:i+1, :]

        ## print
        print(poly.index.values)

        ## zonal hist
        dfba= extractBufferZonalHist(poly, buffer_lengths)
        if i==0: # TODO: use for [x in y] syntax with pd.concat.
            df = dfba
        else:
            df = df.append(dfba)
        
        ## Save checkpoint
        if i % 10 == 0:
            df.to_excel(xlsx_out_pth)

    ## Sort based on join index, which refers to original entries in shapefile
    # df.set_index('Join_idx')

    print('done')

    ## reset index
    df.set_index('Lake_name', inplace=True)

    ## write out
    df.to_excel(xlsx_out_pth)
    print(f'Wrote output: {xlsx_out_pth}')

def normalizeTimeSeries():

    ## Load
    print('Normalizing land cover...')
    df = pd.read_excel(xlsx_out_pth)

    ## find littoral percent of water areas (TODO: ensure it only comes from largest/central water body within buffer)
    df['Littorals_pct'] = df['Shallows/littoral'] / df.loc[:,classes_wet].sum(axis=1)*100

    ## Find wetland percent, like michela does, by taking: (L+B+F)/(L+B+F+W)*100
    # TODO

    ## Find class percent of dry areas
    # normDry = lambda var: df[var] / df.loc[:,classes_dry].sum(axis=1)*100 # just keeping lambda function for practice
    for var in classes_dry:
        df[var + '_pct'] = df[var] / df.loc[:,classes_dry].sum(axis=1)*100
    
    ## Write out
    df.to_excel(xlsx_out_norm_pth)
    print(f'Wrote normalized output table: {xlsx_out_norm_pth}')

def plotTimeSeries():
    ## vars
    buf_len = buffer_lengths[0] # use the smallest (90 m) buffer for plotting

    ## Load
    print('Plotting land cover...')
    value_name = 'Ha' # 'Percent'
    df = pd.read_excel(xlsx_out_norm_pth, index_col=0)
    dfg = df.groupby('Lake_name') # ['Lake_name', 
    group = dfg.get_group('Balloon lake') # formerly ('Balloon lake', buf_len)
    
    ## Plot with mpl
    # fig, ax = plt.subplots()
    # group.plot(x='Year', y='Littorals_pct', ax=ax)
    # plt.savefig(os.path.join(plot_dir, 'time-series-1.png'))

    ## Try facet grid in seaborn
    # dfl = pd.melt(group, id_vars=['Year', 'Buffer_m'], value_vars=df.columns[1:16], var_name = 'Class', value_name=value_name)# data frame long format # use df.columns[-14:] for normalized vals
    # g = sns.FacetGrid(dfl, col="Class", hue="Buffer_m", col_wrap=4)
    # g.map(sns.lineplot, 'Year', value_name)
    # g.add_legend(title="Buffer (m)")
    # plt.show()
    # g.savefig(os.path.join(plot_dir, 'time-series-facets-1.png'))

    ## Plot for all lakes!
    for lake in dfg.groups:
        group = dfg.get_group(lake)
        dfl = pd.melt(group, id_vars=['Year', 'Buffer_m'], value_vars=df.columns[1:16], var_name = 'Class', value_name=value_name)# data frame long format # use df.columns[-14:] for normalized vals
        g = sns.FacetGrid(dfl, col="Class", hue="Buffer_m", col_wrap=4)
        g.map(sns.lineplot, 'Year', value_name)
        g.add_legend(title="Buffer (m)")
        g.fig.subplots_adjust(top=0.93) # adjust the Figure to add super title
        g.fig.suptitle(lake)
        # plt.show()
        plt.close()
        g.savefig(os.path.join(plot_dir, 'time-series-by-lake', f'time-series-facets-{lake}.png').replace(' ','-'))
        print(lake)
    print('Done plotting.')

if __name__ == '__main__':
    # extractTimeSeriesForLakes()
    # normalizeTimeSeries()
    plotTimeSeries()