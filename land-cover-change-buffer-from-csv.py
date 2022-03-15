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
'''

import os
import numpy as np
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
# pth_shp_in = '/mnt/g/Land-cover-whitefish-lk-ak/whitefish-lk-albers.shp'
pth_shp_in = '/mnt/f/ABoVE2021/Mapping/shp/ABOVE_coordinates_for_Ethan_10-19-21.shp'
# pth_roi_in = '/mnt/g/Land-cover-whitefish-lk-ak/roi_albers.shp'
# pth_lc_in = '/mnt/f/Wang-above-land-cover/ABoVE_LandCover_Bh04v01.tif'
pth_lc_in = '/mnt/f/Wang-above-land-cover/ABoVE_LandCover_5km_buffer.vrt'
pth_lc_in_simp = '/mnt/f/Wang-above-land-cover/ABoVE_LandCover_Simplified_Bh04v01.tif' # simplified 10-calss landcover

## out
# pth_csv_out_inner = '/mnt/g/Land-cover-whitefish-lk-ak/out/stats/whitefish-lk-land-cover-change-b279-tmp.csv'
# pth_csv_out_outer = '/mnt/g/Land-cover-whitefish-lk-ak/out/stats/whitefish-lk-land-cover-change-b1350-tmp.csv'
xlsx_out_pth = '/mnt/f/ABoVE2021/Mapping/out/xlsx/' + os.path.basename(pth_shp_in)[:-4] + '_landCoverBuffers.xlsx'

## buffers
# narrowBufferCount = 2 # number of buffer iterations. Length = (narrowBufferCount+1)  * buffer_lengths[2]
# wideBufferCount = 14 # etc
buffer_lengths = (270, 1350) # in m

## classes for land cover 
classes = ['Evergreen Forest','Deciduous Forest',	'Mixed Forest',	'Woodland',	'Low Shrub',	'Tall Shrub',	'Open Shrubs',	'Herbaceous',	'Tussock Tundra',	'Sparsely Vegetated',	'Fen',	'Bog',	'Shallows/littoral',	'Barren',	'Water']
classes_simp = ['Evergreen Forest','Deciduous Forest',	'Shrubland', 'Herbaceous',	'Sparsely Vegetated',	'Barren',	'Fen',	'Bog',	'Shallows/littoral', 'Water']
years = np.arange(1984, 2014+1)
## dynamic values
if use_simplified_classes:
    pth_lc_in = pth_lc_in_simp
    classes = classes_simp
    # pth_csv_out_inner = pth_csv_out_inner.replace('land-cover-change', 'land-cover-change-10class')
    # pth_csv_out_outer = pth_csv_out_outer.replace('land-cover-change', 'land-cover-change-10class')
    
nBuffers = len(buffer_lengths)
nclasses = len(classes)

## validate
print('Paths:')
print(xlsx_out_pth)
print(f'\nUse simplified classes: {use_simplified_classes}')

############### FUNCTIONS #############################
## Create custom function for zonal stats that better resembles the arc/Q version
def my_hist(lc):
    ''' Gives counts for each integer-valued landcover class, with total number hard-coded in as nclasses.'''
    return np.histogram(np.ndarray.flatten(lc), range=[1,nclasses+1], bins = nclasses)[0] # bin counts

## Function
def extractBufferZonalHist(point, buffer_lengths):
    ''' Buffer_lengths is in map units (probably m).'''
    ## buffer pts 
    buffers = pd.concat([point.buffer(length) for length in buffer_lengths])

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

    # ## Zonal statistics

    ## Run zonal stats
    # stat = zonal_stats(shp, lc[:,:,0], affine=lc_transform, stats='count unique', add_stats = {'histogram':my_hist}, nodata=255)
    # print(stat)
    # stat[0]['histogram']
    # stat

    ## Loop over all years and most buffer lengths (can be sped up by vectorizing buffers: run zonal_stats on multiple features at once)
    # array=np.full([nYears, nclasses, nBuffers], np.nan, dtype='uint32') # init array for outpu
    
    ## Init
    dfba = pd.DataFrame(columns = classes + ['Year', 'Buffer_m', 'Join_idx']) # 'df buffer append' # classes.extend(['Year', 'Buffer_m'])
    ## Loop
    n = 0 # init
    for j, ring in enumerate(buffers):
        for i, year in enumerate(range(nYears)):
            ## Zonal stats. Source: https://automating-gis-processes.github.io/CSC/notebooks/L5/zonal-statistics.html
            stat = zonal_stats(ring, lc[:,:,i], affine=lc_transform, stats='count unique', add_stats = {'histogram':my_hist}, nodata=255)
            # array[i,:,j] = stat[0]['histogram']
            # dfb = pd.DataFrame(stat[0]['histogram'][np.newaxis, :] * np.prod(src_res)/10000, columns = classes)
            dfba = dfba.append(pd.DataFrame(stat[0]['histogram'][np.newaxis, :] * np.prod(src_res)/10000, columns=classes), ignore_index=True, verify_integrity=True)
            dfba.loc[n, 'Year'] = years[i]
            dfba.loc[n, 'Buffer_m'] = buffer_lengths[j]
            dfba.loc[n, 'Join_idx'] = point.index.values
            n += 1
        ## convert to pd df in ha
        # dfb = pd.DataFrame(array[:,:,j] * np.prod(src_res)/10000, index = np.arange(1984, 2014+1), columns = classes)  
        # dfb['year'] = np.arange(1984, 2014+1)

        ## append
        # dfba.append(dfb, ignore_index=True, verify_integrity=True)


    ## convert to ha
    # array = array*np.prod(src_res)/10000

    # ## convert to pd df
    # dfb = pd.DataFrame(array[:,:,narrowBufferCount], index = np.arange(1984, 2014+1), columns = classes)
    # dfb['Buffer_m'] = (narrowBufferCount+1) 
    # dfb['Lat'] = point.geometry # TODO: query actual x/y vals. Etc for rest. OR actually add all attribute sof 'point' or 'ring' or join lookup ID
    # dfb['Long'] = point.geometry

    # dfbWide = pd.DataFrame(array[:,:,wideBufferCount], index = np.arange(1984, 2014+1), columns = classes)
    # dfbWide['Buffer_m'] = (wideBufferCount+1) * buffer_lengths[2]
    # dfbWide['Lat'] = point.geometry 
    # dfbWide['Long'] = point.geometry

    return dfba

############### END FUNCTIONS #########################

## roi for cropping
# with fiona.open(pth_roi_in, "r") as shapefile:
#     roi = [feature["geometry"] for feature in shapefile]

## load points
points = gpd.read_file(pth_shp_in) # geodataframe of all lake centers

## Run function in loop
for i in range(len(points)): # i, (_, point) in enumerate(points[:3].iterrows()): # range(4)
    point = points.iloc[i:i+1, :]
    dfba= extractBufferZonalHist(point, buffer_lengths)
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
## write out
df.to_excel(xlsx_out_pth)
print(f'Wrote output: {xlsx_out_pth}')
