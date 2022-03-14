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
buffer_lengths=np.arange(90, 7000, 90)
narrowBufferCount = 2 # number of buffer iterations. Length = (narrowBufferCount+1)  * buffer_lengths[2]
wideBufferCount = 14 # etc

## classes for land cover 
classes = ['Evergreen Forest','Deciduous Forest',	'Mixed Forest',	'Woodland',	'Low Shrub',	'Tall Shrub',	'Open Shrubs',	'Herbaceous',	'Tussock Tundra',	'Sparsely Vegetated',	'Fen',	'Bog',	'Shallows/littoral',	'Barren',	'Water']
classes_simp = ['Evergreen Forest','Deciduous Forest',	'Shrubland', 'Herbaceous',	'Sparsely Vegetated',	'Barren',	'Fen',	'Bog',	'Shallows/littoral', 'Water']

## dynamic values
if use_simplified_classes:
    pth_lc_in = pth_lc_in_simp
    classes = classes_simp
    # pth_csv_out_inner = pth_csv_out_inner.replace('land-cover-change', 'land-cover-change-10class')
    # pth_csv_out_outer = pth_csv_out_outer.replace('land-cover-change', 'land-cover-change-10class')
    
nBuffers = buffer_lengths.size
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

    ## buffer pts 
    for i, length in enumerate(buffer_lengths):
        if i==0:
            buffers = point.buffer(length) # HERE todo: buffer on just geometry? Try changing back to "enumerate" statement
        else:
            buffers = buffers.append(point.buffer(length), True)

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
    array=np.full([nYears, nclasses, nBuffers], np.nan, dtype='uint32') # init array for outpu
    for j, ring in enumerate(buffers[0:16]):
        for i, year in enumerate(range(nYears)):
            stat = zonal_stats(ring, lc[:,:,i], affine=lc_transform, stats='count unique', add_stats = {'histogram':my_hist}, nodata=255)
            array[i,:,j] = stat[0]['histogram']
    # Source: https://automating-gis-processes.github.io/CSC/notebooks/L5/zonal-statistics.html

    ## convert to ha
    array = array*np.prod(src_res)/10000

    ## convert to pd df
    dfbNarrow = pd.DataFrame(array[:,:,narrowBufferCount], index = np.arange(1984, 2014+1), columns = classes)
    dfbNarrow['Buffer_m'] = (narrowBufferCount+1) * buffer_lengths[2]
    dfbNarrow['Lat'] = point.geometry # TODO: query actual x/y vals. Etc for rest. OR actually add all attribute sof 'point' or 'ring' or join lookup ID
    dfbNarrow['Long'] = point.geometry

    dfbWide = pd.DataFrame(array[:,:,wideBufferCount], index = np.arange(1984, 2014+1), columns = classes)
    dfbWide['Buffer_m'] = (wideBufferCount+1) * buffer_lengths[2]
    dfbWide['Lat'] = point.geometry 
    dfbWide['Long'] = point.geometry

    return dfbNarrow, dfbWide

############### END FUNCTIONS #########################

## roi for cropping
# with fiona.open(pth_roi_in, "r") as shapefile:
#     roi = [feature["geometry"] for feature in shapefile]

## load points
points = gpd.read_file(pth_shp_in) # geodataframe of all lake centers

## Run function in loop
for i in range(2,4): # i, (_, point) in enumerate(points[:3].iterrows()): #TODO: change back
    point = points.iloc[i:i+1, :]
    dfbNarrow, dfbWide = extractBufferZonalHist(point, (narrowBufferCount, wideBufferCount))
    if i==0:
        df = pd.concat(dfbNarrow, dfbWide)
    else:
        df = pd.concat(df, pd.concat(dfbNarrow, dfbWide))
    
    ## Save checkpoint
    if i % 10 == 0:
        df.to_csv(pth_csv_out_outer)

print('done')
## write out
df.to_excel(xlsx_out_pth)
