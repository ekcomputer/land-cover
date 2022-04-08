#!/usr/bin/env python
# coding: utf-8

'''
Summary
For calculating land cover change within polygons. Used for whitefish lake for Karen Wang and Jonathan Wang ABoVE land cover dataset- multitemporal. Use 'scratch' environment.
EDK 2021.11.19
TODO: 
* Vectorizing: Can use zonal statistics on multiple polygons (buffers) at once, instead of in loop (fine for now, bc quite fast regardless).
* Add original csv/shp attributes from join based on index.
* Check that water normalization only refers to largest/central lake within buffer.
* Compute actual stats, based on /mnt/c/Users/ekyzivat/Dropbox/Matlab/ABoVE/UAVSAR/analysis/lake-timeseries-stats.ipynb
* Loops for ts stats over multiple buffers
* Add watershed buffer
* Load in csv to join in Location (site) names
* Calc mean dist from shoreline in first processing function (need to write custom script)
* IMPORTANT: Find a way to automatically include Lat/Long and any note columns in final spreadsheet (perhaps join in?) Right now, I'm just using a quick fix in Excel.
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
from scipy.stats.mstats import theilslopes
import pymannkendall

## I/O

## Switches (uncomment)
# use_simplified_classes=True
use_simplified_classes=False

## in: base dir = F:\ABoVE2021\Mapping
# pth_shp_in = '/mnt/f/ABoVE2021/Mapping/shp/ABOVE_coordinates_for_Ethan_10-19-21.shp' # points # temp until I get lake geometries for full dataset!
# pth_shp_in = '/mnt/f/ABoVE2021/Mapping/shp/polygon_geom/ABOVE_coordinates_for_Ethan_10-19-21_jn_PADLakesVis.shp' # polygons
pth_shp_in = '/mnt/f/ABoVE2021/Mapping/shp/polygon_geom/combined/ABOVE_coordinates_for_Ethan_10-19-21_geom.shp' # polygons
pth_lc_in = '/mnt/f/Wang-above-land-cover/ABoVE_LandCover_5km_buffer.vrt'
# pth_lc_in_simp = '/mnt/f/Wang-above-land-cover/ABoVE_LandCover_Simplified_Bh04v01.tif' # simplified 10-calss landcover

## out
xlsx_out_pth = '/mnt/f/ABoVE2021/Mapping/out/xlsx/' + os.path.basename(pth_shp_in)[:-4] + '_landCoverBuffers.xlsx' # e.g. /mnt/f/ABoVE2021/Mapping/out/xlsx/ABOVE_coordinates_for_Ethan_10-19-21_jn_PADLakesVis_landCoverBuffers.xlsx
# shp_projected_out_pth = pth_shp_in.replace('_geom.shp', '_albers_geom.shp')
plot_dir = '/mnt/d/pic/above-land-cover'

## buffers
buffer_lengths = (90, 990) # in m # 90, 990 # 1350

## classes for land cover 
classes =       ['Evergreen Forest','Deciduous Forest',	'Mixed Forest',	'Woodland',	'Low Shrub',	'Tall Shrub',	'Open Shrubs',	'Herbaceous',	'Tussock Tundra',	'Sparsely Vegetated',	'Fen',	'Bog',	'Shallows/littoral',	'Barren',	'Water']
classes_dry =   ['Evergreen Forest','Deciduous Forest',	'Mixed Forest',	'Woodland',	'Low Shrub',	'Tall Shrub',	'Open Shrubs',	'Herbaceous',	'Tussock Tundra',	'Sparsely Vegetated',	'Fen',	'Bog',	'Barren']
classes_dry_rn = [item.replace(' ','_').replace('/','_') for item in classes_dry] # rename var too 
classes_wet =   ['Shallows/littoral', 'Water']
classes_simp = ['Evergreen Forest','Deciduous Forest',	'Shrubland', 'Herbaceous',	'Sparsely Vegetated',	'Barren',	'Fen',	'Bog',	'Shallows/littoral', 'Water']
years = np.arange(1984, 2014+1)

## dynamic values
if use_simplified_classes:
    pth_lc_in = pth_lc_in_simp
    classes = classes_simp

nBuffers = len(buffer_lengths)
nclasses = len(classes)
xlsx_out_norm_pth = xlsx_out_pth.replace('.xlsx', '_norm.xlsx') # e.g. /mnt/f/ABoVE2021/Mapping/out/xlsx/ABOVE_coordinates_for_Ethan_10-19-21_jn_PADLakesVis_landCoverBuffers_norm.xlsx
xlsx_out_time_series_features_pth = xlsx_out_pth.replace('.xlsx', '_tsFeatures.xlsx') # e.g. /mnt/f/ABoVE2021/Mapping/out/xlsx/ABOVE_coordinates_for_Ethan_10-19-21_jn_PADLakesVis_landCoverBuffers_tsFeatures.xlsx
xlsx_out_time_series_features_core_pth = xlsx_out_time_series_features_pth.replace('_tsFeatures.xlsx', '_core_tsFeatures.xlsx')
## Create custom function for zonal stats that better resembles the arc/Q version
def my_hist(lc):
    ''' Gives counts for each integer-valued landcover class, with total number hard-coded in as nclasses.'''
    return np.histogram(np.ndarray.flatten(lc), range=[1,nclasses+1], bins = nclasses)[0] # bin counts

## Function
def extractBufferZonalHist(poly, buffer_lengths):
    ''' Function to run in-memory that computes zonal histagram for an arbitrary number of buffers (typically 2).
        
        Inputs:
        Poly:               polygon-geometry geodataframe of one lake            
        Buffer_lengths:     is in map units (probably m).
        
        Output:             dfba: a gdf with attributes for buffer length, year, join index for the lake given by 'poly'
        '''
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
            dfba.loc[n, 'Join_idx'] = poly.Join_idx.values #[0]
            dfba.loc[n, 'Area_m2'] = poly.Area_m2[0] # could do by running pd.concat((dfba, poly), axis='rows'), but that would add a few more columns to a very tall dataframe...
            dfba.loc[n, 'Perim_m2'] = poly.Perim_m2[0]
            n += 1
    return dfba

def extractTimeSeriesForLakes():
    '''
    Runs extractBufferZonalHist in a loop and outputs final data to 'xlsx_out_pth'. Also outputs map-projected shapefile to 'shp_projected_out_pth.'
    '''
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

    ## Calc area and perim (TODO: dist from shoreline)
    polys['Area_m2'] = polys.area
    polys['Perim_m2'] = polys.length

    ## Create gdf of unique polygons (they should already be unique though)
    polys_g = polys.groupby('Sample_nam')
    polys_u = polys_g.first() # unique lakes

    ## Join in area and perim (not needed)
    # polys_u = polys_u.merge(polys.loc[:, ['Sample_nam', 'Area_m2', 'Perim_m2']], left_index=True, right_on='Sample_nam', how='left')

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
    '''
    Loads 'xlsx_out_pth', normalizes water classes by total water and land classes by total land (in buffer). Outputs data to 'xlsx_out_norm_pth'.
    '''
    ## Load
    print('Normalizing land cover...')
    df = pd.read_excel(xlsx_out_pth)

    ## find littoral percent of water areas (TODO: ensure it only comes from largest/central water body within buffer)
    df['Littorals_pct'] = df['Shallows/littoral'] / df.loc[:,classes_wet].sum(axis=1)*100

    ## Find wetland percent, like michela does, by taking: (L+B+F)/(L+B+F+W)*100
    df['Littoral_wetland_pct'] = (df['Shallows/littoral'] + df.Bog + df.Fen)/ (df.loc[:,classes_wet].sum(axis=1) + df.Bog + df.Fen)*100

    ## Find class percent of dry areas
    # normDry = lambda var: df[var] / df.loc[:,classes_dry].sum(axis=1)*100 # just keeping lambda function for practice
    for var in classes_dry:
        df[var + '_pct'] = df[var] / df.loc[:,classes_dry].sum(axis=1)*100
    
    ## Rename cols to remove spaces
    mapper = {var: var.replace(' ','_').replace('/','_') for var in df.columns}
    df.rename(columns=mapper, inplace=True) # rename cols

    ## Lump into groups
    df['Total_inun'] = df.Water + df.Shallows_littoral
    df['Trees'] = df.Evergreen_Forest + df.Deciduous_Forest + df.Mixed_Forest + df.Woodland
    df['Shrubs'] = df.Low_Shrub + df.Tall_Shrub + df.Open_Shrubs
    df['Wetlands'] = df.Fen + df.Bog # + df.Shallows_littoral
    df['Graminoid'] = df.Herbaceous + df.Tussock_Tundra
    df['Sparse'] = df.Barren + df.Sparsely_Vegetated

    ## Find class percent of lumped dry areas
    lumped_classes = ['Trees', 'Shrubs', 'Wetlands', 'Graminoid', 'Sparse']
    for var in lumped_classes:
        df[var + '_pct'] = df[var] / df.loc[:,classes_dry_rn].sum(axis=1)*100

    ## Write out
    df.to_excel(xlsx_out_norm_pth)
    print(f'Wrote normalized output table: {xlsx_out_norm_pth}')

def plotTimeSeries():
    '''
    Loads 'xlsx_out_norm_pth', manipulates data, and creates a multi-facted time-series plot for each lake from the ABoVE landcover dataset, plotting in ha, not normalized, by default. Saves plots to 'plot_dir'
    '''
    ## vars
    buf_len = buffer_lengths[0] # use the smallest (90 m) buffer for plotting

    ## Load
    print('Plotting land cover...')
    df = pd.read_excel(xlsx_out_norm_pth, index_col=0)
    dfg = df.groupby('Lake_name') # ['Lake_name', 
    # group = dfg.get_group('Balloon lake') # formerly ('Balloon lake', buf_len)
    
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
    plot_types = { # dict for plotting params
        'Ha': {'slice': slice(1,16), 'col_wrap': 4, 'subdir':'time-series-by-lake'},
        'Normalized area (%)': {'slice': slice(21,36), 'col_wrap': 4, 'subdir':'time-series-by-lake-norm'},
        'Norm. land group area (%)': {'slice': slice(42, 47), 'col_wrap': 2, 'subdir':'time-series-by-lake-grouped-norm'} # NOTE: columns 34-40 are non-normalized groups
        }
    for type in ['Norm. land group area (%)']: #plot_types: # HERE: Switch to modify which type of plot to use or use full dict
        for lake in dfg.groups:
            group = dfg.get_group(lake)
            dfl = pd.melt(group, id_vars=['Year', 'Buffer_m'], value_vars=df.columns[plot_types[type]['slice']], var_name = 'Class', value_name=type)# data frame long format
            g = sns.FacetGrid(dfl, col="Class", hue="Buffer_m", col_wrap=plot_types[type]['col_wrap'])
            g.map(sns.lineplot, 'Year', type)
            g.add_legend(title="Buffer (m)")
            g.fig.subplots_adjust(top=0.93) # adjust the Figure to add super title
            g.fig.suptitle(f'{lake} ({group.Area_m2.mode()[0]/1e6:.2f} $km^2$)') # used mode, but mean, first, med would give same answer
            # plt.show()
            plt.close()
            g.savefig(os.path.join(plot_dir, plot_types[type]['subdir'], f'time-series-facets-{lake}.png').replace(' ','-'))
            print(lake)
    print('Done plotting.')

def extractTimeSeriesFeatures():
    '''
    Loads data from 'xlsx_out_norm_pth' and reduces each time series for the specified buffer (probably smallest buffer) to a series of features/metrics.
    Outputs data to 'xlsx_out_time_series_features_pth'.
    '''

    ## Load
    print('Calculating time series features...')
    df = pd.read_excel(xlsx_out_norm_pth, index_col=0)

    ## Filter by buffer length
    df.query('Buffer_m == @buffer_lengths[0]', inplace=True)

    ## Group by lake
    dfg = df.groupby('Lake_name')

    ## Take last (year 2014) value as initial features for output df
    stats_last = dfg.last()

    ## Remove unnecessary columns
    stats_last.drop('Year', axis=1, inplace=True)

    ## Compute median vals
    meta_columns = ['Buffer_m', 'Join_idx', 'Area_m2', 'Perim_m2'] # metadata
    stats_median = dfg.median().drop(columns=meta_columns)

    ## Rename stats vars for 2014 
    mapper = {var: (var + '_2014') for var in stats_last.drop(meta_columns, axis=1).columns}
    stats_last.rename(columns=mapper, inplace=True) # rename cols

    ## Rename stats vars for median
    mapper = {var: (var + '_med') for var in stats_median.columns}
    stats_median.rename(columns=mapper, inplace=True) # rename cols

    ## Insert median stats into stats df
    stats = pd.concat((stats_last, stats_median), axis='columns')

    ## Reorder to put meta vars first
    [stats.insert(0, col, stats.pop(col)) for col in meta_columns[-1::-1]] # re-order cols
   
    ## Compute more features
    # dropna?
    # 1 Dynamism, 1.5 RSD of water, 2 trend in water and shrubs, 3 trend in dom veg

    grouped_classes = ['Trees', 'Shrubs', 'Wetlands', 'Graminoid', 'Sparse']

    stats['Total_inun_RSD'] = dfg.Total_inun.std()/dfg.Total_inun.mean()
    stats['Total_inun_dyn_pct'] = (dfg.Total_inun.max() - dfg.Total_inun.min()) / dfg.Total_inun.max() * 100
    stats['Hi_water_yr'] = dfg.Total_inun.apply(lambda group: years[np.argmax(group)]) # Cool! Use GroupBy.apply to apply a lambda function over all groups!
    stats['Lo_water_yr'] = dfg.Total_inun.apply(lambda group: years[np.argmin(group)]) # Cool! Use GroupBy.apply to apply a lambda function over all groups!
    stats['Dominant_veg_2014'] = dfg.last().loc[:, classes_dry_rn].apply(lambda lake: classes_dry_rn[np.argmax(lake)], axis='columns')
    stats['Dominant_veg_group_2014'] = dfg.last().loc[:, grouped_classes].apply(lambda lake: grouped_classes[np.argmax(lake)], axis='columns')
    stats['SDF'] = stats.Perim_m2 / (2 * np.sqrt(np.pi * stats.Area_m2))
    stats['Perim_area_ratio'] = stats.Perim_m2 / stats.Area_m2
    # dfg['Year', 'Total_inun'].apply(lambda group: theilslopes(group.Year, group.Total_inun))
    for lcClass in ['Total_inun'] + grouped_classes:
        stats[lcClass + '_change'] = dfg[lcClass].apply(lambda group: theilslopes(group)[0]) # Using method from Kuhn et and Butman 2021, PNAS
        stats[lcClass + '_trend'] = dfg[lcClass].apply(lambda group: pymannkendall.original_test(group)[0])

    ## Write out
    stats.to_excel(xlsx_out_time_series_features_pth, freeze_panes=(1,3))
    print(f'Wrote time series output table: {xlsx_out_time_series_features_pth}')

    ## Save and write out most important stats
    important_vars = [					
        'Area_m2',
        'Perim_m2',
        'Total_inun_2014',
        'Trees_pct_2014',		
        'Shrubs_pct_2014',				
        'Wetlands_pct_2014',				
        'Graminoid_pct_2014',				
        'Sparse_pct_2014',
        'Littorals_pct_2014',
        'Littoral_wetland_pct_2014',
        'Total_inun_RSD',				
        'Total_inun_dyn_pct',			
        'Hi_water_yr',					
        'Lo_water_yr',					
        'Dominant_veg_2014',
        'Dominant_veg_group_2014',
        'SDF',	
        'Perim_area_ratio',						
        'Total_inun_change',			
        'Total_inun_trend'
    ]
    stats.loc[:, important_vars].to_excel(xlsx_out_time_series_features_core_pth, freeze_panes=(1,1))
    print(f'Wrote time series output table (greatest hits): {xlsx_out_time_series_features_core_pth}')
if __name__ == '__main__':
    # extractTimeSeriesForLakes()
    # normalizeTimeSeries()
    # plotTimeSeries()
    extractTimeSeriesFeatures()