#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot hourly SWE departure from start-of-storm values
Mark values based on brightband to help illustrate when its 'raining' or 'snowing'
Add in SWE-soil moisture colocated insets and DEM illustration of 
post-storm snow line vs the lowest snow pillow elevation

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib as mpl
from matplotlib.dates import DateFormatter
import seaborn as sns
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from shapely.ops import cascaded_union
from affine import Affine
import rasterio
import matplotlib as mpl
import os
import cmasher as cmr
from matplotlib.ticker import FormatStrFormatter

wgs = 'EPSG:4326'
utc = dt.timezone.utc

# read in basins
basins = gpd.read_file('/media/kden/LaCie/DATA/geog/sierra_hwbasins/sierra_hwbasins.shp')
bs = basins.loc[basins['name'].isin(['Feather','Yuba','American'])]
laea = 'EPSG:5070'  # -- just about perfect
my_crs = laea
bs['diz'] = 1
bsp = bs.to_crs(my_crs)

# dates -- 
utc = dt.timezone.utc
d1 = dt.datetime(2017,1,1, tzinfo=utc)
d2 = dt.datetime(2017,2,28,23, tzinfo=utc)
ar1_start = dt.datetime(2017,1,7,6, tzinfo=dt.timezone.utc)
ar1_end = dt.datetime(2017,1,13,0, tzinfo=dt.timezone.utc)
ar2_start = dt.datetime(2017,2,6,0, tzinfo=dt.timezone.utc)
ar2_end = dt.datetime(2017,2,11,12, tzinfo=dt.timezone.utc)

# %% process the DEM -- stay in WGS (z)


# zfn = 'E:/DATA/SPIRES/SierraElevationCAAlbersWGS84.tif'
zfn = '/media/kden/LaCie/DATA/geog/dem90m_wus/dem_wgs.tif'
z = rxr.open_rasterio(zfn)
z = z.squeeze() 
# subset to an extent larger than basins
xmin,ymin,xmax,ymax = bs.buffer(1).cascaded_union.bounds
mask_lon = (z.x >= xmin) & (z.x <= xmax)
mask_lat = (z.y >=ymin) & (z.y <=ymax)
z = z.where(mask_lon & mask_lat, drop=True)
# set fill values and negatives
z.data[z.data==z._FillValue] = np.nan
z.data[z.data<0] = 0

# %% # project dem (z2)

z1 = z.rio.reproject(my_crs)
# z1.data[z1.data == -32768] = np.nan
# %subset to an extent larger than basins
# xmin,ymin,xmax,ymax = bsp.buffer(0.01).cascaded_union.bounds
xmin,ymin,xmax,ymax = bsp.buffer(500).cascaded_union.bounds
mask_lon = (z1.x >= xmin) & (z1.x <= xmax)
mask_lat = (z1.y >=ymin) & (z1.y <=ymax)
z2 = z1.where(mask_lon & mask_lat, drop=True)
# set fill values and negatives
z2.data[z2.data > 4200] = np.nan
z2.data[z2.data<0] = 0

# %% functions to crop a raster

def transform_from_latlon(lat, lon):
    lat = np.asarray(lat); lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale
def rasterize(shapes, coords, latitude='y', longitude='x',fill=np.nan, **kwargs):
    transform = transform_from_latlon(coords[latitude], coords[longitude])
    out_shape = (len(coords[latitude]), len(coords[longitude]))
    raster = rasterio.features.rasterize(shapes, out_shape=out_shape, fill=fill, transform=transform, dtype=float, **kwargs)
    spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
    return xr.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))
def add_shape_coord_from_data_array(xr_da, shp, coord_name):
    shapes = [(shape, n) for n, shape in enumerate(shp.geometry)]
    xr_da[coord_name] = rasterize(shapes, xr_da.coords, longitude='x', latitude='y')
    return xr_da
# add all these functions together
def clip_raster(raster, shape):
    # dissolve if there are multiple shapes
    if shape.shape[0] > 1: 
        shape['diz'] = 1
        shp = shape.dissolve(by='diz')
    else: 
        shp = shape
    tmp = add_shape_coord_from_data_array(raster, shp, 'bdry')
    tmp2 = raster.where(raster.bdry==0, other=np.nan)
    xx = raster-raster+tmp2.data
    return xx

# %% clip to basins
dem = clip_raster(z2, bsp.dissolve('diz'))


# %% crop DEM to basins -- Feather
dem_feather = clip_raster(z2, bsp.loc[bsp['name']=='Feather'])
xminF,yminF,xmaxF,ymaxF = bsp.loc[bsp['name']=='Feather'].buffer(250).cascaded_union.bounds
dem_feather = dem_feather.where((dem_feather.x>=xminF) & (dem_feather.x<=xmaxF) & 
                                (dem_feather.y>=yminF) & (dem_feather.y<=ymaxF), 
                                drop=True)

# %% ceop DEM to basins -- Yuba
bs_yuba = bsp.loc[bsp['name'].isin(['Yuba','American'])]
dem_yuba = clip_raster(z2, bs_yuba)
xminY,yminY,xmaxY,ymaxY = bs_yuba.buffer(250).cascaded_union.bounds
dem_yuba = dem_yuba.where((dem_yuba.x>=xminY) & (dem_yuba.x<=xmaxY) & 
                                (dem_yuba.y>=yminY) & (dem_yuba.y<=ymaxY), 
                                drop=True)

# %% read in metadata and data sets

# ref = pd.read_csv('E:/DATA/project_ROSb/station_info_2/sierra_nevada_swe_stations.csv')
ref = pd.read_csv('/home/kden/projects/active/phd2_RosDiffs/station_info_2/sierra_nevada_swe_stations.csv')
ref.columns = ['id','name','basin','county','ferix','lon','lat','elev_ft','elev_m','operator','fmcw','rain']

# read in hourly swe
# swe = pd.read_csv('E:/DATA/project_ROSb/sensors_2/sierra_nevada_swe_sensor3_meters_wy16_wy21.csv', index_col=0)
swe = pd.read_csv('/media/kden/hdd1/phd2_RosDiffs/sensors_2/sierra_nevada_swe_sensor3_meters_wy16_wy21.csv', index_col=0)
swe.index = pd.to_datetime(swe.index)
swe.values[swe.values<-.01] = np.nan
swe.values[swe.values>3] = np.nan
swe.index = swe.index.tz_localize('UTC')
swe = swe.loc[(swe.index>=d1) & (swe.index<=d2)]

# read in daily swe
# swed = pd.read_csv('E:/DATA/project_ROSa/sierra_nevada_swe_sensor3_meters_wy16_wy21_daily.csv', index_col=0)
swed = pd.read_csv('/media/kden/hdd1/phd2_RosDiffs/ferix_daily_swe_feather_yuba_american_inches.csv', index_col=0)
swed.index = pd.to_datetime(swed.index)
swed.index = swed.index.tz_localize('UTC')
swed = swed*2.54/100

# read in brightband
bbh = pd.read_csv('/media/kden/hdd1/phd2_RosDiffs/sensors_2/sierra_nevada_fmcw_km_janfeb2017.csv', index_col=0)
bbh.index = pd.to_datetime(bbh.index).tz_localize('UTC')

# %% list out stations in ferix groups
feather_ids = list(ref['id'].loc[ref['ferix']=='FEATHER'])
yuba_ids = list(ref['id'].loc[ref['ferix']=='YUBA AMERICAN'])

# %% subset data based on ferix assignments

swe_feather = swe.loc[:, swe.columns.isin(feather_ids)]
swe_feather = swe_feather[['KTL','GRZ','PLP','GOL','HMB','HRK','RTL','BKL','FOR']]
swe_yuba = swe.loc[:, swe.columns.isin(yuba_ids)]
swe_yuba = swe_yuba.drop(columns=['LOS'])
# do for daily data
swed_feather = swed.loc[:, swed.columns.isin(feather_ids)]
swed_yuba = swed.loc[:, swed.columns.isin(yuba_ids)]

# %% function(s) to get the CDF of a basin shape

def calc_basin_cdf(mybasin):
    # crop the DEM to mybasin
    zclip = clip_raster(z, mybasin)
    # sort values
    Xs = zclip.data.flatten()
    Xs = np.sort(Xs[~np.isnan(Xs)])
    # get CDF indices
    n = np.arange(len(Xs))/len(Xs)
    return n,Xs

# %% function -- create a melted "swe departure" data frame with BBH classification
def swe_departure(df, fmcw, d1_plot, d2_plot, dbreak): 
    # subset and sort data by station elevation
    ordered_ref = ref.loc[ref['id'].isin(df.columns), 
                          ['id','elev_m']].sort_values('elev_m',ascending=False)
    stid_list_zordered = list(ordered_ref['id'])
    zval_list_zordered = list(ordered_ref['elev_m'])
    df_sub = df.loc[(df.index>=d1_plot) & (df.index<=d2_plot), 
                    list(stid_list_zordered)] * 1000  # <-- express in mm
    
    # resample brightband data to df timestep
    tstep = np.mean(np.diff(df_sub.index)).total_seconds()/3600  # hours
    bbh_sub = bbh[fmcw].resample(str(int(tstep))+'H').mean()
    bbh_sub = bbh_sub.loc[(bbh_sub.index>=d1_plot) & (bbh_sub.index<=d2_plot)]
    
    # express data as the departure from dbreak
    df_delta = df_sub - df_sub.loc[df_sub.index==dbreak].values.squeeze()
    
    # melt the data frame and populate elevation corresponding to station
    df_delta['time'] = df_delta.index
    dfmelt = pd.melt(df_delta, id_vars='time')
    dfmelt['elev_m'] = [np.nan]*dfmelt.shape[0]
    for irow in range(dfmelt.shape[0]):
        dfmelt['elev_m'].iloc[irow] = ref['elev_m'].loc[ref['id']==dfmelt['variable'].iloc[irow]].values[0]
    
    # classify brightband -- below, none, above ("bbh is <bbh_class> snow pillow")
    dfmelt['bbh_class'] = [np.nan]*dfmelt.shape[0]
    for irow in range(dfmelt.shape[0]): 
        # compare this timestep's bbh to the given row's elevation
        ibbh = bbh_sub.loc[bbh_sub.index == dfmelt.iloc[irow].time].values[0]*1000
        ielev = dfmelt.iloc[irow].elev_m
        # condition for class
        if np.isnan(ibbh): 
            bclass = 'none'
        elif ibbh >= ielev: 
            bclass = 'above'
        elif ibbh < ielev:
            bclass = 'below'
        else: 
            bclass = 'WRONG'
        dfmelt['bbh_class'].iloc[irow] = bclass
    
    # clean up elevations
    dfmelt['elev_m'] = dfmelt['elev_m'].astype(int)
    return dfmelt, stid_list_zordered, zval_list_zordered

# %% function for plotting hourly swe departures

# we should turn this into a function -- same inputs as above
# 2 separate functions for hourly (scatter) vs daily (lines w/o BBH class)
# have additional inputs for whether to add legend and the figure axis to plot on

def plot_hourly_departure(df, fmcw, legtit, d1_plot, d2_plot, d1_ar, d2_ar, dbreak, leg_flag, mycmap, ax): 
    # df -- data set to plot
    # fmcw -- profiler id
    # legtit -- legend title
    # dx_plot -- desired start/end date for plotting window (Jan3;Feb2)
    # dx_ar -- storm sequence start/end dates for shading
    # dbreak -- desired datetime to express SWE departure from
    # leg_flag -- binary, whether or not to add legend(s)
    # mycmap -- string of colormap name to use for station elev gradient
    # ax -- matplotlib figure axis to plot on
    # ---
    # grab melted data for plotting
    dfmelt, stid_list, zval_list = swe_departure(df, fmcw, d1_plot, d2_plot, dbreak)
    # reverse the stid and elev list once
    stid_list.reverse()
    zval_list.reverse()
    # plot data
    s = sns.scatterplot(data=dfmelt, x='time', y='value', hue='elev_m', style='bbh_class',
            markers={'none':'.', 'above':'o', 'below':'^'}, 
            # try classifying by size too
            size='bbh_class', 
            sizes={'none':25, 'above':40, 'below':40},
            palette=sns.color_palette(mycmap, n_colors=len(np.unique(dfmelt.elev_m)))
            #, legend='brief'
            ,ax=ax) 
    # split auto-generated legend into 2 separate legends
    hand_labels = s.get_legend_handles_labels()[1]  # this gets us the list 'elev_m' through 'bbh_class'
    hand_labels[1:(1+len(stid_list))] = [a + ' ('+f"{int(b):,}"+' m)' for a,b in zip(stid_list,zval_list)]
    hand_labels[0] = legtit
    hand_labels[-4:] = ['Melting Level', 'None', 'Below Pillow', 'Above Pillow']
    # split into 2 different sets
    set1_labels = hand_labels[0:len(stid_list)+1]
    set2_labels = hand_labels[-4:]
    set1_handles = s.get_legend_handles_labels()[0][0:len(stid_list)+1]
    set2_handles = s.get_legend_handles_labels()[0][-4:]
    # add the legends to the figure
    if leg_flag==1: 
        leg1 = ax.legend(set1_handles, set1_labels, loc='upper left', 
                         framealpha=0)
        ax.add_artist(leg1)
        ax.legend(set2_handles, set2_labels, loc='lower right', framealpha=0)
    else: 
        s.legend().remove()
    # ancillary lines/shading
    ax.axvline(dbreak, c='dimgrey')
    ax.axhline(0, c='k', ls='--')
    ax.axvspan(d1_ar, d2_ar, color='lightgrey', zorder=0, alpha=0.6)
    ax.set_xlabel('')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ---
    return ax

# %% now, cook up a function to plot the CDF and stations together on an ax

# mycmap should match those of the swe departure plots
def plot_cdf_stations(mybasin, station_list, mycmap, ax): 
    # plot the CDF
    n, Xs = calc_basin_cdf(mybasin)
    ax.plot(Xs, n, c='grey', lw=1)
    # subset swe metadataframe via station_list
    # ! (ensure "ref" is defined in this script, NOT from soilmoisture_v0.py) !
    refsub = ref.loc[ref['id'].isin(station_list),['id','elev_m']].copy()
    # create a column that maps to the nearest CDF value for each station's elev
    refsub['cdf'] = [np.nan]*refsub.shape[0]
    for stid in station_list: 
        # get the index of the nearest elev value
        stid_elev = refsub['elev_m'].loc[refsub['id']==stid].values[0]
        idx = np.argmin(np.abs(Xs-stid_elev))
        # assign it to the appropriate place
        refsub.loc[refsub['id']==stid, 'cdf'] = n[idx]
    refsub = refsub.sort_values('elev_m')
    # plot
    sns.scatterplot(data=refsub, x='elev_m', y='cdf', hue='elev_m', 
                    markers='o', s=105, 
                    palette=sns.color_palette(mycmap, n_colors=len(station_list)), 
                    ax=ax)
    ax.legend([],[], frameon=False)
    # snow line elevations?
    # make the axis only on the right
    ax.yaxis.tick_right()
    # ax.yaxis.set_ticks_position("right")
    # turn off spines on top and left
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('')
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))


# %% 








# %% read in soil moisture data and set up plotting

def read(fname): 
    df = pd.read_csv(fname, index_col=0)
    df.index = pd.to_datetime(df.index)
    # set tz if undefined
    if not df.index.tzinfo: 
        df.index = df.index.tz_localize('UTC')
    # subset
    df2 = df.loc[(df.index>=d1) & (df.index<=d2)].copy()
    return df2

dir1 = '/media/kden/hdd1/phd2_RosDiffs/stations_2/L0/'
csl_fn = '/media/kden/hdd1/phd2_RosDiffs/allsensors_by_station/CSL/soilmoist_pct_wy16_wy21.csv'
csl = read(csl_fn)[['2in']]
blu = read(dir1+'blu.csv')[['sm_10cm_pct']]   # <-- MISSING FEB AND POORLY CALIBRATED

hcsl = csl.resample('H').mean()
hblu = blu.resample('H').mean()

df = pd.concat([hcsl,hblu], axis=1)
df.columns = ['CSL','blu']
soil = df.copy()

d1_plot = dt.datetime(2017,1,3, tzinfo=utc)  # <-- desired start date for plot window (Jan 1, Jan 31)
d2_plot = dt.datetime(2017,1,16, tzinfo=utc) # <-- desired end date for plot window  (Jan 16, Feb 14)
d1_ar = ar1_start         # <-- storm sequence start date (for shading in the figure)
d2_ar = ar1_end           # <-- storm sequence end date (for shading in the figure)

# %% soil-swe plotting function
def plot_swe_soil(stid, d1_plot, d2_plot, d1_ar, d2_ar, ax): 
    if stid=='blu':
        tmp = pd.concat([swe['BLC'], soil[stid]], axis=1)
    else: 
        tmp = pd.concat([swe[stid],soil[stid]], axis=1)
    tmp.columns = ['SWE','VWC']
    ax2 = ax.twinx();
    (tmp['SWE'].loc[(tmp.index>=d1_plot) & (tmp.index<=d2_plot)]*100).plot(ax=ax, c='k')
    tmp['VWC'].loc[(tmp.index>=d1_plot) & (tmp.index<=d2_plot)].plot(ax=ax2, c='C0')
    ax.axvspan(d1_ar, d2_ar, color='lightgrey')
    ax2.tick_params(axis='y',  colors='C0', labelsize=13)
    ax2.set_ylabel('VWC (%)',  color='C0', fontsize=13)
    ax.set_ylabel('SWE (cm)', fontsize=13)
    ax.set_xlabel('')
    ax.tick_params(which='both', labelsize=13)
    ax.minorticks_off()
    ax.patch.set_alpha(0.1)
    return ax

# %% 







# %% set of functions to plot our DEM CDF-map

# creates a mask of the DEM between 2 elevations
def isolate_elevband(demraster, elevmin, elevmax): 
    mask = demraster.copy()
    mask.data[mask.data < elevmin] = np.nan
    mask.data[mask.data > elevmax] = np.nan
    mask.data[(mask.data>=elevmin) & (mask.data<=elevmax)] = 1
    return mask


# shade between two bands
def map_elevband_yuba(demraster, elevmin, elevmax, mycmap, ax): 
    # get area
    mask = isolate_elevband(demraster, elevmin, elevmax)
    # plot
    # try summer and cool, in addition to autumn... maybe the magma and viridis
    mask.plot(ax=ax, cmap=mycmap, add_colorbar=False, alpha=0.5)
    bsp.loc[bsp['name'].isin(['Yuba','American'])].dissolve('diz').plot(ax=ax, fc='none', ec='k'); 
    # plot snowline elevations separately?
    # dem.plot.contour(ax=ax, levels=[elevmin], colors=('r'), lw=0.1, alpha=0.4); 
    # dem.plot.contour(ax=ax, levels=[elevmax], colors=('b'), lw=0.1, alpha=0.4); 
    # remove boundaries
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    return ax
    
def map_elevband_feather(demraster, elevmin, elevmax, mycmap, ax): 
    # get area
    mask = isolate_elevband(demraster, elevmin, elevmax)
    # plot
    # try summer and cool, in addition to autumn... maybe the magma and viridis
    mask.plot(ax=ax, cmap=mycmap, add_colorbar=False, alpha=0.5)
    bsp.loc[bsp['name']=='Feather'].dissolve('diz').plot(ax=ax, fc='none', ec='k'); 
    # plot snowline elevations separately?
    # dem.plot.contour(ax=ax, levels=[elevmin], colors=('r'), lw=0.1, alpha=0.4); 
    # dem.plot.contour(ax=ax, levels=[elevmax], colors=('b'), lw=0.1, alpha=0.4); 
    # remove boundaries
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    return ax


# %% 


# %% put it all together -- part 1 -- "base" plots of time series and CDFs
plt.close('all')
fig = plt.figure(tight_layout=True, figsize=(16,11))
gs = fig.add_gridspec(2,3)
# top row -- feather 
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1], sharey=ax1)
# bottom row -- yuba-american
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1], sharey=ax3)
# ---- plot ---
# feather -- 7J event
plot_hourly_departure(swe_feather, 'ovl', 'Feather', 
                      dt.datetime(2017,1,3, tzinfo=utc), dt.datetime(2017,1,16, tzinfo=utc), 
                      ar1_start, ar1_end, 
                      dt.datetime(2017,1,7,1, tzinfo=utc), 1, 'magma_r', ax1)
# feather -- 6F event
plot_hourly_departure(swe_feather, 'ovl', 'Feather', 
                      dt.datetime(2017,2,2,tzinfo=utc), dt.datetime(2017,2,14,tzinfo=utc),
                      ar2_start, ar2_end, 
                      dt.datetime(2017,2,5,1,tzinfo=utc), 0, 'magma_r', ax2)
# yuba-american -- 7J
plot_hourly_departure(swe_yuba, 'ovl', 'Yuba-American', 
                      dt.datetime(2017,1,3, tzinfo=utc), dt.datetime(2017,1,16, tzinfo=utc), 
                      ar1_start, ar1_end, 
                      dt.datetime(2017,1,7,1, tzinfo=utc), 1, 'viridis_r', ax3)
# yuba-american -- 6F event
plot_hourly_departure(swe_yuba, 'ovl', 'Yuba-American', 
                      dt.datetime(2017,2,2,tzinfo=utc), dt.datetime(2017,2,14,tzinfo=utc),
                      ar2_start, ar2_end, 
                      dt.datetime(2017,2,5,1,tzinfo=utc), 0, 'viridis_r', ax4)

# % next --- plot CDFs (but soils have to be done before it, 
# ... otherwise yaxis disappears...idky)
ax5 = fig.add_subplot(gs[0,2])
ax6 = fig.add_subplot(gs[1,2], sharey=ax5, sharex=ax5)
# [left, bottom, width, height]
axA = fig.add_axes([0.67, 0.78, 0.1, 0.15])
plot_swe_soil('blu', dt.datetime(2017,1,3, tzinfo=utc), 
              dt.datetime(2017,1,16, tzinfo=utc), ar1_start, ar1_end, axA)
axB = fig.add_axes([0.675, 0.31, 0.1, 0.15])
plot_swe_soil('CSL', dt.datetime(2017,2,2, tzinfo=utc), 
              dt.datetime(2017,2,14, tzinfo=utc), ar2_start, ar2_end, axB)
plot_cdf_stations(bs.loc[bs['name'].isin(['Feather'])], feather_ids, 'magma_r', ax5)
plot_cdf_stations(bs.loc[bs['name'].isin(['Yuba','American'])], yuba_ids, 'viridis_r', ax6)

# % add regional snow lines
ax5.axvline(1256, c='k', ls='--')  # <-- Jan 13 snowline
ax6.axvline(1256, c='k', ls='--')
ax5.axvline(1518, c='k', ls='--') # <-- Feb 11 snowline
ax6.axvline(1518, c='k', ls='--')

# %% format axes labels/ticks

ax1.set_xticklabels('')
ax2.set_xticklabels('')
# label axes
ax1.set_ylabel('$\Delta$SWE (mm)', fontsize=14)
ax3.set_ylabel('$\Delta$SWE (mm)', fontsize=14)
ax2.set_ylabel('')
ax4.set_ylabel('')
ax6.set_xlabel('Elevation (m)', fontsize=14)
fig.text(1.001, 0.75, 'Basin Area Fraction', rotation=90, va='center', ha='center', fontsize=14)
fig.text(1.001, 0.25, 'Basin Area Fraction', rotation=90, va='center', ha='center', fontsize=14)

# tick labels
date_form = DateFormatter("%d-%b")
ax3.set_xticklabels(ax2.get_xticks(), fontsize=14)
ax3.xaxis.set_major_formatter(date_form)
ax4.set_xticklabels(ax4.get_xticks(), fontsize=14)
ax4.xaxis.set_major_formatter(date_form)
# 
ax5.yaxis.set_ticklabels(ax5.get_yticks(), fontsize=14)
ax6.yaxis.set_ticklabels(ax6.get_yticks(), fontsize=14)
ax5.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

for ax in [ax1,ax2,ax3,ax4]: 
    ax.set_yticklabels([int(x) for x in ax.get_yticks()], fontsize=14)

ax6.set_xticklabels([int(x) for x in ax6.get_xticks()], fontsize=14)

# %% # %% annotate the snow lines


ax5.annotate('Snowline \n 13-Jan', (1000, 0.4), ha='left', va='top', rotation=90, fontsize=14)
ax5.annotate('Snowline \n 11-Feb', (1570, 0.1), ha='left', va='bottom', rotation=90, fontsize=14)
# lastly lastly, annotate the station of soil moisture plots
blu_text = axA.annotate('BLC', (dt.datetime(2017,1,3,12,tzinfo=utc),320), fontsize=14)
blu_text.set_alpha(1)
axB.annotate('CSL', (dt.datetime(2017,2,2,12,tzinfo=utc),1400), fontsize=14)

# %% annotate axes

ax1.annotate('A', (-0.03, 1.01), fontsize=18, xycoords='axes fraction'); 
ax2.annotate('B', (-0.03, 1.01), fontsize=18, xycoords='axes fraction'); 
ax5.annotate('C', (-0.1, 1.01), fontsize=18, xycoords='axes fraction'); 
ax3.annotate('D', (-0.03, 1.01), fontsize=18, xycoords='axes fraction'); 
ax4.annotate('E', (-0.03, 1.01), fontsize=18, xycoords='axes fraction');
ax6.annotate('F', (-0.1, 1.01), fontsize=18, xycoords='axes fraction'); 
# %% annotate letters
ax1.annotate('A', (-0.03, 1.01), fontsize=18, xycoords='axes fraction'); 
ax2.annotate('B', (-0.03, 1.01), fontsize=18, xycoords='axes fraction'); 
ax5.annotate('C', (-0.1, 1.01), fontsize=18, xycoords='axes fraction'); 
ax3.annotate('D', (-0.03, 1.01), fontsize=18, xycoords='axes fraction'); 
ax4.annotate('E', (-0.03, 1.01), fontsize=18, xycoords='axes fraction');
ax6.annotate('F', (-0.1, 1.01), fontsize=18, xycoords='axes fraction'); 

# %% plot the DEM cdf map!

# Feather (1585.6 to 2541.4)
axF = fig.add_axes([0.855, 0.55, 0.11, 0.2])
map_elevband_feather(dem_feather, 1585.6, 2541.4, 'Greys', axF)
map_elevband_feather(dem_feather, 1256, 1585.6, 'magma_r', axF)
# dem_feather.plot.contour(ax=axF, levels=[1256], colors=('grey'), lw=0.1, alpha=0.4); 
axF.set_title('')

# Yuba-American (1569.720 to 2667.000)
axY = fig.add_axes([0.835,0.08, 0.13, 0.25])
map_elevband_yuba(dem_yuba, 1569.72, 2667, 'Greys', axY)
map_elevband_yuba(dem_yuba, 1256, 1569.72, 'viridis_r', axY)
# dem_yuba.plot.contour(ax=axY, levels=[1256], colors=('grey'), lw=0.1, alpha=0.4)
axY.set_title('')

# %% lastly, re-do the text annotations for soil-swe insets

fig.text(0.675, 0.91, 'BLC', fontsize=13)
fig.text(0.68, 0.44, 'CSL', fontsize=13)

# %% save
ofname = '/home/kden/projects/active/phd2_RosDiffs/figures/timeseries/'
ofname += '3_swe_departures_hourly_v3a.png'

plt.savefig(ofname, bbox_inches='tight', dpi=400)


# %% 
















