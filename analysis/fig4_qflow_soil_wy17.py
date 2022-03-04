#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 13:17:56 2021

@author: kden

Plot ensemble of log-scale winter daily streamflow and soil moisture in smaller
subplots beneath the ensemble -- to show that streamflow recession does not 
get a chance during the winter -- revealing the increasing importance of the 
subsurface in augmenting runoff in the Feb event and throughout the winter


"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime as dt
from Hydrograph.hydrograph import sepBaseflow
import matplotlib.ticker as mticker
import seaborn as sns


wgs = 'EPSG:4326'
utc = dt.timezone.utc

ar1_start = dt.datetime(2017,1,7,6, tzinfo=utc)
ar1_end = dt.datetime(2017,1,13,0, tzinfo=utc)
ar2_start = dt.datetime(2017,2,6,0, tzinfo=utc)
ar2_end = dt.datetime(2017,2,11,12, tzinfo=utc)

# %% data-read functions
def get_gdf(df, lat_name, lon_name): 
    # lat/lon_name == column names for lat/lon coordinates in df
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_name], df[lat_name]))
    gdf.set_crs(wgs, inplace=True)
    return gdf

def read(fname): 
    df = pd.read_csv(fname, index_col=0)
    df.index = pd.to_datetime(df.index)
    # set tz if undefined
    if not df.index.tzinfo: 
        df.index = df.index.tz_localize('UTC')
    return df

# %% read in metadata
datroot = '/home/kden/projects/active/phd2_RosDiffs/station_info_2/'
# --- basins shape
basins = gpd.read_file('/media/kden/LaCie/DATA/geog/sierra_hwbasins/sierra_hwbasins.shp')
bs = basins.loc[basins['name'].isin(['Feather','Yuba','American'])]
bs['diz'] = 1
# --- stream gages shape
# gages_df = pd.read_csv(datroot+'sierra_nevada_discharge_stations.txt', sep='\t')
gages_df = pd.read_csv(datroot+'sierra_nevada_discharge_daily_stations.txt', sep='\t')
gages = get_gdf(gages_df, 'lat','lon')
# redefine the IDs as a string to match the GAGESII IDs
gages['id'] = [str(x) for x in gages['id']]

# %% read in shapes for gages ii -- 
gagesii_shp_fname = '/home/kden/projects/active/phd2_RosDiffs/station_info_2/shapefiles/'
gagesii_shp_fname += 'usgs_gagesii_daily_boundaries.shp'
gagesii_shp = gpd.read_file(gagesii_shp_fname)

# %% read points for gages ii 

gagesii_pts = gpd.read_file('/media/kden/LaCie/DATA/USGS/GAGESII/points_shape/gagesII_9322_sept30_2011.shp')
gagesii_pts = gagesii_pts.loc[gagesii_pts['STAID'].isin(gages['id'])]
gagesii_pts = gagesii_pts.to_crs(wgs)


# %% load in daily streamflow data

qfname = '/media/kden/hdd1/phd2_RosDiffs/sensors_2/sierra_nevada_discharge_daily_m3s_wy17.csv'
qd = read(qfname)

# get a "normalized" flow data -- divide thru by basin size --> depth
qdm = pd.DataFrame({'time':qd.index}).set_index('time')
for igage in qd.columns: 
    
    # convert m3/s rate to daily total depth m
    area_km2 = gagesii_pts['DRAIN_SQKM'].loc[gagesii_pts['STAID']==igage].values[0]
    qdm[igage] = qd[igage] * 86400/ (area_km2*(1000**2))

qdmm = qdm*1000   # milimeters per day

# %% 

# test 
qdmm.plot(c='grey', legend=False, logy=True);

# %% load in soil moisture
# blu, bbd, UCCA, CSL, ALP, SCN -- in-house
# (BLK in the Mokelumne is such a great example, but let's not cherry-pick)
ref = pd.DataFrame({'id':['CSL','bbd','nvc',
                          'SCN','ALP','UCCA','blu','ata','cmn'], 
                    'basin':['Yuba','Yuba','Yuba',
                             'American','American','American','American','American','American'], 
                    'elev':[2103, 1754, 1055, 
                            2675, 2316, 1841, 1604, 1048, 1006]})

dir1 = '/media/kden/hdd1/phd2_RosDiffs/stations_2/L0/'
csl_fn = '/media/kden/hdd1/phd2_RosDiffs/allsensors_by_station/CSL/soilmoist_pct_wy16_wy21.csv'
scn_fn = '/media/kden/hdd1/phd2_RosDiffs/allsensors_by_station/not_in_use_20210603/SCN/allvars_wy16_wy21_level0.csv'
alp_fn = '/media/kden/hdd1/phd2_RosDiffs/allsensors_by_station/not_in_use_20210603/ALP/allvars_wy16_wy21_level0.csv'

blu = read(dir1+'blu.csv')[['sm_10cm_pct','sm_15cm_pct']]
blu.columns = ['10 cm', '15 cm']
bbd = read(dir1+'bbd.csv')[['sm_10cm_pct','sm_15cm_pct']]
bbd.columns = ['10 cm', '15 cm']
ucca = read(dir1+'UCCA.csv')[['sm_5cm_pct']]*100
ucca.columns = ['5 cm']
csl = read(csl_fn)[['2in','8in','20in']]
csl.columns = ['5 cm', '20 cm', '50 cm']
alp = read(alp_fn)[['sm_30cm_pct','sm_60cm_pct']]
alp.columns = ['30 cm', '60 cm']
scn = read(scn_fn)[['sm_30cm_pct','sm_60cm_pct']]
scn.columns = ['30 cm', '60 cm']

# quick-screen ucca's mid-november outlier
ucca.loc[ucca.index==dt.datetime(2016,11,15,18,30,tzinfo=utc)] = np.nan

# %% 

# function to plot soil moisture on an axis bt two dates
def plot_soilmoisture(df, stid, d1, d2, ax, legloc): 
    # use # columns in df to grab colors -- plot on log scale
    df.loc[(df.index>=d1)&(df.index<=d2)].plot(
        #logy=True, 
        color=mypal[:df.shape[1]], ax=ax).legend(loc=legloc);  
    # add AR sequences
    ax.axvspan(ar1_start, ar1_end, color='lightgrey')
    ax.axvspan(ar2_start, ar2_end, color='lightgrey')
    zval = ref['elev'].loc[ref['id']==stid].values[0]
    catch = ref['basin'].loc[ref['id']==stid].values[0]
    ax.set_title(stid+' '+f'{zval:,}'+' m ('+catch+')')
    ax.set_xlabel('')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# %% 

# try using seaborn to sample 3 colors from palette
mypal = sns.color_palette('magma_r', n_colors=3)
d1 = dt.datetime(2016,11,2, tzinfo=utc)
d2 = dt.datetime(2017,3,31,23, tzinfo=utc)

# %% test

fig,axs = plt.subplots(2,3, figsize=(10,5), tight_layout=True, sharex=True)

plot_soilmoisture(blu, 'blu', d1, d2, axs[0,0])
plot_soilmoisture(bbd, 'bbd', d1, d2, axs[0,1])
plot_soilmoisture(ucca, 'UCCA', d1, d2, axs[0,2])
plot_soilmoisture(csl, 'CSL', d1, d2, axs[1,0])
plot_soilmoisture(alp, 'ALP', d1, d2, axs[1,1])
plot_soilmoisture(scn, 'SCN', d1, d2, axs[1,2])

# %% function to plot the "ensemble" and median of gages between dates
def plot_daily_qflowens(df, d1,d2, ax): 
    # df should be qdmm
    df = df.loc[(df.index>=d1) & (df.index<=d2)]
    dfbar = df.median(axis=1)
    df.plot(ax=ax, c='lightsteelblue',alpha=0.8, legend=False, logy=True)
    dfbar.plot(ax=ax, c='C0', lw=2.5, logy=True, legend=False)
    ax.axvspan(ar1_start, ar1_end, color='lightgrey')
    ax.axvspan(ar2_start, ar2_end, color='lightgrey')
    ax.set_xlabel('')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# %% test

fig,ax = plt.subplots()
plot_daily_qflowens(qdmm, d1,d2, ax)




# %% plot on gridspec

plt.close('all')
fig = plt.figure(tight_layout=True, figsize=(12,8))
gs = fig.add_gridspec(7,3)
# ^ use 1st 2 rows as streamflow plot
a = fig.add_subplot(gs[0:3,:])
b = fig.add_subplot(gs[3:5,0])
c = fig.add_subplot(gs[3:5,1], sharex=b)
d = fig.add_subplot(gs[3:5,2], sharex=b)
e = fig.add_subplot(gs[5:,0], sharex=b)
f = fig.add_subplot(gs[5:,1], sharex=b)
g = fig.add_subplot(gs[5:,2], sharex=b)
# lay it on
plot_daily_qflowens(qdmm, d1,d2, a)
plot_soilmoisture(blu, 'blu', d1, d2, b, 'upper right')
plot_soilmoisture(bbd, 'bbd', d1, d2, c, 'upper right')
plot_soilmoisture(ucca, 'UCCA', d1, d2, d, 'upper right')
plot_soilmoisture(csl, 'CSL', d1, d2, e, 'lower right')
plot_soilmoisture(alp, 'ALP', d1, d2, f, 'upper right')
plot_soilmoisture(scn, 'SCN', d1, d2, g, 'upper right')

# %% format lettering / labels
a.set_ylabel('Discharge (mm d$^{-1}$)', fontsize=13)
b.set_ylabel('VWC (%)', fontsize=13)
e.set_ylabel('VWC (%)', fontsize=13)
for ax in [a,b,c,d,e,f,g]: 
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=13)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=13)
# %% add subplot lettering
a.annotate('A', (dt.datetime(2016,11,3,tzinfo=utc), 115), fontsize=18)
b.annotate('B', (dt.datetime(2016,11,5,tzinfo=utc),61), fontsize=18)
c.annotate('C', (dt.datetime(2016,11,5,tzinfo=utc),43), fontsize=18)
d.annotate('D', (dt.datetime(2016,11,5,tzinfo=utc),44), fontsize=18)
e.annotate('E', (dt.datetime(2016,11,5,tzinfo=utc),42), fontsize=18)
f.annotate('F', (dt.datetime(2016,11,5,tzinfo=utc),39.5), fontsize=18)
g.annotate('G', (dt.datetime(2016,11,5,tzinfo=utc),34), fontsize=18)

# %% save

ofname = '/home/kden/projects/active/phd2_RosDiffs/figures/'
ofname+= 'timeseries/4_qflow_soil_winter_succession_v1.png'
plt.savefig(ofname, bbox_inches='tight', dpi=400)























































