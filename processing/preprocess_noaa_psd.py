#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 16:19:27 2021

Process raw NOAA PSD data (namely blu and perhaps bdd).

Metadata found at --

blu: https://psl.noaa.gov/data/obs/data/view_data_type_info.php?SiteID=blu&DataOperationalID=4728&OperationalID=1913
bdd: https://psl.noaa.gov/data/obs/data/view_data_type_info.php?SiteID=bbd&DataOperationalID=2109&OperationalID=1005
nvc: https://psl.noaa.gov/data/obs/data/view_data_type_info.php?SiteID=nvc&DataOperationalID=310&OperationalID=105

pressure offset applies to bbd and blu, (blu in <= WY2017 only)

Soil moisture is in reflectometry; use equations to convert to VWC
Likely just use the standard coefficients ... 
Table 4 in https://psl.noaa.gov/data/obs/instruments/SoilWaterContent.pdf

# file structure goes --
# stid/raw/
# - calendar_year (YYYY)
# -- julian_day (JJJ)
# --- FILES: stidYYJJJ.HH

@author: kden
"""
import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import geopandas as gpd

froot = '/media/kden/LaCie/DATA/noaa_psd/'
utc = dt.timezone.utc

# %% function -- read raw file given the hourly datetime and stid

def read_psd_hour(stid, date):
    # construct filename
    fname = froot + stid + '/raw/' + str(date.year) + '/' + date.strftime('%j') + '/'
    fname += stid + str(date.year)[2:] + date.strftime('%j') + '.' + '%02d' % date.hour +'m'
    # set column names based on web metadata (specific to station)
    if stid=='blu':
        colnames = ['logid', 'year', 'jday', 'minute', 'p_mb', 'tair_c', 'rh_pct', 
                    'wv_scalar_mps', 'wv_vector_mps', 'wdir_deg', 'wdir_std_deg', 
                    'rnet_wm2', 'battery_v', 'ppt_mm', 'wv_max_mps', 'snowdepth_m', 
                    'tsoil_10cm_c', 'tsoil_15cm_c', 'soilrefl_10cm_usec', 
                    'soilrefl_15cm_usec']
    elif stid=='bbd':
        colnames = ['logid', 'year', 'jday', 'minute', 'p_mb', 'tair_c', 'rh_pct', 
                    'wv_scalar_mps', 'wv_vector_mps', 'wdir_deg', 'wdir_std_deg', 
                    'rnet_wm2', 'battery_v', 'ppt_mm', 'wv_max_mps', 'tsoil_10cm_c', 
                    'tsoil_15cm_c', 'soilrefl_10cm_usec', 'soilrefl_15cm_usec', 
                    'tsnow','snowdepth_m']
    elif stid=='nvc':
        colnames = ['logid', 'year', 'jday', 'minute', 'tair_c', 'rh_pct', 'battery_v', 
                    'ppt_mm', 'tsoil_10cm_c', 'tsoil_15cm_c', 
                    'soilrefl_10cm_usec', 'soilrefl_15cm_usec']
    elif stid=='ata': 
        colnames = ['logid','year','jday','minute','tair_c','rh_pct','battery_v', 
                    'ppt_mm', 'tsoil_10cm_c', 'tsoil_15cm_c', 
                    'soilrefl_10cm_usec', 'soilrefl_15cm_usec']
    elif stid=='cmn':
        colnames = ['logid','year','jday','minute','tair_c','rh_pct','battery_v', 
                    'ppt_mm', 'tsoil_10cm_c', 'tsoil_15cm_c', 
                    'soilrefl_10cm_usec', 'soilrefl_15cm_usec']
    # get full index of 2-minute data
    this_hour = pd.date_range(date, date+dt.timedelta(seconds=3600-120), freq='120S')
    # important -- the file should be all NAs if the raw directory or file doesn't exist
    if not os.path.exists(fname): 
        f = pd.DataFrame(index=this_hour, columns=colnames)
        f.index.rename('time', inplace=True)
    else: 
        f = pd.read_csv(fname)
        f.columns = colnames
        # construct time (PSD's hoursminutes column has uneven spacing, so just use
        # the tail 2 characters, which works even if minutes are single-digit)
        f['time'] = [dt.datetime(date.year, date.month, date.day, date.hour, 
                     int(str(x)[-2:]), tzinfo=dt.timezone.utc) for x in f['minute']]
        f.set_index('time', inplace=True)
        # reindex to get the full range of 2-minute data
        f = f[~f.index.duplicated()]
        f = f.reindex(this_hour)
    # calculate VWC with the standard quatratic calib eqn 
    c0 = -0.0663
    c1 = -0.0063
    c2 = 0.0007
    f['sm_10cm_pct'] = [100*(c0 + c1*x + c2*x**2) for x in f['soilrefl_10cm_usec']]
    f['sm_15cm_pct'] = [100*(c0 + c1*x + c2*x**2) for x in f['soilrefl_15cm_usec']]
    # return relevant columns (assumes consistent # columns in each raw file)
    df = f.loc[:, f.columns.isin(['p_mb','tair_c','rh_pct','wv_vector_mps','wdir_deg','rnet_wn2', 
                    'ppt_mm','tsoil_10cm_c','tsoil_15cm_c', 'sm_10cm_pct', 'rnet_wm2',
                    'sm_15cm_pct', 'snowdepth_m'])].copy()    
    # ** quick QC **
    # soil moisture above 80 percent (allow some poor calibration)
    if 'sm_10cm_pct' in df.columns: 
        df['sm_10cm_pct'].loc[df['sm_10cm_pct'] > 80] = np.nan
    if 'sm_15cm_pct' in df.columns: 
        df['sm_15cm_pct'].loc[df['sm_15cm_pct'] > 80] = np.nan
    # relative humidity out of bounds
    if 'rh_pct' in df.columns:
        df['rh_pct'].loc[(df['rh_pct']<0)] = np.nan
        df['rh_pct'].loc[df['rh_pct']>100] = 100
    # air temperature out of 40 C
    if 'tair_c' in df.columns: 
        df['tair_c'].loc[(df['tair_c']<-40) | (df['tair_c']>40)] = np.nan
    # cold soil temperatures
    if 'tsoil_10cm_c' in df.columns: 
        df['tsoil_10cm_c'].loc[df['tsoil_10cm_c'] < -5] = np.nan
    if 'tsoil_15cm_c' in df.columns: 
        df['tsoil_15cm_c'].loc[df['tsoil_15cm_c'] < -5] = np.nan
    # incremental precip
    if 'ppt_mm' in df.columns: 
        df['ppt_mm'].loc[df['ppt_mm']>10] = np.nan
        df['ppt_mm'].loc[df['ppt_mm']<-10] = np.nan
        df['ppt_mm'].loc[df['ppt_mm']<0] = 0
    # 
    # ** end quick QC **
    return df


# %% ROUTINE -- concat 2-minute data for the time period for a station
# nvc -- works; spotty soil for Jan AR
# blu -- cleaned up raw files -- no soil moisture at Feb AR! :(
# bbd -- clean!
# cmn -- clean
# ata -- clean

for stid in ['ata','cmn','nvc','blu','bbd']: 
    # stid = 'nvc'
    d1 = dt.datetime(2016,11,1,0, tzinfo=utc)
    d2 = dt.datetime(2017,3,31,23, tzinfo=utc)
    dates = pd.date_range(d1,d2, freq='H')
    # loop through dates to construct data set
    for idate in dates: 
        tmp = read_psd_hour(stid, idate)
        if idate==dates[0]:
            df = tmp
        else: 
            df = pd.concat([df,tmp], axis=0)
        # verbose progress
        if all([np.mod(idate.day,10)==0, idate.hour==0]): 
            print('catted '+idate.strftime('%d-%b')+'...')
    
    # offset (and quick-QC) pressure
    if 'p_mb' in df.columns: 
        df['p_mb'] = df['p_mb'] + 400
        df['p_mb'].loc[df['p_mb'] < 500] = np.nan
    
    # save to file
    ofname = '/media/kden/LaCie/DATA/noaa_psd/'+stid+'_2min.csv'
    df.to_csv(ofname)
    print('Saved '+stid)

# %% 






# %% map these out to determine which basin they belong in 
# (adding these to our met station list)

wgs = 'EPSG:4326'

def get_gdf(df, lat_name, lon_name): 
    # lat/lon_name == column names for lat/lon coordinates in df
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_name], df[lat_name]))
    gdf.set_crs(wgs, inplace=True)
    return gdf

def read_df(fname): 
    df = pd.read_csv(fname, index_col=0)
    df.index = pd.to_datetime(df.index)
    # set tz if undefined
    if not df.index.tzinfo: 
        df.index = df.index.tz_localize('UTC')
    return df

# %% read in basins

bfname = '/media/kden/LaCie/DATA/geog/sierra_hwbasins/sierra_hwbasins.shp'
basins = gpd.read_file(bfname)

# %% construct gdf of psd data

psdf = pd.DataFrame(
    {'id':['ata','cmn','nvc','blu','bbd'], 
     'lat':[39.1983, 38.7353, 39.3853, 39.2759, 39.309], 
     'lon':[-120.8155, -120.6644, -120.9782, -120.709, -120.518], 
     'elev_m':[1048, 1006, 1055, 1604, 1754], 
     'basin':['American','American','Yuba','American','Yuba']})
psd = get_gdf(psdf, 'lat', 'lon')

# %% 

ax = basins.plot(fc='none', ec='r')
psd.plot(ax=ax)

for x,y,lab in zip(psd.lon, psd.lat, psd.id): 
    ax.annotate(lab, xy=(x,y))

# %% 







