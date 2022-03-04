#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subplot time series of precip, brightband, streamflow, snow line, 
and storm accumulations of precip and streamflow. 
Add in basin DEM-cdf-maps at a few points in the BBH time series to illustrate
melt level travel

14 Feb 2022 KH

"""
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.dates import DateFormatter
from Hydrograph.hydrograph import sepBaseflow


from shapely.ops import cascaded_union
from affine import Affine
import rasterio
import matplotlib as mpl
import os

wgs = 'EPSG:4326'
utc = dt.timezone.utc

d1 = dt.datetime(2017,1,1, tzinfo=utc)
d2 = dt.datetime(2017,2,28,23, tzinfo=utc)

ar1_start = dt.datetime(2017,1,7,6, tzinfo=utc)
ar1_end = dt.datetime(2017,1,13,0, tzinfo=utc)

ar2_start = dt.datetime(2017,2,6,0, tzinfo=utc)
ar2_end = dt.datetime(2017,2,11,12, tzinfo=utc)

# %% functions to help read data

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

# %% read in metadata

datroot = '/home/kden/projects/active/phd2_RosDiffs/station_info_2/'
# --- basins
basins = gpd.read_file('/media/kden/LaCie/DATA/geog/sierra_hwbasins/sierra_hwbasins.shp')

# --- stream gages
gages_df = pd.read_csv(datroot+'sierra_nevada_discharge_stations.txt', sep='\t')
gages = get_gdf(gages_df, 'lat','lon')

# --- met stations
met_df = pd.read_csv(datroot+'sierra_nevada_met_stations.txt', sep='\t')
met = get_gdf(met_df, 'lat','lon')

# --- rain gages
ppt_df = pd.read_csv(datroot+'sierra_nevada_precip_stations.txt', sep='\t')
ppt = get_gdf(ppt_df, 'lat','lon')

# --- snow pillows
pillows_df = pd.read_csv(datroot+'sierra_nevada_swe_stations.csv')
pillows = get_gdf(pillows_df, 'Latitude', 'Longitude')

# %% # %% read in data sets

datroot = '/media/kden/hdd1/phd2_RosDiffs/sensors_2/'

# --- FMCW brightband
bbh = read_df(datroot+'sierra_nevada_fmcw_km_janfeb2017.csv')

# --- precipitation
ppt60 = read_df(datroot+'sierra_nevada_precip_60min_meters_wy17.csv')
ppt30 = read_df(datroot+'sierra_nevada_precip_30min_meters_wy17.csv')
ppt20 = read_df(datroot+'sierra_nevada_precip_20min_meters_wy17.csv')
ppt15 = read_df(datroot+'sierra_nevada_precip_15min_meters_wy17.csv')
ppt10 = read_df(datroot+'sierra_nevada_precip_10min_meters_wy17.csv')
ppt2 = read_df(datroot+'sierra_nevada_precip_2min_meters_wy17.csv')

# --- stream discharge
q = read_df(datroot+'sierra_nevada_discharge_m3s_wy17.csv')

# --- RSLE
# rsle = read_df('/media/kden/hdd1/phd2_RosDiffs/supplemental/rsle_scatol_10_tmpfix.csv')
rsle = read_df('/media/kden/hdd1/phd2_RosDiffs/supplemental/rsle_scatol_10_fixed.csv')

# %% discharge calculations -- baseflow separation and cumulative flow

# output data frame of discharge + baseflow in meters
def baseflow_separation(stid): 
    area_km2 = gages['drain_sqkm'].loc[gages['id']==stid].values[0]
    # isolate data frame to prep for baseflow separation algo
    tmp = q[[stid]].copy()
    tmp.columns = ['Total runoff [m^3 s^-1]']
    # execute algo
    tmp_bsep = sepBaseflow(tmp, 15, area_km2)  # 15-minute intervals
    qflow = tmp_bsep[['Total runoff [m^3 s^-1]','Baseflow [m^3 s^-1]']].copy()
    qflow.columns = ['discharge','baseflow']
    # subset 
    qflow = qflow.loc[(qflow.index>=d1) & (qflow.index<=d2)]
    return qflow

# process for a few gages
    
gage_feather = '11402000'
gage_yuba_up = '11413000'
gage_yuba_down = '11418500'
gage_american = '11427000'

qflow_feather = baseflow_separation(gage_feather)
qflow_yuba_up = baseflow_separation(gage_yuba_up)
qflow_yuba_down = baseflow_separation(gage_yuba_down)
qflow_american = baseflow_separation(gage_american)

# express in meters per timestep (to make cumsum easy) (900 seconds per 15 min)
qflow_feather_mpt = qflow_feather * 900 / (gages['drain_sqkm'].loc[gages['id']==gage_feather].values[0]*(1000**2))
qflow_yuba_up_mpt = qflow_yuba_up * 900 / (gages['drain_sqkm'].loc[gages['id']==gage_yuba_up].values[0]*(1000**2))
qflow_yuba_down_mpt = qflow_yuba_down * 900 / (gages['drain_sqkm'].loc[gages['id']==gage_yuba_down].values[0]*(1000**2))
qflow_american_mpt = qflow_american * 900 / (gages['drain_sqkm'].loc[gages['id']==gage_american].values[0]*(1000**2))

# subset and calculate cumulative discharges at each gage
# Feather
feather_cum_ar1 = qflow_feather_mpt['discharge'].loc[(qflow_feather_mpt.index>=ar1_start) & 
                                        (qflow_feather_mpt.index<=ar1_end)].cumsum()
feather_cum_ar2 = qflow_feather_mpt['discharge'].loc[(qflow_feather_mpt.index>=ar2_start) & 
                                        (qflow_feather_mpt.index<=ar2_end)].cumsum()
# Yuba up (goodyears bar)
yuba_up_cum_ar1 = qflow_yuba_up_mpt['discharge'].loc[(qflow_yuba_up_mpt.index>=ar1_start) & 
                                        (qflow_yuba_up_mpt.index<=ar1_end)].cumsum()
yuba_up_cum_ar2 = qflow_yuba_up_mpt['discharge'].loc[(qflow_yuba_up_mpt.index>=ar2_start) & 
                                        (qflow_yuba_up_mpt.index<=ar2_end)].cumsum()
# Yuba down (deer creek)
yuba_down_cum_ar1 = qflow_yuba_down_mpt['discharge'].loc[(qflow_yuba_down_mpt.index>=ar1_start) & 
                                        (qflow_yuba_down_mpt.index<=ar1_end)].cumsum()
yuba_down_cum_ar2 = qflow_yuba_down_mpt['discharge'].loc[(qflow_yuba_down_mpt.index>=ar2_start) & 
                                        (qflow_yuba_down_mpt.index<=ar2_end)].cumsum()
# American
american_cum_ar1 = qflow_american_mpt['discharge'].loc[(qflow_american_mpt.index>=ar1_start) & 
                                        (qflow_american_mpt.index<=ar1_end)].cumsum()
american_cum_ar2 = qflow_american_mpt['discharge'].loc[(qflow_american_mpt.index>=ar2_start) & 
                                        (qflow_american_mpt.index<=ar2_end)].cumsum()

# gather accumulations altogether
qflow_cum_ar1 = pd.concat([feather_cum_ar1, yuba_up_cum_ar1, 
                       yuba_down_cum_ar1, american_cum_ar1], axis=1)
qflow_cum_ar2 = pd.concat([feather_cum_ar2, yuba_up_cum_ar2, 
                       yuba_down_cum_ar2, american_cum_ar2], axis=1)

# %% precip calculations -- all-station median and cumulative precip

precip_ids = list(ppt['id'].loc[ppt['basin'].isin(
    ['SACTO VLY NE', 'FEATHER R','YUBA R','AMERICAN R','BEAR R'])])

# ! -- needed to drop a few stations based on failed swe vs precip test -- 
precip_ids = [x for x in precip_ids if x not in 
              ['RNYC1','DAV','CYVC1','SVL','CSL','OWC']]
# ! --- end edit --- 

# resample subhourly data
ppt2_sub = ppt2.loc[(ppt2.index>=d1) & (ppt2.index<=d2), 
          ppt2.columns.isin(precip_ids)].resample('H').sum()

ppt10_sub = ppt10.loc[(ppt10.index>=d1) & (ppt10.index<=d2), 
          ppt10.columns.isin(precip_ids)].resample('H').sum()

ppt20_sub = ppt20.loc[(ppt20.index>=d1) & (ppt20.index<=d2), 
          ppt20.columns.isin(precip_ids)].resample('H').sum()

ppt15_sub = ppt15.loc[(ppt15.index>=d1) & (ppt15.index<=d2), 
          ppt15.columns.isin(precip_ids)].resample('H').sum()

ppt30_sub = ppt30.loc[(ppt30.index>=d1) & (ppt30.index<=d2), 
          ppt30.columns.isin(precip_ids)].resample('H').sum()

ppt60_sub = ppt60.loc[(ppt60.index>=d1) & (ppt60.index<=d2), 
                      ppt60.columns.isin(precip_ids)]

ppt_all_sub = pd.concat([ppt2_sub, ppt10_sub, ppt15_sub, ppt20_sub, ppt30_sub, ppt60_sub], axis=1)
ppt_all_med = ppt_all_sub.median(axis=1)

# calculate cumulative precip
ppt_cum_ar1 = ppt_all_sub.loc[(ppt_all_sub.index>=ar1_start) & (ppt_all_sub.index<=ar1_end)].cumsum()
ppt_cum_ar2 = ppt_all_sub.loc[(ppt_all_sub.index>=ar2_start) & (ppt_all_sub.index<=ar2_end)].cumsum()
ppt_cum_med_ar1 = ppt_all_med.loc[(ppt_all_med.index>=ar1_start) & (ppt_all_med.index<=ar1_end)].cumsum()
ppt_cum_med_ar2 = ppt_all_med.loc[(ppt_all_med.index>=ar2_start) & (ppt_all_med.index<=ar2_end)].cumsum()


# %% plotting functions

# shade the AR sequences of interest
def shade_storms(ax, ar1_start, ar1_end, ar2_start, ar2_end):
    ax.axvspan(ar1_start, ar1_end, color='grey', alpha=0.4, ec='none')
    ax.axvspan(ar2_start, ar2_end,  color='grey', alpha=0.4, ec='none')
    ax.tick_params(labelsize=12)
    return ax

# plot incremental precip - use ppt_all_med
def plot_pptinc(df, ax): 
    ax.fill_between(df.index, df*0, df*1000, facecolor='cornflowerblue', step='pre')
    # (df*1000).plot(ax=ax, kind='area', color='cornflowerblue')
    ax.tick_params(axis='y', labelsize=12, colors='cornflowerblue')
    ax.set_ylabel('Precipitation (mm)', fontsize=14, color='cornflowerblue')
    ax.spines['top'].set_visible(False)
    return ax

# plot brightband heights
def plot_bbh(bbh, ax): 
    ax.scatter(bbh.index, bbh['ovl'], marker='x', fc='grey', s=30, zorder=10)
    ax.scatter(bbh.index, bbh['cff'], marker='.', fc='none', ec='k', s=45, zorder=15)
    ax.set_ylim(0,3.4)
    ax.legend(['Oroville','Colfax'], fancybox=False, fontsize=12, 
               loc='upper right', bbox_to_anchor=(1,1))
    ax.tick_params(labelsize=12)
    ax.set_ylabel('Snow level (km)', fontsize=14)
    ax.spines['top'].set_visible(False)
    return ax


# plot the discharge in m^3 s^-1 -- use qflow_yuba_up
def plot_discharge(df, ax): 
    ax.plot(df.index, df['baseflow'], c='k', ls='-')
    ax.plot(df.index, df['discharge'], c='k', ls='--')
    ax.legend(['Baseflow','Peak flow'], fancybox=False, fontsize=12, 
              loc='upper right', bbox_to_anchor=(1,1))
    ax.tick_params(labelsize=12)
    ax.set_ylabel('Discharge ($m^3 s^{-1}$)', fontsize=14)
    ax.spines['top'].set_visible(False)
    return ax

# plot the RSLE (regional snow line elevation)
def plot_rsle(df,ax): 
    ax.scatter(df.index, df.values.ravel()/1000, marker='s', s=45, c='cornflowerblue')
    ax.tick_params(labelsize=12, axis='y', colors='cornflowerblue')
    ax.set_ylabel('Snowline Elevation (km)', fontsize=14, color='cornflowerblue')
    ax.spines['top'].set_visible(False)
    return ax

# plot the cumulative precip and discharge together
# (shade the ranges, plot the median)
def plot_accumulations(ppt, pptmed, qflow, ax, leg_flag): 
    # ppt and qflow should be subsetted data to the AR sequence
    # columns are each unique gage
    # --- plot precip
    ax.fill_between(ppt.index, ppt.min(axis=1), ppt.max(axis=1), color='lightgrey', alpha=0.8);
    ax.plot(pptmed.index, pptmed.values.ravel(), c='k', lw=2)
    # --- plot discharge
    ax.fill_between(qflow.index, qflow.min(axis=1), qflow.max(axis=1), 
                    color='cornflowerblue', alpha=0.4);
    ax.plot(qflow.index, qflow.median(axis=1), c='C0', lw=2)
    # --- format axis
    date_form = DateFormatter("%b-%d")
    ax.set_xticklabels(ax.get_xticks(), rotation=30, ha='right', fontsize=15)
    ax.xaxis.set_major_formatter(date_form)
    # --- legend
    if leg_flag==1: 
        ax.legend(['Precipitation','Discharge'], fancybox=False, fontsize=12, 
                  loc='upper center')
    
    # will have to format y axis later
    ax.tick_params(labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

# %% 



# %% now, set up mapping bright band change illustrations across basins

bs = basins.loc[basins['name'].isin(['Feather','Yuba','American'])]
laea = 'EPSG:5070'  # -- just about perfect
my_crs = laea
bs['diz'] = 1
# %% functions to crop raster
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

# %% read in and process DEM

zfn = '/media/kden/LaCie/DATA/geog/dem90m_wus/dem_wgs.tif'
z = rxr.open_rasterio(zfn)
z = z.squeeze() 
# project basins
bsp = bs.to_crs(my_crs)
# project dem
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

# clip to basins
dem = clip_raster(z2, bsp.dissolve('diz'))

# %% function to map out elev band areas

# creates a mask of the DEM between 2 elevations
def isolate_elevband(demraster, elevmin, elevmax): 
    mask = demraster.copy()
    mask.data[mask.data < elevmin] = np.nan
    mask.data[mask.data > elevmax] = np.nan
    mask.data[(mask.data>=elevmin) & (mask.data<=elevmax)] = 1
    return mask

def map_elevband_area(elevmin, elevmax, ax): 
    # get area
    mask = isolate_elevband(dem, elevmin, elevmax)
    masksnow = isolate_elevband(dem, elevmax, 5000)
    # plot
    mask.plot(ax=ax, cmap='autumn', add_colorbar=False, alpha=0.5)
    masksnow.plot(ax=ax, cmap='winter', add_colorbar=False, alpha=0.4)
    bsp.dissolve('diz').plot(ax=ax, fc='none', ec='k'); 
    dem.plot.contour(ax=ax, levels=[elevmin], colors=('r'), lw=0.1, alpha=0.4); 
    dem.plot.contour(ax=ax, levels=[elevmax], colors=('b'), lw=0.1, alpha=0.4); 
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

# %% 






# %% 




# %% plot with gridspec
plt.close('all');
fig = plt.figure(tight_layout=True, figsize=(7.5,9))
gs = fig.add_gridspec(3,2)

ax1 = fig.add_subplot(gs[0,:])
ax2 = fig.add_subplot(gs[1,:])
ax3 = fig.add_subplot(gs[-1,0])
ax4 = fig.add_subplot(gs[-1,1], sharey=ax3)

# plot precip and brightband together
ax1_2 = ax1.twinx()
plot_pptinc(ppt_all_med, ax1)
plot_bbh(bbh, ax1_2)
shade_storms(ax1, ar1_start, ar1_end, ar2_start, ar2_end)
ax1.set_xticklabels('');
ax1_2.set_xticklabels('');

# plot discharge and RSLE together
ax2_2 = ax2.twinx()
plot_rsle(rsle, ax2)
plot_discharge(qflow_yuba_up, ax2_2)
shade_storms(ax2, ar1_start, ar1_end, ar2_start, ar2_end)

# plot cumulative precip and discharge
plot_accumulations(ppt_cum_ar1, ppt_cum_med_ar1, qflow_cum_ar1, ax3, 1)
plot_accumulations(ppt_cum_ar2, ppt_cum_med_ar2, qflow_cum_ar2, ax4, 0)
ax3.set_ylabel('(m)', fontsize=14)

# format dates
date_form = DateFormatter("%d-%b")
ax2.set_xticklabels(ax2.get_xticks(), rotation=30, ha='right', fontsize=15)
ax2.xaxis.set_major_formatter(date_form)

ax3.set_xticklabels(ax3.get_xticks(), rotation=40, ha='right', fontsize=15)
ax3.xaxis.set_major_formatter(date_form)

ax4.set_xticklabels(ax4.get_xticks(), rotation=40, ha='right', fontsize=15)
ax4.xaxis.set_major_formatter(date_form)

# format any other axes
ax1.set_xticklabels(ax1.get_xticks(), rotation=0, fontsize=1)
plt.setp(ax1.get_xticklabels(), visible=False)

# %% add in DEM-brightband illustrations

axA = fig.add_axes([0.25, 0.9, 0.1, 0.1])
axA.patch.set_alpha(0.1)
map_elevband_area(945,3059, axA)  # quick 7J AR rise

axB = fig.add_axes([0.42, 0.8, 0.1, 0.1])
axB.patch.set_alpha(0.1)
map_elevband_area(12,1319,axB)  # avg mid-Jan snow 

axC = fig.add_axes([0.48, 0.91, 0.1, 0.1])
axC.patch.set_alpha(0.1)
map_elevband_area(1915, 3169 ,axC)  # quick 6F AR rise

# %% add subplot letters
# move to upper left corner
ax1.annotate('B', (dt.datetime(2016,12,31,23,tzinfo=utc), 9.5), ha='right', va='top', fontsize=18)
ax2_2.annotate('C', (dt.datetime(2016,12,31,23,tzinfo=utc), 680), ha='right', va='top', fontsize=18)
ax3.annotate('D', (dt.datetime(2017,1,7,15,tzinfo=utc), 0.6), ha='right', va='top', fontsize=18)
ax4.annotate('E', (dt.datetime(2017,2,6,9,tzinfo=utc), 0.6), ha='right', va='top', fontsize=18)

# %% save to file

ofn = '/home/kden/projects/active/phd2_RosDiffs/figures/timeseries/'
ofn += '1_region_hydro_timeseries_v6_python.png'
# ofn += 'test_hydro_timeseries.png'
plt.savefig(ofn, dpi=600, bbox_inches='tight')

























































