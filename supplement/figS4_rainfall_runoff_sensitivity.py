#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 12:59:33 2021

@author: kden

'Experiment' comparing CNRFC rainfall (parted w/ brightband) and discharge 
accumulations. Here do a sensitivity test for 'bending' of the melt level
downward. Minder et al. 2011 says the bending is on average 170 m

"""

import geopandas as gpd
import pandas as pd
import xarray as xr
import rioxarray as rxr
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
from affine import Affine
import rasterio
from Hydrograph.hydrograph import sepBaseflow
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib
from matplotlib.dates import DateFormatter
from skimage import exposure
import matplotlib.patheffects as pe

utc = dt.timezone.utc
wgs = 'EPSG:4326'

ar1_start = dt.datetime(2017,1,7,6, tzinfo=utc)
ar1_end = dt.datetime(2017,1,13,0, tzinfo=utc)

ar2_start = dt.datetime(2017,2,6,0, tzinfo=utc)
ar2_end = dt.datetime(2017,2,11,12, tzinfo=utc)


# %% quick functions to read point data
def read_bbh(fname): 
    df = pd.read_csv(fname, sep='\t', index_col=0)
    df.index = pd.to_datetime(df.index)
    if not df.index.tzinfo: 
        df.index = df.index.tz_localize('UTC')
    return df
def read_df(fname): 
    df = pd.read_csv(fname, index_col=0)
    df.index = pd.to_datetime(df.index)
    # set tz if undefined
    if not df.index.tzinfo: 
        df.index = df.index.tz_localize('UTC')
    return df
def get_gdf(df, lat_name, lon_name): 
    # lat/lon_name == column names for lat/lon coordinates in df
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_name], df[lat_name]))
    gdf.set_crs(wgs, inplace=True)
    return gdf

# %% read bbh and streamflow data -- subset to 1-Nov thru 30-April

d_wy1 = dt.datetime(2016,11,1,0, tzinfo=utc)
# d_wy2 = dt.datetime(2017,4,30,23, tzinfo=utc)
d_wy2 = dt.datetime(2017,5,6,23, tzinfo=utc)

# stream discharge (subset after baseflow separation)
q = read_df('/media/kden/hdd1/phd2_RosDiffs/sensors_2/sierra_nevada_discharge_m3s_wy17.csv')

# brightband -- concat oroville and colfax radars
ovl16 = read_bbh('/media/kden/LaCie/DATA/profilers/FMCW/ovl/bbh_processed/ovl.bbh.2016.txt')
ovl17 = read_bbh('/media/kden/LaCie/DATA/profilers/FMCW/ovl/bbh_processed/ovl.bbh.2017.txt')
cff16 = read_bbh('/media/kden/LaCie/DATA/profilers/FMCW/cff/bbh_processed/cff.bbh.2016.txt')
cff17 = read_bbh('/media/kden/LaCie/DATA/profilers/FMCW/cff/bbh_processed/cff.bbh.2017.txt')
ovl = pd.concat([ovl16,ovl17], axis=0).resample('H').mean()
cff = pd.concat([cff16,cff17], axis=0).resample('H').mean()
bbh = pd.concat([ovl,cff], axis=1)
bbh.columns = ['ovl','cff']
bbh = bbh.loc[(bbh.index>=d_wy1) & (bbh.index<=d_wy2)]

# %% read in study boundary shapes and metadata

basins = gpd.read_file('/media/kden/LaCie/DATA/geog/sierra_hwbasins/sierra_hwbasins.shp')
bs = basins.loc[basins['name'].isin(['Feather','Yuba','American'])]

datroot = '/home/kden/projects/active/phd2_RosDiffs/station_info_2/'
gages_df = pd.read_csv(datroot+'sierra_nevada_discharge_stations.txt', sep='\t')
gages = get_gdf(gages_df, 'lat','lon')
gages = gages.loc[gages['source']=='USGS']
gages = gages.loc[gages['basin'].isin(['Feather','Yuba','American'])]

# %% read in USGS GAGESII features
# points 
gagesii_pts = gpd.read_file('/media/kden/LaCie/DATA/USGS/GAGESII/points_shape/gagesII_9322_sept30_2011.shp')
gagesii_pts = gagesii_pts.to_crs(wgs)
gagesii_pts = gagesii_pts.loc[(gagesii_pts['STATE']=='CA') & 
                              (gagesii_pts['AGGECOREGI']=='WestMnts')]
gagesii_pts = gpd.clip(gagesii_pts, bs)
gagesii_pts15 = gagesii_pts.loc[gagesii_pts['STAID'].isin(gages['id'])]
# shapes
gagesii_shp = gpd.read_file('/media/kden/LaCie/DATA/USGS/GAGESII/boundaries-shapefiles-by-aggeco/bas_nonref_WestMnts.shp')
gagesii_shp = gagesii_shp.loc[gagesii_shp['GAGE_ID'].isin(gagesii_pts['STAID'])]
gagesii_shp = gagesii_shp.to_crs(wgs)
gagesii_shp15 = gagesii_shp.loc[gagesii_shp['GAGE_ID'].isin(gages['id'])]

# %% sanity check -- plot gauges, basins, tributary shapes
fig,ax = plt.subplots(); 
bs.plot(ax=ax, fc='none', ec='r');
gagesii_shp15.plot(ax=ax, fc='lightgrey');
gages.plot(ax=ax, c='k');
gagesii_pts15.apply(lambda x: ax.annotate(s=x.STAID, xy=x.loc['geometry'].coords[0]), axis=1)

# %% read in and extract ERA5 z0c data to profiler points
z0_base = '/media/kden/LaCie/DATA/ERA5/z0/era5_z0_'
fname_dates = ['201611','201612','201701','201702','201703','201704','201705']
z0_fnames = [z0_base+x+'.nc' for x in fname_dates]
# read files
z0_raster = xr.open_mfdataset(z0_fnames, concat_dim='time').deg0l
z0_raster.rio.set_crs(wgs)
# extract to profilers
z0_ovl = z0_raster.sel(longitude = -121.487600, latitude = 39.531800, method='nearest')
z0_cff = z0_raster.sel(longitude = -120.937859, latitude = 39.079756, method='nearest')
# covnert to a data frame
z0 = pd.DataFrame({'time':z0_ovl.time.data, 'ovl':z0_ovl.data, 
                   'cff':z0_cff.data}).set_index('time')
z0.index = z0.index.tz_localize('UTC')

# %% read in the QPE data set
qpe = rxr.open_rasterio('/media/kden/hdd1/phd2_RosDiffs/QPE/qpe_wy2017_6hr.tif')
qpe.data[qpe.data==qpe._FillValue] = np.nan

# redefine time -- NOTE THAT THESE ARE THE END OF THE ACCUMULATION PERIOD
qpe_end_times = pd.date_range(start = dt.datetime(2016,11,1,6, tzinfo=utc), 
                          end = dt.datetime(2017,5,1,0, tzinfo=utc), 
                          freq = '6H')
# LET'S EXPRESS IN START-OF-PERIOD TIME
qpe_times = pd.date_range(start = dt.datetime(2016,11,1,0, tzinfo=utc), 
                          end = dt.datetime(2017,4,30,18, tzinfo=utc), 
                          freq = '6H')
qpe = qpe.rename({'band':'time'})
qpe = qpe.assign_coords(time= qpe.time.astype('datetime64[ns]'))

# %% read in DEM data -- subset and interp to the CNRFC grid
z = rxr.open_rasterio('/media/kden/LaCie/DATA/geog/dem90m_wus/dem_wgs.tif')
z = z.squeeze()
z = z.astype(float)
z = z.where((z.y>=36) & (z.y<=41.5) & (z.x>=-123) & (z.x<=-117), drop=True)
z.data[z.data==z._FillValue] = np.nan
#chop to the CNRFC grid to the DEM and interp the DEM to the grid
qpe = qpe.where((qpe.y>=z.y.data.min()) & (qpe.y<=z.y.data.max()) & 
                (qpe.x>=z.x.data.min()) & (qpe.x<=z.x.data.max()))
dem = z.interp(x=qpe.x.data, y=qpe.y.data, method='linear')

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
# %% 
# separate baseflow from 15-minute gage data given the id -- return df of gauge
def baseflow_separation(stid): 
    area_km2 = gages['drain_sqkm'].loc[gages['id']==stid].values[0]
    # isolate data frame to prep for baseflow separation algo
    tmp = q[[stid]].copy()
    tmp.columns = ['Total runoff [m^3 s^-1]']
    # execute algo
    tmp_bsep = sepBaseflow(tmp, 15, area_km2)  # 15-minute intervals
    qflow = tmp_bsep[['Total runoff [m^3 s^-1]','Baseflow [m^3 s^-1]']].copy()
    qflow.columns = ['discharge','baseflow']
    # let's calculate just the peaks
    qflow['peaks'] = qflow['discharge']-qflow['baseflow']
    # subset 
    # qflow = qflow.loc[(qflow.index>=dstart) & (qflow.index<=dfinish)]
    return qflow

# input is the gage ID for 15 minute data ONLY -- returns qflow6
# these are discharge TOTALS for 6-hrs given 15-minute inputs
# returns qflow6, expressed in meters
def get_6hour_discharge(usgs_id):
    qflow = baseflow_separation(usgs_id)
    qflow_mpt = qflow * 900 / (gages['drain_sqkm'].loc[gages['id']==usgs_id].values[0]*(1000**2))
    qflow6 = qflow_mpt.resample('6H').sum()
    return qflow6

#function to gap-fill BBH given the FMCW ID
# (z0 should be defined from other script)
# returns bb6
def gapfill_bbh(pid):
    # set up regression
    # tmpdf = pd.concat([bbh[[pid]],z0], axis=1)
    tmpdf = pd.concat([bbh[[pid]],z0[[pid]]], axis=1)
    tmpdf = tmpdf.dropna()
    tmpdf.columns = ['bbh','z0c']
    X = np.array(tmpdf['z0c']).reshape(-1,1)
    y = np.array(tmpdf['bbh'])
    # OLS regression
    # reg = linear_model.LinearRegression()
    # reg.fit(X, y)
    # print(reg.score(X,y))
    # try the actualy OSL package from statsmodels (to get more results than rsquared)
    ols = sm.OLS(y, X)
    ols_result = ols.fit()
    print(ols_result.summary())
    # bbh_hat = reg.predict(np.array(z0).reshape(-1,1))
    bbh_hat = ols_result.predict(np.array(z0[pid]).reshape(-1,1))
    df = pd.DataFrame({'time':bbh.index, 'bbh_obs':bbh[pid].values.flatten(),
                       'z0c':z0[pid].values.flatten(), 'bbh_est':bbh_hat}).set_index('time')
    # add a column to gap-fill the observed BBH
    df['bbh_filled'] = df['bbh_obs'].copy()
    df['bbh_filled'].loc[np.isnan(df['bbh_filled'])] = df['bbh_est']
    # resample to 6-hourly
    bb6 = df[['bbh_filled']].resample('6H').mean()
    return bb6

# calculating the brightband-aware rainfall above theg gage
# (gagesii_shp should be defined, and the qpe data set)
# returns qperain
def calculate_qpe_rainfall(usgs_id, pid, bbbend, dem_basin):
    shp = gagesii_shp.loc[gagesii_shp['GAGE_ID']==usgs_id]
    qpe_basin = clip_raster(qpe, shp)
    qperain = pd.DataFrame({'time':bb6.index, 
                            'rain_mm':[np.nan]*len(bb6.index)}).set_index('time')
    for i in range(qperain.shape[0]):
        tmp = qpe_basin[[i]].squeeze()
        brightband = bb6[pid].iloc[i] - bbbend
        tmp.data[dem_basin.data > brightband] = np.nan
        qperain['rain_mm'].iloc[i] = np.nanmean(tmp.data)
    qperain['rain_mm'].loc[np.isnan(qperain['rain_mm'])] = 0
    return qperain

# quick function to plot results on an axis
def plot_accumulations(qflow6, qperain, d1, d2, ax):
    (1000*qflow6[['discharge']].loc[
        (qflow6.index>=d1) & (qflow6.index<=d2)]).cumsum().plot(ax=ax, x_compat=True)
    qperain.loc[
        (qperain.index>=d1) & (qperain.index<=d2)].cumsum().plot(ax=ax, x_compat=True)
    ax.set_xlabel('')
    ax.set_ylabel('')
    return ax

# how about a function for multiple columns of bbh bending?
def calculate_qpe_rainfall_multiple_bbh(usgs_id, pid, bbbend_list, dem_basin):
    # bbbend_list -- list of integer values for bbbend in meters
    for i in range(len(bbbend_list)):
        tmp = calculate_qpe_rainfall(usgs_id, pid, bbbend_list[i], dem_basin)
        tmp.columns = ['bb_bend_'+str(bbbend_list[i])]
        if i==0:
            fulldf = tmp
        else: 
            fulldf = pd.concat([fulldf,tmp], axis=1)
    return fulldf

# %%  create a 6-hourly bbh data set
ovl6 = gapfill_bbh('ovl')  # n=658, R^2=0.963, SE=0.010m, AIC/BIC = 9608/9612
cff6 = gapfill_bbh('cff')  # n=668, R^2=0.968, SE=0.010m, AIC/BIC = 9727/9731

bb6 = pd.concat([ovl6,cff6], axis=1)
bb6.columns = ['ovl','cff']


# %% get 6-hourly flows to ease processing time later

gage_feather = '11402000'   #  ovl
gage_yuba = '11413000'      # ovl
gage_yuba2 = '11418500'  # smaller one, cff
gage_american = '11427000'  # cff

qflow6_feather = get_6hour_discharge(gage_feather)
qflow6_yuba = get_6hour_discharge(gage_yuba)
qflow6_yuba2 = get_6hour_discharge(gage_yuba2)
qflow6_american = get_6hour_discharge(gage_american)

# %% ^ determine gage-radar assignments by proximity

slrs_df = pd.DataFrame({'id':['ovl','cff'], 'lat':[39.531800,39.079756], 
                        'lon':[-121.487600,-120.937859]})
slrs = gpd.GeoDataFrame(slrs_df, geometry=gpd.points_from_xy(slrs_df.lon,slrs_df.lat))
slrs.set_crs(wgs, inplace=True)

# %% get the basin DEMs

shp_feather = gagesii_shp.loc[gagesii_shp['GAGE_ID']==gage_feather]
shp_yuba = gagesii_shp.loc[gagesii_shp['GAGE_ID']==gage_yuba]
shp_yuba2 = gagesii_shp.loc[gagesii_shp['GAGE_ID']==gage_yuba2]
shp_american = gagesii_shp.loc[gagesii_shp['GAGE_ID']==gage_american]

dem_feather = clip_raster(dem, shp_feather)  # [1047 to 2257 m]
dem_yuba = clip_raster(dem, shp_yuba)  # [972 to 2266 m]
dem_yuba2 = clip_raster(dem, shp_yuba2)# [257 to 1283 m]
dem_american = clip_raster(dem, shp_american) # [270 to 2404 m]


# %% function to plot experiment results
# accumulations from rainfall-runoff experiment
def plot_multiple_accumulations(qflow6, qflowvar, qperain, d1, d2, ax, mycmap, legflag):
    cmap = matplotlib.cm.get_cmap(mycmap)
    myqflow6 = 1000*qflow6[[qflowvar]].loc[(qflow6.index>=d1)&(qflow6.index<=d2)]
    myqflow6.cumsum().plot(ax=ax, x_compat=True, lw=2, zorder=100)
    qperain['bb_bend_200'].loc[
        (qperain.index>=d1) & (qperain.index<=d2)].cumsum().plot(
            ax=ax, x_compat=True, 
            color=cmap(0.25))
    qperain['bb_bend_400'].loc[
        (qperain.index>=d1) & (qperain.index<=d2)].cumsum().plot(
            ax=ax, x_compat=True, 
            color=cmap(0.4))
    qperain['bb_bend_600'].loc[
        (qperain.index>=d1) & (qperain.index<=d2)].cumsum().plot(
            ax=ax, x_compat=True, 
            color=cmap(0.6))
    if legflag==1:
        ax.legend(['Discharge', 'Rain ($BBH-200 m$)', 
                   'Rain ($BBH-400 m$)', 'Rain ($BBH-600 m$)'], 
                  fontsize=11, frameon=False)
    else: 
        ax.get_legend().remove()
    ax.set_xlabel('')
    ax.set_ylabel('')
    date_form = DateFormatter("%d-%b")
    ax.xaxis.set_major_formatter(date_form)
    ax.tick_params(labelsize=13)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax

# %% 



# %% 

bbbend_list = [200, 400, 600]

# %% inset a worldview 'antecedent' image into subplot 

def proc_worldview(d): 
    # d should be a datetime object
    # --- develop filname and import
    fname = '/media/kden/hdd1/phd2_RosDiffs/worldview/snapshot-'+d.strftime('%Y-%m-%dT00_00_00Z')+'.tiff'
    f = rxr.open_rasterio(fname)
    f.rio.set_crs(wgs)
    # --- prep color
    rgb = np.dstack(f.data)
    p2, p98 = np.percentile(rgb, (1, 99))                            # Calculate 2nd,98th percentile for updating min/max vals
    rgbStretched = exposure.rescale_intensity(rgb, in_range=(p2, p98)) # Perform contrast stretch on RGB range
    rgbStretched = exposure.adjust_gamma(rgbStretched, 0.5)            # Perform Gamma Correction
    f2 = f.copy()
    f2.data = np.moveaxis(rgbStretched, -1, 0)
    return f2

# %% 
def add_wv_inset(d, basinshape, ax): 
    f = proc_worldview(d)
    basins.plot(ax=ax, fc='none', ec='c')
    basinshape.plot(ax=ax, fc='none', ec='r')
    f.plot.imshow(ax=ax)
    # ax.text(-119.35, 40.55, d.strftime('%d-%b'), ha='right', va='top', fontsize=14, 
    #         bbox={'edgecolor':'k', 'facecolor':'white'})
    ax.set_title('')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # set a buffer around basinshape
    xmin,ymin,xmax,ymax = tuple(basinshape.buffer(0.2).bounds.values.squeeze())
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    # ax.set_xlim(-122.05230251850705, -119.23618972068871)
    # ax.set_ylim(38.46603076742303, 40.635822595250275)
    # ax.text(xmin, ymin, d.strftime('%d-%b'), ha='left', va='bottom', fontsize=10, 
    #         color='C1', fontweight='bold')
    ax.text(xmin, ymin, d.strftime('%d-%b'), ha='left', va='bottom', fontsize=10, 
            color='k', path_effects=[pe.withStroke(linewidth=3, foreground="w")])
            # , bbox={'edgecolor':'lightgrey','facecolor':'C5'})
    
    ax.set_title('')
    return ax
# %%



# %% 
def plot_extended_results(igage, ishp, idem, iqflow6, pid, axs, legflags): 
    # set up the experiment for the gage/shape
    bbbend_list = [200, 400, 600]
    mycmap = 'magma_r'
    qflowvar = 'discharge'
    qperain = calculate_qpe_rainfall_multiple_bbh(igage, pid, bbbend_list, idem)
    # set dates for plotting windows
    d1a = dt.datetime(2016,11,18, tzinfo=utc)
    d2a = dt.datetime(2016,12,31,18,tzinfo=utc)
    
    
    d1b = dt.datetime(2017,1,1, tzinfo=utc)
    d2b = dt.datetime(2017,1,31,18, tzinfo=utc)
    
    d1c = dt.datetime(2017,2,1, tzinfo=utc)
    # d2c = dt.datetime(2017,2,28,18, tzinfo=utc)
    d2c = dt.datetime(2017,3,4,18, tzinfo=utc)

    
    # d1d = dt.datetime(2017,3,1, tzinfo=utc)
    d1d = dt.datetime(2017,3,19, tzinfo=utc)
    # d2d = dt.datetime(2017,4,30,18,tzinfo=utc)
    d2d = dt.datetime(2017,5,6,18,tzinfo=utc)
    # # 1st panel -- early winter Nov-Dec
    plot_multiple_accumulations(iqflow6, qflowvar, qperain, d1a, d2a, axs[0], mycmap, legflags[0])
    # 2nd -- Jan 
    plot_multiple_accumulations(iqflow6, qflowvar, qperain, d1b, d2b, axs[1], mycmap, legflags[1])
    # 3rd -- Feb 
    plot_multiple_accumulations(iqflow6, qflowvar, qperain, d1c, d2c, axs[2], mycmap, legflags[2])
    # 4th -- Mar-April
    plot_multiple_accumulations(iqflow6, qflowvar, qperain, d1d, d2d, axs[3], mycmap, legflags[3])
    # shade in ARs
    axs[1].axvspan(ar1_start, ar1_end, color='lightgrey')
    axs[2].axvspan(ar2_start, ar2_end, color='lightgrey')
# 
def plot_wv_insets(ishp, axs, my_inwidth, my_inheight): 
    my_wvdates = [dt.datetime(2016,11,17), dt.datetime(2016,12,31), dt.datetime(2017,1,27), dt.datetime(2017,3,13)]
    for i in range(4): 
        axL,axB,axW,axH = axs[i].get_position().bounds
        # iax = fig.add_axes([axL+axW-my_inwidth, axB, my_inwidth, my_inheight])
        # ^ change to place in upper left corner
        iax = fig.add_axes([axL, axB+axH-my_inheight, my_inwidth, my_inheight])
        add_wv_inset(my_wvdates[i], ishp, iax)


# %% 





# %% a grand figure -- use gridspec; initialize
plt.close('all')
fig = plt.figure(tight_layout=True, figsize=(15,14))
gs = fig.add_gridspec(4,4)
# top row -- feather 
A1 = fig.add_subplot(gs[0,0])
A2 = fig.add_subplot(gs[0,1])
A3 = fig.add_subplot(gs[0,2])
A4 = fig.add_subplot(gs[0,3])
# 2nd row -- yuba 1
B1 = fig.add_subplot(gs[1,0], sharex=A1)
B2 = fig.add_subplot(gs[1,1], sharex=A2)
B3 = fig.add_subplot(gs[1,2], sharex=A3)
B4 = fig.add_subplot(gs[1,3], sharex=A4)
# 3rd row -- yuba 2
C1 = fig.add_subplot(gs[2,0], sharex=A1)
C2 = fig.add_subplot(gs[2,1], sharex=A2)
C3 = fig.add_subplot(gs[2,2], sharex=A3)
C4 = fig.add_subplot(gs[2,3], sharex=A4)
# 4th row -- american
D1 = fig.add_subplot(gs[3,0], sharex=A1)
D2 = fig.add_subplot(gs[3,1], sharex=A2)
D3 = fig.add_subplot(gs[3,2], sharex=A3)
D4 = fig.add_subplot(gs[3,3], sharex=A4)
# % plot rain+flow
# --- plot Feather
plot_extended_results(gage_feather, shp_feather, dem_feather, 
                      qflow6_feather, 'ovl', [A1,A2,A3,A4], [0,0,1,0])
# --- plot Yuba 1
plot_extended_results(gage_yuba, shp_yuba, dem_yuba, 
                      qflow6_yuba, 'ovl', [B1,B2,B3,B4], [0,0,0,0])
# --- plot Yuba 2
plot_extended_results(gage_yuba2, shp_yuba2, dem_yuba2, 
                      qflow6_yuba2, 'cff', [C1,C2,C3,C4], [0,0,0,0])
# --- plot American
plot_extended_results(gage_american, shp_american, dem_american, 
                      qflow6_american, 'cff', [D1,D2,D3,D4], [0,0,0,0])

# %% plot worldview
my_inwidth = 0.061
my_inheight = 0.091
plot_wv_insets(shp_feather, [A1,A2,A3,A4], my_inwidth, my_inheight)
plot_wv_insets(shp_yuba, [B1,B2,B3,B4], my_inwidth, my_inheight)
plot_wv_insets(shp_yuba2, [C1,C2,C3,C4], my_inwidth, my_inheight)
plot_wv_insets(shp_american, [D1,D2,D3,D4], my_inwidth, my_inheight)

# %% annotate the gauge numbers underneath WV images

# annotate gauge numbers beneath Wv images
date1 = dt.datetime(2016,11,16)
date2 = dt.datetime(2016,12,30,12)
date3 = dt.datetime(2017,1,31)
date4 = dt.datetime(2017,3,17)

A1.text(date1, 215, gage_feather, fontsize=13)
A2.text(date2, 330, gage_feather, fontsize=13)
A3.text(date3, 210, gage_feather, fontsize=13)
A4.text(date4, 125, gage_feather, fontsize=13)
B1.text(date1, 240, gage_yuba, fontsize=13)
B2.text(date2, 300, gage_yuba, fontsize=13)
B3.text(date3, 410, gage_yuba, fontsize=13)
B4.text(date4, 240, gage_yuba, fontsize=13)
C1.text(date1, 190, gage_yuba2, fontsize=13)
C2.text(date2, 250, gage_yuba2, fontsize=13)
C3.text(date3, 260, gage_yuba2, fontsize=13)
C4.text(date4, 150, gage_yuba2, fontsize=13)
D1.text(date1, 230, gage_american, fontsize=13)
D2.text(date2, 320, gage_american, fontsize=13)
D3.text(date3, 330, gage_american, fontsize=13)
D4.text(date4, 200, gage_american, fontsize=13)
# %% add ylabels
fig.text(-0.009, 0.175, 'mm', fontsize=14, rotation=90)
fig.text(-0.009, 0.425, 'mm', fontsize=14, rotation=90)
fig.text(-0.009, 0.65, 'mm', fontsize=14, rotation=90)
fig.text(-0.009, 0.875, 'mm', fontsize=14, rotation=90)
# %% subplot lettering
fig.text(-0.005, 0.99, 'A', fontsize=18)
fig.text(-0.005, 0.76, 'B', fontsize=18)
fig.text(-0.005, 0.52, 'C', fontsize=18)
fig.text(-0.005, 0.28, 'D', fontsize=18)
for axes_list in [[A1,A2,A3,A4],[B1,B2,B3,B4],[C1,C2,C3,C4],[D1,D2,D3,D4]]: 
    for i,ax in zip(['i','ii','iii','iv'], axes_list): 
        ax.annotate(i, xy=(1,0), xycoords='axes fraction', fontsize=16, ha='right', va='bottom')

# %% save

ofname = '/home/kden/projects/active/phd2_RosDiffs/figures/'
# ofname += '5_conceptual_extended_v0.png'
ofname += 'supplemental/S_conceptual_extended_bbh_sensitivity_v2.png'
plt.savefig(ofname, bbox_inches='tight', dpi=400)





