#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

ERA5 synoptic analysis comparing the two events. 

Make an array of subplots 
1st row -- 7J event (a) gph+v500, (b) precip.water+v700, (c) int.T.flux+v850
2nd row -- 6F event (a, b, c) ^^
3rd row -- delta-event:: (a) Tair500, (b) Tair700, (c) Tair850
           ^^also test -- moist static energy for these levels^^

Test sensitiity to the dates that represent when the storm is in full-swing.

Also, develop a table comparing storm-integrated quantities
during the hours that BBH was above a certain height (~1500 m).
- integrated kinetic energy
- precipitable water
- integrated thermal energy
- change in MSE ? -- MSE = cp*T + Lv*q + Phi(<-- geopotential height)

Created on Mon Oct 25 14:10:17 2021

@author: kden

"""


import numpy as np
import pandas as pd
import geopandas as gpd
import datetime as dt
import os
import matplotlib.pyplot as plt
import xarray as xr

# define important times/proj
utc = dt.timezone.utc
wgs = 'EPSG:4326'
ar1_start = dt.datetime(2017,1,7,6, tzinfo=utc)
ar1_end = dt.datetime(2017,1,13,0, tzinfo=utc)
ar2_start = dt.datetime(2017,2,6,0, tzinfo=utc)
ar2_end = dt.datetime(2017,2,11,12, tzinfo=utc)

# %% read in spatial data

# grab basins
basins = gpd.read_file( '/media/kden/LaCie/DATA/geog/sierra_hwbasins/sierra_hwbasins.shp')  # already in wgs
bs = basins.loc[basins['name'].isin(['Feather','Yuba','American'])]
# also, grab the centeroid coordinates
bs_centroid = bs.cascaded_union.centroid

# grab states/north america
# read in state shapefiles to plot over and the metadata
states = gpd.read_file('/media/kden/LaCie/DATA/geog/states/states.shp')
wus_names = ['Washington','Oregon','California','Nevada','Montana',
             'Idaho','Utah','New Mexico','Colorado','Arizona','Wyoming']
wus = states.loc[states['STATE_NAME'].isin(wus_names)]
wus.to_crs(wgs, inplace=True)

# north america
na_fname = '/media/kden/LaCie/DATA/geog/states/north_america_areas/usa_can_mex.shp'
na = gpd.read_file(na_fname)
# na = gpd.read_file('E:/DATA/geog/states/north_america/boundary_l_v2.shp')
na = na.loc[na['COUNTRY'].isin(['CAN','CAN USA','MEX','MEX USA','USA'])].to_crs(wgs)

# %% read in ERA5 data sets

jan = xr.open_dataset('/media/kden/hdd1/phd2_RosDiffs/era5/era5_pressure_jan17.nc')
jan.rio.set_crs(wgs)
feb = xr.open_dataset('/media/kden/hdd1/phd2_RosDiffs/era5/era5_pressure_feb17.nc')
feb.rio.set_crs(wgs)
# combine -- this will take a minute
era = xr.concat([jan,feb], dim='time')
era.rio.set_crs(wgs)

# will need to grab geopotential height separately
gph = xr.open_dataset('/media/kden/hdd1/phd2_RosDiffs/era5/era5_pressure_geopotential_janfeb17.nc')
gph.rio.set_crs(wgs)
gph = gph.z


# %% read in brightband height data




# %% quick function -- grab a slice of data at a time and pressure

# grab a variable at a pressure level
def era_slice(d, p, v): 
    # d-- datetime object -- preferrably timezone-naive
    # p -- pressure level in hPa
    # v -- string of abbreviated variable (q, t, u, v)
    return era[v].sel({'time':d, 'level':p})

# grab the geopotential height at a pressure level
def gph_slice(d, p): 
    return gph.sel({'time':d, 'level':p})  # m2/s2 (divide by gravity to get in m)

# %% functions -- integrated calculations at a time step

# integrated vapor transport
def calc_ivt(d):
    # d -- datetime object -- preferrably timezone-naive
    # -- 
    q = era['q'].sel({'time':d})    # specific humidity (kg/kg)
    u = era['u'].sel({'time':d})    # zonal wind (m/s)
    v = era['v'].sel({'time':d})    # meridional wind (m/s)
    pressures = q.level.data*100    # pressure levels array (Pa)
    # --- wind-weighted humidity; redefine pressures in Pa
    qv = (q*v).assign_coords(level=pressures)
    qu = (q*u).assign_coords(level=pressures)
    # --- integrate across pressures
    iqu = qu.integrate('level')/9.81
    iqv = qv.integrate('level')/9.81
    ivt = np.sqrt(iqu**2 + iqv**2)
    return ivt   # kg m**-1 s**-1

# integrated (approx) heat transport (temperature*spec.heat.of.dry.air)
def calc_itt(d): 
    # d -- datetime object -- preferrably timezone-naive
    # -- 
    t = era['t'].sel({'time':d})    # air temperature (K)
    u = era['u'].sel({'time':d})    # zonal wind (m/s)
    v = era['v'].sel({'time':d})    # meridional wind (m/s)
    pressures = t.level.data*100    # pressure levels array (Pa)
    # --- wind-weighted temperature; redefine pressures in Pa
    tv = (t*v).assign_coords(level=pressures)
    tu = (t*u).assign_coords(level=pressures)
    # --- integrate across pressures
    itv = tv.integrate('level')*(1005/9.81)
    itu = tu.integrate('level')*(1005/9.81)
    itt = np.sqrt(itv**2 + itu**2) * 1e-6
    return itt  # MJ m**-1 s**-1

# precipitable water (vertically integrated humidity)
def calc_pw(d): 
    q = era['q'].sel({'time':d})    # specific humidity (kg/kg)
    pressures = q.level.data*100    # pressure levels array (Pa)
    # integrate across pressures 
    q = q.assign_coords(level=pressures)
    iq = q.integrate('level')/9.81/1000
    # ^ scale by rho_w to go from kg m**-2 to m of water
    return iq
    
# vertically integrated temperature (express in thermal energy)
def calc_thermal(d): 
    t = era['t'].sel({'time':d})    # air temperature (K)
    pressures = t.level.data*100    # pressure levels array (Pa)
    t = t.assign_coords(level=pressures)
    # integrate across pressures
    it = t.integrate('level')*(1005/9.81)  # J m**-2
    return it

# vertically-integrated kinetic energy
def calc_ke(d): 
    u = era['u'].sel({'time':d})    # zonal wind (m/s)
    v = era['v'].sel({'time':d})    # meridional wind (m/s)
    pressures = u.level.data*100    # pressure levels array (Pa)
    # calc kinetic energy at each level
    ke = ((u**2 + v**2)/2).assign_coords(level=pressures)
    # integrate across pressures
    ike = ke.integrate('level')/9.81   # J m**-2
    return ike

# %% derived quantities at a pressure level

# moist static energy
def calc_mse(d, p):
    t = era_slice(d, p, 't')        # air temperature (K)   
    q = era_slice(d, p, 'q')        # specific humidity (kg/kg)
    g = gph_slice(d, p)             # geopotential height (m2/s2, or J/kg)
    cp = 1005  # specific heat capacity of dry air, const. pressure (J kg**-1 K**-1)
    Lv = 2.5e6  # latent heat of vaporization (J kg**-1)
    # *** EDIT -- WE NEED POTENTIAL TEMPERATURE *** 
    Rd = 287  # dry air gas constant (J kg**-1 K**-1)
    p0 = 100000  # reference air pressure (Pa)
    theta = t * ((100*p/p0)**(-Rd/cp))
    mse = cp*theta + Lv*q + g
    # *** END EDIT ***
    # mse = cp*t + Lv*q + g
    return mse   # J kg**-1

# %% a quick function to coarsen WGS data (wind field arrows need to be coarser)
def aggregate(x, n): 
    # x: input xarray DataArray
    # n: desired spatial resolution in the x-y data units
    xreduced = (
        x
        .groupby(((x.longitude//n)+.5)*n).mean(dim='longitude')
        .groupby(((x.latitude//n)+.5)*n).mean(dim='latitude')
    )
    # re-define CRS
    xreduced.rio.set_crs(wgs)
    return xreduced

# %% 



# %% plotting functions 

# constrain the x and y limits of an axis with era5 data
def constrain_ax(ax):
    ax.set_xlim(era.longitude.data.min(), era.longitude.data.max())
    ax.set_ylim(era.latitude.data.min(), era.latitude.data.max())
    ax.set_xlabel('')
    ax.set_ylabel('')
    return ax

# plot the winds at a given level/date on an axis, 
# but only over a certain threshold speed and coarsened to 2 degrees or so
def plot_wind_vectors(d, p, wvmin, res, ax):
    # d -- datetime
    # p -- pressure level in hPa
    # wvmin -- minimum wind speed threshold to display
    # res -- spatial resolution to coarsen data to
    # ---
    # grab winds
    iu = era_slice(d, p, 'u')
    iv = era_slice(d, p, 'v')
    # aggregate
    wv = aggregate(xr.merge([iu,iv]), res)
    # mask lower winds (requires calculating magnitude)
    ws = np.sqrt(wv.u**2 + wv.v**2)
    wv.u.data[ws.data < wvmin] = np.nan
    wv.v.data[ws.data < wvmin] = np.nan
    # lastly, plot
    quiv = ax.quiver(wv.longitude.data, wv.latitude.data, 
                     wv.u.data, wv.v.data, headwidth=4,
                     angles='xy', scale_units='xy', color='r', scale=20)
    # quiverkey might need to be placed ad-hoc
    # ax.quiverkey(quiv, 0.2, .2, 50, coordinates='figure', 
    #              label='50 m $s^{-1}$')
    return quiv


def add_quiver_key(quiv, x, y, wvmag, ax):
    ax.quiverkey(quiv, x, y, wvmag, coordinates='figure', 
                 label=str(round(wvmag,1))+ ' m $s^{-1}$')
    return ax


def add_shapes(ax):
    na.plot(ax=ax, fc='none', ec='grey')
    ax.plot(bs_centroid.x, bs_centroid.y, marker='*', c='w', markersize=18, 
            markeredgecolor='k')


# %% plotting functions panel-by-panel

# GPH and winds
def plot_gph500(d, wvmin, wvmag, x, y, ax, cmap): 
    h = gph_slice(d, 500)
    (h/9.81).plot.contourf(ax=ax, vmin=4900, vmax=6000, levels=11, cmap=cmap,
                           cbar_kwargs={'label':''}, extend='neither')
    quiv = plot_wind_vectors(d, 500, wvmin, res, ax)
    ax.quiverkey(quiv, x, y, wvmag, coordinates='axes', 
                       label='500 hPa\n'+str(round(wvmag,1))+ 'm '+\
                           r'$ \mathbf{s^{-1}}$', 
                       fontproperties={'weight':600})
    

# precipitable water
def plot_pw(d, wvmin, wvmag, x, y, ax, cmap):
    pw = calc_pw(d)
    pw.data[pw.data<0.02] = np.nan
    (pw*1000).plot.contourf(ax=ax, vmin=20, vmax=42, levels=12, cmap=cmap,
                            cbar_kwargs={'label':''}, extend='neither')
    quiv = plot_wind_vectors(d, 700, wvmin, res, ax)
    ax.quiverkey(quiv, x, y, wvmag, coordinates='axes', 
                       label='700 hPa\n'+str(round(wvmag,1))+ 'm '+\
                           r'$ \mathbf{s^{-1}}$', 
                       fontproperties={'weight':600})


# integrated temperature transport
def plot_itt(d, wvmin, wvmag, x, y, ax, cmap):
    tmp = calc_itt(d)/1000  # kJ/m2
    tmp.data[tmp.data<40] = np.nan
    tmp.plot.contourf(ax=ax, vmin=40, vmax=100, levels=11, cmap=cmap,
                      cbar_kwargs={'label':''})
    quiv = plot_wind_vectors(d, 850, wvmin, res, ax)
    ax.quiverkey(quiv, x, y, wvmag, coordinates='axes', 
                       label='850 hPa\n'+str(round(wvmag,1))+ 'm '+\
                           r'$ \mathbf{s^{-1}}$', 
                       fontproperties={'weight':600})


# temperature difference at a level
def plot_delta_tair(d1, d2, p, ax):
    t500A = era_slice(d1, p, 't')   # era_slice or calc_mse
    t500B = era_slice(d2, p, 't')
    dt1 = t500B - t500A
    dt1.plot.contourf(ax=ax, vmin=-16, vmax=16, levels=17, cmap='coolwarm', 
                      cbar_kwargs={'label':''})


# %% plot figure

# original 
d1 = dt.datetime(2017,1,8, 21)
d2 = dt.datetime(2017,2,7, 19)
# sensitivity testing
# d1 = dt.datetime(2017,1,8, 23)
# d2 = dt.datetime(2017,2,7, 23)

wvmag = 40
wvmin = 20
res = 2

x=0.88
y=0.02

cmap='magma_r'
gcmap = 'YlGnBu_r'

plt.close('all')
fig,axs = plt.subplots(3,3, figsize=(15,12), sharex=True, sharey=True
                       # ,tight_layout=True
                       )

# ------- 1st row -- Jan AR u500+GPH;  u700+PW;  u850+Tflux ---- 
plot_gph500(d1, wvmin, wvmag, x, y, axs[0,0], gcmap)
plot_pw(d1, wvmin, wvmag, x, y, axs[0,1], cmap)
plot_itt(d1, wvmin, wvmag, x, y, axs[0,2], cmap)

# ---------  2nd row -- Feb AR  ----------------
plot_gph500(d2, wvmin, wvmag, x, y, axs[1,0], gcmap)
plot_pw(d2, wvmin, wvmag, x, y, axs[1,1], cmap)
plot_itt(d2, wvmin, wvmag, x, y, axs[1,2], cmap)

#  ------- 3rd row -- delta-event:: (a) Tair500, (b) Tair700, (c) Tair850
plot_delta_tair(d1, d2, 500, axs[2,0])
plot_delta_tair(d1, d2, 700, axs[2,1])
plot_delta_tair(d1, d2, 850, axs[2,2])

# format axes 
for ax in axs.flatten(): 
    add_shapes(ax)
    constrain_ax(ax)

# add labels
axs[0,0].set_title('500-hPa Geopotential Height')
axs[0,1].set_title('Precipitable Water')
axs[0,2].set_title('Integrated $T_{air}$')

axs[1,0].set_title('500-hPa Geopotential Height')
axs[1,1].set_title('Precipitable Water')
axs[1,2].set_title('Integrated $T_{air}$')

axs[2,0].set_title('500 hPa')
axs[2,1].set_title('700 hPa')
axs[2,2].set_title('850 hPa')


# add "titles" above colorbars
fig.text(0.32, 0.894, '(m)')
fig.text(0.32, 0.62, '(m)')
fig.text(0.32, 0.35, '(K)')
fig.text(0.59, 0.894, '(mm)')
fig.text(0.59, 0.62, '(mm)')
fig.text(0.59, 0.35, '(K)')
fig.text(0.86, 0.894, '(kJ $m^{-2}$)')
fig.text(0.86, 0.62, '(kJ $m^{-2}$)')
fig.text(0.865, 0.35, '(K)')

# add "letter" labels for each row
fig.text(0.12, 0.9, 
         '(a) '+d1.strftime('%d-%b %H00Z'), 
         fontsize=12, ha='right', va='bottom')
fig.text(0.12, 0.62, 
         '(b) '+d2.strftime('%d-%b %H00Z'), 
         fontsize=12, ha='right', va='bottom')
fig.text(0.12, 0.35, 
         '(c) $\Delta T_{air}$ (b - a)     ', 
         fontsize=12, ha='right', va='bottom')

# %% save 

ofname = '/home/kden/projects/active/phd2_RosDiffs/figures/'
ofname += '4_synoptic_analysis_v0.png'
plt.savefig(ofname, bbox_inches='tight', dpi=600)











# %% read in function to clip raster
from affine import Affine
import rasterio

def transform_from_latlon(lat, lon):
    lat = np.asarray(lat); lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale

def rasterize(shapes, coords, latitude='latitude', 
              longitude='longitude',fill=np.nan, **kwargs):
    transform = transform_from_latlon(coords[latitude], coords[longitude])
    out_shape = (len(coords[latitude]), len(coords[longitude]))
    raster = rasterio.features.rasterize(shapes, out_shape=out_shape, 
                                         fill=fill, transform=transform, dtype=float, **kwargs)
    spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
    return xr.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))

def add_shape_coord_from_data_array(xr_da, shp, coord_name):
    shapes = [(shape, n) for n, shape in enumerate(shp.geometry)]
    xr_da[coord_name] = rasterize(shapes, xr_da.coords, 
                                  longitude='longitude', latitude='latitude')
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


# %% analysis 

# calculate time-integrated (or averaged) quantities over the basin
# quantities of interest --
# -- vapor flux          (function of moisture and wind) -- SUM
# -- thermal flux        (function of Tair and wind)     -- SUM
# -- dMSE 850-500 mb     (function of gph, Tair, q)      -- AVERAGE
#       ^ signifies stability (Dettinger 2004 JHM "orographic ratios")


# procedure --
# -- identify timesteps when BBH is high during storms
# -- concat maps of vertically-integrated quantities for those timesteps
# -- sum up those maps across time (be wary of time units)
#       ^ for MSE, we'll want the temporal average
# -- clip the resulting map over the basins
# -- take the resulting spatial average

# %% functions to read/plot the brightband

def read_df(fname): 
    df = pd.read_csv(fname, index_col=0)
    df.index = pd.to_datetime(df.index)
    # set tz if undefined
    if not df.index.tzinfo: 
        df.index = df.index.tz_localize('UTC')
    return df

def shade_storms(ax, ar1_start, ar1_end, ar2_start, ar2_end):
    ax.axvspan(ar1_start, ar1_end, color='grey', alpha=0.4)
    ax.axvspan(ar2_start, ar2_end,  color='grey', alpha=0.4)
    ax.tick_params(labelsize=12)
    return ax

# plot brightband heights
def plot_bbh(bbh, ax): 
    ax.scatter(bbh.index, bbh['ovl'], marker='x', fc='limegreen', s=30, zorder=10)
    ax.scatter(bbh.index, bbh['cff'], marker='.', fc='none', ec='k', s=45, zorder=15)
    ax.set_ylim(0,3.4)
    ax.legend(['Oroville','Colfax'], fancybox=False, fontsize=12, 
              loc='upper center', bbox_to_anchor=(0.45, 0.99))
    ax.tick_params(labelsize=12)
    ax.set_ylabel('Snow level (km)', fontsize=14)
    return ax

datroot = '/media/kden/hdd1/phd2_RosDiffs/sensors_2/'
bbh = read_df(datroot+'sierra_nevada_fmcw_km_janfeb2017.csv')

bbh_hourly = bbh.resample('H').mean()

bbh_ar1 = bbh_hourly.loc[(bbh_hourly.index>=ar1_start) & 
                         (bbh_hourly.index<=ar1_end), ['ovl','cff']]

bbh_ar2 = bbh_hourly.loc[(bbh_hourly.index>=ar2_start) & 
                         (bbh_hourly.index<=ar2_end), ['ovl','cff']]

# %% 


# %% grab the timesteps associated with each AR being above BBH

z_threshold = 1.6 # km

times_ar1 = bbh_ar1.loc[(bbh_ar1.ovl>=z_threshold) | (bbh_ar1.cff>=z_threshold)].index
times_ar2 = bbh_ar2.loc[(bbh_ar2.ovl>=z_threshold) | (bbh_ar2.cff>=z_threshold)].index

print(len(times_ar1))
print(len(times_ar2))

# for requiring ALL BBHs to be reading and above a threshold --
# for 1600 m -- AR1 has 38 hours; AR2 has 34
# for 2000 m -- AR1=24; AR2=28

# for requiring ANY BBHs to be reading and above a threshold --
# 1600 m -- 58, 62
# 2000 m -- 39, 50

# %% 

# %% functions to get the time-integrated or averaged map for a string of dates


# total IVT over timesteps (kg m**-1)
def aggregate_ivt(my_dates):
    my_dates = [dt.datetime(x.year, x.month, x.day, x.hour) for x in my_dates]
    for i in range(len(my_dates)):
        idate = my_dates[i]
        # *** metric calculation***
        tmp = calc_ivt(idate)*3600   # this gives us accumulated (kg/m) for the hour
        if i==0:
            tmp2 = tmp
        else: 
            tmp2 = xr.concat([tmp,tmp2], dim='time')
    # *** metric aggregation ***
    tmp2agg = tmp2.sum(dim='time')
    tmp2agg.rio.set_crs(wgs)
    return tmp2agg   # this is in kg/m

# total Tair transport over timesteps (MJ m**-1)
def aggregate_itt(my_dates):
    my_dates = [dt.datetime(x.year, x.month, x.day, x.hour) for x in my_dates]
    for i in range(len(my_dates)):
        idate = my_dates[i]
        # *** metric calculation***
        tmp = calc_itt(idate)*3600   # this gives us accumulated (kg/m) for the hour
        if i==0:
            tmp2 = tmp
        else: 
            tmp2 = xr.concat([tmp,tmp2], dim='time')
    # *** metric aggregation ***
    tmp2agg = tmp2.sum(dim='time')
    tmp2agg.rio.set_crs(wgs)
    return tmp2agg 


# average difference in moist static energy (J kg**-1)
# 500 minus 850 hPa --> bigger gradient ~ less stable against convection/uplift
def aggregate_mse(my_dates):
    my_dates = [dt.datetime(x.year, x.month, x.day, x.hour) for x in my_dates]
    for i in range(len(my_dates)):
        idate = my_dates[i]
        # *** metric calculation***
        mse_upper = calc_mse(idate, 500)   # higher-in-atmosphere
        mse_lower = calc_mse(idate, 850)   # lower-in-atmosphere
        tmp = mse_upper - mse_lower
        if i==0:
            tmp2 = tmp
        else: 
            tmp2 = xr.concat([tmp,tmp2], dim='time')
    # *** metric aggregation ***
    tmp2agg = tmp2.mean(dim='time')
    tmp2agg.rio.set_crs(wgs)
    return tmp2agg 


# %% testbed

# tmp1 = aggregate_ivt(times_ar1)
# tmp2 = aggregate_ivt(times_ar2)

# tmp1 = aggregate_itt(times_ar1)
# tmp2 = aggregate_itt(times_ar2)

# tmp1 = aggregate_ike(times_ar1)
# tmp2 = aggregate_ike(times_ar2)

tmp1 = aggregate_mse(times_ar1)
tmp2 = aggregate_mse(times_ar2)

# %% testbed
fig,axs = plt.subplots(1,2, figsize=(9,4), sharex=True, sharey=True)
tmp1.plot(ax=axs[0], vmin=44000, vmax=72000)
tmp2.plot(ax=axs[1], vmin=44000, vmax=72000)
# tmp1.plot.contourf(ax=axs[0], vmin=44000, vmax=72000)
# tmp2.plot.contourf(ax=axs[1], vmin=44000, vmax=72000)
axs[0].set_title('AR 1')
axs[1].set_title('AR 2')

for ax in axs.flatten(): 
    add_shapes(ax)
    constrain_ax(ax)

# %% ROUTINE -- from elev-tolerance to maps and output statistics


def table_analysis(z_threshold):
    times_ar1 = bbh_ar1.loc[(bbh_ar1.ovl>=z_threshold) | (bbh_ar1.cff>=z_threshold)].index
    times_ar2 = bbh_ar2.loc[(bbh_ar2.ovl>=z_threshold) | (bbh_ar2.cff>=z_threshold)].index
    print('Analyzing ARs for moments BBH is above '+str((z_threshold))+' km ...')
    print('... Calculating quantities...')
    tmp1v = aggregate_ivt(times_ar1)
    tmp2v = aggregate_ivt(times_ar2)
    tmp1t = aggregate_itt(times_ar1)
    tmp2t = aggregate_itt(times_ar2)
    tmp1k = aggregate_ike(times_ar1)
    tmp2k = aggregate_ike(times_ar2)
    tmp1m = aggregate_mse(times_ar1)
    tmp2m = aggregate_mse(times_ar2)
    print('... Creating maps ...')
    fig,axs = plt.subplots(4, 2, figsize=(9, 14), sharex=True, sharey=True, tight_layout=True)
    tmp1v.plot(ax=axs[0,0])
    tmp2v.plot(ax=axs[0,1])
    tmp1t.plot(ax=axs[1,0])
    tmp2t.plot(ax=axs[1,1])
    tmp1k.plot(ax=axs[2,0])
    tmp2k.plot(ax=axs[2,1])
    tmp1m.plot(ax=axs[3,0])
    tmp2m.plot(ax=axs[3,1])
    axs[0,0].set_title('AR1 -- $\Sigma$(IVT)')
    axs[0,1].set_title('AR2 -- $\Sigma$(IVT)')
    axs[1,0].set_title('AR1 -- $\Sigma$(ITT)')
    axs[1,1].set_title('AR2 -- $\Sigma$(ITT)')
    axs[2,0].set_title('AR1 -- $\overline{IKE}$')
    axs[2,1].set_title('AR2 -- $\overline{IKE}$')
    axs[3,0].set_title('AR1 -- $\overline{dMSE}$')
    axs[3,1].set_title('AR2 -- $\overline{dMSE}$')
    for ax in axs.flatten(): 
        add_shapes(ax)
        constrain_ax(ax)
    # Calculate statistics
    c1v = round(np.nanmean(clip_raster(tmp1v, bs)))
    c2v = round(np.nanmean(clip_raster(tmp2v, bs)))
    c1t = round(np.nanmean(clip_raster(tmp1t, bs)))
    c2t = round(np.nanmean(clip_raster(tmp2t, bs)))
    c1k = round(np.nanmean(clip_raster(tmp1k, bs)))
    c2k = round(np.nanmean(clip_raster(tmp2k, bs)))
    c1m = round(np.nanmean(clip_raster(tmp1m, bs)))
    c2m = round(np.nanmean(clip_raster(tmp2m, bs)))
    dur1 = len(times_ar1)
    dur2 = len(times_ar2)
    
    print('\n\n----- STATISTICS FOR '+str((z_threshold))+' km ------\n')
    print('AR 1: \n    hours = '+str(dur1)+'\n    IVT = '+str(c1v)+' kg/m \n    ITT = ' \
          + str(c1t)+' MJ/m \n    IKE = '+str(c1k)+' J/m2 \n    dMSE = '+str(c1m)+' J/kg')
    print('AR 2: \n    hours = '+str(dur2)+'\n    IVT = '+str(c2v)+' kg/m \n    ITT = ' \
          + str(c2t)+' MJ/m \n    IKE = '+str(c2k)+' J/m2 \n    dMSE = '+str(c2m)+' J/kg \n\n')
    print('---------------------------------')

# %% 
# threshold in km
# z_threshold = 1.6
# z_threshold = 1.8
z_threshold = 2
table_analysis(z_threshold)


# %% next, consider station data

# define metadata and functions from tinker.met.fluxes.py
# met -- is the geodataframe of metadata
# sub_ar_variable(stid, ar, varname, missing_tolerance) -- return hourly time series

# what if we analyze this in different elevation bands?
# try <1200 (n=8), 1.2-1.6 (n=12), 1.6-2.0 (n=11), >2000 (n=10)

# quick function to take list of stations and concat variable
def station_data(stations, varname, ar):
    stations = list(stations)
    missing_tolerance = 10
    for stid in stations: 
        tmp = sub_ar_variable(stid, ar, varname, missing_tolerance)
        tmp.columns = [stid]
        if stid == stations[0]:
            tmp2 = tmp
        else: 
            tmp2 = pd.concat([tmp2,tmp], axis=1)
    
    df = tmp2.dropna(axis=1, how='all')
    return df

# %% separate into elevation band and calc distribution of a variable

stations1 = met['id'].loc[met['elev_m']<1200]
stations2 = met['id'].loc[(met['elev_m']>=1200) & (met['elev_m']<1600)]
stations3 = met['id'].loc[(met['elev_m']>=1600) & (met['elev_m']<2000)]
stations4 = met['id'].loc[met['elev_m']>=2000]

def station_analysis(stations, varname, ar):
    df = station_data(stations, varname, ar)
    if ar==1:
        my_times = times_ar1
    else: 
        my_times = times_ar2
    df = df.loc[df.index.isin(my_times)]
    print('SAMPLE SIZE = '+str(df.shape[1]))
    q25 = round(np.nanquantile(df.values, 0.25),2)
    q50 = round(np.nanquantile(df.values, 0.5),2)
    q75 = round(np.nanquantile(df.values, 0.75),2)
    print(varname +' distribution for AR '+str(ar)+': \n    Q25 = '+str(q25)+'\n    Q50 = '+str(q50) + 
          '\n    Q75 = '+str(q75))

# %% 

ar = 2
varname = 'tair_c'

print('\n \n---- below 1200 m ----')
station_analysis(stations1, varname, ar)
print('\n \n --- between 1200 -1600 m ---- ')
station_analysis(stations2, varname, ar)
print('\n \n --- between 1600 - 2000 m ----')
station_analysis(stations3, varname, ar)
print('\n \n --- above 2000 m')
station_analysis(stations4, varname, ar)
















