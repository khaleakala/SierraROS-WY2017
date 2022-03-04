#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 08:13:49 2021

@author: kden

Calculate hourly CNRFC data using uniform scaling -- save to a data set

Then, calculate wind-corrected precip using ERA5 -- save to a data set

Also plotted sensitivity to roughness length, concluded 6-12 percent increase

"""

import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from affine import Affine

wgs = 'EPSG:4326'
utm11 = 'EPSG:32611'

# %% read in sapefiles 
states = gpd.read_file('/media/kden/LaCie/DATA/geog/states/states_wgs.shp')
basins = gpd.read_file('/media/kden/LaCie/DATA/geog/sierra_hwbasins/sierra_hwbasins.shp')  
# ^ these are already in wgs

# %% define date range applied to each cnrfc raster layer
utc = dt.timezone.utc
d1 = dt.datetime(2017,1,1,6, tzinfo=utc)
d2 = dt.datetime(2017,3,1,0, tzinfo=utc)
dates = pd.date_range(d1, d2, freq='6H')

# %% read in cnrfc and era5 data sets
cnrfc = xr.open_dataset('/media/kden/hdd1/phd2_RosDiffs/QPE/qpe_janfeb2017_6hr.tif')
cnrfc = cnrfc.rename({'x':'lon', 'y':'lat', 'band':'time'})
cnrfc = cnrfc.assign_coords(time=dates)
cnrfc = cnrfc.assign_coords(time = cnrfc.time.data.astype('datetime64[ns]'))
cnrfc = cnrfc.to_array().squeeze()

era5 = xr.open_dataset('/media/kden/hdd1/phd2_RosDiffs/era5/era5_land_vars.nc')
era5 = era5.tp
era5.rio.set_crs(cnrfc.rio.crs)

# %% 

# %% resample 6-hourly cnrfc to hourly values

# this takes about a minute
for i in range(len(dates)): 
    # create 6 layers for each timestep ... 
    foo = cnrfc[[i]].copy()
    tmp = xr.concat([foo,foo,foo,foo,foo,foo], dim='time')
    # assign the dates corresponding to the given 6-hr timestep 
    idate6 = pd.to_datetime(cnrfc[[i]].time.data[0])
    idates1 = np.flip([idate6-dt.timedelta(hours=x) for x in range(1,7)])
    tmp = tmp.assign_coords(time=idates1)
    # concatenate these 6-hr slices together
    if i==0: 
        cnrfc_1rep = tmp
    else: 
        cnrfc_1rep = xr.concat([cnrfc_1rep, tmp], dim='time')

cnrfc_1hr = cnrfc_1rep/6



# %% save this

ofname = '/media/kden/hdd1/phd2_RosDiffs/QPE/qpe_janfeb2017_1hr_uniform.tif'
cnrfc_1hr = cnrfc_1hr.rename({'lat':'y', 'lon':'x'})
cnrfc_1hr.rio.to_raster(ofname)

x = xr.open_dataset(ofname)
x # this is now hourly from Jan 1 0100 through Feb 28 2300 Z, in WGS84

# %% 









# %%







# %% now, calc wind correction

# read in wind vectors
era5 = xr.open_dataset('/media/kden/hdd1/phd2_RosDiffs/era5/era5_land_vars.nc')
u10 = era5.u10
v10 = era5.v10

# calculate the scalar wind speed
wv_era5 = np.sqrt(u10**2 + v10**2)

# re-scale to CNRFC grid
wv_cnrfc = wv_era5.interp(longitude=cnrfc_1hr.x.data, latitude=cnrfc_1hr.y.data, method='nearest')

# %% as per masuda et al. 2019 

# CR = f(U) = 1 / (1 + mU),  m = 0.0454, U = wind AT RAIN GAGE HEIGHT
# U = Uobs * ((logZ1 - logZ0)/(logZ2 - logZ0)), 

z0 = 0.6    # relative roughness for forest (to be conservative)
z1 = 2      # rain gage height
z2 = 10     # anemometer height

m = 0.0454  # correction coefficient for heated tipping bucket gage

fact = (np.log(z1) - np.log(z0)) / (np.log(z2) - np.log(z0))
print(fact)


# %% turn obtaining wind-corrected recip into a function

def calc_windcorr_ppt(z0, z1, z2, m, wv_cnrfc, cnrfc_1hr): 
    fact = (np.log(z1) - np.log(z0)) / (np.log(z2) - np.log(z0))
    cr = 1/(wv_cnrfc*m*fact + 1)
    cr = cr.rename({'latitude':'y', 'longitude':'x'})
    cnrfc_1hr_wcorr = cnrfc_1hr/cr
    return cnrfc_1hr_wcorr


# %% \
# %% correct the hourly precip

cnrfc_1hr_wcorr = calc_windcorr_ppt(z0, z1, z2, m, wv_cnrfc, cnrfc_1hr)

ppt = cnrfc_1hr.copy()
wppt = cnrfc_1hr_wcorr.copy()

# %% # %% functions to clip a raster to basin extent

# a few long functions to clip raster data to a shapefile
def transform_from_latlon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale

def rasterize(shapes, coords, latitude='y', longitude='x',
              fill=np.nan, **kwargs):
    transform = transform_from_latlon(coords[latitude], coords[longitude])
    out_shape = (len(coords[latitude]), len(coords[longitude]))
    raster = rasterio.features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
    return xr.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))

def add_shape_coord_from_data_array(xr_da, shp, coord_name):
    # 2. create a list of tuples (shapely.geometry, id)
    #    this allows for many different polygons within a .shp file (e.g. States of US)
    shapes = [(shape, n) for n, shape in enumerate(shp.geometry)]
    # 3. create a new coord in the xr_da which will be set to the id in `shapes`
    xr_da[coord_name] = rasterize(shapes, xr_da.coords, 
                               longitude='x', latitude='y')
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


# %% function to calculate precip totals for a basin, date range, and wv-corr parameters
def compare_precip_totals(basin, d1, d2,  z0, z1, z2, m): 
    # subset precip and wind data from dates
    idates = pd.date_range(d1,d2, freq='H')
    pptsub = ppt.loc[np.isin(ppt.time.data, idates)]
    wvsub = wv_cnrfc.loc[np.isin(wv_cnrfc.time.data, idates)]
    # calculate wind correction
    wpptsub = calc_windcorr_ppt(z0,z1,z2,m, wvsub, pptsub)
    # calculate range-wide total
    ptot = pptsub.sum('time')
    wptot = wpptsub.sum('time')
    # clip to basin shape
    pclip = clip_raster(ptot, basin)
    wclip = clip_raster(wptot, basin)
    # calculate the region-averaged precip depth over the time period
    pdepth = np.nanmean(pclip)/1000  # IN METERS
    wdepth = np.nanmean(wclip)/1000  # IN METERS
    return pdepth, wdepth




# %% check out wind-correction difference

z0 = 0.6    # relative roughness for forest (to be conservative)
z1 = 2      # rain gage height
z2 = 10     # anemometer height
m = 0.0454  # correction coefficient for heated tipping bucket gage


basin = basins.loc[basins['name'].isin(['Feather','Yuba','American'])]

# keep tz naive

# d1 = dt.datetime(2017,1,7,6)
# d2 = dt.datetime(2017,1,13,0)

d1 = dt.datetime(2017,2,6,0)
d2 = dt.datetime(2017,2,11,12)

pdepth,wdepth = compare_precip_totals(basin, d1, d2,  z0, z1, z2, m)

pctdiff = 100* (wdepth/pdepth - 1)
print(pctdiff)

# 7.0 percent for entire region in Jan AR (completely conservative values)
# 6.6 percent for Feb AR


# %% how do these change with, say, roughness length? 

zlist = np.arange(0.02, 0.81, 0.01)
pjan = []
pfeb = [] # vectors for percentage

# loop through zlist to calculate ratios of corrected-to-raw precip sensitivity to roughness length
for z in zlist: 
    pdjan, wdjan = compare_precip_totals(basin, dt.datetime(2017,1,7,6), 
                                         dt.datetime(2017,1,13,0),  z, z1, z2, m)
    pjan.append(wdjan/pdjan)
    pdfeb, wdfeb = compare_precip_totals(basin, dt.datetime(2017,2,6,0), 
                                         dt.datetime(2017,2,11,12),  z, z1, z2, m)
    pfeb.append(wdfeb/pdfeb)

df = pd.DataFrame({'z0':zlist, 'pjan':pjan, 'pfeb':pfeb}).set_index('z0')

# %%

df.plot()
plt.legend(['7J', '6F'])
plt.xlabel('Roughness (m)')
plt.ylabel('$PPT_{adj}$ / PPT')

# %% save the figure

ofname = '/home/kden/projects/active/phd2_RosDiffs/figures/supplemental/S_pptwind_vs_roughness.png'
plt.savefig(ofname, dpi=600, bbox_inches='tight')

























































