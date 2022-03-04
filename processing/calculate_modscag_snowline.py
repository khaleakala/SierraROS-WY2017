# -*- coding: utf-8 -*-
"""
Process modscag tile -- and calculate the snow line

"""
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
import geopandas as gpd
import datetime as dt
from shapely.ops import cascaded_union
from affine import Affine
import rasterio
import matplotlib as mpl
import os

wgs = 'EPSG:4326'

# %% function -- Get the tile data in native CRS; raw (fill values still there)
def read_tile(tile, date, varname):
    # directory of files given the day
    fdir = mroot + 'raw/modscag-historic/2017/'+date.strftime('%j')+'/'
    flist = os.listdir(fdir)
    # isolate filename
    fbasename = [x for x in flist if all([tile in x, varname in x])][0]
    f = rxr.open_rasterio(fdir+fbasename)
    f.data = f.data.astype(float)
    # fproj = f.rio.reproject(wgs)
    # fproj.data[fproj.data==fproj._FillValue] = np.nan
    # f.data[f.data==f._FillValue] = np.nan
    return f

# %% function -- combine 2 tiles h08v04 and h08v05
def combine_tiles(date, varname): 
    f1 = read_tile('h08v04', date, varname)
    f2 = read_tile('h08v05', date, varname)
    fcom = f1.combine_first(f2)
    return fcom

# %% READ IN DATA

# SPIRES DEM gives us some funny business reprojecting... use another
# cut the 90-m wus DEM to our general extent
z = rxr.open_rasterio('/media/kden/LaCie/DATA/geog/dem90m_wus/dem_wgs.tif')
z = z.squeeze()
z = z.astype(float)
z = z.where((z.y>=36) & (z.y<=41.5) & (z.x>=-123) & (z.x<=-117), drop=True)
z.data[z.data==z._FillValue] = np.nan

# read in a dummy modscag file (to interp the DEM to)
mds = combine_tiles(dt.datetime(2017,1,24), 'cloud')
z2 = z.rio.reproject(mds.rio.crs)
z2.data[z2.data==z2._FillValue] = np.nan
# ^ success! now we just need to chop mds to size and interp the dem
mds = mds.where((mds.y>=z2.y.data.min()) & (mds.y<=z2.y.data.max()) & 
                (mds.x>=z2.x.data.min()) & (mds.x<=z2.x.data.max()))
dem = z2.interp(x=mds.x.data, y=mds.y.data, method='linear')

# load in basins for reference -- project to modscag
basins = gpd.read_file( '/media/kden/LaCie/DATA/geog/sierra_hwbasins/sierra_hwbasins.shp')  # already in wgs
bs = basins.loc[basins['name'].isin(['Feather','Yuba','American'])]
bsp = bs.to_crs(dem.rio.crs)

# modscag directory
mroot = '/media/kden/LaCie/DATA/MODSCAG/'

# %% 


# %% functions -- clip raster to a basin
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

# %% function -- prep files for RSLE calculation (masking)

def prep_rsle_data(date, basin): 
    # grab sca and subset to the DEM
    sca = combine_tiles(date, 'snow_fraction')
    sca = sca.where((sca.x>=dem.x.data.min()) & (sca.x<=dem.x.data.max()) & 
                    (sca.y>=dem.y.data.min()) * (sca.y<=dem.y.data.max()))
    sca.data[sca.data>100] = np.nan
    # clip dem and sca to basin
    demclip = clip_raster(dem, basin)
    scaclip = clip_raster(sca, basin)
    # get a mask of the basin
    basin_area = xr.ones_like(demclip)
    basin_area = clip_raster(basin_area, basin)
    return scaclip, demclip, basin_area

# %% function -- calculate the RSLE, and return the clearsky fraction
def snowline(date, basin, scatol): 
    # prep input files -- input SCA, DEM, and basin should be in the same CRS ready to go
    scaclip, demclip, basin_area = prep_rsle_data(date, basin)
    # what's the proportion of cloud cover? 
    # (non-nan SCA pixels / non-nan basin pixels ==> clearsky fraction)
    clearsky_fraction = np.sum(~np.isnan(scaclip.data)) / np.sum(~np.isnan(basin_area.data))
    # round the DEM to integers to reduce computational load
    demclip.data = np.round(demclip.data)
    # get snow-free pixels
    snowfree = scaclip.copy().squeeze()
    snowfree.data[snowfree.data > scatol] = np.nan
    snowfree.data[snowfree.data == 0] = 1
    # get sca
    sca = scaclip.copy().squeeze()
    sca.data[sca.data > 100] = np.nan
    sca.data[sca.data >= scatol] = 1
    sca.data[sca.data < 1] = np.nan
    # hack -- remove high-elev SCA terrain to reduce computational load
    sca_elev_median = np.nanmedian(demclip.data[sca.data>0])
    dem_below = demclip.copy()
    dem_below.data[dem_below.data > sca_elev_median] = np.nan
    # set vector of elevations to test for RSLE
    elevs = np.unique(dem_below.data[~np.isnan(dem_below)])
    # --- compute --- 
    # preallocate vectors to load metrics for each elev
    p_land = np.zeros(elevs.shape)
    p_snow = np.zeros(elevs.shape)
    i_scatter = np.zeros(elevs.shape)
    # loop through elevs --
    for i in range(len(elevs)): 
        # grab number of snow-free pixels above the given elevation
        p_land[i] = np.nansum(snowfree.data[demclip.data>=elevs[i]])
        # grab number of snow-cvered pixels below the given elevation
        p_snow[i] = np.nansum(sca.data[demclip.data<=elevs[i]])
        # grab "index of scatter"
        i_scatter[i] = 100 * ((p_land[i] + p_snow[i]) / np.nansum(basin_area.data))
        # verbose progress
        if np.mod(i,100)==0: 
            print('processed '+str(i)+' elevs of '+str(len(elevs))+'...')
    # return RSLE value, the is-elev relation, and clearsky fraction
    idx = np.where(i_scatter==np.min(i_scatter))[0][0]
    rsle = elevs[idx]
    t = pd.DataFrame({'elev':elevs, 'Is':i_scatter})
    return rsle, clearsky_fraction, t

# %% test calc
# inputs -- date, basin, and TOLERANCE for snow-free sca
date = dt.datetime(2017,1,16)
basin = bsp.loc[bs['name']=='Feather'].copy()
scatol = 10
rsle, cs, t = snowline(date, basin, scatol)

# %% test plots
scaclip, demclip, basin_area = prep_rsle_data(date, basin)
fig,ax = plt.subplots(); 
scaclip.plot(ax=ax, cmap='Blues')
basin.plot(ax=ax, fc='none', ec='r')
dem.plot.contour(ax=ax, levels=[rsle], colors=('grey',))

# %% 


# %% ROUTINE -- compute RSLE for all dates for a given tolerance
# save 2 files -- 1 RSLE, 1 clearsky fraction 
# rows = dates, columns = basins [all, feather, yuba, american]
dates = pd.date_range(dt.datetime(2017,1,1), dt.datetime(2017,2,28))
for scatol in [5, 25, 50]:
    scatol = 10
    ofdir = '/home/kden/projects/active/phd2_RosDiffs/'
    ofname_rsle = ofdir+'rsle_scatol_'+str(scatol)+'.csv'
    ofname_cs = ofdir+'clearsky_fraction.csv'
    
    df_rlse = pd.DataFrame({'time':dates, 'all':[np.nan]*len(dates), 
                            'feather':[np.nan]*len(dates), 'yuba':[np.nan]*len(dates), 
                            'american':[np.nan]*len(dates)}).set_index('time')
    df_cs = df_rlse.copy()
    # loop through dates
    for idate in dates: 
        print('\n--------------------------------------------\n')
        print('---------- PROCESSING '+idate.strftime('%d-%b')+' ---------------')
        print('\n---------------------------------------------\n')
        print('\nall basins...\n')
        r1,cs1,t1 = snowline(idate, bsp, scatol)
        print('\nfeather...\n')
        r2,cs2,t2 = snowline(idate, bsp.loc[bs['name']=='Feather'].copy(), scatol)
        print('\nyuba...\n')
        r3,cs3,t3 = snowline(idate, bsp.loc[bs['name']=='Yuba'].copy(), scatol)
        print('\namerican...\n')
        r4,cs4,t4 = snowline(idate, bsp.loc[bs['name']=='American'].copy(), scatol)
        # populate data frames
        df_rlse.loc[df_rlse.index==idate, 'all'] = r1
        df_rlse.loc[df_rlse.index==idate, 'feather'] = r2
        df_rlse.loc[df_rlse.index==idate, 'yuba'] = r3
        df_rlse.loc[df_rlse.index==idate, 'american'] = r4
        df_cs.loc[df_cs.index==idate, 'all'] = cs1
        df_cs.loc[df_cs.index==idate, 'feather'] = cs2
        df_cs.loc[df_cs.index==idate, 'yuba'] = cs3
        df_cs.loc[df_cs.index==idate, 'american'] = cs4
    print('\n--------------------- DONE -------------------\n')
    
    # Save
    df_rlse.to_csv(ofname_rsle)
# df_cs.to_csv(ofname_cs)

# %% 
df_rlse.plot();
df_cs.plot();
# %% 


# %% 


































































