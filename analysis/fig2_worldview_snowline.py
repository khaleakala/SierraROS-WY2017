#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 13:51:36 2021

@author: kden

Plot truecolor worldview sequence, but overlay the snow line on 'clear' days


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
from skimage import exposure
import matplotlib as mpl
import os
import cmasher as cmr

wgs = 'EPSG:4326'

# read in basins
basins = gpd.read_file('/media/kden/LaCie/DATA/geog/sierra_hwbasins/sierra_hwbasins.shp')
bs = basins.loc[basins['name'].isin(['Feather','Yuba','American'])]

# read in RSLE
rsle = pd.read_csv('/media/kden/hdd1/phd2_RosDiffs/supplemental/rsle_scatol_10_fixed.csv', 
                   index_col=0)
rsle.index = pd.to_datetime(rsle.index)

# %% process the DEM

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


# %% worldview functions

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

# perhaps a function to plot f on an axis
def plot_worldview(d, ax, rsle_flag): 
    f = proc_worldview(d)
    basins.plot(ax=ax, fc='none', ec='grey')
    bs.plot(ax=ax, fc='none', ec='r')
    f.plot.imshow(ax=ax)
    # ax.set_title(d.strftime('%d-%b'))
    ax.text(-119.35, 40.55, d.strftime('%d-%b'), ha='right', va='top', fontsize=14, 
            bbox={'edgecolor':'k', 'facecolor':'white'})
    ax.set_title('')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # add routine to overlay RSLE
    if rsle_flag==1: 
        rsleval = rsle['all'].iloc[rsle.index==d][0]
        z.plot(ax=ax, cmap='Greys', zorder=0, add_colorbar=False)
        z.plot.contour(ax=ax, levels=[rsleval], colors=('C1',), lw=0.1, alpha=0.35)
    ax.set_xlim(-122.05230251850705, -119.23618972068871)
    ax.set_ylim(38.46603076742303, 40.635822595250275)
    ax.set_title('')
    return ax

# %% 

# %% plot -- part cells to allow image to render


mydates = [dt.datetime(2016,12,31), dt.datetime(2017,1,6), dt.datetime(2017,1,13), 
           dt.datetime(2017,1,16), dt.datetime(2017,1,24), dt.datetime(2017,1,27), 
           dt.datetime(2017,1,30), dt.datetime(2017,2,5), dt.datetime(2017,2,12)]


fig,axs = plt.subplots(3,3, figsize=(9, 8), tight_layout=True)

# plot worldview tiles on leftmost subplots
plot_worldview(mydates[0], axs[0,0], 0)
plot_worldview(mydates[1], axs[0,1], 0)
plot_worldview(mydates[2], axs[0,2], 1)
plot_worldview(mydates[3], axs[1,0], 1)
plot_worldview(mydates[4], axs[1,1], 1)
plot_worldview(mydates[5], axs[1,2], 1)
plot_worldview(mydates[6], axs[2,0], 0)
plot_worldview(mydates[7], axs[2,1], 0)
plot_worldview(mydates[8], axs[2,2], 1)

# %% add in labels
bprops = {'edgecolor':'k', 'facecolor':'lightgrey'}

# (not sure "priming" events are relevant to the meat of the paper...? eh)
# between 31 Dec and 6 Jan -- the "priming" storm before the 7J AR
lab1 = '7J Priming'
# fig.text(0.333, 0.7, lab1, fontsize=14, bbox=bprops, ha='center')
# 6 Jan to 13 jan -- that's the 7J AR
lab2 = '7J AR'
fig.text(0.667, 0.7, lab2, fontsize=14, bbox=bprops, ha='center')
# 16 Jan to 24 Jan -- the snowstorm between our ARs
lab3 = 'Mid-Jan Snow'
fig.text(0.35, 0.375, lab3, fontsize=14, bbox=bprops, ha='center')
# snowpack recession after storm
lab4 = 'Ephemeral Melt'
fig.text(0.67, 0.375, lab4, fontsize=13.5, bbox=bprops, ha='center')
lab5 = '6F Priming'
# fig.text(0.333, 0.048, lab5, fontsize=14, bbox=bprops, ha='center')
lab6 = '6F AR'
fig.text(0.667, 0.048, lab6, fontsize=14, bbox=bprops, ha='center')
# %% add in subplot lettering
# letters = 'abcdefghi'
letters = 'ABCDEFGHI'
axes = axs.flatten()
for i in range(len(letters)): 
    ax = axes[i]
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    ax.text(xmin,ymax, letters[i], ha='left', va='top', fontsize=14, 
            bbox={'edgecolor':'w', 'facecolor':'white'})

# %% save to file

ofn = '/home/kden/projects/active/phd2_RosDiffs/figures/'
ofn+= '2_worldview_sequence_v3c.png'
plt.savefig(ofn, dpi=500, bbox_inches='tight')

























































