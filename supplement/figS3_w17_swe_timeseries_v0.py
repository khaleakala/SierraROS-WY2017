# -*- coding: utf-8 -*-
"""
Plot entire season SWE for feather and yuba-american groups -- color by elev

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib as mpl
from matplotlib.dates import DateFormatter


# read in metadata and data sets
ref = pd.read_csv('E:/DATA/project_ROSb/station_info_2/sierra_nevada_swe_stations.csv')
ref.columns = ['id','name','basin','county','ferix','lon','lat','elev_ft','elev_m','operator','fmcw','rain']

# read in hourly swe
swe = pd.read_csv('E:/DATA/project_ROSb/sensors_2/sierra_nevada_swe_sensor3_meters_wy16_wy21.csv', index_col=0)
swe.index = pd.to_datetime(swe.index)
swe.values[swe.values<-1] = np.nan
swe.values[swe.values>3] = np.nan



# %% plot  time series over the year

d1 = dt.datetime(2016,11,1)
d2 = dt.datetime(2017,7,31)
swe_feather = 100*swe[['KTL','GRZ','PLP','GOL','HMB','HRK','RTL','BKL','FOR']].loc[(swe.index>=d1) & (swe.index<=d2)]
swe_yuba = 100*swe[['RBP','BLC','GKS','RBB','RCC','HYS','VVL','CSL','SIL','FRN','ALP','CAP','SCN']].loc[(swe.index>=d1) & (swe.index<=d2)]


# %% 
import seaborn as sns

swe_feather['time'] = swe_feather.index
feather_melt = swe_feather.melt(id_vars='time')
feather_melt['elev_m'] = [np.nan]*feather_melt.shape[0]
for i in range(feather_melt.shape[0]): 
    feather_melt['elev_m'].iloc[i] = ref['elev_m'].loc[ref['id']==feather_melt['variable'].iloc[i]].values[0]

swe_yuba['time'] = swe_yuba.index
yuba_melt = swe_yuba.melt(id_vars = 'time')
yuba_melt['elev_m'] = [np.nan]*yuba_melt.shape[0]
for i in range(yuba_melt.shape[0]): 
    yuba_melt['elev_m'].iloc[i] = ref['elev_m'].loc[ref['id']==yuba_melt['variable'].iloc[i]].values[0]

# %% get elevations and station ids
feather_ids = ['KTL','GRZ','PLP','GOL','HMB','HRK','RTL','BKL','FOR']
yuba_ids = ['RBP','BLC','GKS','RBB','RCC','HYS','VVL','CSL','SIL','FRN','ALP','CAP','SCN']

feather_id_ordered = list(ref.loc[ref['id'].isin(feather_ids)].sort_values('elev_m')['id'])
yuba_id_ordered = list(ref.loc[ref['id'].isin(yuba_ids)].sort_values('elev_m')['id'])
feather_z_ordered = list(ref.loc[ref['id'].isin(feather_ids)].sort_values('elev_m')['elev_m'])
yuba_z_ordered = list(ref.loc[ref['id'].isin(yuba_ids)].sort_values('elev_m')['elev_m'])

ar1_start = dt.datetime(2017,1,7,6)#, tzinfo=dt.timezone.utc)
ar1_end = dt.datetime(2017,1,13,0)#, tzinfo=dt.timezone.utc)
ar2_start = dt.datetime(2017,2,6,0)#, tzinfo=dt.timezone.utc)
ar2_end = dt.datetime(2017,2,11,12)#, tzinfo=dt.timezone.utc)
# %% 



# %% plot
plt.close('all')
fig,axs = plt.subplots(2,1, figsize=(9,8), sharex=True, tight_layout=True)

s1 = sns.lineplot(data=feather_melt, x='time', y='value', hue='elev_m', #legend='brief',
                  palette=sns.color_palette('magma_r', n_colors=len(np.unique(feather_melt.elev_m))), 
                  ax=axs[0])
fhand_labels = s1.get_legend_handles_labels()[1]
fhand_labels = [a+' ('+f"{int(b):,}"+' m)' for a,b in zip(feather_id_ordered,feather_z_ordered)]
axs[0].legend(fhand_labels)

s2 = sns.lineplot(data=yuba_melt, x='time', y='value', hue='elev_m', #legend='brief',
                  palette=sns.color_palette('viridis_r', n_colors=len(np.unique(yuba_melt.elev_m))), 
                  ax=axs[1])
yhand_labels = s2.get_legend_handles_labels()[1]
yhand_labels = [a+' ('+f"{int(b):,}"+' m)' for a,b in zip(yuba_id_ordered,yuba_z_ordered)]
axs[1].legend(yhand_labels)

axs[0].axvspan(ar1_start, ar1_end, color='lightgrey', zorder=0, alpha=0.6)
axs[1].axvspan(ar1_start, ar1_end, color='lightgrey', zorder=0, alpha=0.6)
axs[0].axvspan(ar2_start, ar2_end, color='lightgrey', zorder=0, alpha=0.6)
axs[1].axvspan(ar2_start, ar2_end, color='lightgrey', zorder=0, alpha=0.6)

axs[1].set_xlabel('')
axs[0].set_ylabel('SWE (cm)');
axs[1].set_ylabel('SWE (cm)')

# %% save

ofname = 'C:/Users/Kayden/research/project2_RosDiffs/figures/supplemental/'
ofname += 'S_swe_freix_wy2017_v0.png'
plt.savefig(ofname, dpi=400, bbox_inches='tight')



