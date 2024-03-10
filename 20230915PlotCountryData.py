# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:00:14 2023

@author: Konin045
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
import csv
from scipy.optimize import curve_fit
import math
import time
from datetime import date
import seaborn as sns
import sys
import os
import pickle
import geopandas as gpd
import plotly.express as px
import plotly
import json
import kaleido
from urllib.request import urlopen

def plotmap(variable, label, fileatt, vmin, vmax):
    if vmin < -0.5: #vmin = -1 then simply use the default min max
        ax = gdf_rg0_joined.plot(column=variable, figsize=(20,15), cmap=mpl.colormaps['Greens'], legend=True, legend_kwds={"label": label, "orientation":"vertical"})    
    else:
        ax = gdf_rg0_joined.plot(column=variable, figsize=(20,15), cmap=mpl.colormaps['Greens'], vmin=vmin, vmax=vmax, legend=True, legend_kwds={"label": label, "orientation":"vertical"})
    #plot all the regions except for the ones for which epsCR1 is not nan
    gdf_rg0_joined[gdf_rg0_joined[variable].isna()].plot(ax=ax, color='grey')
    plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    ax.set_xlim(2.5e6, 6.6e6)
    ax.set_ylim(1.2e6, 5.5e6)
    fig = ax.figure
    cb_ax = fig.axes[1]
    cb_ax.tick_params(labelsize=35)
    gdf_bn0.plot(figsize=(20,15), ax=ax, color="grey", lw=2)
    plt.savefig(timestamp + fileatt, dpi=600,bbox_inches='tight')
    return 0


timestamp = '20231230 2346 2u  0.00'
alpha_values=[1.0,0.6,0.3] 

countries = ['NL','AT','BE',
'BG',
'HR',
'CY',
'CZ',
'DK',
'EE',
'FI',
'FR',
'DE',
'EL',
'HU',
'IE',
'IT',
'LV',
'LT',
'LU',
'MT',
'PL',
'PT',
'RO',
'SK',
'SI',
'ES',
'SE']
#countries = ['NL']

plt.close('all')
size = len(countries)    
eps_matrix = np.zeros((size, len(alpha_values)))
rho_matrix = np.zeros((size, len(alpha_values)))
r_star_matrix = np.zeros((size, len(alpha_values)))
e_star_matrix = np.zeros((size, len(alpha_values)))  
cost_star_matrix = np.zeros((size, len(alpha_values))) 
f_star_hydro_matrix = np.zeros((size, len(alpha_values))) 
r_star_hydro_matrix = np.zeros((size, len(alpha_values))) 
e_star_hydro_matrix = np.zeros((size, len(alpha_values)))
cost_star_hydro_matrix = np.zeros((size, len(alpha_values))) 
for i in range(0,size): 
    country = countries[i]
    with open(country + '\\' + timestamp + '\\data_for_run.dat', "rb") as input_file:
        eps, rho, e_star, r_star, e_optH, cost_star, r_optH, f_star_hydro, r_star_hydro, e_star_hydro, cost_star_hydro  = pickle.load(input_file)
    eps_matrix[i,]=eps
    rho_matrix[i,]=rho
    r_star_matrix[i,]=r_star
    e_star_matrix[i,]=e_star
    cost_star_matrix[i,]=cost_star
    f_star_hydro_matrix[i,] = f_star_hydro 
    r_star_hydro_matrix[i,] = r_star_hydro
    e_star_hydro_matrix[i,] = e_star_hydro 
    cost_star_hydro_matrix[i,] = cost_star_hydro

plt.close('all')
if not os.path.exists(timestamp):
    os.mkdir(timestamp)    
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
plt.ylabel(r'electrolyzer-to-renewables build out ratio $\epsilon$') 
ax.bar(countries,eps_matrix[:,0])
plt.savefig(timestamp + '\\eps vs country', dpi=600,bbox_inches='tight')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
plt.ylabel(r'renewables threshold value $\rho$') 
ax.bar(countries,rho_matrix[:,0])
plt.savefig(timestamp + '\\rho vs country', dpi=600,bbox_inches='tight')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
plt.ylabel(r'electrolyzer capacity at full decarbonization $e^*$') 
ax.bar(countries,e_star_matrix[:,0])
plt.savefig(timestamp + '\\e star vs country', dpi=600,bbox_inches='tight')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
plt.ylabel(r'renewables generation at full decarbonization $r^*$')
ax.bar(countries,r_star_matrix[:,0])
plt.savefig(timestamp + '\\r star vs country', dpi=600,bbox_inches='tight')
plt.close('all')

column_values = ['rstarCR1', 'rstarCR2', 'rstarCR3']
df1 = pd.DataFrame(data = r_star_matrix.tolist(), 
                  columns = column_values)
column_values = ['estarCR1', 'estarCR2', 'estarCR3']
df2 = pd.DataFrame(data = e_star_matrix.tolist(), 
                  columns = column_values)
column_values = ['epsCR1', 'epsCR2', 'epsCR3']
df3 = pd.DataFrame(data = eps_matrix.tolist(), 
                  columns = column_values)
column_values = ['rhoCR1', 'rhoCR2', 'rhoCR3']
df4 = pd.DataFrame(data = rho_matrix.tolist(), 
                  columns = column_values)
column_values = ['cstarCR1', 'cstarCR2', 'cstarCR3']
df5 = pd.DataFrame(data = cost_star_matrix.tolist(), 
                  columns = column_values)
column_values = ['fstarhydroCR1', 'fstarhydroCR2', 'fstarhydroCR3']
df6 = pd.DataFrame(data = f_star_hydro_matrix.tolist(), 
                  columns = column_values)
column_values = ['rstarhydroCR1', 'rstarhydroCR2', 'rstarhydroCR3']
df7 = pd.DataFrame(data = r_star_hydro_matrix.tolist(), 
                  columns = column_values)
column_values = ['estarhydroCR1', 'estarhydroCR2', 'estarhydroCR3']
df8 = pd.DataFrame(data = e_star_hydro_matrix.tolist(), 
                  columns = column_values)
column_values = ['cstarhydroCR1', 'cstarhydroCR2', 'cstarhydroCR3']
df9 = pd.DataFrame(data = cost_star_hydro_matrix.tolist(), 
                  columns = column_values)
frames = [df1, df2, df3, df4, df5, df6, df7, df8, df9]
df = pd.concat(frames, axis=1)
df['country'] = pd.DataFrame(countries)




path_rg = "NUTS_RG_20M_2021_3035.geojson"
gdf_rg = gpd.read_file(path_rg)
gdf_rg0 = gdf_rg.loc[gdf_rg['LEVL_CODE'] == 0]
gdf_rg0_joined =pd.merge(gdf_rg0,df, left_on='id', right_on='country', how='left')
path_bn = "NUTS_BN_20M_2021_3035.geojson"
gdf_bn = gpd.read_file(path_bn)
gdf_bn0 = gdf_bn.loc[gdf_bn['LEVL_CODE'] == 0]



plotmap('rstarCR1', r'$r^{*}$', '\\r star vs country map', -1,-1) 
plotmap('estarCR1', r'$e^{*}$', '\\e star vs country map',-1,-1)
plotmap('epsCR1', r'$\epsilon$', '\\eps vs country map',-1,-1)
plotmap('rhoCR1', r'$\rho$', '\\rho vs country map',-1,-1)
plotmap('cstarCR1', r'$C^{*}$', '\\cost star vs country map',-1,-1)
plotmap('fstarhydroCR1', r'$f^{*}_{hydro}$', '\\f star hydro vs country map',-1,-1)
plotmap('rstarhydroCR1', r'$r^{*}_{hydro}$', '\\r star hydro vs country map',-1,-1)
plotmap('estarhydroCR1', r'$e^{*}_{hydro}$', '\\e star hydro vs country map',-1,-1)
plotmap('cstarhydroCR1', r'$C^{*}_{hydro}$', '\\cost star hydro vs country map',-1,-1)
plt.close('all')
#now plot the variables of interest on the same scale
mi = min(df1.min())
ma = max(df1.max())
plotmap('rstarCR1', r'$r^{*}$', '\\r star vs country map 1', mi, ma)
plotmap('rstarCR2', r'$r^{*}$', '\\r star vs country map 2', mi, ma)
plotmap('rstarCR3', r'$r^{*}$', '\\r star vs country map 3', mi, ma)
mi = min(df2.min())
ma = max(df2.max())
plotmap('estarCR1', r'$e^{*}$', '\\e star vs country map 1', mi, ma)
plotmap('estarCR2', r'$e^{*}$', '\\e star vs country map 2', mi, ma)
plotmap('estarCR3', r'$e^{*}$', '\\e star vs country map 3', mi, ma)
mi = min(df3.min())
ma = max(df3.max())
plotmap('epsCR1', r'$\epsilon$', '\\eps vs country map 1', mi, ma)
plotmap('epsCR2', r'$\epsilon$', '\\eps vs country map 2', mi, ma)
plotmap('epsCR3', r'$\epsilon$', '\\eps vs country map 3', mi, ma)
mi = min(df4.min())
ma = max(df4.max())
plotmap('rhoCR1', r'$\rho$', '\\rho vs country map 1', mi, ma)
plotmap('rhoCR2', r'$\rho$', '\\rho vs country map 2', mi, ma)
plotmap('rhoCR3', r'$\rho$', '\\rho vs country map 3', mi, ma)
mi = min(df5.min())
ma = max(df5.max())
plt.close('all')
plotmap('cstarCR1', r'$C^{*}$', '\\cost star vs country map 1', mi, ma)
plotmap('cstarCR2', r'$C^{*}$', '\\cost star vs country map 2', mi, ma)
plotmap('cstarCR3', r'$C^{*}$', '\\cost star vs country map 3', mi, ma)
plt.close('all')
mi = min(df6.min())
ma = max(df6.max())
plotmap('fstarhydroCR1', r'$f^{*}_{hydro}$', '\\f star hydro vs country map 1', mi, ma)
plotmap('fstarhydroCR2', r'$f^{*}_{hydro}$', '\\f star hydro vs country map 2', mi, ma)
plotmap('fstarhydroCR3', r'$f^{*}_{hydro}$', '\\f star hydro vs country map 3', mi, ma)
mi = min(df7.min())
ma = max(df7.max())
plotmap('rstarhydroCR1', r'$r^{*}_{hydro}$', '\\r star hydro vs country map 1', mi, ma)
plotmap('rstarhydroCR2', r'$r^{*}_{hydro}$', '\\r star hydro vs country map 2', mi, ma)
plotmap('rstarhydroCR3', r'$r^{*}_{hydro}$', '\\r star hydro vs country map 3', mi, ma)
mi = min(df8.min())
ma = max(df8.max())
plotmap('estarhydroCR1', r'$e^{*}_{hydro}$', '\\e star hydro vs country map 1', mi, ma)
plotmap('estarhydroCR2', r'$e^{*}_{hydro}$', '\\e star hydro vs country map 2', mi, ma)
plotmap('estarhydroCR3', r'$e^{*}_{hydro}$', '\\e star hydro vs country map 3', mi, ma)
mi = min(df9.min())
ma = max(df9.max())
plotmap('cstarhydroCR1', r'$C^{*}_{hydro}$', '\\cost star hydro vs country map 1', mi, ma)
plotmap('cstarhydroCR2', r'$C^{*}_{hydro}$', '\\cost star hydro vs country map 2', mi, ma)
plotmap('cstarhydroCR3', r'$C^{*}_{hydro}$', '\\cost star hydro vs country map 3', mi, ma)
plt.close('all')



folder_country_data = 'GA_annual_data_country_level'
with open(folder_country_data + '\\GA_annual_data_country_level.dat', "rb") as input_file:
    solar_list, off_list, on_list, river_list, countries, cap_fact_pv_list, cap_fact_on_list, cap_fact_off_list, cap_fact_river_list, CR_correction_list, cap_hydro_list, f_star_hydro_iea_list, method_hydro = pickle.load(input_file)    
dfGA = pd.DataFrame(list(zip(solar_list, off_list, on_list, river_list,countries, cap_fact_pv_list, cap_fact_on_list, cap_fact_off_list, cap_fact_river_list, CR_correction_list, cap_hydro_list)),
               columns =['solar_list' , 'off_list', 'on_list', 'river_list','countries', 'cap_fact_pv_list', 'cap_fact_on_list', 'cap_fact_off_list', 'cap_fact_river_list','CR_correction_list', 'cap_hydro_list'])
gdf_rg0_joined =pd.merge(gdf_rg0_joined,dfGA, left_on='id', right_on='countries', how='left')
plotmap('solar_list', r'solar fraction', '\\solar vs country map', -1,-1)
plotmap('on_list', r'onshore wind fraction', '\\on vs country map', -1,-1)
plotmap('off_list', r'offshore wind fraction', '\\offshore vs country map', -1,-1)
plotmap('river_list', r'river fraction', '\\river vs country map', -1,-1)
plotmap('cap_fact_pv_list', r'capacity factor solar', '\\cap pv vs country map', -1,-1)
plotmap('cap_fact_on_list', r'capacity factor onshore wind', '\\cap on vs country map', -1,-1)
plotmap('cap_fact_off_list', r'capacity factor offshore wind', '\\cap off vs country map', -1,-1)
plotmap('cap_fact_river_list', r'capacity factor river', '\\cap river vs country map', -1,-1)
plotmap('CR_correction_list', r'CR factor', '\\CR correction vs country map', -1,-1)
plotmap('cap_hydro_list', r'capacity hydropower', '\\cap hydropower vs country map', -1,-1)



#temporary. this could works under the assumption that C=3 for renewables and C=1 for electrolyzers.
#just to check if computation above is consistent
gdf_rg0_joined['investment_cost'] = 3.0*gdf_rg0_joined['CR_correction_list']*gdf_rg0_joined['rstarCR1'] + 1.0* gdf_rg0_joined['estarCR1']
plotmap('investment_cost', r'investment cost', '\\investment cost vs country map', -1,-1)
plt.close('all')
#plotmap('CR1')
#plotmap('epsCR1')
#plotmap('epsCR1')


















# path_rg2 = "europe.geojson"
# gdf_rg2 = gpd.read_file(path_rg2)
# gdf_rg2.to_crs(3035)
# gdf_rg2.plot()
# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# world.plot()
# world = world[(world.name == "Kosovo")]
# world = world.to_crs({'init': 'epsg:3035'})
# world.plot()
