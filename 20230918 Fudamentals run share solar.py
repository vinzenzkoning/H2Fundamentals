# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:22:27 2023

@author: Konin045
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import A20230918FundamentalsHydrogen as fundamentals
import pickle
from matplotlib.animation import PillowWriter
from PIL import Image
import time
from datetime import date
import os
import sys

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

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

alpha_values=[1.0,0.6,0.3] 
CRsNL = np.array([3.0, 6.0, 2.0]) #cost ratio between renewables production and electrolyzer capacity 
CRs = CRsNL
u = 0.0
PVshare = 0.0
battery_size = 0.0
scenario = 'NSWPH'

countries = ['NL']
scenario = 'NSWPH-NL'

today = date.today()
currentdate = str(today.year)+str(today.month)+str(today.day)
hhmm = str(time.localtime(time.time()).tm_hour) + str(time.localtime(time.time()).tm_min)
hhmmss = str(time.localtime(time.time()).tm_hour) + str(time.localtime(time.time()).tm_min) + ' ' + str(time.localtime(time.time()).tm_sec)
datehhmm = currentdate+' '+hhmm
datehhmmss = currentdate+' '+hhmmss
cwd = os.getcwd()


if scenario == 'Mix' or scenario == 'Wind_GAR2040' or scenario == 'Offshore_GAR2040' or scenario == 'PV_GAR2040' or scenario == '2040EHR' or scenario == '2050EHR' or scenario == '2040GAR' or scenario == '2050GAR' or scenario =='Offshore' or scenario =='PV':
    T = 8736
elif scenario == 'NSWPH' or scenario == 'NSWPH-NL':
    T = 8760
else: 
    T = 8760

for country in countries:
    if scenario == 'Mix':
        filename = 'Wind_GAR2040' + '.csv'
        df =pd.read_csv(filename, delimiter=',', skiprows=1)
        filename = 'PV_GAR2040' + '.csv'
        df2 =pd.read_csv(filename, delimiter=',', skiprows=1)
        df['p']=(1.0-PVshare)*df['p']/np.mean(df['p'])+PVshare*df2['p']/np.mean(df2['p']) #division by means is to normalize the input data such that we indeed compute based on the right solar fraction
        folder = scenario + ' PVshare %.2f' %PVshare  + ' ' + datehhmmss + ' u  %.2f' % u
    elif scenario == 'NSWPH' or scenario == 'NSWPH-NL': 
        with open(country + '\\dataframe.dat', "rb") as input_file:
            df, PVshare, off, on, river, ee, cap_fact_pv, cap_fact_wind, cap_fact_on, cap_fact_off, cap_fact_river, CR_correction, cap_hydro, f_star_hydro_iea, method_hydro= pickle.load(input_file)
        folder = country + '\\' + datehhmmss + 'u  %.2f' % u
        CRs = CRsNL * CR_correction
    else:
        filename = scenario + '.csv'
        df =pd.read_csv(filename, delimiter=',', skiprows=1) 
        folder = scenario + ' ' + datehhmmss + 'u  %.2f' % u
    
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    
    Synergy_coproduction, Vp, VH2, C_integrated_prod, rho, eps, r_star, e_star, eps_sa, C_H2_sa_min, P_conv, cost_star  = fundamentals.analyse_power_and_hydrogen(df, folder, T, u, alpha_values, scenario, PVshare, CRs, cap_hydro, f_star_hydro_iea, method_hydro, battery_size, 0.0)

#uncomment next line for additional analyses
sys.exit()

defaultcolour = 'darkgreen'
country = 'NL'
battery_size = 0.0
size=41
size = 2
size_level = 29
r_for_bat_ref_level = np.linspace(1.2, 1.48, size_level)
#r_for_bat_ref_level = np.linspace(1.33, 1.33, 1)
bat_ref_level_array =-(1.-1./r_for_bat_ref_level)
# = -(1.-1./r_for_bat_ref_level)
battery_size_array = np.linspace(0.5,1.0,size)
eps_matrix = np.zeros((size_level, size, len(alpha_values)))
rho_matrix = np.zeros((size_level, size, len(alpha_values)))
r_star_matrix = np.zeros((size_level, size, len(alpha_values)))
e_star_matrix = np.zeros((size_level, size, len(alpha_values)))
cost_star_matrix = np.zeros((size_level, size, len(alpha_values)))
folder_gen = country + '\\' + datehhmmss + 'u  %.2f' % u
if not os.path.exists(folder_gen):
    os.mkdir(folder_gen)
for j in range(0,len(bat_ref_level_array)):
    for i in range(0,len(battery_size_array)):
        battery_size = battery_size_array[i]
        bat_ref_level = bat_ref_level_array[j]
        folder = scenario + ' battery %.2f' %battery_size  + ' ref level %.4f' %bat_ref_level  +' ' + datehhmmss + ' u  %.2f' % u
        if not os.path.exists(folder):
            os.mkdir(folder)    
        Synergy_coproduction, Vp, VH2, C_integrated_prod, rho, eps, r_star, e_star, eps_sa, C_H2_sa_min, P_conv, cost_star  = fundamentals.analyse_power_and_hydrogen(df, folder, T, u, alpha_values, scenario, PVshare, CRs, cap_hydro, f_star_hydro_iea, method_hydro, battery_size, bat_ref_level)
        eps_matrix[j,i,]=eps
        rho_matrix[j,i,]=rho
        r_star_matrix[j,i,]=r_star
        e_star_matrix[j,i,]=e_star
        cost_star_matrix[j,i,]=cost_star
scenario_names = [' (reference)', ' (future)', ' (current)'] 


with open(folder_gen + '\\results battery variation.dat', 'wb') as f:
    pickle.dump([battery_size_array, r_for_bat_ref_level, eps_matrix,rho_matrix, r_star_matrix, e_star_matrix, cost_star_matrix], f)

r_bat = np.zeros((size, len(alpha_values)))
e_bat = np.zeros((size, len(alpha_values)))
cost_bat = np.zeros((size, len(alpha_values)))
for i in range(0,len(battery_size_array)):
    for k in range(0,len(CRs)):
        min_cost_index = np.argmin(cost_star_matrix[:,i,k])
        #min_r_diff_index = np.argmin(np.abs(r_star_matrix[:,i,k]-r_for_bat_ref_level))
        print(min_cost_index)
        #print(min_r_diff_index)
        r_bat[i,k] = r_star_matrix[min_cost_index, i, k]
        e_bat[i,k] = e_star_matrix[min_cost_index, i, k]
        cost_bat[i,k] = cost_star_matrix[min_cost_index, i, k]


# 15
# 13
# 20
# 12
# 9
# 18



sys.exit()
plt.figure()
for k in range(0,len(CRs)):
    labelname = r'$\mathrm{CR}=%.0f$' % CRs[k] + scenario_names[k]     
    #plt.scatter(battery_size_array, rho_matrix[:,k], alpha=alpha_values[k], color=defaultcolour, s=5)
    #plt.legend(loc='lower left', framealpha=1.0)
    #plt.plot(movingaverage(battery_size_array,2), movingaverage(rho_matrix[:,k],2),label=labelname,alpha=alpha_values[k], color=defaultcolour)
    plt.plot(battery_size_array, rho_matrix[:,k],label=labelname,alpha=alpha_values[k], color=defaultcolour)
plt.xlabel(r'battery power capacity $p_b/r$', fontsize =14)
plt.ylabel(r'renewables threshold value $\rho$' +"\n", fontsize =14)   
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.xlim(0.0,2.0)
plt.ylim(ymin=0.0)
plt.ylim(0.0,1.35)
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,0,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='lower left', framealpha=1.0, fontsize=14)
plt.savefig(folder_gen + '\\rho vs battery power capacity' + scenario, dpi=600,bbox_inches='tight')    

plt.figure()
for k in range(0,len(CRs)):
    labelname = r'$\mathrm{CR}=%.0f$' % CRs[k] + scenario_names[k]     
    plt.plot(battery_size_array, eps_matrix[:,k], alpha=alpha_values[k], color=defaultcolour, label=labelname)
plt.xlabel(r'battery power capacity $p_b/r$', fontsize =14)
plt.ylabel('electrolyser-to-renewables \n'+ r'build out ratio $\varepsilon$', fontsize =14)   
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.xlim(0.0,2.0)
plt.ylim(ymin=0.0)
plt.ylim(0.0,4.1)
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,0,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='lower left', framealpha=1.0, fontsize=14)
plt.savefig(folder_gen + '\\eps vs battery power capacity' + scenario, dpi=600,bbox_inches='tight')    

plt.figure()
for k in range(0,len(CRs)):
    labelname = r'$\mathrm{CR}=%.0f$' % CRs[k] + scenario_names[k]     
    #lt.scatter(battery_size_array, eps_matrix[:,k], alpha=alpha_values[k], color=defaultcolour, s=5)
    #plt.legend(loc='lower left', framealpha=1.0)
    plt.plot(battery_size_array, cost_star_matrix[:,k], alpha=alpha_values[k], color=defaultcolour, label=labelname)
plt.xlabel(r'battery power capacity $p_b/r$', fontsize =14)
plt.ylabel(r'   cost $C^{*} \ (€/ W)$'+ "\nof full decarbonisation", fontsize =14) 
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.xlim(0.0,2.0)
plt.ylim(ymin=0.0)
plt.ylim(0.0,10)
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,0,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='lower left', framealpha=1.0, fontsize=14)
plt.savefig(folder_gen + '\\cost vs battery power capacity' + scenario, dpi=600,bbox_inches='tight')    

plt.figure()
for k in range(0,len(CRs)):
    labelname = r'$\mathrm{CR}=%.0f$' % CRs[k] + scenario_names[k]     
    plt.plot(battery_size_array, e_star_matrix[:,k], alpha=alpha_values[k], color=defaultcolour, label=labelname)
plt.xlabel(r'battery power capacity $p_b/r$', fontsize =14)
plt.ylabel(r'   electrolyser capacity $e^{*} \ (€/ W)$'+ "\nat full decarbonisation", fontsize =14) 
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.xlim(0.0,2.0)
plt.ylim(ymin=0.0)
#plt.ylim(0.0,10)
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,0,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='lower left', framealpha=1.0, fontsize=14)
plt.savefig(folder_gen + '\\e star vs battery power capacity' + scenario, dpi=600,bbox_inches='tight')    

plt.figure()
for k in range(0,len(CRs)):
    labelname = r'$\mathrm{CR}=%.0f$' % CRs[k] + scenario_names[k]     
    plt.plot(battery_size_array, r_star_matrix[:,k], alpha=alpha_values[k], color=defaultcolour, label=labelname)
plt.xlabel(r'battery power capacity $p_b/r$', fontsize =14)
plt.ylabel(r' electrolyser capacity $r^{*} \ (€/ W)$'+ "\nat full decarbonisation", fontsize =14) 
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.xlim(0.0,2.0)
plt.ylim(ymin=0.0)
plt.ylim(0.0,1.7)
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,0,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='lower left', framealpha=1.0, fontsize=14)
plt.savefig(folder_gen + '\\r star vs battery power capacity' + scenario, dpi=600,bbox_inches='tight')    

fig=plt.figure()
A = np.linspace(0,15.75,10) #14,9
x=np.linspace(0,10,11)
for a in A:
    plt.plot(x,-CRsNL[0]*x+a, color='k', alpha=0.35, lw=0.5)
for k in range(0,len(CRs)):
    labelname = r'$CR=%.0f$' % CRs[k]      
    plt.scatter(r_star_matrix[:,k], e_star_matrix[:,k],  alpha=alpha_values[k], color=defaultcolour, s=3)
    plt.scatter(r_star_matrix[0,k], e_star_matrix[0,k], alpha=alpha_values[k], color=defaultcolour, s=30, marker = 'D')
    plt.scatter(r_star_matrix[-1,k], e_star_matrix[-1,k], alpha=alpha_values[k], color=defaultcolour, s=30, marker = 'D')
    plt.plot(movingaverage(r_star_matrix[:,k],5), movingaverage(e_star_matrix[:,k],5),label=labelname,alpha=alpha_values[k], color=defaultcolour)
#plt.legend(loc='upper left', framealpha=1.0)
plt.xlabel(r'renewables generation at full decarbonization $r^{*}$', fontsize =14)
plt.ylabel(r'   electrolyzer capacity $e^{*}$' + "\nat full decarbonisation", fontsize =14) 
plt.ylim(0.0,3.5)
plt.xlim(0.0,2.5)
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.savefig(folder_gen + '\\e en r star vs battery power capacity' + scenario, dpi=600,bbox_inches='tight') 
fig.set_size_inches(5,5)
#plt.figure().set_figheight(2)
plt.ylim(0.5,1.2)
plt.xlim(1.05,1.75)
plt.xticks([1.1, 1.3, 1.5, 1.7])
plt.yticks([0.5, 0.7, 0.9, 1.1])
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20) 
plt.xlabel('')
plt.ylabel('') 
plt.savefig(folder_gen + '\\e en r star vs battery power capacity zoom' + scenario, dpi=600,bbox_inches='tight')


with open(folder_gen + '\\results battery variation.dat', 'wb') as f:
    pickle.dump([battery_size_array, eps_matrix,rho_matrix, r_star_matrix, e_star_matrix, cost_star_matrix], f)
    
#folder_gen = 'NL\\2024118 028 51u  0.00'
#filename = folder_gen + '\\results battery variation.dat'
#with open(filename, 'rb') as f:
#    battery_size_array, eps_matrix,rho_matrix, r_star_matrix, e_star_matrix, cost_star_matrix = pickle.load(f)


sys.exit()

###################  USER INPUT  ################### 
scenario = 'Mix NSWHP' #'2040GAR' #choices: 2040EHR, 2050EHR, 2040GAR, 2050GAR, Offshore', 'PV', PVNL, PV_GAR2040
####################################################
battery_size = 0.0
country = 'NL'
with open(country + '\\dataframe.dat', "rb") as input_file:
    df_original, solar, off, on, river, ee, cap_fact_pv, cap_fact_wind, cap_fact_on, cap_fact_off, cap_fact_river, CR_correction, cap_hydro, f_star_hydro_iea, method_hydro= pickle.load(input_file)
CRs = CRsNL * CR_correction
frames = []
#alpha_values=[1.0,0.6,0.3] 
u = 0.0
size = 101
eps_matrix = np.zeros((size, len(alpha_values)))
rho_matrix = np.zeros((size, len(alpha_values)))
r_star_matrix = np.zeros((size, len(alpha_values)))
e_star_matrix = np.zeros((size, len(alpha_values)))
cost_star_matrix = np.zeros((size, len(alpha_values)))
eps_sa_matrix = np.zeros((size, len(alpha_values)))
C_H2_sa_min_matrix = np.zeros((size, len(alpha_values)))
P_conv_matrix = np.zeros((size, len(alpha_values)))
PVshare_array = np.linspace(0.0,1.0,size)
for i in range(0,len(PVshare_array)):
    PVshare = PVshare_array[i]
    #adjust the p values to account for the different share of solar 
    df = df_original
    df['p']=(1.0-PVshare)*df_original['total_wind']/np.mean(df_original['total_wind'])+PVshare*df_original['energy_power_solar_pv_electricity_supply']/np.mean(df_original['energy_power_solar_pv_electricity_supply']) #division by means is to normalize the input data such that we indeed compute based on the right solar fraction
    folder = scenario + ' PVshare %.2f' %PVshare  + ' ' + datehhmmss + ' u  %.2f' % u
    if not os.path.exists(folder):
        os.mkdir(folder)
    Synergy_coproduction, Vp, VH2, C_integrated_prod, rho, eps, r_star, e_star, eps_sa, C_H2_sa_min, P_conv, cost_star  = fundamentals.analyse_power_and_hydrogen(df, folder, T, u, alpha_values, scenario, PVshare, CRs, cap_hydro, f_star_hydro_iea, method_hydro, battery_size)
    #Synergy_coproduction, Vp, VH2, C_integrated_prod, folder, rho, eps, r_star, e_star, eps_sa, C_H2_sa_min, P_conv  = fundamentals.analyse_power_and_hydrogen(u, alpha_values, scenario, PVshare)
    eps_matrix[i,]=eps
    rho_matrix[i,]=rho
    r_star_matrix[i,]=r_star
    e_star_matrix[i,]=e_star
    cost_star_matrix[i,]=cost_star
    eps_sa_matrix[i,]=eps_sa
    C_H2_sa_min_matrix[i,]=C_H2_sa_min
    P_conv_matrix[i,]=P_conv
    img = folder + '\\flevelsin e r space ' + scenario + '.png'    
    new_frame = Image.open(img)
    frames.append(new_frame)
#now make the video
folder_gen = country + '\\' + datehhmmss + 'u  %.2f' % u
if not os.path.exists(folder_gen):
    os.mkdir(folder_gen)
frames[0].save(folder_gen + '\\e-vs-r movie4.gif', format='GIF',
               append_images=frames,
               save_all=True, loop = 0) #duration=300,


#k=0 
#plt.scatter(PVshare_array,eps_matrix[:,k])
#plt.plot(np.convolve(eps_matrix[:,k], np.ones(10)/10, mode='valid'))
#plt.plot(movingaverage(eps_matrix[:,k],10))
#plt.plot(movingaverage(PVshare_array,10), movingaverage(eps_matrix[:,k],10))

#CRs = np.array([3.0, 6.0, 2.0]) #cost ratio between renewables production and electrolyzer capacity
defaultcolour = 'darkgreen'
plt.figure()
for k in range(0,len(CRs)):
    labelname = r'$CR=%.0f$' % CRs[k]      
    plt.scatter(PVshare_array, eps_matrix[:,k], alpha=alpha_values[k], color=defaultcolour, s=5, edgecolor = 'None')
    plt.plot(movingaverage(PVshare_array,5), movingaverage(eps_matrix[:,k],5),label=labelname,alpha=alpha_values[k], color=defaultcolour)
#plt.legend(loc='lower left', framealpha=1.0)
plt.xlabel(r'fraction of solar $s$', fontsize =14)
plt.ylabel("electrolyzer-to-renewables\n" + r'build out ratio $\epsilon$', fontsize =14) 
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.ylim(ymin=0.0)
plt.xlim(0.0,1.0)
plt.savefig(folder_gen + '\\eps vs solar fraction' + scenario, dpi=600,bbox_inches='tight')


scenario_names = [' (reference)', ' (future)', ' (current)'] 
plt.figure()
for k in range(0,len(CRs)):
    labelname = r'$CR=%.0f$' % CRs[k] + scenario_names[k]     
    plt.scatter(PVshare_array, rho_matrix[:,k], alpha=alpha_values[k], color=defaultcolour, s=5)
    plt.plot(movingaverage(PVshare_array,5), movingaverage(rho_matrix[:,k],5),label=labelname,alpha=alpha_values[k], color=defaultcolour)
#plt.legend(loc='lower left', framealpha=1.0)
plt.xlabel(r'fraction of solar $s$', fontsize =14)
plt.ylabel(r'renewables threshold value $\rho$' +"\n", fontsize =14)   
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.xlim(0.0,1.0)
plt.ylim(ymin=0.0)
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,0,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='lower left', framealpha=1.0, fontsize=14)
plt.savefig(folder_gen + '\\rho vs solar fraction' + scenario, dpi=600,bbox_inches='tight')    
 
plt.figure()
for k in range(0,len(CRs)):
    labelname = r'$CR=%.0f$' % CRs[k]      
    plt.scatter(PVshare_array, r_star_matrix[:,k], alpha=alpha_values[k], color=defaultcolour, s=5)
    plt.plot(movingaverage(PVshare_array,5), movingaverage(r_star_matrix[:,k],5),label=labelname,alpha=alpha_values[k], color=defaultcolour)
#plt.legend(loc='lower left', framealpha=1.0)
plt.xlabel(r'fraction of solar $s$', fontsize =14)
plt.ylabel(r'   renewables generation $r^*$'+ "\nat full decarbonisation", fontsize =14) 
plt.ylim(ymin=0.0)
plt.xlim(0.0,1.0)
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.savefig(folder_gen + '\\r star vs solar fraction' + scenario, dpi=600,bbox_inches='tight')    

plt.figure()
for k in range(0,len(CRs)):
    labelname = r'$CR=%.0f$' % CRs[k]      
    plt.scatter(PVshare_array, e_star_matrix[:,k],  alpha=alpha_values[k], color=defaultcolour, s=5)
    plt.plot(movingaverage(PVshare_array,5), movingaverage(e_star_matrix[:,k],5),label=labelname,alpha=alpha_values[k], color=defaultcolour)
#plt.legend(loc='upper left', framealpha=1.0)
plt.xlabel(r'fraction of solar $s$', fontsize =14)
#plt.ylabel(r'electrolyzer capacity $e^*$ at full decarbonization $e^*$', fontsize =14) 
plt.ylabel(r'   electrolyzer capacity $e^{*}$' + "\nat full decarbonisation", fontsize =14)
plt.ylim(0.0,None)
plt.xlim(0.0,1.0)
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.savefig(folder_gen + '\\e star vs solar fraction' + scenario, dpi=600,bbox_inches='tight') 


plt.figure()
A = np.linspace(0,15.75,10) #14,9
x=np.linspace(0,10,11)
for a in A:
    plt.plot(x,-CRsNL[0]*x+a, color='k', alpha=0.35, lw=0.5)
for k in range(0,len(CRs)):
    labelname = r'$CR=%.0f$' % CRs[k]      
    plt.scatter(r_star_matrix[:,k], e_star_matrix[:,k],  alpha=alpha_values[k], color=defaultcolour, s=3)
    plt.scatter(r_star_matrix[0,k], e_star_matrix[0,k], alpha=alpha_values[k], color=defaultcolour, s=30, marker = 'D')
    plt.scatter(r_star_matrix[-1,k], e_star_matrix[-1,k], alpha=alpha_values[k], color=defaultcolour, s=30, marker = 'D')
    plt.plot(movingaverage(r_star_matrix[:,k],5), movingaverage(e_star_matrix[:,k],5),label=labelname,alpha=alpha_values[k], color=defaultcolour)
#plt.legend(loc='upper left', framealpha=1.0)
plt.xlabel(r'renewables generation at full decarbonization $r^{*}$', fontsize =14)
plt.ylabel(r'   electrolyzer capacity $e^{*}$' + "\nat full decarbonisation", fontsize =14) 
plt.ylim(0.0,8.)
plt.xlim(0.0,3.)
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.savefig(folder_gen + '\\e en r star vs solar fraction' + scenario, dpi=600,bbox_inches='tight') 


plt.figure()
for k in range(0,len(CRs)):
    labelname = r'$CR=%.0f$' % CRs[k]      
    plt.scatter(PVshare_array, cost_star_matrix[:,k], alpha=alpha_values[k], color=defaultcolour, s=5, edgecolor = 'None')
    plt.plot(movingaverage(PVshare_array,5), movingaverage(cost_star_matrix[:,k],5),label=labelname,alpha=alpha_values[k], color=defaultcolour)
#plt.legend(loc='upper left', framealpha=1.0)
plt.xlabel(r'fraction of solar $s$', fontsize =14)
plt.ylabel(r'   cost $C^{*} \ (€/ W)$'+ "\nof full decarbonisation", fontsize =14) 
plt.ylim(ymin=0.0)
plt.xlim(0.0,1.0)
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.savefig(folder_gen + '\\cost star vs solar fraction' + scenario, dpi=600,bbox_inches='tight')

plt.figure()
for k in range(0,len(CRs)):
    labelname = r'$CR=%.0f$' % CRs[k]      
    plt.scatter(PVshare_array, eps_sa_matrix[:,k], alpha=alpha_values[k], color=defaultcolour)
    plt.plot(movingaverage(PVshare_array,5), movingaverage(e_star_matrix[:,k],5),label=labelname,alpha=alpha_values[k], color=defaultcolour)
plt.legend(loc='lower right', framealpha=1.0)
plt.xlabel(r'fraction of solar $s$', fontsize =14)
plt.ylabel(r'stand-alone electrolyzer-to-renewables ratio $\epsilon^{sa}$', fontsize =14) 
plt.ylim(0.0,None)
plt.xlim(0.0,1.0)
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.savefig(folder_gen + '\\eps sa vs solar fraction' + scenario, dpi=600,bbox_inches='tight') 

plt.figure()
for k in range(0,len(CRs)):
    labelname = r'$CR=%.0f$' % CRs[k]      
    plt.scatter(PVshare_array, 100.*P_conv_matrix[:,k], label=labelname, alpha=alpha_values[k], color=defaultcolour)
plt.legend(loc='lower right', framealpha=1.0)
plt.xlabel(r'fraction of solar $s$', fontsize =14)
plt.ylabel(r'power converted $P_{conv} \left(\%\right)$', fontsize =14) 
plt.ylim(0.0,100.0)
plt.xlim(0.0,1.0)
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.savefig(folder_gen + '\\Pconv vs solar fraction' + scenario, dpi=600,bbox_inches='tight') 

plt.figure()
for k in range(0,len(CRs)):
    labelname = r'$CR=%.0f$' % CRs[k]      
    plt.scatter(PVshare_array, C_H2_sa_min_matrix[:,k], label=labelname, alpha=alpha_values[k], color=defaultcolour)
plt.legend(loc='lower right', framealpha=1.0)
plt.xlabel(r'fraction of solar $s$', fontsize =14)
plt.ylabel(r'cost of hydrogen production $C^{sa}_{H} / H^{sa} (€/W_H)$', fontsize =14) 
plt.ylim(0.0,None)
plt.xlim(0.0,1.0)
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.savefig(folder_gen + '\\cost stand alone vs solar fraction' + scenario, dpi=600,bbox_inches='tight') 

with open(folder_gen + '\\results solar fraction variation.dat', 'wb') as f:
    pickle.dump([PVshare_array, eps_matrix,rho_matrix, r_star_matrix, e_star_matrix, cost_star_matrix, eps_sa_matrix, C_H2_sa_min_matrix, P_conv_matrix], f)





