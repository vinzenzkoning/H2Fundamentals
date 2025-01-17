# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 22:32:25 2024

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
import A20230918FundamentalsHydrogen as fundamentals
from scipy.optimize import curve_fit

# folder = '20231230 2346 2u  0.00'
# alpha_values=[1.0,0.6,0.3] 
# CRsNL = np.array([3.0, 6.0, 2.0])
# defaultcolour = 'darkgreen'


# countries = ['NL','AT','BE',
# 'BG',
# 'HR',
# 'CY',
# 'CZ',
# 'DK',
# 'EE',
# 'FI',
# 'FR',
# 'DE',
# 'EL',
# 'HU',
# 'IE',
# 'IT',
# 'LV',
# 'LT',
# 'LU',
# 'MT',
# 'PL',
# 'PT',
# 'RO',
# 'SK',
# 'SI',
# 'ES',
# 'SE']

# eps_matrix = np.zeros((len(CRsNL), len(countries)))
# rho_matrix = np.zeros((len(CRsNL), len(countries)))
# s_countries = np.zeros(len(countries))
# CR_correction_countries = np.zeros(len(countries))

# for i in range(0,len(countries)):
#     country = countries[i]
#     with open(country + '//' + folder+'//' +'data_for_run.dat', "rb") as input_file:
#         #eps, rho, e_star, r_star, e_optH, cost_star, r_optH, f_star_hydro, r_star_hydro, e_star_hydro, cost_star_hydro, flevels, hlevels = pickle.load(input_file)
#         eps, rho, e_star, r_star, e_optH, cost_star, r_optH, f_star_hydro, r_star_hydro, e_star_hydro, cost_star_hydro = pickle.load(input_file)
#         eps_matrix[:,i] = eps
#         rho_matrix[:,i] = rho
#     with open(country + '\\dataframe.dat', "rb") as input_file:
#         df, PVshare, off, on, river, ee, cap_fact_pv, cap_fact_wind, cap_fact_on, cap_fact_off, cap_fact_river, CR_correction, cap_hydro, f_star_hydro_iea, method_hydro= pickle.load(input_file)
#         s_countries[i] = PVshare
#         CR_correction_countries[i] = CR_correction 
        

def plot_analysis_countries(folder_gen, scenario, alpha_values, CRsNL, countries, defaultcolour, curtcolour, colorX, eps_matrix, rho_matrix, s_countries, CR_correction_countries, decarbonisation_threshold_matrix, curtailment_matrix, CRs, PV_capacity_matrix, wind_capacity_matrix, epsX_array, rhoX_array):
    eps_matrix = np.transpose(eps_matrix)
    rho_matrix = np.transpose(rho_matrix)
    scenario_names = ['reference', 'future', 'current'] 
    #position of country codes in plot
    text_pos = np.zeros(len(countries))
    for i in range(0,len(text_pos)):
        if i%2==0:
            text_pos[i] = +0.8  
            if i==10:
                text_pos[i]=text_pos[i]+0.2 #small change to prevent country codes from interferring
            elif i==16:
                text_pos[i]=text_pos[i]+0.05
            elif i==0:
                text_pos[i]=text_pos[i]-0.05
        else:
            text_pos[i] = -0.8
    
    plt.figure()    
    for k in range(0,len(CRsNL)):
        labelname = scenario_names[k] 
        plt.scatter(s_countries, eps_matrix[k,:],color=defaultcolour, alpha=alpha_values[k], label = labelname)
        #plt.scatter(s_countries[0], eps_matrix[k,0],color='darkorange', alpha=alpha_values[k])
    for i in range(0,len(countries)):
        plt.plot(s_countries[i]*np.ones(len(CRsNL)), eps_matrix[:,i], color='grey', ls = 'dotted', alpha = 0.7)
        plt.text(np.sort(s_countries)[i]-0.01, eps_matrix[0,np.argsort(s_countries)[i]]+text_pos[i],countries[np.argsort(s_countries)[i]])
    plt.ylabel("electrolyzer-to-renewables\n" + r'build out ratio $\epsilon$', fontsize =14)     
    plt.xlabel(r'country-specific fraction of solar $s_i$', fontsize =14)
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    plt.xlim(0.0,1.0)
    plt.ylim(ymin=0.0)
    plt.ylim(0.0,4.1)
    plt.savefig(folder_gen + '\\eps vs solar fraction country', dpi=900,bbox_inches='tight')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2,0,1]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='lower right', framealpha=1.0, fontsize=12, borderpad=0.1, handletextpad=0.1,labelspacing = 0.4)
    plt.savefig(folder_gen + '\\eps vs solar fraction country legend', dpi=900,bbox_inches='tight')
    
    text_pos = np.zeros(len(countries))
    for i in range(0,len(text_pos)):
        if i%2==0:
            text_pos[i] = +0.8  
            if i==10 or i==16:
                text_pos[i]=text_pos[i]+0.2 #small change to prevent country codes from interferring
            elif i==14:
                text_pos[i]=text_pos[i]+0.25
            elif i==0:
                text_pos[i]=text_pos[i]-0.5
            elif i==2:
                text_pos[i]=text_pos[i]+0.1
            elif i==4:
                text_pos[i]=text_pos[i]+0.9
            elif i==6:
                text_pos[i]=text_pos[i]+0.9
            elif i==12:
                text_pos[i]=text_pos[i]+0.45
            elif i==24:
                text_pos[i]=text_pos[i]-2.0                
        else:
            text_pos[i] = -0.8
            if i==1:
                text_pos[i]=text_pos[i]+1.4
            elif i==3:
                text_pos[i]=text_pos[i]+2.0
            elif i==5:
                text_pos[i]=text_pos[i]+2.1
            elif i==7 or i==9:
                text_pos[i]=text_pos[i]+0.3
            elif i==9 or i==11 or i==15 or i==17 or i==19:
                text_pos[i]=text_pos[i]-0.3
            elif i==13:
                text_pos[i]=text_pos[i]+0.15
    
    # Create a figure with 2 subplots: adjust the subplot heights with gridspec_kw    
    fig, ax = plt.subplots(2, 1, figsize=(5.7, 4.0), sharex=False, 
                           gridspec_kw={'height_ratios': [1.5, 1]})  # Bottom subplot is 1.5 times the height of the top

    # First subplot (Solar Capacity-to-Electrolyser Build-Out Ratio)
    for k in range(len(CRs)):
        labelname = r'$CR=%.0f$' % CRs[k]      
        ax[0].scatter(s_countries, PV_capacity_matrix[:, k], alpha=alpha_values[k], color=defaultcolour, edgecolor='None')
    for i in range(0,len(countries)):
        ax[0].plot(s_countries[i]*np.ones(len(CRsNL)), PV_capacity_matrix[i,:], color='grey', ls = 'dotted', alpha = 0.7)
        ax[0].text(np.sort(s_countries)[i]-0.01, PV_capacity_matrix[np.argsort(s_countries)[i],0]+text_pos[i],countries[np.argsort(s_countries)[i]])        
    #ax[0].scatter(PVshare_array, PV_capacityX_array, color=colorX, s=5, edgecolor='None')

    #ax[0].set_xlabel(r'fraction of solar $s$', fontsize=14)
    
    ax[0].set_ylim(0.0, 4.5)
    ax[0].set_xlim(0.0, 1.0)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[0].text(0.1,4.5-0.56, 'GW solar capacity per GW electrolyser capacity', fontsize=12)


    # Second subplot (Wind Capacity-to-Electrolyser Build-Out Ratio)
            
    #ax[1].scatter(PVshare_array, wind_capacityX_array, color=colorX, s=5, edgecolor='None')

    for i in range(0,len(countries)):
        ax[1].plot(s_countries[i]*np.ones(len(CRsNL)), wind_capacity_matrix[i,:], color='grey', ls = 'dotted', alpha = 0.7)
        #ax[1].text(np.sort(s_countries)[i]-0.01, wind_capacity_matrix[np.argsort(s_countries)[i],0]+text_pos[i],countries[np.argsort(s_countries)[i]])
    for k in range(len(CRs)):
        labelname = r'$CR=%.0f$' % CRs[k]      
        ax[1].scatter(s_countries, wind_capacity_matrix[:, k], alpha=alpha_values[k], color=defaultcolour, edgecolor='None')        

    ax[1].set_xlabel(r'country-specific fraction of solar $s_i$', fontsize=14)
    
    ax[1].set_ylim(0.0, 4.5/1.5)
    ax[1].set_xlim(0.0, 1.0)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].text(0.1,3.-0.56, 'GW wind capacity per GW electrolyser capacity', fontsize=12)
    # Place the text once across both subplots
    fig.text(-0.08, 0.55, 'renewable capacities-to-electrolyser \n' + r'               build out ratio', rotation=90, 
             verticalalignment='center', fontsize=14, color='black')

    # Adjust layout to prevent overlapping elements
    plt.tight_layout()

    # Save the combined figure
    plt.savefig(folder_gen + '\\combined_capacities_vs_solar_fraction_wo_eq' + scenario, dpi=900, bbox_inches='tight')
    ax[0].set_ylabel( r' $\frac{s}{\lambda_s\epsilon}$', fontsize=20)
    ax[1].set_ylabel(r'$\frac{1-s}{\lambda_{vw}\epsilon}$', fontsize=20)
    plt.savefig(folder_gen + '\\combined_capacities_vs_solar_fraction_w_eq' + scenario, dpi=900, bbox_inches='tight')
    
    
    
    #position of country codes in plot
    text_pos = np.zeros(len(countries))
    for i in range(0,len(text_pos)):
        if i%2==0:
            text_pos[i] = +0.35  
            if i==12:
                text_pos[i]=text_pos[i]+0.05 #small change to prevent country codes from interferring
            elif i==0:
                text_pos[i]=text_pos[i]+0.15
            elif i==2:
                text_pos[i]=text_pos[i]+0.1    
            elif i==4:
                text_pos[i]=text_pos[i]+0.11
            elif i==24 or i==26 or i==22:
                text_pos[i]=text_pos[i]-0.07
        else:
            text_pos[i] = -0.35
            if i==1 or i==3:
                text_pos[i]=text_pos[i]-0.12
            if i==5:
                text_pos[i]=text_pos[i]-0.08
            
    plt.figure()    
    for k in range(0,len(CRsNL)):
        labelname = scenario_names[k]  
        plt.scatter(s_countries, rho_matrix[k,:],color=defaultcolour, alpha=alpha_values[k], label=labelname)
        #plt.scatter(s_countries[0], rho_matrix[k,0],color='darkorange', alpha=alpha_values[k])
    for i in range(0,len(countries)):
        plt.plot(s_countries[i]*np.ones(len(CRsNL)), rho_matrix[:,i], color='grey', ls = 'dotted', alpha = 0.7)
        plt.text(np.sort(s_countries)[i]-0.01, rho_matrix[0,np.argsort(s_countries)[i]]+text_pos[i],countries[np.argsort(s_countries)[i]])
    plt.xlabel(r'country-specific fraction of solar $s_i$', fontsize =14)
    plt.ylabel(r'renewables threshold value $\rho$' +"\n", fontsize =14)   
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    plt.xlim(0.0,1.0)
    plt.ylim(ymin=0.0)
    plt.ylim(0.0,1.5)
    plt.savefig(folder_gen + '\\rho vs solar fraction country', dpi=900,bbox_inches='tight')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2,0,1]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper right', framealpha=1.0, fontsize=12, borderpad=0.1, handletextpad=0.1,labelspacing = 0.4)
    plt.savefig(folder_gen + '\\rho vs solar fraction country legend', dpi=900,bbox_inches='tight')
    

    # #position of country codes in plot
    # text_pos = np.zeros(len(countries))
    # for i in range(0,len(text_pos)):
    #     if i%2==0:
    #         text_pos[i] = +0.15  
    #         if i==12:
    #             text_pos[i]=text_pos[i]+0.05 #small change to prevent country codes from interferring
    #         elif i==0:
    #             text_pos[i]=text_pos[i]+0.15
    #         elif i==2:
    #             text_pos[i]=text_pos[i]+0.1    
    #         elif i==4 or i==22:
    #             text_pos[i]=text_pos[i]+0.15
    #         elif i==16:
    #             text_pos[i]=text_pos[i]+0.11
    #         elif i==18:
    #             text_pos[i]=text_pos[i]+0.05
    #         elif i==24 or i==26:
    #             text_pos[i]=text_pos[i]+0.25
    #     else:
    #         text_pos[i] = -0.6
    #         if i==1:
    #             text_pos[i]=text_pos[i]-0.05
    #         elif i==3:
    #             text_pos[i]=text_pos[i]-0.09
    #         elif i==5:
    #             text_pos[i]=text_pos[i]-0.03
    #         elif i==15:
    #             text_pos[i]=text_pos[i]+0.05 
    #         elif i ==19 or i==17:
    #             text_pos[i]=text_pos[i]+0.1 
    #         elif i ==23 or i==21:
    #             text_pos[i]=text_pos[i]+0.2 
    #         elif i==25:
    #             text_pos[i]=text_pos[i]+0.3 

    #position of country codes in plot
    text_pos = np.zeros(len(countries))
    for i in range(0,len(text_pos)):
        if i%2==0:
            text_pos[i] = +0.15  
            if i==12:
                text_pos[i]=text_pos[i]+0.05 #small change to prevent country codes from interferring
            elif i==0:
                text_pos[i]=text_pos[i]+0.15
            elif i==2:
                text_pos[i]=text_pos[i]+0.1    
            elif i==4:
                text_pos[i]=text_pos[i]+0.15
            elif i==16:
                text_pos[i]=text_pos[i]+0.11
            elif i==18:
                text_pos[i]=text_pos[i]+0.05
            #elif i==24 or i==26:
                #text_pos[i]=text_pos[i]+0.25
        else:
            text_pos[i] = -0.22
            if i==3:
                text_pos[i]=text_pos[i]-0.12
            elif i==5 or i==1:
                text_pos[i]=text_pos[i]-0.08   
            elif i==25 or i==23 or i==21:
                text_pos[i]=text_pos[i]+0.44  
    
    fig, ax1 = plt.subplots() 
      
    for i in range(0,len(countries)):
        plt.plot(s_countries[i]*np.ones(len(CRsNL)), 100*decarbonisation_threshold_matrix[i,:], color='grey', ls = 'dotted', alpha = 0.7)
        plt.text(np.sort(s_countries)[i]-0.01, 100*decarbonisation_threshold_matrix[np.argsort(s_countries)[i],0]+100*text_pos[i],countries[np.argsort(s_countries)[i]])
    for k in range(0,len(CRs)):
        labelname = r'$CR=%.0f$' % CRs[k]      
        plt.scatter(s_countries, 100*decarbonisation_threshold_matrix[:,k], alpha=alpha_values[k], color=defaultcolour, edgecolor = 'None')  
    plt.xlabel(r'country-specific fraction of solar $s_i$', fontsize =14)
    plt.ylabel('level of grid decarbonisation\n' +r'$1-f(0,\rho)\  (\%)$', fontsize =14) 
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    plt.ylim(ymin=0.0)
    plt.ylim(0.0,100)
    plt.xlim(0.0,1.0)
    
    ax2 = ax1.twinx() 
     
    for i in range(0,len(countries)):
        plt.plot(s_countries[i]*np.ones(len(CRsNL)), 100*curtailment_matrix[i,:], color='grey', ls = 'dotted', alpha = 0.7)
    for k in range(0,len(CRs)):
        labelname = r'$CR=%.0f$' % CRs[k]      
        ax2.scatter(s_countries, 100*curtailment_matrix[:,k], alpha=alpha_values[k], color=curtcolour, edgecolor = 'None')   
    ax2.set_ylabel('renewable power curtailed\n' + r'$\left(\rho+f(0,\rho)-1\right)/\rho \  (\%)$', color=curtcolour, fontsize =14) 
    ax2.set_ylim(0.0,100)
    ax2.tick_params(axis='y', colors=curtcolour, labelsize=14)
#    ax2.set_yticks([0, 10, 20, 30, 40, 50])
    ax2.yaxis.label.set_color(curtcolour)
    ax2.spines['right'].set_color(curtcolour)

    plt.savefig(folder_gen + '\\decarbonisation threshold vs solar fraction country' + scenario, dpi=900,bbox_inches='tight')
    
    
    
    # plt.figure()    
    # for k in range(0,len(CRsNL)):
    #     labelname = scenario_names[k]  
    #     plt.scatter(s_countries, rho_matrix[k,:],color=defaultcolour, alpha=alpha_values[k], label=labelname)
    #     #plt.scatter(s_countries[0], rho_matrix[k,0],color='darkorange', alpha=alpha_values[k])
    # for i in range(0,len(countries)):
    #     plt.plot(s_countries[i]*np.ones(len(CRsNL)), rho_matrix[:,i], color='lightgrey', ls = 'dotted', alpha = 0.7)
    #     plt.text(np.sort(s_countries)[i]-0.01, rho_matrix[0,np.argsort(s_countries)[i]]+text_pos[i],countries[np.argsort(s_countries)[i]])
    # plt.xlabel(r'country-specific fraction of solar $s_i$', fontsize =14)
    # plt.ylabel(r'renewables threshold value $\rho$' +"\n", fontsize =14)   
    # plt.xticks(fontsize=14) 
    # plt.yticks(fontsize=14) 
    # plt.xlim(0.0,1.0)
    # plt.ylim(ymin=0.0)
    # plt.ylim(0.0,1.38)
    # plt.savefig(folder + '\\rho vs solar fraction country', dpi=600,bbox_inches='tight')
    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [2,0,1]
    # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper right', framealpha=1.0, fontsize=12, borderpad=0.1, handletextpad=0.1,labelspacing = 0.4)
    # plt.savefig(folder + '\\rho vs solar fraction country legend', dpi=600,bbox_inches='tight')
    
    
    plt.figure()    
    for k in range(0,len(CRsNL)):
        plt.scatter(rho_matrix[k,s_countries<0.3], eps_matrix[k,s_countries<0.3],color=defaultcolour, alpha=alpha_values[k], marker = '*', label = r'$s_i<0.3$')            
        plt.scatter(rho_matrix[k,((s_countries>0.3) & (s_countries<0.6))], eps_matrix[k,((s_countries>0.3) & (s_countries<0.6))],color=defaultcolour, alpha=alpha_values[k], marker = '^', label = r'$0.3<s_i<0.6$')
        plt.scatter(rho_matrix[k,s_countries>0.6], eps_matrix[k,s_countries>0.6],color=defaultcolour, alpha=alpha_values[k], marker = 'o', label = r'$s_i>0.6$')
        plt.scatter(rhoX_array[s_countries<0.3], epsX_array[s_countries<0.3],color=colorX, alpha=alpha_values[k], marker = '*')    
        plt.scatter(rhoX_array[s_countries>0.6], epsX_array[s_countries>0.6],color=colorX, alpha=alpha_values[k], marker = 'o')
        plt.scatter(rhoX_array[((s_countries>0.3) & (s_countries<0.6))], epsX_array[((s_countries>0.3) & (s_countries<0.6))],color=colorX, alpha=alpha_values[k], marker = '^')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0,1,2]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper right', framealpha=1.0, fontsize=12, borderpad=0.1, handletextpad=0.1,labelspacing = 0.4)
    plt.ylim(ymin=0.0)
    plt.xlim(xmin=0.0)
    plt.ylabel("electrolyzer-to-renewables\n" + r'build out ratio $\epsilon$', fontsize =14)
    plt.xlabel(r'renewables threshold value $\rho$' , fontsize =14) 
    plt.savefig(folder_gen  + '\\rho vs eps', dpi=900,bbox_inches='tight')

    plt.figure()    
    for k in range(0,len(CRsNL)):
        plt.scatter(rho_matrix[k,:], eps_matrix[k,:], 20*s_countries, color=defaultcolour, alpha=alpha_values[k], marker='o')    
    
    plt.figure()
    plt.scatter(rho_matrix[0,:], eps_matrix[0,:], 20*s_countries, color=defaultcolour, alpha=alpha_values[k], marker='o')  
    return 0