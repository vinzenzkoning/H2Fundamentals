# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:24:35 2024

@author: Konin045
"""
#plotting routines for graphs as a function of solar share

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

def plot_analysis_solar_share(folder_gen, scenario, alpha_values, CRsNL, defaultcolour, curtcolour, colorX, eps_matrix, rho_matrix, decarbonisation_threshold_matrix, curtailment_matrix, CRs, PV_capacity_matrix, wind_capacity_matrix, epsX_array, rhoX_array, PVshare_array, PV_capacityX_array, wind_capacityX_array, r_star_matrix, e_star_matrix, PV_capacity_full_matrix, wind_capacity_full_matrix, r_starX_array, e_starX_array, cost_star_matrix, c_r_NL, eps_sa_matrix, P_conv_matrix, C_H2_sa_min_matrix):
    fig, ax1 = plt.subplots() 
    for k in range(0,len(CRs)):
        labelname = r'$CR=%.0f$' % CRs[k]      
        plt.scatter(PVshare_array, 100*decarbonisation_threshold_matrix[:,k], alpha=alpha_values[k], color=defaultcolour, s=5, edgecolor = 'None')
        plt.plot(fundamentals.movingaverage(PVshare_array,5), 100*fundamentals.movingaverage(decarbonisation_threshold_matrix[:,k],5),label=labelname,alpha=alpha_values[k], color=defaultcolour)
    plt.scatter(PVshare_array, 100*rhoX_array, color=colorX, s=5, edgecolor = 'None')#in scenario X we use electrolysers right when there is oversupply, so decarbonisation threshold = rho for this scenario
    plt.plot(fundamentals.movingaverage(PVshare_array,5), 100*fundamentals.movingaverage(rhoX_array,5),label='X', color=colorX)
    #plt.legend(loc='lower left', framealpha=1.0)
    plt.xlabel(r'fraction of solar $s$', fontsize =14)
    plt.ylabel('level of grid decarbonisation\n' +r'$1-f(0,\rho)\  (\%)$', fontsize =14) 
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    plt.ylim(ymin=0.0)
    plt.ylim(0.0,100)
    plt.xlim(0.0,1.0)
    ax2 = ax1.twinx() 
    for k in range(0,len(CRs)):
        labelname = r'$CR=%.0f$' % CRs[k]      
        ax2.scatter(PVshare_array, 100*curtailment_matrix[:,k], alpha=alpha_values[k], color=curtcolour, s=5, edgecolor = 'None')
        ax2.plot(fundamentals.movingaverage(PVshare_array,5), 100*fundamentals.movingaverage(curtailment_matrix[:,k],5),label=labelname,alpha=alpha_values[k], color=curtcolour)
    ax2.set_ylabel('renewable power curtailed\n' + r'$\left(\rho+f(0,\rho)-1\right)/\rho \  (\%)$', color=curtcolour, fontsize =14) 
    ax2.set_ylim(0.0,100)
    ax2.tick_params(axis='y', colors=curtcolour, labelsize=14)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.yaxis.label.set_color(curtcolour)
    ax2.spines['right'].set_color(curtcolour)
    plt.savefig(folder_gen + '\\decarbonisation threshold vs solar fraction' + scenario, dpi=600,bbox_inches='tight')

    plt.figure()
    for k in range(0,len(CRs)):
        labelname = r'$CR=%.0f$' % CRs[k]      
        plt.scatter(PVshare_array, eps_matrix[:,k], alpha=alpha_values[k], color=defaultcolour, s=5, edgecolor = 'None')
        plt.plot(fundamentals.movingaverage(PVshare_array,5), fundamentals.movingaverage(eps_matrix[:,k],5),label=labelname,alpha=alpha_values[k], color=defaultcolour)
    plt.scatter(PVshare_array, epsX_array, color=colorX, s=5, edgecolor = 'None')
    plt.plot(fundamentals.movingaverage(PVshare_array,5), fundamentals.movingaverage(epsX_array,5),label='X', color=colorX)
    #plt.legend(loc='lower left', framealpha=1.0)
    plt.xlabel(r'fraction of solar $s$', fontsize =14)
    plt.ylabel("electrolyser-to-renewables\n" + r'build out ratio $\epsilon$'+"\n", fontsize =14) 
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    plt.ylim(ymin=0.0)
    plt.xlim(0.0,1.0)
    plt.savefig(folder_gen + '\\eps vs solar fraction' + scenario, dpi=600,bbox_inches='tight')


    # Create a figure with 2 subplots: adjust the subplot heights with gridspec_kw
    fig, ax = plt.subplots(2, 1, figsize=(5.45, 4.0), sharex=False, 
                           gridspec_kw={'height_ratios': [3, 1]})  # Bottom subplot is 3 times the height of the top

    # First subplot (Solar Capacity-to-Electrolyser Build-Out Ratio)
    for k in range(len(CRs)):
        labelname = r'$CR=%.0f$' % CRs[k]      
        ax[0].scatter(PVshare_array, PV_capacity_matrix[:, k], alpha=alpha_values[k], color=defaultcolour, s=5, edgecolor='None')
        ax[0].plot(fundamentals.movingaverage(PVshare_array, 5), fundamentals.movingaverage(PV_capacity_matrix[:, k], 5), label=labelname, alpha=alpha_values[k], color=defaultcolour)
    ax[0].scatter(PVshare_array, PV_capacityX_array, color=colorX, s=5, edgecolor='None')
    ax[0].plot(fundamentals.movingaverage(PVshare_array, 5), fundamentals.movingaverage(PV_capacityX_array, 5), label=labelname, color=colorX)
    #ax[0].set_xlabel(r'fraction of solar $s$', fontsize=14)

    ax[0].set_ylim(0.0, 6.0)
    ax[0].set_xlim(0.0, 1.0)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[0].text(0.1,6.0-0.5, r'GW solar capacity per GW electrolyser capacity', fontsize=12)


    # Second subplot (Wind Capacity-to-Electrolyser Build-Out Ratio)
    for k in range(len(CRs)):
        labelname = r'$CR=%.0f$' % CRs[k]      
        ax[1].scatter(PVshare_array, wind_capacity_matrix[:, k], alpha=alpha_values[k], color=defaultcolour, s=5, edgecolor='None')
        ax[1].plot(fundamentals.movingaverage(PVshare_array, 5), fundamentals.movingaverage(wind_capacity_matrix[:, k], 5), label=labelname, alpha=alpha_values[k], color=defaultcolour)
    ax[1].scatter(PVshare_array, wind_capacityX_array, color=colorX, s=5, edgecolor='None')
    ax[1].plot(fundamentals.movingaverage(PVshare_array, 5), fundamentals.movingaverage(wind_capacityX_array, 5), label=labelname, color=colorX)
    ax[1].set_xlabel(r'fraction of solar $s$', fontsize=14)

    ax[1].set_ylim(0.0, 6./3.)
    ax[1].set_xlim(0.0, 1.0)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].text(0.1,6./3.-0.5, 'GW wind capacity per GW electrolyser capacity', fontsize=12)
    # Place the text once across both subplots
    fig.text(-0.1, 0.52, 'renewable capacities-to-electrolyser \n' + r'               build out ratio ', rotation=90, 
             verticalalignment='center', fontsize=14, color='black')

    # Adjust layout to prevent overlapping elements
    plt.tight_layout()

    # Save the combined figure
    plt.savefig(folder_gen + '\\combined_capacities_vs_solar_fraction_wo_eq' + scenario, dpi=600, bbox_inches='tight')
    ax[0].set_ylabel( r' $\frac{s}{\lambda_s\epsilon}$', fontsize=20)
    ax[1].set_ylabel(r'$\frac{1-s}{\lambda_{vw}\epsilon}$', fontsize=20)
    plt.savefig(folder_gen + '\\combined_capacities_vs_solar_fraction_w_eq' + scenario, dpi=600, bbox_inches='tight')


    scenario_names = [' (reference)', ' (future)', ' (current)'] 
    plt.figure()
    plt.scatter(PVshare_array, rhoX_array, color=colorX, s=5, edgecolor = 'None')
    #plt.plot(movingaverage(PVshare_array,5), movingaverage(rhoX_array,5),label ='CR'+ r'$=\infty$' + ' (X: zero cost electrolysis)', color=colorX)
    plt.plot(fundamentals.movingaverage(PVshare_array,5), fundamentals.movingaverage(rhoX_array,5),label ='CR'+ r'$=\infty$' + ' (X)', color=colorX)
    for k in range(0,len(CRs)):
        labelname = 'CR' + r'$=%.0f$' % CRs[k] + scenario_names[k]     
        plt.scatter(PVshare_array, rho_matrix[:,k], alpha=alpha_values[k], color=defaultcolour, s=5)
        plt.plot(fundamentals.movingaverage(PVshare_array,5), fundamentals.movingaverage(rho_matrix[:,k],5),label=labelname,alpha=alpha_values[k], color=defaultcolour)
    #plt.legend(loc='lower left', framealpha=1.0)
    plt.xlabel(r'fraction of solar $s$', fontsize =14)
    plt.ylabel(r'renewables threshold value $\rho$' +"\n", fontsize =14)   
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    plt.xlim(0.0,1.0)
    plt.ylim(ymin=0.0)
    plt.savefig(folder_gen + '\\rho vs solar fraction wo legend' + scenario, dpi=600,bbox_inches='tight')    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3,1,2,0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper left', framealpha=1.0, fontsize = 11, frameon=False)
    plt.ylim(0.0,1.45)
    plt.savefig(folder_gen + '\\rho vs solar fraction' + scenario, dpi=600,bbox_inches='tight')    
     

    plt.figure()
    for k in range(0,len(CRs)):
        labelname = r'$CR=%.0f$' % CRs[k]      
        plt.scatter(rho_matrix[:,k], eps_matrix[:,k], alpha=alpha_values[k], color=defaultcolour, s=5, edgecolor = 'None')
        plt.plot(fundamentals.movingaverage(rho_matrix[:,k],5), fundamentals.movingaverage(eps_matrix[:,k],5),label=labelname,alpha=alpha_values[k], color=defaultcolour)
    #plt.legend(loc='lower left', framealpha=1.0)
    plt.scatter(rhoX_array, epsX_array, color=colorX, s=5, edgecolor = 'None')
    plt.plot(fundamentals.movingaverage(rhoX_array,5), fundamentals.movingaverage(epsX_array,5),label ='CR'+ r'$=\infty$' + ' (X)', color=colorX)
    plt.xlabel(r'renewables threshold value $\rho$', fontsize =14) 
    plt.ylabel("electrolyser-to-renewables\n" + r'build out ratio $\epsilon$', fontsize =14) 
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    plt.ylim(ymin=0.0)
    #plt.xlim(0.0,1.0)
    plt.savefig(folder_gen + '\\eps vs rho for solar fractions' + scenario, dpi=600,bbox_inches='tight')

    plt.figure()
    for k in range(0,len(CRs)):
        labelname = r'$CR=%.0f$' % CRs[k]      
        plt.scatter(PVshare_array, r_star_matrix[:,k], alpha=alpha_values[k], color=defaultcolour, s=5)
        plt.plot(fundamentals.movingaverage(PVshare_array,5), fundamentals.movingaverage(r_star_matrix[:,k],5),label=labelname,alpha=alpha_values[k], color=defaultcolour)
    #plt.legend(loc='lower left', framealpha=1.0)
    plt.xlabel(r'fraction of solar $s$', fontsize =14)
    plt.ylabel(r'average renewable generation $r^*$'+ "\nat full decarbonisation", fontsize =14) 
    plt.ylim(ymin=0.0)
    plt.xlim(0.0,1.0)
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    plt.savefig(folder_gen + '\\r star vs solar fraction' + scenario, dpi=600,bbox_inches='tight')    

    plt.figure()
    for k in range(0,len(CRs)):
        labelname = r'$CR=%.0f$' % CRs[k]      
        plt.scatter(PVshare_array, PV_capacity_full_matrix[:,k], alpha=alpha_values[k], color=defaultcolour, s=5)
        plt.plot(fundamentals.movingaverage(PVshare_array,5), fundamentals.movingaverage(PV_capacity_full_matrix[:,k],5),label=labelname,alpha=alpha_values[k], color=defaultcolour)
    #plt.legend(loc='lower left', framealpha=1.0)
    plt.xlabel(r'fraction of solar $s$', fontsize =14)
    plt.ylabel(r'   solar capacity'+ "\nat full decarbonisation", fontsize =14) 
    plt.ylim(ymin=0.0)
    plt.xlim(0.0,1.0)
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    plt.savefig(folder_gen + '\\solar capacity vs solar fraction' + scenario, dpi=600,bbox_inches='tight')   

    plt.figure()
    for k in range(0,len(CRs)):
        labelname = r'$CR=%.0f$' % CRs[k]      
        plt.scatter(PVshare_array, wind_capacity_full_matrix[:,k], alpha=alpha_values[k], color=defaultcolour, s=5)
        plt.plot(fundamentals.movingaverage(PVshare_array,5), fundamentals.movingaverage(wind_capacity_full_matrix[:,k],5),label=labelname,alpha=alpha_values[k], color=defaultcolour)
    #plt.legend(loc='lower left', framealpha=1.0)
    plt.xlabel(r'fraction of solar $s$', fontsize =14)
    plt.ylabel(r'   wind capacity'+ "\nat full decarbonisation", fontsize =14) 
    plt.ylim(ymin=0.0)
    plt.xlim(0.0,1.0)
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    plt.savefig(folder_gen + '\\wind capacity vs solar fraction' + scenario, dpi=600,bbox_inches='tight')   

    plt.figure()
    for k in range(0,len(CRs)):
        labelname = r'$CR=%.0f$' % CRs[k]      
        plt.scatter(PVshare_array, e_star_matrix[:,k],  alpha=alpha_values[k], color=defaultcolour, s=5)
        plt.plot(fundamentals.movingaverage(PVshare_array,5), fundamentals.movingaverage(e_star_matrix[:,k],5),label=labelname,alpha=alpha_values[k], color=defaultcolour)
    #plt.legend(loc='upper left', framealpha=1.0)
    plt.xlabel(r'fraction of solar $s$', fontsize =14)
    #plt.ylabel(r'electrolyzer capacity $e^*$ at full decarbonization $e^*$', fontsize =14) 
    plt.ylabel(r'   electrolyser capacity $e^{*}$' + "\nat full decarbonisation", fontsize =14)
    plt.ylim(0.0,None)
    plt.xlim(0.0,1.0)
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    plt.savefig(folder_gen + '\\e star vs solar fraction' + scenario, dpi=600,bbox_inches='tight') 


    plt.figure()
    #A = np.linspace(0,15.75,10) #14,9
    A = np.linspace(0,19.25,12) #A and x for lines of equal cost
    x=np.linspace(0,10,11)
    for a in A:
        plt.plot(x,-CRsNL[0]*x+a, color='k', alpha=0.35, lw=0.5)#draw lines of equal cost
    plt.scatter(r_starX_array, e_starX_array, color=colorX, s=5, edgecolor = 'None')
    plt.scatter(r_starX_array[0], e_starX_array[0], color=colorX, s=30, marker = 'D')
    plt.scatter(r_starX_array[-1], e_starX_array[-1], color=colorX, s=30, marker = 'D')
    plt.plot(fundamentals.movingaverage(r_starX_array,5), fundamentals.movingaverage(e_starX_array,5),label='X', color=colorX)
    for k in range(0,len(CRs)):
        labelname = r'$CR=%.0f$' % CRs[k]      
        plt.scatter(r_star_matrix[:,k], e_star_matrix[:,k],  alpha=alpha_values[k], color=defaultcolour, s=3)
        plt.scatter(r_star_matrix[0,k], e_star_matrix[0,k], alpha=alpha_values[k], color=defaultcolour, s=30, marker = 'D') #diamonds for wind-only and solar-only
        plt.scatter(r_star_matrix[-1,k], e_star_matrix[-1,k], alpha=alpha_values[k], color=defaultcolour, s=30, marker = 'D')
        plt.plot(fundamentals.movingaverage(r_star_matrix[:,k],5), fundamentals.movingaverage(e_star_matrix[:,k],5),label=labelname,alpha=alpha_values[k], color=defaultcolour)
    #plt.legend(loc='upper left', framealpha=1.0)
    plt.xlabel(r'average renewable generation $r^{*}$'+ "\nat full decarbonisation", fontsize =14)
    plt.ylabel(r'   electrolyser capacity $e^{*}$' + "\nat full decarbonisation", fontsize =14) 
    #plt.ylim(0.0,8.)
    plt.ylim(0.0,11.7)
    plt.xlim(0.0,3.)
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    plt.savefig(folder_gen + '\\e en r star vs solar fraction' + scenario, dpi=600,bbox_inches='tight') 


    plt.figure()
    for k in range(0,len(CRs)):
        labelname = r'$CR=%.0f$' % CRs[k]      
        plt.scatter(PVshare_array, cost_star_matrix[:,k], alpha=alpha_values[k], color=defaultcolour, s=5, edgecolor = 'None')
        plt.plot(fundamentals.movingaverage(PVshare_array,5), fundamentals.movingaverage(cost_star_matrix[:,k],5),label=labelname,alpha=alpha_values[k], color=defaultcolour)
    #plt.legend(loc='upper left', framealpha=1.0)
    plt.scatter(PVshare_array, r_starX_array*c_r_NL, color=colorX, s=5, edgecolor = 'None')
    plt.plot(fundamentals.movingaverage(PVshare_array,5), fundamentals.movingaverage(r_starX_array*c_r_NL,5),label ='CR'+ r'$=\infty$' + ' (X)', color=colorX)
    plt.xlabel(r'fraction of solar $s$', fontsize =14)
    plt.ylabel(r'   cost $C^{*} \ (€/ W_a)$'+ "\nof full decarbonisation", fontsize =14) 
    plt.ylim(ymin=0.0)
    plt.xlim(0.0,1.0)
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    plt.savefig(folder_gen + '\\cost star vs solar fraction' + scenario, dpi=600,bbox_inches='tight')

    plt.figure()
    for k in range(0,len(CRs)):
        labelname = r'$CR=%.0f$' % CRs[k]      
        plt.scatter(PVshare_array, eps_sa_matrix[:,k], alpha=alpha_values[k], color=defaultcolour)
        plt.plot(fundamentals.movingaverage(PVshare_array,5), fundamentals.movingaverage(eps_sa_matrix[:,k],5),label=labelname,alpha=alpha_values[k], color=defaultcolour)
    plt.legend(loc='lower right', framealpha=1.0)
    plt.xlabel(r'fraction of solar $s$', fontsize =14)
    plt.ylabel(r'stand-alone electrolyser-to-renewables ratio $\epsilon^{sa}$', fontsize =14) 
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
    return 0