# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:57:32 2025

@author: Konin045
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

def calc_analytic_eps_rho():
    scenario='analytic'
    alpha_values=[1.0,0.6,0.3] 
    T = 8760
    country = 'NL'
    defaultcolour = 'darkgreen'
    T = 8760
    country='NL'
    with open(country + '\\dataframe.dat', "rb") as input_file:
        df_original, solar, off, on, river, ee, cap_fact_pv, cap_fact_wind, cap_fact_on, cap_fact_off, cap_fact_river, CR_correction, cap_hydro, f_star_hydro_iea, method_hydro= pickle.load(input_file)
    size_duration_curves = 11
    PVshare_array = np.linspace(0.0,1.0,size_duration_curves)
    #first, let's see how the curves look like
    plt.figure()
    for i in range(0,len(PVshare_array)):
        PVshare = PVshare_array[i]
        #adjust the p values to account for the different share of solar 
        df = df_original
        df['p']=(1.0-PVshare)*df_original['total_wind']/np.mean(df_original['total_wind'])+PVshare*df_original['energy_power_solar_pv_electricity_supply']/np.mean(df_original['energy_power_solar_pv_electricity_supply']) #division by means is to normalize the input data such that we indeed compute based on the right solar fraction
        p_unsorted = df['p'].values.copy()
        p_unsorted = p_unsorted / np.mean(p_unsorted)
        p_sorted = np.sort(p_unsorted)[::-1]
        Tau =  np.divide(list(range(T)), T)+1.0/T # extra term to make it 1 for complete year
        plt.plot(Tau, p_sorted, color = defaultcolour, alpha=1./3. +2.*(1-PVshare)/3.)     
        plt.xlabel('duration '+ r'$\tau$' + ' (years)', fontsize=14)
        plt.ylabel('renewable power generation '+ r'$\bar{q}$' + '\n (normalised)', fontsize=14)
        plt.xlim([0,1])
        plt.ylim([0,6])
        plt.xticks(fontsize=14) 
        plt.yticks(fontsize=14)
        plt.tight_layout()
        #print(np.sum(p_sorted)/8760)
    plt.plot(Tau, 5.-12.5*Tau, color = 'darkorange', ls='dashed',label='solar-dominated', lw='2'),
    plt.plot(Tau, 2.-2.*Tau, color = 'darkorange', alpha=1,label='wind-dominated', lw='2')
    plt.text(0.03,5, r'$s=1$', color=defaultcolour, alpha=1./3.)
    plt.text(0.32,1.75, r'$s=0$', color=defaultcolour, alpha=1)
    plt.legend()
    plt.savefig(country + '\\power duration curve vs solar fraction ' + scenario, dpi=600,bbox_inches='tight')
    
    eta_c=0.35
    plt.figure()
    costratio=np.linspace(0.01, 16, 100)
    rho_w = 0.25+0.25*(1.+4./(costratio*eta_c))**0.5
    rho_s = 0.1+0.1*(1.+10./(costratio*eta_c))**0.5
    plt.plot(costratio, rho_w, color = 'darkorange', alpha=1, label='wind-dominated', lw='2')
    plt.scatter(3.,0.68068598, color=defaultcolour, alpha=alpha_values[0], label=r'$\rho$' + ' for ' + '$s=0$') #hardcoded values from solar variation analysis
    plt.scatter(6.,0.5797603, color=defaultcolour, alpha=alpha_values[1])
    plt.scatter(2.,0.78268281, color=defaultcolour, alpha=alpha_values[2])
    plt.plot(costratio, rho_s, color = 'darkorange', ls='dashed',label='solar-dominated', lw='2')
    plt.xlabel('cost ratio CR', fontsize=14)
    plt.ylabel('renewables threshold value ' + r'$\rho$', fontsize=14)
    plt.xlim([0,16])
    plt.ylim([0,1.45])
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14)
    plt.scatter(3.,0.41831283, color=defaultcolour, alpha=alpha_values[0], label=r'$\rho$' + ' for ' + '$s=1$', marker='s')
    plt.scatter(6.,0.33237113, color=defaultcolour, alpha=alpha_values[1], marker='s')
    plt.scatter(2.,0.53181499, color=defaultcolour,alpha=alpha_values[2], marker='s' )
    plt.legend()
    plt.savefig(country + '\\rho vs cost ratio analytic ' + scenario, dpi=600,bbox_inches='tight')
    
    plt.figure()
    eps_w = (4.*rho_w - 1)/(rho_w+1./costratio)
    eps_s = (10.*rho_s - 1)/(rho_s+1./costratio)
    plt.plot(costratio, eps_w, color = 'darkorange', alpha=1, label='wind-dominated', lw='2')
    plt.scatter(3.,1.879053, color=defaultcolour, alpha=alpha_values[0], label=r'$\epsilon$' + ' for ' + '$s=0$')
    plt.scatter(6.,1.92826385, color=defaultcolour, alpha=alpha_values[1])
    plt.scatter(2.,1.79600958, color=defaultcolour, alpha=alpha_values[2])
    plt.plot(costratio, eps_s, color = 'darkorange', ls='dashed',label='solar-dominated', lw='2')
    plt.xlabel('cost ratio CR', fontsize=14)
    plt.ylabel('electrolyser-to-renewables ' + '\n build out ratio ' + r'$\epsilon$', fontsize=14)
    plt.xlim([0,16])
    plt.ylim([0,7.3])
    plt.xticks(fontsize=14) 
    plt.yticks([0,1,2,3,4,5],fontsize=14)
    plt.scatter(3.,3.2884782, color=defaultcolour, alpha=alpha_values[0], label=r'$\epsilon$' + ' for ' + '$s=1$', marker='s')
    plt.scatter(6.,3.92706313, color=defaultcolour, alpha=alpha_values[1], marker='s')
    plt.scatter(2.,3.01831042, color=defaultcolour, alpha=alpha_values[2], marker='s')
    plt.legend(loc='upper left')
    plt.savefig(country + '\\eps vs cost ratio analytic ' + scenario, dpi=600,bbox_inches='tight')
    return 0