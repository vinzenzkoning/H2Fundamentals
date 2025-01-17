# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:21:27 2025

@author: Konin045
"""

import numpy as np
import A20230918FundamentalsHydrogen as fundamentals
import pickle
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def varybatteries(datehhmmss):
    alpha_values=[1.0,0.6,0.3] 
    CRsNL = np.array([3.0, 6.0, 2.0]) #cost ratio between renewables production and electrolyzer capacity 
    CRs = CRsNL
    u = 0.0
    T = 8760
    scenario = 'NSWPH'
    defaultcolour = 'darkgreen'
    colourfit = 'olive'
    defaultcolour2 = 'darkorange'
    colourfit2 = 'maroon'
    countries = ['NL','ES']
    scenario_names = [' (reference)', ' (future)', ' (current)'] 
    for country in countries:
        battery_size = 0.0
        size = 2 #number of different sizes for battery storage
        if country == 'NL':    
            size_level = 101
            r_for_bat_ref_level = np.linspace(0.5, 1.50, size_level)
        elif country == 'ES':
            size_level = 121
            r_for_bat_ref_level = np.linspace(0.6, 1.80, size_level)
        bat_ref_level_array =-(1.-1./r_for_bat_ref_level)
        battery_size_array = np.linspace(0.5,1.0,size)
        eps_matrix = np.zeros((size_level, size, len(alpha_values)))
        rho_matrix = np.zeros((size_level, size, len(alpha_values)))
        r_star_matrix = np.zeros((size_level, size, len(alpha_values)))
        e_star_matrix = np.zeros((size_level, size, len(alpha_values)))
        cost_star_matrix = np.zeros((size_level, size, len(alpha_values)))
        r_star_hydro_matrix = np.zeros((size_level, size, len(alpha_values)))
        e_star_hydro_matrix = np.zeros((size_level, size, len(alpha_values)))
        cost_star_hydro_matrix = np.zeros((size_level, size, len(alpha_values)))
        
        with open(country + '\\dataframe.dat', "rb") as input_file:
            df, PVshare, off, on, river, ee, cap_fact_pv, cap_fact_wind, cap_fact_on, cap_fact_off, cap_fact_river, CR_correction, cap_hydro, f_star_hydro_iea, method_hydro= pickle.load(input_file)
        folder_gen = country + '\\' + datehhmmss + 'battery' 
        CRs = CRsNL * CR_correction
        if not os.path.exists(folder_gen):
            os.mkdir(folder_gen)
        for i in range(0,len(battery_size_array)):
            for j in range(0,len(bat_ref_level_array)):        
                battery_size = battery_size_array[i]
                bat_ref_level = bat_ref_level_array[j]
                folder = scenario + ' battery %.2f' %battery_size  + ' ref level %.4f' %bat_ref_level  +' ' + datehhmmss + ' u  %.2f' % u
                if not os.path.exists(folder):
                    os.mkdir(folder)    
                Synergy_coproduction, Vp, VH2, C_integrated_prod, rho, eps, r_star, e_star, eps_sa, C_H2_sa_min, P_conv, cost_star, r_star_hydro, e_star_hydro, cost_star_hydro, flevels, hlevels, e_optH, r_optH, epsX, rhoX, r_starX, e_starX, f_star_hydro  = fundamentals.analyse_power_and_hydrogen(df, folder, T, u, alpha_values, scenario, PVshare, CRs, cap_hydro, f_star_hydro_iea, method_hydro, cap_fact_pv, cap_fact_wind, battery_size, bat_ref_level)
                eps_matrix[j,i,]=eps
                rho_matrix[j,i,]=rho
                r_star_matrix[j,i,]=r_star
                e_star_matrix[j,i,]=e_star
                cost_star_matrix[j,i,]=cost_star
                r_star_hydro_matrix[j,i,]=r_star_hydro
                e_star_hydro_matrix[j,i,]=e_star_hydro
                cost_star_hydro_matrix[j,i,]=cost_star_hydro
    
      
        #store relevant data. r_star, e_star, cost_star are presented in supplementary tables
        with open(folder_gen + '\\results battery variation.dat', 'wb') as f:
            pickle.dump([battery_size_array, r_for_bat_ref_level, eps_matrix,rho_matrix, r_star_matrix, e_star_matrix, cost_star_matrix, cost_star_hydro_matrix], f)
    
    
        #find and store the right values for e and r for full system decarbonisation
        #results are only valid when r_for_bat_ref_level matches the r that is found through the optimisation
        r_bat_hydro = np.zeros((size, len(alpha_values)))
        e_bat_hydro = np.zeros((size, len(alpha_values)))
        cost_bat_hydro = np.zeros((size, len(alpha_values)))
        r_bat = np.zeros((size, len(alpha_values)))
        e_bat = np.zeros((size, len(alpha_values)))
        cost_bat = np.zeros((size, len(alpha_values)))
        for i in range(0,len(battery_size_array)):
            for k in range(0,len(CRs)):
                min_cost_index_hydro = np.argmin(cost_star_hydro_matrix[:,i,k])
                r_bat_hydro[i,k] = r_star_hydro_matrix[min_cost_index_hydro, i, k]
                e_bat_hydro[i,k] = e_star_hydro_matrix[min_cost_index_hydro, i, k]
                cost_bat_hydro[i,k] = cost_star_hydro_matrix[min_cost_index_hydro, i, k]
                min_cost_index = np.argmin(cost_star_matrix[:,i,k])
                r_bat[i,k] = r_star_matrix[min_cost_index, i, k]
                e_bat[i,k] = e_star_matrix[min_cost_index, i, k]
                cost_bat[i,k] = cost_star_matrix[min_cost_index, i, k]            
        
        #now do this structurally along the trajectory. Reload the relevant data.
            battery_size = battery_size_array[i]
            for j in range(0,size_level):
                bat_ref_level = bat_ref_level_array[j]   
                folder = scenario + ' battery %.2f' %battery_size  + ' ref level %.4f' %bat_ref_level  +' ' + datehhmmss + ' u  %.2f' % u  
                with open(folder+'//' +'data_for_run.dat', "rb") as input_file:
                    eps, rho, e_star, r_star, e_optH, cost_star, r_optH, f_star_hydro, r_star_hydro, e_star_hydro, cost_star_hydro, flevels, hlevels = pickle.load(input_file)
                if j==0:
                    len_real_flevels = len(flevels[len(hlevels):])
                    r_sol_matrix = np.zeros((size_level, len(alpha_values),len_real_flevels))
                    e_sol_matrix = np.zeros((size_level, len(alpha_values),len_real_flevels))
                    cost_sol_matrix = np.zeros((size_level, len(alpha_values),len_real_flevels))        
                total_levels = len(hlevels)+len_real_flevels
                r_sol_matrix[j,:,:]=r_optH[:,len(hlevels):total_levels]
                e_sol_matrix[j,:,:]=e_optH[:,len(hlevels):total_levels]
                for k in range(0, len(CRs)):
                    cost_sol_matrix[j,k,:]=CRs[k]*r_optH[k,len(hlevels):total_levels]+ e_optH[k,len(hlevels):total_levels] #cost proxy
    
    
            r_sols = np.zeros((len(alpha_values),len_real_flevels))
            e_sols = np.zeros((len(alpha_values),len_real_flevels))
            for k in range(0,len(CRs)):
                for ilevel in range(len_real_flevels):        
                    #min_cost_index2 = np.argmin(abs(r_sol_matrix[:,k, ilevel]-r_for_bat_ref_level[:size_battery_ref_level_loop]))#alternative way
                    min_cost_index = np.argmin(cost_sol_matrix[:,k, ilevel]) 
                    e_bat_sol = e_sol_matrix[min_cost_index, k, ilevel]
                    r_bat_sol = r_sol_matrix[min_cost_index, k, ilevel]
                    if abs(r_bat_sol - r_for_bat_ref_level[min_cost_index]) > 0.05 and e_bat_sol>0.0: #for some high f levels r is indifferent
                        print('r optimal does not match r used for battery-induced modification of supply profile')
                    r_sols[k,ilevel] = r_bat_sol
                    e_sols[k,ilevel] = e_bat_sol
        
        
            eps_bat = np.zeros(len(CRs))
            rho_bat = np.zeros(len(CRs))
            for k in range(0,len(CRs)): #find epsilon and rho through curve fitting
                popt, pcov = curve_fit(fundamentals.func, r_sols[k,:], e_sols[k,:])
                eps_bat[k] = popt[0]
                rho_bat[k] = popt[1]       
        
            
                    
            #store optima.
            with open(folder_gen + '\\results battery sols' +' battery %.2f' %battery_size +'.dat', 'wb') as f:
                pickle.dump([eps_bat, rho_bat, e_sols, r_sols], f)
            
        
            Synergy_coproduction, Vp, VH2, C_integrated_prod, rho, eps, r_star, e_star, eps_sa, C_H2_sa_min, P_conv, cost_star, r_star_hydro, e_star_hydro, cost_star_hydro, flevels, hlevels, e_optH, r_optH, epsX, rhoX, r_starX, e_starX, f_star_hydro  = fundamentals.analyse_power_and_hydrogen(df, folder, T, u, alpha_values, scenario, PVshare, CRs, cap_hydro, f_star_hydro_iea, method_hydro, cap_fact_pv, cap_fact_wind, 0.0, 0.0)
        
            #now make the graph for the electrolyser-renewables deployment trajectory for both batteries and no batteries.
            plt.figure()
            ylimmax = 3.5
            plt.ylim(0.0,ylimmax)
            plt.xlim(0.0,2.5)
            for k in range(0,len(CRs)):
                labelname = 'CR'+ r'$=%.1f$' % CRs[k] + scenario_names[k]
                plt.scatter(r_optH[k,len(hlevels):],e_optH[k,len(hlevels):], label=labelname, alpha=alpha_values[k], color=defaultcolour)
                labelname = r'$\epsilon=%5.2f, \rho=%5.2f$' % (eps[k],rho[k])
                plt.plot(r_optH[k,len(hlevels):], fundamentals.func(r_optH[k,len(hlevels):], eps[k], rho[k]), color = colourfit, ls='dashed',label=labelname, alpha=alpha_values[k])
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [4,5,0,1,2,3]
            legend1 = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper left', framealpha=1.0, fontsize = 8.5) 
            for k in range(0,len(CRs)):
                labelname = 'CR'+ r'$=%.1f$' % CRs[k] + scenario_names[k]
                plt.scatter(r_sols[k,:],e_sols[k,:], label=labelname, alpha=alpha_values[k], color=defaultcolour2)
                labelname = r'$\epsilon=%5.2f, \rho=%5.2f$' % (eps_bat[k],rho_bat[k])
                plt.plot(r_sols[k,:], fundamentals.func(r_sols[k,:], eps_bat[k], rho_bat[k]), color = colourfit2, ls='dashed',label=labelname, alpha=alpha_values[k])             
            plt.xlabel(r'average renewable power generation $r$')
            plt.ylabel(r'electrolyser capacity $e$') 
        
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [4+6,5+6,0+6,1+6,2+6,3+6]
            plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper right', framealpha=1.0, fontsize = 8.5)
            plt.gca().add_artist(legend1)
            plt.tight_layout()     
            plt.savefig(country + '\\flevelsin e r space wo title bat' + scenario +  datehhmmss +' battery %.2f' %battery_size + '.png', dpi=600,bbox_inches='tight')


    
   