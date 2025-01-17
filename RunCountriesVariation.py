# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:42:29 2025

@author: Konin045
"""

import numpy as np
import A20230918FundamentalsHydrogen as fundamentals
import pickle
import os
import PlotCountryData as PlotCountryData
import AnalysisCountries as analysisCountries

def varycountries(datehhmmss, countries):
    alpha_values=[1.0,0.6,0.3] 
    CRsNL = np.array([3.0, 6.0, 2.0]) #cost ratio between renewables production and electrolyzer capacity 
    CRs = CRsNL
    u = 0.0
    PVshare = 0.0
    battery_size = 0.0
    T = 8760
    scenario = 'NSWPH'
    defaultcolour = 'darkgreen'
    curtcolour = 'saddlebrown'
    colorX = 'navy'
    
    if len(countries)<2:
        scenario = 'NSWPH-NL' #if only want to generate NL and no orange line in plot
    
     
        
    eps_matrix = np.zeros((len(countries),len(CRsNL)))
    rho_matrix = np.zeros((len(countries),len(CRsNL)))
    epsX_array = np.zeros(len(countries))
    rhoX_array = np.zeros(len(countries))
    decarbonisation_threshold_matrix = np.zeros((len(countries), len(alpha_values)))
    curtailment_matrix = np.zeros((len(countries), len(alpha_values)))
    s_countries = np.zeros(len(countries))
    CR_correction_countries = np.zeros(len(countries))
    cap_fact_pv_matrix = np.zeros((len(countries), len(alpha_values)))
    cap_fact_wind_matrix = np.zeros((len(countries), len(alpha_values)))
    size = len(countries)
    r_star_matrix = np.zeros((size, len(alpha_values)))
    e_star_matrix = np.zeros((size, len(alpha_values)))  
    cost_star_matrix = np.zeros((size, len(alpha_values))) 
    f_star_hydro_matrix = np.zeros((size, len(alpha_values))) 
    r_star_hydro_matrix = np.zeros((size, len(alpha_values))) 
    e_star_hydro_matrix = np.zeros((size, len(alpha_values)))
    cost_star_hydro_matrix = np.zeros((size, len(alpha_values))) 
    
    #for country in countries:
    for i in range(0,len(countries)):
        country = countries[i]
        with open(country + '\\dataframe.dat', "rb") as input_file:
            df, PVshare, off, on, river, ee, cap_fact_pv, cap_fact_wind, cap_fact_on, cap_fact_off, cap_fact_river, CR_correction, cap_hydro, f_star_hydro_iea, method_hydro= pickle.load(input_file)
        folder = country + '\\' + datehhmmss + 'u  %.2f' % u
        CRs = CRsNL * CR_correction
        s_countries[i] = PVshare
        CR_correction_countries[i] = CR_correction 
    
        
        if not os.path.exists(folder):
            os.mkdir(folder)
            
        Synergy_coproduction, Vp, VH2, C_integrated_prod, rho, eps, r_star, e_star, eps_sa, C_H2_sa_min, P_conv, cost_star, r_star_hydro, e_star_hydro, cost_star_hydro, flevels, hlevels, e_optH, r_optH, epsX, rhoX, r_starX, e_starX, f_star_hydro  = fundamentals.analyse_power_and_hydrogen(df, folder, T, u, alpha_values, scenario, PVshare, CRs, cap_hydro, f_star_hydro_iea, method_hydro, cap_fact_pv, cap_fact_wind, battery_size, 0.0)
        eps_matrix[i,:] = eps
        rho_matrix[i,:] = rho
        epsX_array[i] = epsX
        rhoX_array[i] = rhoX
        decarbonisation_threshold,curtailment = fundamentals.calc_grid_decarb_level(CRs,rho,r_optH,flevels)
        decarbonisation_threshold_matrix[i,]=decarbonisation_threshold
        curtailment_matrix[i,]=curtailment
        cap_fact_pv_matrix[i,]=cap_fact_pv
        cap_fact_wind_matrix[i,]=cap_fact_wind 
        r_star_matrix[i,]=r_star
        e_star_matrix[i,]=e_star
        cost_star_matrix[i,]=cost_star
        f_star_hydro_matrix[i,] = f_star_hydro 
        r_star_hydro_matrix[i,] = r_star_hydro
        e_star_hydro_matrix[i,] = e_star_hydro 
        cost_star_hydro_matrix[i,] = cost_star_hydro
        
    #capacities for epsilon 
    PV_capacity_matrix = np.zeros((len(countries), len(alpha_values)))
    wind_capacity_matrix = np.zeros((len(countries), len(alpha_values)))
    for k in range(0,len(CRs)):
        PV_capacity_matrix[:,k] = s_countries / (cap_fact_pv_matrix[:,k] * eps_matrix[:,k]) #solar capacity per GW of electrolysis
        wind_capacity_matrix[:,k] = (1.-s_countries) / (cap_fact_wind_matrix[:,k] * eps_matrix[:,k]) #wind capacity per GW electrolysis
    
        
    
    if not os.path.exists('multiplecountries'):
        os.mkdir('multiplecountries')    
    folder_gen = 'multiplecountries' + '\\' + datehhmmss + 'u  %.2f' % u
    if not os.path.exists(folder_gen):
        os.mkdir(folder_gen)
    analysisCountries.plot_analysis_countries(folder_gen, scenario, alpha_values, CRsNL, countries, defaultcolour, curtcolour, colorX, eps_matrix, rho_matrix, s_countries, CR_correction_countries, decarbonisation_threshold_matrix, curtailment_matrix, CRs, PV_capacity_matrix, wind_capacity_matrix, epsX_array, rhoX_array)
    PlotCountryData.plotcountrydata(folder_gen, alpha_values, countries, eps_matrix, rho_matrix, e_star_matrix, r_star_matrix, f_star_hydro_matrix, cost_star_matrix, r_star_hydro_matrix, e_star_hydro_matrix, cost_star_hydro_matrix)
    
    #save data
    with open(folder_gen + '\\results countries variation for country analysis.dat', 'wb') as f:
        pickle.dump([folder_gen, scenario, alpha_values, CRsNL, countries, defaultcolour, curtcolour, colorX, eps_matrix, rho_matrix, s_countries, CR_correction_countries, decarbonisation_threshold_matrix, curtailment_matrix, CRs, PV_capacity_matrix, wind_capacity_matrix, epsX_array, rhoX_array], f)
    with open(folder_gen + '\\results countries variation for country plots.dat', 'wb') as f:
        pickle.dump([folder_gen, alpha_values, countries, eps_matrix, rho_matrix, e_star_matrix, r_star_matrix, f_star_hydro_matrix, cost_star_matrix, r_star_hydro_matrix, e_star_hydro_matrix, cost_star_hydro_matrix], f)
    return 0

