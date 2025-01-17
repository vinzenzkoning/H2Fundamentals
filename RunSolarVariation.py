# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:34:08 2025

@author: Konin045

This code 
"""
import Analysissolarshare as plotsolarshare
from PIL import Image
import numpy as np
import A20230918FundamentalsHydrogen as fundamentals
import pickle
import os


def varysolar(datehhmmss):
    ###################  USER INPUT  ################### 
    scenario = 'Mix NSWHP' #'2040GAR' #choices: 2040EHR, 2050EHR, 2040GAR, 2050GAR, Offshore', 'PV', PVNL, PV_GAR2040
    ####################################################
    battery_size = 0.0
    c_r_NL = 4.5
    alpha_values=[1.0,0.6,0.3] 
    CRsNL = np.array([3.0, 6.0, 2.0]) #cost ratio between renewables production and electrolyzer capacity 
    T = 8760
    country = 'NL'
    defaultcolour = 'darkgreen'
    curtcolour = 'saddlebrown'
    colorX = 'navy'
    with open(country + '\\dataframe.dat', "rb") as input_file:
        df_original, solar, off, on, river, ee, cap_fact_pv, cap_fact_wind, cap_fact_on, cap_fact_off, cap_fact_river, CR_correction, cap_hydro, f_star_hydro_iea, method_hydro= pickle.load(input_file)
    CRs = CRsNL * CR_correction
    frames = []
    u = 0.0
    size = 101
    PVshare_array = np.linspace(0.0,1.0,size)
    eps_matrix = np.zeros((size, len(alpha_values)))
    rho_matrix = np.zeros((size, len(alpha_values)))
    decarbonisation_threshold_matrix = np.zeros((size, len(alpha_values)))
    curtailment_matrix = np.zeros((size, len(alpha_values)))
    r_star_matrix = np.zeros((size, len(alpha_values)))
    e_star_matrix = np.zeros((size, len(alpha_values)))
    r_starX_array = np.zeros(size)
    e_starX_array = np.zeros(size)
    epsX_array = np.zeros(size)
    rhoX_array = np.zeros(size)
    cost_star_matrix = np.zeros((size, len(alpha_values)))
    eps_sa_matrix = np.zeros((size, len(alpha_values)))
    C_H2_sa_min_matrix = np.zeros((size, len(alpha_values)))
    P_conv_matrix = np.zeros((size, len(alpha_values)))
    for i in range(0,len(PVshare_array)):
        PVshare = PVshare_array[i]
        #adjust the p values to account for the different share of solar 
        df = df_original
        df['p']=(1.0-PVshare)*df_original['total_wind']/np.mean(df_original['total_wind'])+PVshare*df_original['energy_power_solar_pv_electricity_supply']/np.mean(df_original['energy_power_solar_pv_electricity_supply']) #division by means is to normalize the input data such that we indeed compute based on the right solar fraction
        folder = scenario + ' PVshare %.2f' %PVshare  + ' ' + datehhmmss + ' u  %.2f' % u
        if not os.path.exists(folder):
            os.mkdir(folder)
        Synergy_coproduction, Vp, VH2, C_integrated_prod, rho, eps, r_star, e_star, eps_sa, C_H2_sa_min, P_conv, cost_star, r_star_hydro, e_star_hydro, cost_star_hydro, flevels, hlevels, e_optH, r_optH, epsX, rhoX, r_starX, e_starX, f_star_hydro  = fundamentals.analyse_power_and_hydrogen(df, folder, T, u, alpha_values, scenario, PVshare, CRs, cap_hydro, f_star_hydro_iea, method_hydro, cap_fact_pv, cap_fact_wind, battery_size, 0.0)
        decarbonisation_threshold,curtailment = fundamentals.calc_grid_decarb_level(CRs,rho,r_optH,flevels)
        eps_matrix[i,]=eps
        rho_matrix[i,]=rho
        decarbonisation_threshold_matrix[i,]=decarbonisation_threshold
        curtailment_matrix[i,]=curtailment
        r_star_matrix[i,]=r_star
        e_star_matrix[i,]=e_star
        r_starX_array[i,] = r_starX
        e_starX_array[i,] = e_starX
        epsX_array[i,] = epsX
        rhoX_array[i,] = rhoX
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
    
    #capacities for epsilon 
    PV_capacity_matrix = np.zeros((size, len(alpha_values)))
    wind_capacity_matrix = np.zeros((size, len(alpha_values)))
    for k in range(0,len(CRs)):
        PV_capacity_matrix[:,k] = PVshare_array / (cap_fact_pv * eps_matrix[:,k]) #solar capacity per GW of electrolysis
        wind_capacity_matrix[:,k] = (1.-PVshare_array) / (cap_fact_wind * eps_matrix[:,k]) #wind capacity per GW electrolysis
    PV_capacityX_array= PVshare_array / (cap_fact_pv * epsX_array)
    wind_capacityX_array =  (1.-PVshare_array) / (cap_fact_wind * epsX_array)
    
    #capacities for full power system decarbonisation    
    PV_capacity_full_matrix = np.zeros((size, len(alpha_values)))
    wind_capacity_full_matrix = np.zeros((size, len(alpha_values)))   
    for k in range(0,len(CRs)):
        PV_capacity_full_matrix[:,k] = PVshare_array / (cap_fact_pv * r_star_matrix[:,k]) #solar capacity per GW of electrolysis
        wind_capacity_full_matrix[:,k] = (1.-PVshare_array) / (cap_fact_wind * r_star_matrix[:,k]) #wind capacity per GW electrolysis
    
    
    plotsolarshare.plot_analysis_solar_share(folder_gen, scenario, alpha_values, CRsNL, defaultcolour, curtcolour, colorX, eps_matrix, rho_matrix, decarbonisation_threshold_matrix, curtailment_matrix, CRs, PV_capacity_matrix, wind_capacity_matrix, epsX_array, rhoX_array, PVshare_array, PV_capacityX_array, wind_capacityX_array, r_star_matrix, e_star_matrix, PV_capacity_full_matrix, wind_capacity_full_matrix, r_starX_array, e_starX_array, cost_star_matrix, c_r_NL, eps_sa_matrix, P_conv_matrix, C_H2_sa_min_matrix)
    
    
    with open(folder_gen + '\\results solar fraction variation.dat', 'wb') as f:
        pickle.dump([PVshare_array, eps_matrix,rho_matrix, r_star_matrix, e_star_matrix, cost_star_matrix, eps_sa_matrix, C_H2_sa_min_matrix, P_conv_matrix], f)
    
    with open(folder_gen + '\\results solar fraction variation for plots.dat', 'wb') as f:
        pickle.dump([folder_gen, scenario, alpha_values, CRsNL, defaultcolour, curtcolour, colorX, eps_matrix, rho_matrix, decarbonisation_threshold_matrix, curtailment_matrix, CRs, PV_capacity_matrix, wind_capacity_matrix, epsX_array, rhoX_array, PVshare_array, PV_capacityX_array, wind_capacityX_array, r_star_matrix, e_star_matrix, PV_capacity_full_matrix, wind_capacity_full_matrix, r_starX_array, e_starX_array, cost_star_matrix, c_r_NL, eps_sa_matrix, P_conv_matrix, C_H2_sa_min_matrix], f)

