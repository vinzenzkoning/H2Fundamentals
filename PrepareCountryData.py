# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import sys
import os
import pickle


def preparecountries():
    method_hydro = 'IEA'
    solar_list=[]
    off_list=[] 
    on_list=[] 
    river_list=[]
    cap_fact_pv_list =[]
    cap_fact_on_list=[]
    cap_fact_off_list=[]
    cap_fact_river_list=[]
    CR_correction_list=[]
    cap_hydro_list=[]
    f_star_hydro_iea_list=[]
    
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
    #countries = ['NL','SK']
    
    elec_gen_files = os.listdir('IEA electricity generation')
    elec_gen_countries= []
    for file in elec_gen_files:
        elec_gen_countries.append(file[35:-4])
    
    dict_countries = {'Austria' : 'AT',
     'Belgium': 'BE',
     'Bulgaria': 'BG',
     'Croatia': 'HR',
     'Cyprus': 'CY',
     'Czech Republic': 'CZ',
     'Denmark': 'DK',
     'Estonia': 'EE',
     'Finland': 'FI',
     'France':'FR',
     'Germany':'DE',
     'Luxembourg':'LU',
     'Greece': 'EL',
     'Hungary':'HU',
     'Ireland':'IE',
     'Italy':'IT',
     'Latvia':'LV',
     'Lithuania': 'LT',
     'Malta': 'MT',
     'Netherlands' : 'NL',
     'Poland':'PL',
     'Portugal':'PT',
     'Romania':'RO',
     'Slovak Republic': 'SK',
     'Slovenia': 'SI',
     'Spain':  'ES',
     'Sweden':  'SE'}
    inv_dict_countries = {v: k for k, v in dict_countries.items()}
    
    elec_gen_countries_codes= []
    for country in elec_gen_countries:
        elec_gen_countries_codes.append(dict_countries[country])
    
    
        
    
        
    
    folder_country_data = 'GA_annual_data_country_level'
    #get capacity data        
    dict_df = pd.read_excel('GA_annual_data_country_level.xlsx', 
                       sheet_name=countries)  
    
    
    
    
    
    
    files = os.listdir('2040')
    for country in countries:
        #first get iea statistics on hydropower
        country_iea = inv_dict_countries[country]
        filename_iea = 'IEA electricity generation\\Electricity generation by source - '+country_iea+'.csv'
        df_iea = pd.read_csv(filename_iea, delimiter=',', skiprows=[0,1,2])  
        if 'Hydro' in df_iea.columns:     
            hydropower_iea = df_iea['Hydro'].tail(1).iloc[0] #last year, usually 2021, sometimes 2020   
            power_iea = df_iea.iloc[-1,1:-1] 
            power_tot_iea = power_iea.sum()
            f_star_hydro_iea = (hydropower_iea / power_tot_iea)
        else:
            f_star_hydro_iea = (0.0)
    
    
        #get time profiles
        dfNL = pd.DataFrame(columns = ['energy_power_solar_pv_electricity_supply', 'energy_power_wind_offshore_electricity_supply', 'energy_power_wind_onshore_electricity_supply', 'energy_power_hydro_run_electricity_supply','total_wind', 'p','d'])
        counter = 0
        counteroff = 0 #offshore wind is not present for landlocked regions, so needs separate treatment
        counteron = 0 #onshore wind is not present in Brussel
        counterriver = 0 #may not be run of river hydropower
        counterd = 0 # counter for demand
        for file in files:
            if file.startswith(country):
                filename = '2040\\' + file
                df=pd.read_csv(filename, delimiter=',', skiprows=[0,1,2,4,5])
                if 'energy_power_solar_pv_electricity_supply' in df.columns:
                    if counter == 0:
                        dfNL.energy_power_solar_pv_electricity_supply = df.energy_power_solar_pv_electricity_supply            
                        counter = 1
                    else:
                        dfNL.energy_power_solar_pv_electricity_supply = dfNL.energy_power_solar_pv_electricity_supply + df.energy_power_solar_pv_electricity_supply     
                if 'energy_power_wind_offshore_electricity_supply' in df.columns:
                    if counteroff == 0:
                        dfNL.energy_power_wind_offshore_electricity_supply = df.energy_power_wind_offshore_electricity_supply                        
                        counteroff = 1
                    else:
                        dfNL.energy_power_wind_offshore_electricity_supply = dfNL.energy_power_wind_offshore_electricity_supply + df.energy_power_wind_offshore_electricity_supply
                if 'energy_power_wind_onshore_electricity_supply' in df.columns:
                    if counteron == 0:
                        dfNL.energy_power_wind_onshore_electricity_supply = df.energy_power_wind_onshore_electricity_supply                        
                        counteron = 1
                    else:
                        dfNL.energy_power_wind_onshore_electricity_supply = dfNL.energy_power_wind_onshore_electricity_supply + df.energy_power_wind_onshore_electricity_supply
                if 'energy_power_hydro_run_electricity_supply' in df.columns:
                    if counterriver == 0:
                        dfNL.energy_power_hydro_run_electricity_supply = df.energy_power_hydro_run_electricity_supply                        
                        counterriver = 1
                    else:
                        dfNL.energy_power_hydro_run_electricity_supply = dfNL.energy_power_hydro_run_electricity_supply + df.energy_power_hydro_run_electricity_supply
                #now look at demand            
                df2=pd.read_csv(filename, delimiter=',', skiprows=[2,4,5], header=[0,1,2])
                dfD = df2.electricity.demand.sum(axis=1)
                if counterd == 0:
                    dfNL.d = dfD
                    counterd = 1
                else:    
                    dfNL.d = dfNL.d + dfD
                
        #normalize demand, average should be equal to 1
        dfNL.d = dfNL.d.size * dfNL.d/dfNL.d.sum()
                
        #compute total wind supply and total power supply per hour
        if dfNL['energy_power_wind_offshore_electricity_supply'].isnull().values.any(): #some countries are landlocked and do not have offshore
            print('There is no offshore wind in ' + country)
            dfNL['energy_power_wind_offshore_electricity_supply'] = 0.0
            landlocked = 1
        else:
            landlocked = 0
        #check if run of river present    
        if dfNL['energy_power_hydro_run_electricity_supply'].isnull().values.any(): #some countries are landlocked and do not have offshore
            #print('There is no run of river hydropower in ' + country)
            dfNL['energy_power_hydro_run_electricity_supply'] = 0.0
            no_run_of_river = 1
        else:
            no_run_of_river = 0
            
        #check fractions of solar, offshore and onshore wind
        dfNL.total_wind = dfNL.energy_power_wind_offshore_electricity_supply + dfNL.energy_power_wind_onshore_electricity_supply
        if method_hydro =='IEA':
            dfNL.p = dfNL.total_wind + dfNL.energy_power_solar_pv_electricity_supply 
            river = 0.0
        else:
            river = dfNL.energy_power_hydro_run_electricity_supply.sum()/dfNL.p.sum()
            dfNL.p = dfNL.total_wind + dfNL.energy_power_solar_pv_electricity_supply + dfNL.energy_power_hydro_run_electricity_supply
        solar = dfNL.energy_power_solar_pv_electricity_supply.sum()/dfNL.p.sum()
        wind = dfNL.total_wind.sum() /dfNL.p.sum()
        off = dfNL.energy_power_wind_offshore_electricity_supply.sum()/dfNL.p.sum()        
        on = dfNL.energy_power_wind_onshore_electricity_supply.sum()/dfNL.p.sum() 
        
    
        #compute capacity factors. Need it for the computation of CR
        cap_pv = dict_df[country].loc[63].at["2040"] #gw pv
        cap_off = dict_df[country].loc[64].at["2040"] #gw offshore
        cap_on = dict_df[country].loc[65].at["2040"] #gw onshore
        cap_river = dict_df[country].loc[51].at["2040"] #gw run of river
        cap_fact_pv =  dfNL.energy_power_solar_pv_electricity_supply.sum() / (8760 *cap_pv)
        cap_fact_wind = dfNL.total_wind.sum() / (8760 * (cap_on + cap_off)) 
        if landlocked == 1:
            cap_fact_off = float('nan')
        else:
            cap_fact_off = dfNL.energy_power_wind_offshore_electricity_supply.sum() / (8760 * cap_off)    
        cap_fact_on = dfNL.energy_power_wind_onshore_electricity_supply.sum() / (8760 * cap_on)
        if no_run_of_river == 1:
            cap_fact_river = float('nan')
        else:
            cap_fact_river = dfNL.energy_power_hydro_run_electricity_supply.sum()  / (8760 * cap_river)    
        
        if method_hydro == 'IEA':
            cap_river = 0.0
        total_cap = cap_pv + cap_on + cap_off + cap_river
        cap_pv_fr = cap_pv / (total_cap)
        cap_on_fr = cap_on / (total_cap)
        cap_off_fr = cap_off / (total_cap)
        cap_river_fr = cap_river / total_cap
        
        #compute correction to CR. Assumption is that in NL wind and solar electricity are approximately as costly
        if country == 'NL': #first store reference value. this procedure works because NL is first in the list
            cap_fact_pv_NL = cap_fact_pv
            cap_fact_on_NL = cap_fact_on
            cap_fact_off_NL = cap_fact_off
            cap_fact_river_NL = cap_fact_river
        if landlocked == 1 and no_run_of_river == 1:
            CR_correction = solar * cap_fact_pv_NL / cap_fact_pv + on * cap_fact_on_NL / cap_fact_on 
        elif landlocked == 0 and no_run_of_river == 1:
            CR_correction = solar * cap_fact_pv_NL / cap_fact_pv + on * cap_fact_on_NL / cap_fact_on + off * cap_fact_off_NL / cap_fact_off 
        elif landlocked == 1 and no_run_of_river == 0:
            CR_correction = solar * cap_fact_pv_NL / cap_fact_pv + on * cap_fact_on_NL / cap_fact_on + river * cap_fact_river_NL / cap_fact_river 
        else:
            CR_correction = solar * cap_fact_pv_NL / cap_fact_pv + on * cap_fact_on_NL / cap_fact_on + off * cap_fact_off_NL / cap_fact_off + river * cap_fact_river_NL / cap_fact_river
        print(country)
        print(CR_correction)
        
        #get hydropower data.  reservoir hydro
        cap_hydro_abs =  dict_df[country].loc[62].at["2040"]
        #this needs to be scaled with average electricity demand
        dem_total = 0
        for i in range(29):
            dem = (dict_df[country].loc[i].at["2040"])/8.76
            dem_total = dem_total + dem
        cap_hydro = cap_hydro_abs/dem_total
        
        
        solar_list.append(solar)
        off_list.append(off) 
        on_list.append(on) 
        river_list.append(river) 
        cap_fact_pv_list.append(cap_fact_pv)
        cap_fact_on_list.append(cap_fact_on)
        cap_fact_off_list.append(cap_fact_off)
        cap_fact_river_list.append(cap_fact_river)
        CR_correction_list.append(CR_correction)
        cap_hydro_list.append(cap_hydro)
        f_star_hydro_iea_list.append(f_star_hydro_iea)
        
        #save file
        if not os.path.exists(country):
            os.mkdir(country)
        with open(country + '\\dataframe.dat', 'wb') as f:
            pickle.dump([dfNL, solar, off, on, river, country, cap_fact_pv, cap_fact_wind, cap_fact_on, cap_fact_off, cap_fact_river, CR_correction, cap_hydro, f_star_hydro_iea, method_hydro], f)    
            




