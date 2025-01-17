# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 16:28:56 2025

@author: Konin045
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

def varydemand():
    #read filenames
    files = os.listdir('2040')
    country = 'NL'
    
    #get time profiles
    dfNL = pd.DataFrame(columns = ['d','industry', 'residential', 'services', 'transport', 'other'])
    #counters for reading the files
    counterd = 0 # counter for demand
    counterindustry = 0
    counterservices = 0
    countertransport = 0
    counterresidential = 0
    counterother = 0
    for file in files:
        if file.startswith(country):
            filename = '2040\\' + file
            df=pd.read_csv(filename, delimiter=',', skiprows=[0,1,2,4,5])
           #now look at demand            
            df['industry'] = df.industry_aluminium_electricity_demand	+df.industry_chemicals_energetic_electricity_demand	+df.industry_data_centres_electricity_demand	+df.industry_fertilizers_energetic_electricity_demand	+df.industry_food_electricity_demand	+df.industry_metals_electricity_demand	+df.industry_other_electricity_demand	+df.industry_paper_electricity_demand	+df.industry_refineries_electricity_demand	+df.industry_steel_electricity_demand
    
                   
            df['residential'] = df.residential_appliances_electricity_demand	+df.residential_collective_heating_heatpump_electricity_demand	+df.residential_heating_heatpump_electricity_demand	+df.residential_heating_hybrid_heatpump_electricity_demand	+df.residential_heating_other_electricity_demand
            df['services'] = df.services_appliances_electricity_demand	+ df.services_heating_heatpump_electricity_demand	+ df.services_heating_hybrid_heatpump_electricity_demand	+ df.services_heating_other_electricity_demand
            df['transport'] = df.transport_bus_electricity_demand	+df.transport_car_electricity_demand	+df.transport_train_electricity_demand	+df.transport_truck_electricity_demand	+df.transport_van_electricity_demand
            df['other']= df.other_electricity_demand + df.agriculture_electricity_demand	+ df.energy_own_use_electricity_demand
            if 'transport_domestic_aviation_electricity_demand' in df.columns: #not all parts of the country have domestic aviation
                df.transport = df.transport + df.transport_domestic_aviation_electricity_demand 
            if 'transport_international_aviation_electricity_demand' in df.columns:
                df.transport = df.transport + df.transport_international_aviation_electricity_demand  
                
            if 'industry' in df.columns:
                if counterindustry == 0:
                    dfNL.industry = df.industry                        
                    counterindustry = 1
                else:
                    dfNL.industry = dfNL.industry + df.industry
            if 'residential' in df.columns:
                if counterresidential == 0:
                    dfNL.residential = df.residential                        
                    counterresidential = 1
                else:
                    dfNL.residential = dfNL.residential + df.residential
            if 'services' in df.columns:
                if counterservices == 0:
                    dfNL.services = df.services                        
                    counterservices = 1
                else:
                    dfNL.services = dfNL.services + df.services
            if 'transport' in df.columns:
                if countertransport == 0:
                    dfNL.transport = df.transport                        
                    countertransport = 1
                else:
                    dfNL.transport = dfNL.transport + df.transport   
            if 'other' in df.columns:
                if counterother == 0:
                    dfNL.other = df.other                        
                    counterother = 1
                else:
                    dfNL.other = dfNL.other + df.other                  
            #alternative way to compute total demand
            df2=pd.read_csv(filename, delimiter=',', skiprows=[2,4,5], header=[0,1,2])
            dfD = df2.electricity.demand.sum(axis=1)
            if counterd == 0:
                dfNL.d = dfD
                counterd = 1
            else:    
                dfNL.d = dfNL.d + dfD
                
                
    #normalize demand, average total demand should be equal to 1
    total_d = dfNL.d.sum()
    total_d_alt = dfNL.other.sum() + dfNL.transport.sum() +dfNL.industry.sum() +dfNL.services.sum() +dfNL.residential.sum() #check. should be the same
    dfNL.d = dfNL.d.size * dfNL.d /total_d
    dfNL.other = dfNL.d.size *  dfNL.other/total_d
    dfNL.transport = dfNL.d.size * dfNL.transport/total_d
    dfNL.industry = dfNL.d.size * dfNL.industry/total_d
    dfNL.services = dfNL.d.size * dfNL.services/total_d
    dfNL.residential = dfNL.d.size * dfNL.residential/total_d      
    
    #now make plot
    fig, ax1 = plt.subplots()
    plt.subplot(2,2,1)
    plt.plot(dfNL.index/dfNL.d.size, dfNL.residential, color='darkgreen', label = 'residential', lw=0.1)
    plt.ylim(0,0.4)
    plt.xlim(0,1)
    plt.text(0.33, 0.35, 'residential', size=11)
    plt.ylabel('demand ', size = 12)
    plt.xlabel('time ' + r'$t$' + ' (years)', size = 12)
    plt.subplot(2,2,2)
    plt.plot(dfNL.index/dfNL.d.size, dfNL.services, color='darkgreen', label = 'services', lw=0.1)
    plt.ylim(0,0.4)
    plt.xlim(0,1)
    plt.ylabel('demand ', size = 12)
    plt.xlabel('time ' + r'$t$' + ' (years)', size = 12)
    plt.text(0.33, 0.35, 'services', size=11)
    plt.subplot(2,2,3)
    plt.plot(dfNL.index/dfNL.d.size, dfNL.transport, color='darkgreen', label = 'transport', lw=0.1)
    plt.ylim(0,0.4)
    plt.xlim(0,1)
    plt.ylabel('demand ', size = 12)
    plt.xlabel('time ' + r'$t$' + ' (years)', size = 12)
    plt.text(0.33, 0.35, 'transport', size=11)
    plt.subplot(2,2,4)
    plt.ylim(0,0.4)
    plt.xlim(0,1)
    plt.ylabel('demand ', size = 12)
    plt.xlabel('time ' + r'$t$' + ' (years)', size = 12)
    plt.plot(dfNL.index/dfNL.d.size, dfNL.industry, color='darkgreen', label = 'industry', lw=0.1)
    plt.text(0.33, 0.05, 'industry', size=11)
    plt.subplots_adjust(wspace=0.05,hspace=0.05)
    plt.tight_layout()
    plt.savefig('NL\\demand_plots ', dpi=600,bbox_inches='tight')
    
    plt.figure()
    plt.plot(dfNL.index/dfNL.d.size, dfNL.d, color='darkorange', label = 'residential', lw=0.1)
    plt.ylim(0,1.5)
    plt.xlim(0,1)
    plt.yticks([0,0.5,1,1.5])
    plt.text(0.42, 1.4, 'total', size=11)
    plt.xlabel('time ' + r'$t$' + ' (years)', size = 12)
    plt.ylabel('demand '+ r'$d$', size=12)
    plt.savefig('NL\\totaldemand ', dpi=600,bbox_inches='tight')

