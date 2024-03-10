# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 19:08:25 2024

@author: Konin045
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
import csv
from scipy.optimize import curve_fit
import math
import time
from datetime import date
import seaborn as sns
import sys
import os
import pickle

#country = 'NL'
#with open(country + '\\dataframe.dat', "rb") as input_file:
#    df, solar, off, on, river, ee, cap_fact_pv, cap_fact_wind, cap_fact_on, cap_fact_off, cap_fact_river, CR_correction, cap_hydro, f_star_hydro_iea, method_hydro= pickle.load(input_file)
#resLoad = 1.0 - df['p'].values/np.mean(df['p'].values) #1.0 because we assume flat demand

def p_after_battery(resLoad, Pc):
    dT = 1. #measure time in hours
    
    etaE2S = 1.0#0.96 #efficiency from electricity to storage
    etaS2E = 1.0#0.96 #efficiency from storage to electricity
    tau = 4.
    #ratio of the ouput power capacity (the max electricity that comes out of storage medium)
    #to input power capacity (the max electricity that goes into storage medium)
    ratioOutInP = 1#discharging can be done with the same rate as charging  
    PcOut = Pc * ratioOutInP # is max amount of electricity that can be released
    
    #specify values for storage medium
    Ec = tau*Pc #in units of d*h=h

    
    #initialize energy and power levels of storage medium
    P = np.zeros(np.size(resLoad)) #power level of electricity in /out (positve when charging, negative when discharging, max Pc, min -Pc)
    E = np.zeros(np.size(resLoad)) #energy level of storage (between 0 and Ec)
    EstartShare = 0.5 #starting level of battery as a share
    
    
    #introduce storage to resLoad
    resLoadS = resLoad.copy()
    for i in range(0, len(resLoadS)):
        if i==0:
            Eprev = EstartShare*Ec#Estart #x*Ec would be alternative where x would indicate how full the battery is at start 
        else:
            Eprev = E[i-1]
        if resLoadS[i] < 0 : 
            #then we charge the storage medium
            if -resLoadS[i]*dT*etaE2S > (Ec - Eprev): #if shortage is larger than remaining energy
                P[i] = min((Ec - Eprev)/(dT*etaE2S), Pc) 
                #print('A')
            else: #if shortage is smaller than remaining energy level
                P[i] = min(-resLoadS[i], Pc)
                #print('B')
            E[i] = Eprev + etaE2S*P[i]*dT #update energy levels storage medium; use E2S (electricity to storage (i.e. hydrogen or energy level battery)) efficiency
        elif resLoadS[i] > 0:
            #then we discharge the storage medium
            if resLoadS[i]*dT > etaS2E*Eprev: #if all res demand cannot be supplied by storage
                P[i] = -min(etaS2E*Eprev/dT, PcOut)  
                #print('C')
            else:                   #if all res demand can be supplied by storage
                P[i] = -min(resLoadS[i], PcOut)
                #print('D')
            E[i] = Eprev + (P[i]/etaS2E)*dT #update energy levels storage medium; use E2S (electricity to storage (i.e. hydrogen or energy level battery)) efficiency. Correction of P by division by S2E efficiency: more storage energy than what is gained as electricity    
        #for each i
        resLoadS[i] = resLoadS[i] + P[i] #adjust remaining residual load
            
    return (1. - resLoadS), E



