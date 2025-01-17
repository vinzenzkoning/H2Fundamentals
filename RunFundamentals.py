# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:22:27 2023

@author: Konin045
"""


import AnalyticCalc
import RunCountriesVariation
import RunSolarVariation 
import PrepareCountryData 
import RunBatteriesVariation
import extrademandplotNL as RunDemand
import time
from datetime import date
import os


def getdatehhmmss():
    today = date.today()
    currentdate = str(today.year)+str(today.month)+str(today.day)
    hhmm = str(time.localtime(time.time()).tm_hour) + str(time.localtime(time.time()).tm_min)
    hhmmss = str(time.localtime(time.time()).tm_hour) + str(time.localtime(time.time()).tm_min) + ' ' + str(time.localtime(time.time()).tm_sec)
    datehhmm = currentdate+' '+hhmm
    datehhmmss = currentdate+' '+hhmmss
    return datehhmmss, datehhmm
 
#user input. Switch to 'yes' if you would like to carry out the analyis.
#in default mode only the Netherlands is computed. 
#Analysis for the Netherlands takes few mins. Analyis for EU takes an hour, solar variation few hours, and batteries a few days.   
Qruncountries = 'no'
Qrunsolar = 'no'
Qrunbatteries = 'no'
Qrunanalytic = 'yes'
Qrundemand = 'yes'

#initialize countries with data
PrepareCountryData.preparecountries()

#get date and directory
datehhmmss, datehhmm  = getdatehhmmss()
cwd = os.getcwd()

#run the analysis for the Netherlands
#and generate the plots in figures 1a, 2, 3, 6, 7b, Extended Data figure 4
RunCountriesVariation.varycountries(datehhmmss, ['NL'])

#run the analyis for all EU countries
#and generate also plots in figure 4 and Extended Data Figures 1,2,3
if Qruncountries == 'yes':
    countries = ['NL','AT','BE','BG','HR','CY','CZ','DK','EE','FI','FR','DE','EL','HU','IE','IT','LV','LT','LU','MT','PL','PT','RO','SK','SI','ES','SE']
    RunCountriesVariation.varycountries(datehhmmss, countries)

#run analysis in which solar share of the Netherlands is varied
#and generate plots in Figure 4
if Qrunsolar == 'yes':
    RunSolarVariation.varysolar(datehhmmss)

#visualise the analytic treatment
#i.e. generate the plots for Supplementary Discussion 3  
if Qrunsolar == 'yes':  
    AnalyticCalc.calc_analytic_eps_rho()

#analyse demand fluctuations for the Netherlands
#and generate plots for Supplementary Discussion 2
if Qrundemand == 'yes':     
    RunDemand.varydemand()    

#run analysis for varying battery size for Netherlands and Spain
#and generate plots for figures in Supplementary Discussion 1
if Qrunbatteries == 'yes':    
    RunBatteriesVariation.varybatteries(datehhmmss)




      












