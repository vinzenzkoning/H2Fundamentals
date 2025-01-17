# Fundamentals of hydrogen production and use in a renewable power system
# About
With this code one can carry out the analysis presentend in the manuscript 'Fundamentals of hydrogen production and use in a decarbonising power system' by Gert Jan Kramer, Wouter van de Graaf, Vinzenz Koning. 
All figures in both the main text and the supplementary information can be recreated.

# Author
Vinzenz Koning. For questions please contact me at v.koning@uu.nl

# Instruction
To run the analysis please run RunFundamentals.py. This will call all the other python functions. The only exception is Sankey.py, which is a stand-alone scripot with which Sankey diagram was made for Figure 1b of the paper.
In the default settings, the code carries out the analysis for the Netherlands. This takes a few minutes on a laptop with the current high resolution settings and generates the plots in figures 1a, 2, 3, 6, 7b, Extended Data figure 4, and the figures in Supplementary Discussion 3.
If you would like to run the analysis for other EU countries and generate Figure 5 and Extended Data Figures 1,2,3, please set Qruncountries to 'yes' in line 32 of RunFundamentals.py. This will take about an hour.
If you would like to run the analysis for other solar fractions for the Netherlands and generate Figure 4, please set QrunQrunsolar to 'yes' in line 33 of RunFundamentals.py. This will take a few hours.  
If you would like to run the battery sensitivity analyis for the Netherlands and Spain and generate Figures in Supplementary Discussion 1, 
please set QrunQrunbatteries to 'yes' in line 34 of RunFundamentals.py. Since the method was not designed for batteries, this analysis will take significantly longer, namely 2 days.  

# Data
The code reads data from the folder '2040', which is retrieved from 
Haan, M. den, Uden, J. van, Rennuit-Mortensen, A. W. & Kawale, D. Scenario Data: Hourly Demand and Supply per Region (NSWPH regions, weather years 2011-2019). Zenodo https://doi.org/10.5281/zenodo.7892927 (2023)
The geojson files contain data on the EU countries, which are needed for the maps (Figure 8 and Figure 5e). 

# Libraries
The code was run using Python 3.9.18 and makes use of the following libraries: 
time, datetime, os, PIL, numpy, pickle, matplotlib, scipy, pandas, geopandas
