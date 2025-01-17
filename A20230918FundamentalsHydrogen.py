# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 19:48:25 2022

@author: Konin045
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle
import A20240103battery as battery



####################################################


#################### PARAMETERS ####################
eta_H2e = 0.5
eta_e = 0.7
eta_c = eta_H2e * eta_e  #efficiency energy conversion p2h2p
c_r_NL = 4.5
CRsNL = np.array([3.0, 6.0, 2.0])
alt_cost_factor = 0.5 #factor by which the cost come down in alternative scenario compared with the original

####################################################

def func(x, eps, rho):
    return eps * (x - rho) * np.heaviside(x - rho, 0.0)

def func_q(x, beta, eps_sa):
    return (eps_sa * x - 1.0 - beta / x ) * np.heaviside(eps_sa * x - 1.0 - beta / x, 0.0)

def func_g(x, beta, eps_sa):
    return eps_sa - beta / x**2 

def calc_grid_decarb_level(CRs,rho,r_optH,flevels):    #from rho, find the grid decarbonisation level, i.e. 1 - f_rho    
    f_rho = np.zeros(len(CRs))
    for k in range(0,len(CRs)):
        f_rho[k] = np.interp(-rho[k], -r_optH[k,:], flevels) #minus sign is because np.interp assumes increasing array 
    decarbonisation_threshold = 1. - f_rho
    curtailment = (rho + f_rho - 1.) / rho
    return decarbonisation_threshold, curtailment

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def analyse_power_and_hydrogen(df, folder, T, u, alpha_values, scenario, PVshare, CRs, cap_hydro, f_star_hydro_iea, method_hydro, cap_fact_pv, cap_fact_wind, battery_size, bat_ref_level):
    c_r = c_r_NL * CRs/CRsNL
    c_e = c_r/CRs
    defaultcolour = 'darkgreen'
    colourfit = 'olive'
    scenario_names = [' (reference)', ' (future)', ' (current)']
    d_unsorted = df['d'].values.copy()
    p_unsorted = df['p'].values.copy()
    p_unsorted = p_unsorted / np.mean(p_unsorted)
    p_unsorted_no_bat = p_unsorted.copy()
    if battery_size > 0.0:
        #bat_ref_level = -(1.-1./1.31)
        p_unsorted, Ebat = battery.p_after_battery(1.0 - p_unsorted + bat_ref_level, battery_size) 
        p_unsorted = p_unsorted + bat_ref_level
    else:
        Ebat = np.zeros(len(p_unsorted))
    mean_p_unsorted = np.mean(p_unsorted+1e-12)
    p_unsorted = (p_unsorted+1e-12) / mean_p_unsorted #1e-12 just to get rid of zeros in 100% solar scenario's, just to avoid numerical issues
    p_unsorted_no_bat = (p_unsorted_no_bat+1e-12) / mean_p_unsorted 
    p_array = np.sort(p_unsorted)[::-1]
    p_array_no_bat = np.sort(p_unsorted_no_bat)[::-1]

    
    plt.figure()
    plt.plot(df.p, label='power profile')
    plt.xlabel(r'$t$' + ' (hours)')
    plt.ylabel(r'$p / D$')
    plt.ylabel(r'$p$')
    plt.tight_layout()
    plt.savefig(folder + '\\power profile ' + scenario, dpi=600,bbox_inches='tight')
    
        
    Tau =  np.divide(list(range(T)), T)+1.0/T # extra term to make it 1 for complete year
    plt.figure()
    plt.plot(Tau, p_array, label='power duration curve') 
    plt.xlabel(r'$\tau$' + ' (years)')
    #plt.ylabel(r'$\bar{p} / D$')
    plt.ylabel(r'$\bar{p}$')
    #plt.legend()
    plt.tight_layout()
    plt.savefig(folder + '\\power duration curve ' + scenario, dpi=600,bbox_inches='tight')
    plt.close('all')
    
    Iup = np.zeros(T)
    Idown = np.zeros(T)
    b = np.zeros(T)
    for tau_b in range (0,T):
        Iup[tau_b] = 1.0/T*(p_array[0:tau_b+1].sum() - tau_b*p_array[tau_b])
        Idown[tau_b] = 1.0/T*((T-tau_b)*p_array[tau_b] - p_array[tau_b:T].sum() )
        b[tau_b]=p_array[tau_b]
        
        
    plt.figure()
    plt.plot(Iup, label='Integral up')
    plt.plot(Idown, label='Integral down')
    plt.xlabel(r'$\tau$' + ' (hours)')
    plt.ylabel(r'$I_{up/down}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder + '\\Integralvstau ' + scenario, dpi=600,bbox_inches='tight')  
    
    fig, ax1 = plt.subplots()
    left, bottom, width, height = [0.48, 0.53, 0.41, 0.33]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax1.plot(Tau, p_array, label='power duration curve', color='k') 
    ax1.fill_between(Tau, p_array[4400], p_array, facecolor='C0', alpha=0.3, interpolate=True, edgecolor = 'None')
    ax1.axhline(y = p_array[4400], color = 'k')
    ax1.axvline(x = 4400/len(Tau), ymax=.36, color = 'k', linestyle = '--')
    ax1.text(0.12, p_array[4400]+0.2, r'$i_\uparrow$', dict(size=20))
    ax1.text(4400/len(Tau)+0.35, p_array[4400]-0.5, r'$i_\downarrow$', dict(size=20))
    ax1.text(0.01, p_array[4400]+0.05, r'$b$', dict(size=10))
    ax1.text(4400/len(Tau)+0.01, 0.05, r'$\tau\left(b\right)$', dict(size=10))
    ax1.set_xlabel(r'$\tau$' + ' (years)', dict(size=15))
    ax1.set_ylabel(r'$\bar{p}$', dict(size=15))
    ax1.set_yticks([])
    ax1.set_xticks([0,1])
    ax1.set_xlim(0,1)
    ax1.set_ylim(ymin=0)
    ax2.plot(b, Iup, label='Integral up', color='k')
    ax2.plot(b, Idown, label='Integral down', color = 'gray')
    ax2.set_xlabel(r'$b$')
    ax2.set_ylabel(r'$I$')
    ax2.text(2, 0.1, r'$I_\uparrow$', dict(size=10), color='k')
    ax2.text(2, 1.3, r'$I_\downarrow$', dict(size=10), color='gray')
    ax2.set_xlim(0,np.max(b))
    ax2.set_ylim(ymin=0)
    ax2.set_xticks([0,1,2])
    ax2.set_yticks([0,1])
    plt.savefig(folder + '\\Integralvstau plus inset' + scenario, dpi=600,bbox_inches='tight')  

    
    
    plt.figure()
    plt.plot(b, Iup, label='Integral up')
    plt.plot(b, Idown, label='Integral down')
    plt.xlabel(r'$b$', dict(size=15))
    plt.ylabel(r'$I_{up/down}$', dict(size=15))
    plt.legend()
    plt.tight_layout()  
    plt.savefig(folder + '\\Integralvsb ' + scenario, dpi=300,bbox_inches='tight')  
    

    r = (1.0+u)/b
    fY = 1.0 - r * (1.0 - Iup)
    fX = 1.0 - r * (1.0 - (1-eta_c)*Iup)
    e= np.matmul(np.transpose(r).reshape(T,1), np.transpose(b).reshape(1,T))-1.0-u
    f=0.0*e
    for i in range (0,T): #counter for different values of 1/r
        print(i)
        for j in range (0,T): #counter for different values of (1+e)/r
            f[i,j]= (1.0 - r[i]) + r[i] * ( (1.0-eta_c)*Iup[i] + eta_c * Iup[j])
    
    indexX = (fX>0.0).argmin()
    indexZ = 4222 #a choice such that, typically, Z is in between X0 and Y0
    fZ = f[indexX,indexZ]
    if fZ > fY[indexX]:
        indexZ = 2222 #adjust choice
        fZ = f[indexX,indexZ]#this tends to happen for PV   
    eZ = e[indexX,indexZ]
    plt.figure()
    ylimmax = 2.5
    plt.ylim(0.0,ylimmax)
    plt.xlim(0.0,1.0)
    plt.plot(1.0-fY, r, label='Y', color='k')
    plt.plot(1.0-fX, r, label='X', color='k')
    plt.fill_betweenx(r, 1.0-fY, 1.0-fX, facecolor='lightgray', alpha=0.3, interpolate=True, edgecolor = 'None')
    ftrivial = np.linspace(fX[0], 1.0,20)
    plt.plot(1.0-ftrivial, 1.0-ftrivial, label='Y', color='k')
    plt.scatter(1.0-fX[0],1.0-fX[0], color='k')
    delta_text = 0.05
    plt.text(1.0-fX[0],1.0-fX[0]+delta_text, r'$O$', dict(size=10), color='k')
    plt.scatter(1.0-fX[indexX],r[indexX], color='k')
    plt.text(1.0-fX[indexX]-delta_text,r[indexX], r'$X_0$', dict(size=10), color='k')
    plt.scatter(1.0-fY[indexX],r[indexX], color='k')
    plt.text(1.0-fY[indexX]-delta_text,r[indexX], r'$Y_0$', dict(size=10), color='k')
    plt.scatter(1.0-fZ,r[indexX], color='k')
    plt.text(1.0-fZ-0.7*delta_text,r[indexX], r'$Z$', dict(size=10), color='k')
    shift=1000
    if r[indexX+shift]<ylimmax:
        plt.text(1.0-fY[indexX+shift]-0.7*delta_text,r[indexX+shift], r'$Y$', dict(size=10), color='k')
    shift=-800
    plt.text(1.0-fX[indexX+shift]+delta_text,r[indexX+shift], r'$X$', dict(size=10), color='k')
    plt.xlabel(r'level of decarbonisation $1-f$', dict(size=11))
    plt.ylabel(r'average renewable power generation $r$', dict(size=11))
    #plt.tight_layout()  
    plt.savefig(folder + '\\fvsr ' + scenario, dpi=600,bbox_inches='tight')  
    plt.close('all')
    
    
    
    
    #now plot specifics for point Z, Y0, X0, and O 
    posX = 0.45
    posY = 2.1

    plt.figure(figsize=(6.4, 4.8))
        
    ax=plt.subplot(2,3,4)
    plt.plot(Tau, r[indexX]*p_array, color = 'k') #label='power duration curve'
    plt.ylabel(r'renewable power $\bar{p}$', dict(size=10))
    plt.xlabel(r'duration $\tau$' + ' (years)', dict(size=10))
    plt.xlim(0.0,1.0)
    plt.ylim(0.0,4.0)
    plt.yticks([0,1,2,3,4])
    plt.xticks([0,1])
    plt.axhline(y = 1.0, color = 'k', linestyle = '--')
    plt.text(posX,posY, r'point $Y_0$', dict(size=10), color='k')
    plt.fill_betweenx(np.append(r[indexX]*p_array, 0.0), 0.0, np.append(Tau, 1.0), where=(np.append(r[indexX]*p_array, 0.0) <= 1.0), facecolor='darkgreen', alpha=0.7, interpolate=True, edgecolor = 'None', label = 'direct use')
    plt.fill_between(Tau, 1.0, r[indexX]*p_array, where=(r[indexX]*p_array >= 1.0), facecolor='wheat', alpha=0.3, interpolate=True, edgecolor = 'None', label = 'curtailed')
    plt.fill_between(Tau, 1.0, r[indexX]*p_array, where=(r[indexX]*p_array <= 1.0), facecolor='dimgray', alpha=0.7, interpolate=True, edgecolor = 'None', label = 'fossil/firm')
    ax.xaxis.labelpad = -10
    
    ax=plt.subplot(2,3,5)
    fZ = f[indexX,indexZ]
    eZ = e[indexX,indexZ]
    plt.plot(Tau, r[indexX]*p_array, color = 'k') #label='power duration curve'
    plt.xlabel(r'duration $\tau$' + ' (years)', dict(size=10))
    plt.xlim(0.0,1.0)
    plt.ylim(0.0,4.0)
    plt.yticks([])
    plt.xticks([0,1])    
    plt.axhline(y = 1.0, color = 'k', linestyle = '--')
    plt.axhline(y = 1.0+eZ, xmax = Tau[indexZ], color = 'k', linestyle = '--')
    #indexZ2 = 6300 #just for visualazation purposes. needs to be computed exactly for general purposes
    diff_array = np.absolute(r[indexX]*Idown-fZ)
    indexZ2 = diff_array.argmin()
    plt.axhline(y = r[indexX]*p_array[indexZ2], xmin = Tau[indexZ2], color = 'k', linestyle = '--') #just for visualazation purposes. needs to be computed exactly for general purposes
    plt.text(posX,posY, r'point $Z$', dict(size=10), color='k')
    plt.fill_betweenx(np.append(r[indexX]*p_array, 0.0), 0.0, np.append(Tau, 1.0), where=(np.append(r[indexX]*p_array, 0.0) <= 1.0), facecolor='darkgreen', alpha=0.7, interpolate=True, edgecolor = 'None', label = 'direct use')
    plt.fill_between(Tau, 1.0, r[indexX]*p_array, where=(r[indexX]*p_array >= 1.0) & (r[indexX]*p_array <= 1.0+eZ), facecolor='olivedrab', alpha=0.5, interpolate=True, edgecolor = 'None', label = 'power-to-hydrogen')
    plt.fill_between(Tau, 1.0, 1.0+eZ, where=(r[indexX]*p_array >= 1.0+eZ), facecolor='olivedrab', alpha=0.5, interpolate=True, edgecolor = 'None')
    plt.fill_between(Tau, 1.0+eZ, r[indexX]*p_array, where=(r[indexX]*p_array >= 1.0+eZ), facecolor='wheat', alpha=0.3, interpolate=True, edgecolor = 'None', label = 'curtailed')
    plt.fill_betweenx(r[indexX]*p_array,Tau, 1.0, where=(r[indexX]*p_array <=1.0) & (r[indexX]*p_array >= r[indexX]*p_array[indexZ2] ), facecolor='olive', alpha=0.7, interpolate=True, edgecolor = 'None', label = 'hydrogen-to-power')
    plt.fill_between(Tau, r[indexX]*p_array[indexZ2], r[indexX]*p_array, where=(r[indexX]*p_array <= r[indexX]*p_array[indexZ2]), facecolor='dimgray', alpha=0.7, interpolate=True, edgecolor = 'None', label = 'fossil/firm')
    plt.legend(fontsize="10", bbox_to_anchor=(-0.05, 1.21), loc='lower left') #1.25, -0.25
    ax.xaxis.labelpad = -10



    ax=plt.subplot(2,3,1)
    rO = 1.0-fX[0]
    plt.plot(Tau, rO*p_array, color = 'k') #label='power duration curve'
    plt.xlabel(r'duration $\tau$' + ' (years)', dict(size=10))
    plt.ylabel(r'renewable power $\bar{p}$',dict(size=10))
    plt.xlim(0.0,1.0)
    plt.ylim(0.0,4.0)
    plt.yticks([0,1,2,3,4])
    plt.xticks([0,1])
    plt.axhline(y = 1.0, color = 'k', linestyle = '--')
    plt.text(posX,posY, r'point $O$', dict(size=10), color='k')
    plt.fill_betweenx(np.append(rO*p_array, 0.0), 0.0, np.append(Tau, 1.0), facecolor='darkgreen', alpha=0.7, interpolate=True, edgecolor = 'None', label = 'direct use')
    plt.fill_between(Tau, 1.0, rO*p_array, facecolor='dimgray', alpha=0.7, interpolate=True, edgecolor = 'None', label = 'fossil/firm')
    ax.xaxis.labelpad = -10
    
    ax=plt.subplot(2,3,6)
    plt.plot(Tau, r[indexX]*p_array, color = 'k') #label='power duration curve'
    plt.xlabel(r'duration $\tau$' + ' (years)', dict(size=10))
    plt.xlim(0.0,1.0)
    plt.ylim(0.0,4.0)
    plt.yticks([])
    plt.xticks([0,1])
    plt.axhline(y = 1.0, color = 'k', linestyle = '--')
    plt.text(posX,posY, r'point $X_0$', dict(size=10), color='k')
    plt.fill_betweenx(np.append(r[indexX]*p_array, 0.0), 0.0, np.append(Tau, 1.0), where=(np.append(r[indexX]*p_array, 0.0) <= 1.0), facecolor='darkgreen', alpha=0.7, interpolate=True, edgecolor = 'None', label = 'direct use')
    plt.fill_between(Tau, 1.0, r[indexX]*p_array, where=(r[indexX]*p_array >= 1.0), facecolor='olivedrab', alpha=0.5, interpolate=True, edgecolor = 'None', label = 'power-to-hydrogen')
    plt.fill_between(Tau, 1.0, r[indexX]*p_array, where=(r[indexX]*p_array <= 1.0), facecolor='olive', alpha=0.7, interpolate=True, edgecolor = 'None', label = 'hydrogen-to-power')
    ax.xaxis.labelpad = -10
    
    #plt.subplots_adjust(wspace=0.08,hspace=0.08)
    plt.subplots_adjust(wspace=0.12,hspace=0.25)
    plt.savefig(folder + '\\fvsr plus subplots ' + scenario, dpi=900,bbox_inches='tight')
    
    
    
    #find contours myself
    flevels_pos = np.linspace(0.0, 0.8, 33)
    hlevels = np.linspace(-1.0, 0.0, 21)
    flevels= np.concatenate((hlevels,flevels_pos))
    e_contour = np.zeros((T, len(flevels)))
    index_j = np.zeros((T, len(flevels)), dtype=int)
    for level in range(0,len(flevels)):
        for i in range(0,T):
            # calculate the difference array
            difference_array = np.absolute(f[i,:]-flevels[level])
            # find the index (j) of minimum element from the array
            index = difference_array.argmin()
            e_contour[i,level] = e[i,index]
            index_j[i, level] = int(index)
    #do the same to get the 'e_contour' for f_hydro such that it can be drawn later
    e_contour_hydro = np.zeros(T)
    for i in range(0,T):
        # calculate the difference array
        difference_array_hydro = np.absolute(f[i,:]-f_star_hydro_iea)
        index_hydro = difference_array_hydro.argmin()
        e_contour_hydro[i] = e[i,index_hydro]        
    #some flevels are such that all e values are negative, hence they fall out of phase space
    #and need to be removed
    lenflevelsO = len(flevels) #original size flevels
    lenflevels_posO = len(flevels_pos) 
    for leveltemp in range(0,len(flevels)):
        level = lenflevelsO - 1 - leveltemp #reverse order
        level_pos = lenflevels_posO - 1 - leveltemp #reverse order
        if np.max(e_contour[:,level])< 1e-9:
            print(level)
            flevels = np.delete(flevels, level)
            flevels_pos = np.delete(flevels_pos, level_pos)
            e_contour = np.delete(e_contour, level, axis=1)
            index_j = np.delete(index_j, level, axis=1)
    
    
    eX = r * p_array.max() - (1.0+u) #line X in e,r space
    
    #find optimum
    
    e_opt = np.zeros((len(CRs),len(flevels)))
    index_j_opt = np.zeros((len(CRs),len(flevels)), dtype=int)
    r_opt = np.zeros((len(CRs),len(flevels)))
    intersect_array = np.zeros((len(CRs),len(flevels)))
    index_opt = np.zeros((len(CRs),len(flevels)), dtype=int)
    r_0 = np.zeros((len(CRs),len(flevels)))
    index_0 = np.zeros((len(CRs),len(flevels)), dtype=int)
    for k in range(0,len(CRs)):    
        cost = np.zeros(T) #proxy for costs, taking into account cost ratio CR
        for level in range(0,len(flevels)): 
            #cost = CRs[k] * r + e_contour[:,level]
            cost = c_r[k] * r + c_e[k] * e_contour[:,level]
            #first find the intersection point between contour and line X
            intersect = e_contour[:,level].argmax()
            intersect_array[:,level] = intersect
            #then find the optimal e and r 
            index = cost[intersect:].argmin()
            index_opt[k, level] = int(index+intersect)
            r_opt[k, level] = r[index+intersect]    
            e_opt[k, level] = e_contour[index+intersect, level]
            index_j_opt[k, level] = index_j[index+intersect, level]
            #find intersect of contour for f and e=0 (r) axis; this is to eliminate negative (unphysical) optimal electrolyzer capacity; only relevant for negative flevels, not hlevels
            contour_intersect_temp = np.argmax(e_contour[intersect:,level]<0.0)
            if contour_intersect_temp == 0:
                print('yes')
                contour_intersect0 = T-1 #there is really no intersect, so make it max it can be
            else:
                contour_intersect0 = np.argmax(e_contour[intersect:,level]<0.0)+intersect-1
            index_0[k, level] = contour_intersect0
            r_0[k, level] = r[contour_intersect0] 
    
             
    
    #check contours
    plt.figure()
    plt.plot(e_contour)
    plt.scatter(index_0[0,:], 0*index_0[0,:])        
            
    #curve fitting: linear case for positive f, LATER:quadratic case for all f including negative f (hlevels)
    eps = np.zeros(len(CRs))
    rho = np.zeros(len(CRs))
    e_optH = e_opt*np.heaviside(e_opt,0.0) #because negative e is unphysical
    r_optH = r_opt*np.heaviside(r_0 - r_opt,1.0) + r_0*np.heaviside(r_opt - r_0,0.0) #r_optH does not work for hlevels
    for k in range(0,len(CRs)): 
    #only for positive f, no h
        popt, pcov = curve_fit(func, r_optH[k,len(hlevels):], e_optH[k,len(hlevels)::])
        eps[k] = popt[0]
        rho[k] = popt[1]
        
    #determine epsilon for X 
    popt, pcov = curve_fit(func, r, eX)
    epsX = popt[0]
    rhoX = popt[1] #note that one can derive that epsX = 1/rhoX    
    
    decarbonisation_threshold,curtailment = calc_grid_decarb_level(CRs,rho,r_optH,flevels)

    e_star = np.zeros(len(CRs))
    r_star = np.zeros(len(CRs))
    fig = plt.figure()
    ylimmax = 3.5
    plt.ylim(0.0,ylimmax)
    plt.xlim(0.0,2.5)
    for level in range(0,len(flevels_pos)):
        if level % 4 == 0:
            labelname = r'$f=%.1f$' % flevels_pos[level]
            plt.plot(r,e_contour[:,len(hlevels)+level], 'k')
            index_e_contour_on_X = np.argmax(e_contour[:,len(hlevels)+level])
            e_contour_X_draw = 0.05+e_contour[index_e_contour_on_X,len(hlevels)+level]
            if e_contour_X_draw < ylimmax-0.24:
                plt.text(-0.23+r[index_e_contour_on_X],e_contour_X_draw, labelname, dict(size=10), color='k')
        elif level % 4 == 2:
            plt.plot(r,e_contour[:,len(hlevels)+level], 'k--')
    if scenario == 'NSWPH':
        plt.plot(r,e_contour_hydro, color='darkorange')
    plt.plot(r,eX, color='navy') #X
    plt.axhline(y = 0.01, color = 'navy') #Y
    eX_draw = 0.05+eX[5800]
    if eX_draw < ylimmax-0.22:
        plt.text(-0.1+r[5800],eX_draw, r'$X$', dict(size=10), color='navy')   
    plt.text(2.3,-0.2, r'$Y$', dict(size=10), color='navy')
    plt.text(-0.1+rO,-0.2, r'$O$', dict(size=10), color='navy')
    for k in range(0,len(CRs)):
        labelname = 'CR'+ r'$=%.1f$' % CRs[k] + scenario_names[k]
        plt.scatter(r_optH[k,len(hlevels):],e_optH[k,len(hlevels):], label=labelname, alpha=alpha_values[k], color=defaultcolour)
        labelname = r'$\epsilon=%5.2f, \rho=%5.2f$' % (eps[k],rho[k])
        plt.plot(r_opt[k,len(hlevels):], func(r_opt[k,len(hlevels):], eps[k], rho[k]), color = colourfit, ls='dashed',label=labelname, alpha=alpha_values[k])             
    plt.xlabel(r'average renewable power generation $r$')
    plt.ylabel(r'electrolyser capacity $e$') 
    plt.fill_between(r, 0.0, eX, where=(r >= r[0]), facecolor='lightgray', alpha=0.3, interpolate=True, edgecolor = 'None')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [4,5,0,1,2,3]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper right', framealpha=1.0, fontsize = 8.5)
    plt.tight_layout()    
    plt.savefig(folder + '\\flevelsin e r space wo title' + scenario, dpi=600,bbox_inches='tight')
    for k in range(0,len(CRs)):
        plt.scatter(r_optH[k,len(hlevels)],e_optH[k,len(hlevels)], marker='*', s = 100, alpha=alpha_values[k], color=defaultcolour) #plot last one as a star
    plt.savefig(folder + '\\flevelsin e r space star wo title' + scenario, dpi=600,bbox_inches='tight')   
    plt.title(r'$s=%.2f$' %PVshare, dict(size=10), color='k')
    plt.savefig(folder + '\\flevelsin e r space ' + scenario, dpi=600,bbox_inches='tight')

    plt.ylim(0.0,12.0)
    plt.xlim(0.0,7.5)
    plt.savefig(folder + '\\flevelsin e r space wide ' + scenario, dpi=600,bbox_inches='tight') 
    e_star = e_optH[:,len(hlevels)]
    r_star = r_optH[:,len(hlevels)]
    
    plt.figure()
    for k in range(0,len(CRs)):
        labelname = 'CR'+ r'$=%.0f$' % CRs[k]
        plt.scatter(1.0-flevels,e_opt[k,:], label=labelname)
    plt.ylim(0.0,3.0)
    plt.xlim(0.5,1.0)
    plt.xlabel(r'$1-f$')
    plt.ylabel(r'$e_{opt}$') 
    plt.legend() 
    plt.tight_layout()  
    plt.savefig(folder + '\\eopt ' + scenario, dpi=600,bbox_inches='tight')
    plt.ylim(0.0,6.0)
    plt.xlim(0.0,1.0) 
    plt.savefig(folder + '\\eopt wide ' + scenario, dpi=600,bbox_inches='tight')
    
    PV_capacity = PVshare / (cap_fact_pv * eps) #solar capacity per GW of electrolysis
    wind_capacity = (1.-PVshare) / (cap_fact_wind * eps) #wind capacity per GW electrolysis
    print('Per GW of electrolysis the nr of GW of solar capacity required is:')
    print(PV_capacity)
    print('Per GW of electrolysis the nr of GW of wind capacity required is:')
    print(wind_capacity)
    
    index_optH = index_opt*np.heaviside(r_0 - r_opt,1) + index_0*np.heaviside(r_opt - r_0,0)
    index_optH = index_optH.astype('int')
    F = np.linspace(0.0, 1.0, 25)

    plt.figure(figsize=(4.8,4.8))
    P_dir = 1.0 - r_optH * Idown[index_optH]
    P_dem_plus_curt = r_optH+flevels - (1.0-(P_dir+flevels))*(1.0/eta_c-1.0) #from the total (r+f) subtract losses that went into P2H2P converion
    k=2
    plt.plot(1.0-flevels[:-1],r_optH[k,:-1]+flevels[:-1], '--', color=defaultcolour, alpha=alpha_values[k])
    plt.plot(1.0-flevels[:-1],P_dir[k,:-1]+flevels[:-1], '--', color=defaultcolour, alpha=alpha_values[k])
    plt.plot(1.0-flevels[:-1],P_dem_plus_curt[k,:-1], '--', color=defaultcolour, alpha=alpha_values[k])
    k=1
    plt.plot(1.0-flevels[:-1],r_optH[k,:-1]+flevels[:-1], ls = 'dotted', color=defaultcolour, alpha=alpha_values[k])
    plt.plot(1.0-flevels[:-1],P_dir[k,:-1]+flevels[:-1], ls = 'dotted', color=defaultcolour, alpha=alpha_values[k])
    plt.plot(1.0-flevels[:-1],P_dem_plus_curt[k,:-1], ls = 'dotted', color=defaultcolour, alpha=alpha_values[k])
    k=0
    plt.plot(1.0-flevels[:-1],r_optH[k,:-1]+flevels[:-1],'k', label='lost')
    plt.plot(1.0-flevels[:-1],P_dir[k,:-1]+flevels[:-1], 'k', label = 'direct renew')
    plt.plot(1.0-flevels[:-1],P_dem_plus_curt[k,:-1], 'k', label = 'curtailed')
    plt.plot(1-F,F, 'k', label = 'firm')
    plt.plot(1.0-F,1.0 + 0.0*F, 'k', label = 'P2H2P')
    plt.ylim(0.0,1.5)
    plt.xlim(0.0,1.0)
    plt.fill_between(1-F, 0.0, F, facecolor='dimgray', alpha=0.7, interpolate=True, edgecolor = 'None', label = 'fossil/firm')
    plt.fill_between(1.0-flevels[:-1], 1.0, P_dir[k,:-1]+flevels[:-1], facecolor='olive', alpha=0.5, interpolate=True, edgecolor = 'None', label = 'hydrogen-to-power')
    plt.fill_between(1.0-flevels[:-1], P_dem_plus_curt[k,:-1], r_optH[k,:-1]+flevels[:-1], facecolor='olivedrab', alpha=0.5, interpolate=True, edgecolor = 'White', label = 'power-to-hydrogen', hatch='.')
    plt.fill_between(1.0-flevels[:-1], 1.0, P_dem_plus_curt[k,:-1], facecolor='wheat', alpha=0.3, interpolate=True, edgecolor = 'None', label = 'curtailed')
    
    F = np.arange(flevels[-2], 1.0+flevels[1]-flevels[0], flevels[1]-flevels[0])
    plt.fill_between(np.concatenate((1.0-flevels[:-1],1.0-F)), np.concatenate((flevels[:-1],F)), np.concatenate((P_dir[k,:-1]+flevels[:-1],np.ones(len(F)))), facecolor='darkgreen', alpha=0.7, interpolate=True, edgecolor = 'None', label = 'direct use')
    plt.text(0.2,0.3, 'fossil/firm', dict(size=15), color='white')
    plt.text(0.45,0.65, 'direct renewable', dict(size=15), color='white')
    plt.text(0.82,1.1, '  lost in' + "\n" +' P2H2P', dict(size=12))
    plt.text(0.78,1.02, 'curtailed', dict(size=12))
    plt.text(0.89,0.93, 'H2P', dict(size=12))
    plt.xlabel('level of decarbonisation ' +r'$1-f$', size=14)
    plt.ylabel('energy ' + r'$E$', size=14)      
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()  
    plt.savefig(folder + '\\phasediagram  ' + scenario, dpi=600,bbox_inches='tight') 
    

    
    #now as a functio of r for power sector decarbonization scenario
    f_PowerFirst = flevels.copy() #no hydrogen export yet initially, so there f = flevels
    f_PowerFirst[:len(hlevels)]=0.0 #once power sector is decarbonized, f=0
    H_PowerFirst = (f_PowerFirst-flevels)/eta_H2e
    F = np.arange(flevels[-2], 1.0+flevels[1]-flevels[0], flevels[1]-flevels[0])
    P_dem_plus_curt = r_optH + f_PowerFirst - (1.0-(P_dir+f_PowerFirst))*(1.0/eta_c-1.0) - H_PowerFirst*(1.0/eta_e-1.0) #from the total (r+f) subtract losses that went into P2H2P conversion and into P2Hexp 
    plt.figure()
    plt.plot(r_optH[k,:-1],1.0+H_PowerFirst[:-1],'k')#, label='H exp')
    plt.plot(r_optH[k,:-1],r_optH[k,:-1]+f_PowerFirst[:-1],'k')# label='lost')
    plt.plot(r_optH[k,:-1],P_dir[k,:-1]+f_PowerFirst[:-1], 'k')# label = 'direct renew')
    plt.plot(r_optH[k,:-1],P_dem_plus_curt[k,:-1], 'k')#, label = 'curtailed')
    plt.plot(np.concatenate((r_optH[k,:-1],1-F)),np.concatenate((f_PowerFirst[:-1], F)), 'k')#, label = 'firm')
    plt.plot(np.concatenate((r_optH[k,:-1],1-F)),1.0 + 0.0*np.concatenate((f_PowerFirst[:-1], F)), 'k') #, label = 'P2H2P')
    plt.ylim(0.0,3.0)
    plt.xlim(0.0,3.0) 
    plt.fill_between(r_optH[k,:-1], P_dem_plus_curt[k,:-1], r_optH[k,:-1]+f_PowerFirst[:-1], facecolor='olivedrab', alpha=0.5, interpolate=True, edgecolor = 'white', label = 'lost in P2H and H2P', hatch='.')
    plt.fill_between(r_optH[k,:-1], 1.0+H_PowerFirst[:-1], P_dem_plus_curt[k,:-1], facecolor='wheat', alpha=0.3, interpolate=True, edgecolor = 'None', label = 'curtailed')
    plt.fill_between(r_optH[k,:-1], 1.0, 1.0+H_PowerFirst[:-1], facecolor='C0', alpha=0.3, interpolate=True, edgecolor = 'None', label = 'hydrogen export')
    plt.fill_between(r_optH[k,:-1], 1.0, P_dir[k,:-1]+f_PowerFirst[:-1], facecolor='olive', alpha=0.5, interpolate=True, edgecolor = 'None', label = 'hydrogen-to-power')
    plt.fill_between(np.concatenate((r_optH[k,:-1],1.0-F)), np.concatenate((f_PowerFirst[:-1],F)), np.concatenate((P_dir[k,:-1]+f_PowerFirst[:-1],np.ones(len(F)))), facecolor='darkgreen', alpha=0.7, interpolate=True, edgecolor = 'None', label = 'direct use')    
    plt.fill_between(np.concatenate((r_optH[k,:-1],1-F)), 0.0, np.concatenate((f_PowerFirst[:-1], F)), facecolor='dimgray', alpha=0.7, interpolate=True, edgecolor = 'None', label = 'fossil/firm')
    plt.xlabel(r'average renewable power generation $r$', size=12)
    plt.ylabel(r'energy $E$', size=12) 
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()  
    plt.legend(loc = 'upper left', fontsize=11)
    plt.title('power sector decarbonisation first', fontsize=12)
    plt.savefig(folder + '\\phasediagram vs r  - power first' + scenario, dpi=600,bbox_inches='tight')    
    plt.text(2.2,1.3, 'export H'+r'$_2$', dict(size=14), color='k')
    plt.savefig(folder + '\\phasediagram vs r  - power first -with text' + scenario, dpi=600,bbox_inches='tight')    


    #now as a functio of r for hydrogen export first scenario
    f_HFirst = 1 - P_dir[0,:]
    H_exp_HFirst = (f_HFirst-flevels)/eta_H2e
    F = np.arange(flevels[-2], 1.0+flevels[1]-flevels[0], flevels[1]-flevels[0])
    P_dem_plus_curt = r_optH + f_PowerFirst  - H_PowerFirst*(1.0/eta_e-1.0) #from the total (r+f) subtract losses that went into P2H2P conversion and into P2Hexp 
    plt.figure()
    plt.plot(r_optH[k,:-1],1.0+H_exp_HFirst[:-1],'k', label='H exp')
    plt.plot(r_optH[k,:-1],r_optH[k,:-1]+f_HFirst[:-1],'k', label='lost')
    plt.plot(r_optH[k,:-1],P_dir[k,:-1]+f_HFirst[:-1], 'k', label = 'direct renew')
    plt.plot(r_optH[k,:-1],P_dem_plus_curt[k,:-1], 'k', label = 'curtailed')
    plt.plot(np.concatenate((r_optH[k,:-1],1-F)),np.concatenate((f_HFirst[:-1], F)), 'k', label = 'firm')
    plt.plot(np.concatenate((r_optH[k,:-1],1-F)),1.0 + 0.0*np.concatenate((f_HFirst[:-1], F)), 'k', label = 'P2H2P')
    plt.ylim(0.0,3.0)
    plt.xlim(0.0,3.0)
    plt.fill_between(r_optH[k,:-1], 1.0, 1.0+H_exp_HFirst[:-1], facecolor='C0', alpha=0.3, interpolate=True, edgecolor = 'None', label = 'fossil/firm')
    plt.fill_between(np.concatenate((r_optH[k,:-1],1-F)), 0.0, np.concatenate((f_HFirst[:-1], F)), facecolor='dimgray', alpha=0.7, interpolate=True, edgecolor = 'None', label = 'fossil/firm')
    plt.fill_between(r_optH[k,:-1], 1.0, P_dir[k,:-1]+f_HFirst[:-1], facecolor='olive', alpha=0.5, interpolate=True, edgecolor = 'None', label = 'hydrogen-to-power')
    plt.fill_between(r_optH[k,:-1], P_dem_plus_curt[k,:-1], r_optH[k,:-1]+f_HFirst[:-1], facecolor='olivedrab', alpha=0.5, interpolate=True, edgecolor = 'white', label = 'power-to-hydrogen', hatch='.')
    plt.fill_between(r_optH[k,:-1], 1.0+H_exp_HFirst[:-1], P_dem_plus_curt[k,:-1], facecolor='wheat', alpha=0.3, interpolate=True, edgecolor = 'None', label = 'curtailed')
    plt.fill_between(np.concatenate((r_optH[k,:-1],1.0-F)), np.concatenate((f_HFirst[:-1],F)), np.concatenate((P_dir[k,:-1]+f_HFirst[:-1],np.ones(len(F)))), facecolor='darkgreen', alpha=0.7, interpolate=True, edgecolor = 'None', label = 'direct use')   
    plt.xlabel(r'average renewable power generation $r$', size=12)
    plt.ylabel(r'energy $E$', size=12) 
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('non-power sector decarbonisation first', fontsize=12)
    plt.tight_layout()  
    plt.savefig(folder + '\\phasediagram vs r  - H first' + scenario, dpi=600,bbox_inches='tight')    
    plt.text(2.2,1.3, 'export H'+r'$_2$', dict(size=14), color='k')
    plt.savefig(folder + '\\phasediagram vs r  - H first - with text' + scenario, dpi=600,bbox_inches='tight')
    plt.close('all')
    
    F_high = np.linspace(np.max(fY), 1.0, 25)
    P_ren = 1.0 - flevels
    P_renY = 1.0 - fY 
    P_renX = 1.0 - fX
    cost = np.zeros((len(CRs),len(flevels)))
    for k in range(0,len(CRs)):
        #cost of the electrolyzer varies en of renewables is fixed
        cost[k,:] = c_r[k] * r_optH[k,:] + c_e[k]* e_optH[k,:]
    costY = c_r[0] * r #Y is characterized by no electrolysis, so no costs for electrolyzers
    #eX = fX + r - 1.0
    costX = c_r[0] * r #no costs for electrolyzers, because c_e goes to zero for X. It's the trajectory for an abundance of electrolyzers 
    Cm = np.gradient(cost)[1]/np.gradient(P_ren)
    CmY = np.gradient(costY)/np.gradient(P_renY) 
    CmX = np.gradient(costX)/np.gradient(P_renX) 
    CmY[CmY < 1e-9] = np.nan #take out zeros as they are unreal, caused by numerical resolution
    CmX[CmX < 1e-9] = np.nan    
    fig, ax1 = plt.subplots()
    for k in range(0,len(CRs)):
        labelname = 'CR'+ r'$=%.0f, c_e=%.1f$' % (CRs[k], c_e[k])#CRs[0]/CRs[k])
        plt.plot(flevels[:-2], Cm[k,:-2], label=labelname)
    plt.plot(fX, CmX, label='X, '+'CR'+ r'$=%.0f, c_e=0$' % c_r[0])
    plt.plot(fY, CmY, label='Y, '+'CR'+ r'$=%.0f, c_e \rightarrow \infty$' % c_r[0])
    plt.legend()  
    plt.xlim(0.0,1.0)
    ymax = 25
    plt.ylim(0.0,ymax) 
    plt.plot(F_high, c_r[0]*np.ones(len(F_high)), 'k')
    plt.xlabel(r'$f$')
    plt.ylabel(r'$C_m (\$/W)$')   
    ax2 = ax1.twinx()
    ax2.set_ylim(0.0,ymax*50.0/3.0)  #3 dollar/W corresponds with 50 dollar per megawatt, so
    ax2.set_ylabel(r'$C_m (\$/MWh)$') 
    plt.tight_layout()
    plt.savefig(folder + '\\cost', dpi=300,bbox_inches='tight') 
    plt.close('all')
    


    plt.figure(figsize=(4.8,4.8))
    costdif = np.zeros((len(CRs),len(flevels)))
    costabs = np.zeros((len(CRs),len(flevels))) #needed later
    #first compute the cost difference for scenario X (cheap electrolysis)
    costabsX = c_r[0] * r
    costdifX = costabsX / (1 - fX)
    plt.plot(1 - fX,costdifX, color ='navy', label ='CR'+ r'$=\infty$' + ' (X: zero cost electrolysis)' )
    plt.plot(1 - fX,alt_cost_factor*costdifX, color ='navy', ls= 'dashed')
    #now for the scenarios
    for k in range(0,len(CRs)):
        labelname = 'CR'+ r'$=%.0f$' % CRs[k] + scenario_names[k]
        costabs[k,:] = c_r[k] * r_optH[k,:] + c_e[k] * e_optH[k,:]
        costdif[k,:] = ( c_r[k] * r_optH[k,:] + c_e[k] * e_optH[k,:] ) / (1.0-flevels)
        costdif_high =  c_r[k] * np.ones(len(F_high))#(1.0-F_high) /(1.0-F_high) #revisit this line
        f_for_plot = np.concatenate((flevels,F_high))
        costdif_for_plot = np.concatenate((costdif[k,:],costdif_high))
        plt.plot(1.0 - f_for_plot, costdif_for_plot, color=defaultcolour, alpha = alpha_values[k], label=labelname)
        if k<2: #if alternative scenario exist (i.e. reference and future)
            plt.plot(1.0 - f_for_plot, alt_cost_factor*costdif_for_plot, color=defaultcolour, alpha = alpha_values[k], ls= 'dashed')
        #plt.text(1.01,costdif[k,len(hlevels)]-0.01, r'$%5.2f \ €/W$' % (costdif[k,len(hlevels)]), dict(size=10), color='k')    
    plt.text(0.05,costdif_for_plot[-1]-0.4, 'main cost scenrios', dict(size=13), color='k')
    plt.text(0.05,alt_cost_factor*costdif_for_plot[-1]-0.4, 'low renewables cost scenarios', dict(size=13), color='k')
    plt.ylim(0.0,9)
    plt.xlim(0.0,1.0)
    plt.xlabel(r'level of decarbonisation $1-f$', fontsize=13)
  
    plt.ylabel(r'decarbonisation cost $-\Delta C / \Delta f \ \ \left(€/W\right)$', fontsize=13)
    plt.xticks(fontsize=14)
    plt.yticks([0,2,4,6,8],fontsize=14)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3,1,2,0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper left', framealpha=1.0)
    plt.tight_layout()
    plt.savefig(folder + '\\deltacost ' + scenario, dpi=600,bbox_inches='tight')
    plt.close('all')
    
    cost_star = costabs[:,len(hlevels)]
    
    
    f_star_hydro=np.zeros(len(CRs))
    e_star_hydro=np.zeros(len(CRs))
    r_star_hydro=np.zeros(len(CRs))
    cost_star_hydro=np.zeros(len(CRs))
    if method_hydro == 'river':
        for k in range(0,len(CRs)):
            r_hydro = r_optH[k,:].copy()
            e_hydro = e_optH[k,:].copy()
            f_hydro = flevels.copy()
            index_hydro = np.zeros(len(r_hydro))
            #compute 2 f_hydro's and equate them to find the right value for f
            #one follows from the constraint on hydropower capacity (f_hydro2)
            #the other one follows from optimal combination of e and r (f_hydro)
            f_hydro2 = np.zeros(len(r_hydro))
            for m in range(0,len(r_hydro)):
                diff_array_hydro = np.absolute(r_hydro[m]*p_array - cap_hydro)
                index_hydro[m] = diff_array_hydro.argmin()
                f_hydro2[m] = r_hydro[m] * Idown[int(index_hydro[m])]    
            f_hydro_int = np.linspace(min(f_hydro), max(f_hydro), 1001)
            r_hydro_int = np.interp(f_hydro_int, f_hydro, r_hydro)
            e_hydro_int = np.interp(f_hydro_int, f_hydro, e_hydro)
            f_hydro2_int = np.interp(f_hydro_int, f_hydro, f_hydro2)
            diff_f_hydros = np.absolute(f_hydro_int - f_hydro2_int)
            index_diff_f_hydros = diff_f_hydros.argmin()
            
            f_star_hydro[k] = f_hydro2_int[int(index_diff_f_hydros)]
            e_star_hydro[k] = e_hydro_int[int(index_diff_f_hydros)]
            r_star_hydro[k] = r_hydro_int[int(index_diff_f_hydros)]
            cost_star_hydro[k] = c_e[k] * e_star_hydro[k] + c_r[k] * r_star_hydro[k]
        
    else: #method_hydro = 'IEA
        f_star_hydro = f_star_hydro_iea * np.ones(len(CRs)) 
        print('f_star_hydro is ')
        print(f_star_hydro)
        #interpolate to obtain the corresponding values for e and r
        for k in range(0,len(CRs)):
            r_hydro = r_optH[k,:].copy()
            e_hydro = e_optH[k,:].copy()
            e_star_hydro[k] = np.interp(f_star_hydro[k], flevels, e_hydro)
            r_star_hydro[k] = np.interp(f_star_hydro[k], flevels, r_hydro)
            cost_star_hydro[k] = c_e[k] * e_star_hydro[k] + c_r[k] * r_star_hydro[k]
        print('e_star_hydro is ')
        print(e_star_hydro)
        print('rf_star_hydro is ')
        print(r_star_hydro)
        
           

    
    #determine the electrolyser capacity and renewable production for full decarbonisation for scenario X
    fXabs = np.absolute(fX)
    index_fXabs = fXabs.argmin()
    e_starX = eX[index_fXabs]
    r_starX = r[index_fXabs]



    C_H2_sa = np.zeros((len(CRs), len(b)))
    b_min = np.zeros(len(CRs))
    b_min_arg = []
    C_H2_sa_min = np.zeros(len(CRs))

    for k in range(0,len(c_r)): 
        C_H2_sa[k,:] = c_e[k] * (b + CRs[k]) / ( eta_e * (1.0 - Iup)) 
        b_min_arg.append(C_H2_sa[k,:].argmin())
        b_min[k] = b[b_min_arg[-1]]
        C_H2_sa_min[k] = C_H2_sa[k, b_min_arg[-1]]
    fig, ax1 = plt.subplots() 
    for k in range(0,len(CRs)):
        labelname = "CR" + r'$=%.0f$' % CRs[k] + scenario_names[k]

        plt.plot(b, C_H2_sa[k,:], label=labelname, color = defaultcolour, alpha = alpha_values[k])
        plt.scatter(b_min[k],C_H2_sa_min[k], color = defaultcolour, alpha = alpha_values[k])    

    labelname = r'$X$' 

    C_H2_sa_inf = c_r[0] / ( eta_e * (1.0 - Iup)) #since CRs[0]=c_r

    plt.plot(b, C_H2_sa_inf, label=labelname, color='navy')
    plt.scatter(b[C_H2_sa_inf.argmin()], C_H2_sa_inf.min(), color='navy')
    plt.xlabel(r'relative electrolyser size $e/r$')
    plt.ylabel(r'hydrogen cost $C^{sa}_{H} / H^{sa} \  (€/W_H)$')     
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2,0,1,3]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper right', framealpha=1.0)
    ylim_max=25.0
    plt.ylim(0.0,ylim_max)
    plt.xlim(xmin=0.0) 
    plt.tight_layout()  
    ax2 = ax1.twinx() 
    P_conv = 1.0 - Iup 
    ax2.plot(b, 100.0*P_conv,color= 'darkorange')
    ylim_max2 = 160
    ax2.set_ylim(0.0,ylim_max2) 
    ax2.set_ylabel(r'power converted $E_e/r\  (\%)$', color='darkorange') 

    ax2.tick_params(axis='y', colors='darkorange')
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.yaxis.label.set_color('darkorange')
    ax2.spines['right'].set_color('darkorange')
    for k in range(0,len(CRs)):
        plt.axvline(x=b_min[k], ymin=C_H2_sa_min[k]/ylim_max, ymax=P_conv[b_min_arg[k]]*100.0/ylim_max2, ls='dashed', color='gray')
        ax2.scatter(b_min[k], 100.0*P_conv[b_min_arg[k]], color='darkorange')
    plt.axvline(x=b[C_H2_sa_inf.argmin()], ymin=C_H2_sa_inf.min()/ylim_max, ymax=P_conv[C_H2_sa_inf.argmin()]*100.0/ylim_max2, ls='dashed', color='gray')
    ax2.scatter(b[C_H2_sa_inf.argmin()], 100.0*P_conv[C_H2_sa_inf.argmin()], color='darkorange')
    plt.savefig(folder + '\\costH2SA ' + scenario, dpi=600, bbox_inches='tight') 
    plt.close('all')
    
    H_sa_ratio_to_r = eta_e * (1.0 - Iup[b_min_arg])
    utilization = H_sa_ratio_to_r / ( eta_e * b_min )
    
    plt.figure() 
    P_conv = 1.0 - Iup  
    plt.plot(b, P_conv)
    plt.xlabel(r'$e/r$')
    plt.ylabel(r'$P_{conv} (\%)$')  
    plt.savefig(folder + '\\PconvSA ' + scenario, dpi=300,bbox_inches='tight') 
    

    #cost of decarbonizing the power system only for reference scenaario
    Vp = costdif[0, len(hlevels)] #at index len(hlevels): f=0
    VH2 = 0.0*H_PowerFirst[:len(hlevels)]
    Cint = 0.0*H_PowerFirst[:len(hlevels)]
    for i in range(0,len(H_PowerFirst[:len(hlevels)])):
        VH2[i] = C_H2_sa_min[0] * H_PowerFirst[i]
        #now compute cost of integrated system
        Cint[i] = costabs[0,i] 
    Synergy = Vp + VH2 - Cint
    Synergy_rel = Synergy / VH2
    plt.figure()
    plt.plot(H_PowerFirst[:len(hlevels)], 100*Synergy_rel, label=labelname, color=defaultcolour)
    plt.xlabel(r'Exported hydrogen $H$', fontsize=12)
    plt.ylabel(r'Relative synergy $S/V_H (\%)$', fontsize=12)
    plt.xlim(xmin=0.0) 
    plt.ylim(ymin=0.0) 
    plt.savefig(folder + '\\Relative synergy ' + scenario, dpi=600,bbox_inches='tight')

    fig, ax1 = plt.subplots()
    plt.plot(H_PowerFirst[:len(hlevels)], Synergy / H_PowerFirst[:len(hlevels)], label=labelname, color=defaultcolour)
    plt.xlabel(r'hydrogen produced $H/D$',  fontsize=12)
    plt.ylabel(r'relative synergy $S/H \ (€/W_H)$', fontsize=12)  
    plt.xlim(xmin=0.0)
    plt.yticks([0,0.5,1,1.5,2])
    plt.xticks([0,0.5,1,1.5,2])
    max_syn_rel = 0.2
    plt.ylim(0.0, max_syn_rel*C_H2_sa_min[0]) 
    ax2 = ax1.twinx()
    ax2.plot(H_PowerFirst[:len(hlevels)], 100*Synergy_rel, label=labelname, color=defaultcolour)
    ax2.set_ylabel(r'relative synergy $S/V_H \ (\%)$', fontsize=12)  
    ax2.set_ylim(0.0, 100*max_syn_rel)
    ax2.set_yticks([0,4,8,12,16,20])
    plt.savefig(folder + '\\Relative synergy double axes ' + scenario, dpi=600,bbox_inches='tight')


    #cost of decarbonizing the power system only for reference scenaario
    Vp = costdif[0, len(hlevels)] #at index len(hlevels): f=0
    H_indus_size = np.array([0.25, 0.5, 1.0]) #magnitude of non-power H2 in terms of size power system

    H_amount =  H_indus_size
    VH2 = 0.0*H_indus_size
    Cint = 0.0*H_indus_size
    for i in range(0,len(H_indus_size)):
        f_amount = -eta_H2e * H_amount[i]
        #since I work with discrete values for the flevels/hlevels I will adjust H_amount to value closest to it
        difference_array = np.absolute(flevels-f_amount)
        index = difference_array.argmin()
        H_amount[i] = - flevels[index] / eta_H2e
        VH2[i] = C_H2_sa_min[0] * H_amount[i]
        #now compute cost of integrated system
        Cint[i] = costabs[0,index] #factor half: going from negative f to H!!
    Synergy = Vp + VH2 - Cint
    Synergy_rel = Synergy / (Vp+VH2)
    
    Synergy_max = Vp - c_e * (( b_min + CRs ) / (utilization * b_min) - 1.0) #to be completed
    
    barWidth = 0.25
    #plt.subplots(1,2, sharey=False)
    plt.subplot(1,2,1)
    # Set position of bar on X axis
    br1 = np.arange(len(Synergy))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    # Make the plot
    plt.bar(br1, Vp + VH2, color ='navy', width = barWidth,edgecolor ='k', label ='Stand-alone production', alpha=alpha_values[0])
    plt.bar(br2, Cint, color ='navy', width = barWidth, edgecolor ='k', label ='Integrated production', alpha=alpha_values[1])
    #to be calculated:
    plt.bar(br3, 0*Cint, color ='navy', width = barWidth, edgecolor ='k', label ='Incl. flexible use', alpha=alpha_values[2])
    plt.ylabel(r'Cost $C \ (€/W)$')
    plt.xticks([i + barWidth for i in range(len(Synergy))],
            [r'$H=D/4$', r'$H=D/2$', r'$H=E$'])
    plt.legend(fontsize=8, loc= 'upper center')
    plt.ylim(ymax=16)
    plt.subplot(2,2,2)

    br = [r'$H=D/4$', r'$H=D/2$', r'$H=D$']
    plt.bar(br, Synergy, color ='darkgray', width = barWidth,edgecolor ='k', label ='Synergy', alpha=alpha_values[0])
    plt.ylabel(r'Synergy $S \ (€/W)$')
    plt.subplot(2,2,4)
    #To be calculated
    plt.bar(br, 0*Synergy, color ='darkgray', width = barWidth,edgecolor ='k', alpha=alpha_values[0])
    plt.ylabel(r'Value flexible use $F \ (€/W)$')
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(folder + '\\Synergy ' + scenario, dpi=600,bbox_inches='tight') 
    
    
    popt,pcov = curve_fit(lambda x, a: func_q(x, a, b_min[0]), r_optH[0,:], e_optH[0,:])
    g = (e_optH[0,e_optH[0,:] >0.0] + 1.0) / r_optH[0,e_optH[0,:] >0.0]
    
    plt.figure()
    plt.plot(r_optH[0,e_optH[0,:] >0.0], func_g(r_optH[0,e_optH[0,:] >0.0], popt, b_min[0]), color = colourfit, ls='dashed')
    plt.scatter(r_optH[0,e_optH[0,:] >0.0],g)
    plt.plot(r_optH[0,e_optH[0,:] >0.0], b_min[0]*np.ones(len(r_optH[0,e_optH[0,:] >0.0])), 'k--')
    plt.xlabel(r'$r$')
    plt.ylabel(r'$e$') 
    plt.savefig(folder + '\\roptH', dpi=300,bbox_inches='tight') 
    
    plt.figure()
    r_inverse = np.linspace(0.0001, 1.0/np.min(r_optH[0,e_optH[0,:] >0.0]), 20)
    plt.plot(r_inverse, func_g(1.0/r_inverse, popt,b_min[0]), color = colourfit, ls='dashed')
    plt.plot(r_inverse, b_min[0]*np.ones(len(r_inverse)), 'k--')
    plt.scatter(1.0/r_optH[0,e_optH[0,:] >0.0],g)
    plt.xlabel(r'$1/r$')
    plt.ylabel(r'$e$') 
    plt.xlim(0.0, np.max(r_inverse))
    #plt.legend() 
    plt.tight_layout()  
    plt.savefig(folder + '\\quadratic fit ' + scenario, dpi=300,bbox_inches='tight') 
    
    plt.figure()
    for level in range(0,len(hlevels)):
        if level % 5 == 0:
        #if hlevel[level] == 0 or hlevel[level] = -:
            #labelname = r'$f=%.2f$' % hlevels[level]
            labelname = r'$H=%.1f$' % (-hlevels[level]/eta_H2e+0.000001)
            plt.plot(r,e_contour[:,level], color='k')
            #index_e_contour_on_X = np.argmax(e_contour[:,len(hlevels)+level])
            index_e_contour_on_X = np.argmax(e_contour[:,level])
            if hlevels[level] > -0.51:
                plt.text(-0.45+r[index_e_contour_on_X],0.05+e_contour[index_e_contour_on_X,level], labelname, dict(size=10), color='k')

    plt.plot(r,eX, color='navy') #label='X'
    k=0
    labelname = 'CR'+ r'$=%.0f$' % CRs[k]
    plt.scatter(r_optH[k,:],e_optH[k,:], color=defaultcolour)
    plt.plot(r, b_min[k]*r - 1.0, 'k--', label=r'$e = \epsilon^{sa} r - 1$')
    r_prime = np.linspace(0.0, 8, 100)
    plt.plot(r_prime, b_min[k]*r_prime, color='k', ls='dotted', label=r'$e = \epsilon^{sa} r$')
    plt.plot(r_prime,0.0*r_prime, color='navy')
                
    plt.ylim(0.0,7.0)
    plt.xlim(0.0,4.0)
    plt.fill_between(r, 0.0, eX, where=(r >= r[0]), facecolor='lightgray', alpha=0.3, interpolate=True, edgecolor = 'None')
    plt.xlabel(r'$r$')
    plt.ylabel(r'$e$') 
    plt.legend() 
    plt.tight_layout()      
    plt.savefig(folder + '\\fandhlevels ' + scenario, dpi=600,bbox_inches='tight')
    plt.plot(r_opt[0,:], func_q(r_opt[0,:], popt, b_min[0]), color = colourfit, ls='dashed')  
    plt.savefig(folder + '\\fandhlevels with fit' + scenario, dpi=600,bbox_inches='tight')
     
    r_optH0 = r_optH[0,len(hlevels)] 
    e_optH0 = e_optH[0,len(hlevels)] 
    fig, ax1 = plt.subplots()
    left, bottom, width, height = [0.25, 0.67, 0.6, 0.18]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax1.plot(Tau, r_optH0*p_array, label='power duration curve', color = 'k', lw=1.0) 
    ax1.set_xlabel('duration ' + r'$\tau$' + ' (years)', dict(size=12))
    ax1.set_ylabel('renewable power generation ' + r'$\bar{p}$', dict(size=12))
    ax1.set_xlim(0.0,1.0)
    ax1.set_ylim(0.0,4.0)
    ax1.axhline(y = 1.0, color = 'k', lw=1.0)
    ax1.axhline(y = r_optH0, color = 'k', linestyle = '--', lw=1.0)
    ax1.axhline(y = 1.0 + e_optH0, color = 'k', linestyle = 'dotted', lw=1.0)
    ax1.text(0.9, 1.05, r'$d=1$', dict(size=10))
    ax1.text(0.9, 1.0 + e_optH0 + 0.05, r'$1+e$', dict(size=10))
    ax1.text(0.9, r_optH0 + 0.05, r'$r$', dict(size=10))
    ax1.fill_between(Tau, 1.0, r_optH0*p_array, where=(r_optH0*p_array <= 1.0), facecolor=colourfit, alpha=0.7, interpolate=True, edgecolor = 'None')
    ax1.fill_between(Tau, 1.0, r_optH0*p_array, where=(r_optH0*p_array <= 1.0 + e_optH0) & (r_optH0*p_array >= 1.0), facecolor='olivedrab', alpha=0.5, interpolate=True, edgecolor = 'None')
    ax1.fill_between(Tau, 1.0, 1.0 + e_optH0, where=(r_optH0*p_array >= 1.0 + e_optH0), facecolor='olivedrab', alpha=0.5, interpolate=True, edgecolor = 'None')

    ax2.plot(Tau, r_optH0*p_unsorted, label='power profile', lw=0.1, color = 'k')
    ax2.set_xlabel('time ' + r'$t$' + ' (years)')
    ax2.set_ylabel(r'$p$')
    ax2.set_xlim(0.0,1.0)
    ax2.set_ylim(0.0, 4.0)

    plt.savefig(folder + '\\power duration curve filled ' + scenario, dpi=600,bbox_inches='tight')
    ax1.plot(Tau, r_optH0*p_array_no_bat, label='power duration curve', color = 'k', lw=2) 
    ax1.plot(Tau, r_optH0*p_array, label='power duration curve', color = 'darkorange', lw=2)
    ax2.plot(Tau, r_optH0*p_unsorted_no_bat, label='power profile', lw=0.1, color = 'k')
    ax2.plot(Tau, r_optH0*p_unsorted, label='power profile', lw=0.1, color = 'darkorange')
    plt.savefig(folder + '\\power duration curve filled battery' + scenario, dpi=600,bbox_inches='tight')
    batname =  'battery level =%.4f' % bat_ref_level
    plt.title(batname)
    plt.savefig(folder + '\\power duration curve filled battery2' + scenario, dpi=600,bbox_inches='tight')

    fig, ax1 = plt.subplots()
    ax1.plot(Tau, r_optH0*p_array, label='power duration curve', color = 'k', lw=1.0) 
    ax1.set_xlabel('duration ' + r'$\tau$' + ' (years)', dict(size=12))
    ax1.set_ylabel('renewable power generation ' + r'$\bar{p}$', dict(size=12))
    ax1.set_xlim(0.0,1.0)
    ax1.set_ylim(0.0,4.0)
    ax1.axhline(y = 1.0, color = 'k', lw=1.0)
    ax1.axhline(y = r_optH0, color = 'k', linestyle = '--', lw=1.0)
    ax1.axhline(y = 1.0 + e_optH0, color = 'k', linestyle = 'dotted', lw=1.0)
    ax1.text(0.9, 1.05, r'$d=1$', dict(size=10))
    ax1.text(0.9, 1.0 + e_optH0 + 0.05, r'$1+e$', dict(size=10))
    ax1.text(0.9, r_optH0 + 0.05, r'$r$', dict(size=10))
    ax1.fill_between(Tau, 1.0, r_optH0*p_array, where=(r_optH0*p_array <= 1.0), facecolor=colourfit, alpha=0.7, interpolate=True, edgecolor = 'None')
    ax1.fill_between(Tau, 1.0, r_optH0*p_array, where=(r_optH0*p_array <= 1.0 + e_optH0) & (r_optH0*p_array >= 1.0), facecolor='olivedrab', alpha=0.5, interpolate=True, edgecolor = 'None')
    ax1.fill_between(Tau, 1.0, 1.0 + e_optH0, where=(r_optH0*p_array >= 1.0 + e_optH0), facecolor='olivedrab', alpha=0.5, interpolate=True, edgecolor = 'None')
    ax1.plot(Tau, r_optH0*p_array_no_bat, label='power duration curve', color = 'k', lw=2) 
    ax1.plot(Tau, r_optH0*p_array, label='power duration curve', color = 'darkorange', lw=2)
    plt.savefig(folder + '\\power duration curve filled battery without inset' + scenario, dpi=600,bbox_inches='tight')
    batname =  'battery level =%.4f' % bat_ref_level
    plt.title(batname)
    plt.savefig(folder + '\\power duration curve filled battery2 without inset' + scenario, dpi=600,bbox_inches='tight')

    
    plt.figure(figsize=(6.4, 8))
    plt.subplot(8,1,1)
    plt.plot(Tau*365, r_optH0*p_unsorted_no_bat, label='power duration curve', color = 'k', lw=0.8)
    plt.plot(Tau*365, r_optH0*p_unsorted, label='power duration curve', color = 'darkorange', lw=0.8)
    plt.xlim(0., 7)
    plt.ylim(0.0, 4.0)
    plt.ylabel(r'$p$')
    plt.yticks([0,1,2,3,4], ['0', '', '2','',''])
    plt.subplot(8,1,2)
    plt.plot(Tau*365, Ebat/4, label='power duration curve', color = 'darkgray', lw=0.8)
    plt.ylabel(r'$E_b/E_{b,\mathrm{c}}$')
    plt.xlim(0., 7)
    plt.ylim(-0.1, 1.1)
    
    plt.subplot(8,1,3)
    plt.plot(Tau*365, r_optH0*p_unsorted_no_bat, label='power duration curve', color = 'k', lw=0.8)
    plt.plot(Tau*365, r_optH0*p_unsorted, label='power duration curve', color = 'darkorange', lw=0.8)
    snapshot = 31+28+31
    plt.xlim(snapshot, snapshot + 7) 
    plt.ylim(0.0, 4.0)
    plt.ylabel(r'$p$')
    plt.yticks([0,1,2,3,4], ['0', '', '2','',''])
    plt.subplot(8,1,4)
    plt.plot(Tau*365, Ebat/4, label='power duration curve', color = 'darkgray', lw=0.8)
    plt.ylabel(r'$E_b/E_{b,\mathrm{c}}$')
    plt.xlim(snapshot, snapshot + 7) 
    plt.ylim(-0.1, 1.1)
    
    plt.subplot(8,1,5)
    snapshot = snapshot + 30+31+30
    plt.plot(Tau*365, r_optH0*p_unsorted_no_bat, label='power duration curve', color = 'k', lw=0.8)
    plt.plot(Tau*365, r_optH0*p_unsorted, label='power duration curve', color = 'darkorange', lw=0.8)
    plt.xlim(snapshot, snapshot + 7)
    plt.ylim(0.0, 4.0)
    plt.ylabel(r'$p$')
    plt.yticks([0,1,2,3,4], ['0', '', '2','',''])
    plt.subplot(8,1,6)
    plt.plot(Tau*365, Ebat/4, label='power duration curve', color = 'darkgray', lw=0.8)
    plt.ylabel(r'$E_b/E_{b,\mathrm{c}}$')
    plt.xlim(snapshot, snapshot + 7)
    plt.ylim(-0.1, 1.1)
    
    plt.subplot(8,1,7)
    plt.plot(Tau*365, r_optH0*p_unsorted_no_bat, label='power duration curve', color = 'k', lw=0.8)
    plt.plot(Tau*365, r_optH0*p_unsorted, label='power duration curve', color = 'darkorange', lw=0.8)
    snapshot = snapshot + 31+31+30
    plt.xlim(snapshot, snapshot + 7)
    plt.ylim(0.0, 4.0)
    plt.ylabel(r'$p$')
    plt.yticks([0,1,2,3,4], ['0', '', '2','',''])
    plt.subplot(8,1,8)
    plt.plot(Tau*365, Ebat/4, label='power duration curve', color = 'darkgray', lw=0.8)
    plt.ylabel(r'$E_b/E_{b,\mathrm{c}}$')
    plt.xlim(snapshot, snapshot + 7)
    plt.ylim(-0.1, 1.1)
    plt.subplots_adjust(wspace=0.12,hspace=0.5)
    plt.xlabel('time '+r'$t \ (\mathrm{days})$')
    plt.savefig(folder + '\\battery profiles'  +scenario, dpi=600,bbox_inches='tight')
    
    
    fig, ax1 = plt.subplots()
    left, bottom, width, height = [0.25, 0.67, 0.6, 0.18]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax1.plot(Tau, r_optH0*p_array, label='power duration curve', color = 'k', lw=2) 
    ax1.set_xlabel(r'$\tau$' + ' (years)', dict(size=14))
    ax1.set_ylabel(r'$\bar{p}$', dict(size=14))
    ax1.set_xlim(0.0,1.0)
    ax1.set_ylim(0.0,4.0)
    ax1.axhline(y = 1.0, color = 'k')
    ax1.axhline(y = r_optH0, color = 'k', linestyle = '--')
    ax1.axhline(y = 1.0 + u + e_optH0, color = 'k', linestyle = 'dotted')
    ax1.axhline(y = 1.0 + u, color = 'k', linestyle = 'dashdot')
    #ax1.text(0.9, 1.05, r'$d$', dict(size=10))
    ax1.text(0.8, 1.05+u, r'$1+u$', dict(size=10))
    ax1.text(0.8, 1.0 + e_optH0 + u + 0.05, r'$1+u+e$', dict(size=10))
    ax1.text(0.8, r_optH0 + 0.05, r'$r$', dict(size=10))
    ax1.fill_between(Tau, 1.0, r_optH0*p_array, where=(r_optH0*p_array <= 1.0 ), facecolor='C0', alpha=0.3, interpolate=True, edgecolor = 'None')
    ax1.fill_between(Tau, 1.0+u, r_optH0*p_array, where=((r_optH0*p_array <= 1.0 + u + e_optH0) * (r_optH0*p_array >= 1.0+u)), facecolor='C0', alpha=0.3, interpolate=True, edgecolor = 'None')
    ax1.fill_between(Tau, 1.0+u, 1.0 + u + e_optH0, where=(r_optH0*p_array >= 1.0 + u + e_optH0), facecolor='C0', alpha=0.3, interpolate=True, edgecolor = 'None')
    ax1.fill_betweenx(r_optH0*p_array, 0, Tau, where=((r_optH0*p_array>=1.0)*(r_optH0*p_array<=1.0+u)),facecolor='C1', alpha=0.3, interpolate=True, edgecolor = 'None')
    #ax1.set_tight_layout()
    ax2.plot(Tau, r_optH0*p_unsorted, label='power profile', lw=0.1)
    ax2.set_xlabel(r'$t$' + ' (years)')
    ax2.set_ylabel(r'$P$')
    ax2.set_xlim(0.0,1.0)
    ax2.set_ylim(0.0, 4.0)
    plt.savefig(folder + '\\power duration curve filled incl u' + scenario, dpi=600,bbox_inches='tight')
 
    
    fig, ax1 = plt.subplots()
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax1.plot(range(10), color='red')
    ax2.plot(range(6)[::-1], color='green')
    ax1.set_xlabel(r'$\tau$' + ' (years)')
    
    #plots for variation in demand
    delta_d = d_unsorted - 1.0
    p_min_delta_d_unsorted = p_unsorted - delta_d
    p_min_delta_d_array = np.sort(p_min_delta_d_unsorted)[::-1]
    fig, ax1 = plt.subplots()
    left, bottom, width, height = [0.25, 0.67, 0.58, 0.18]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax1.plot(Tau, r_optH0*p_array, label='power duration curve', color = 'k', lw=1.0) 
    ax1.set_xlabel('duration ' + r'$\tau$' + ' (years)', dict(size=12))
    ax1.set_ylabel('net renewable power generation ' + r'$\overline{p-\delta d}$', dict(size=12))
    ax1.set_xlim(0.0,1.0)
    ax1.set_ylim(-0.333,4.0)
    ax1.axhline(y = 1.0, color = 'k', lw=1.0)
    ax1.axhline(y = r_optH0, color = 'k', linestyle = '--', lw=1.0)
    ax1.axhline(y = 1.0 + e_optH0, color = 'k', linestyle = 'dotted', lw=1.0)
    ax1.text(0.9, 1.05, r'$\langle d\rangle=1$', dict(size=10))
    ax1.text(0.9, 1.0 + e_optH0 + 0.05, r'$1+e$', dict(size=10))
    ax1.text(0.9, r_optH0 + 0.05, r'$r$', dict(size=10))
    ax1.fill_between(Tau, 1.0, r_optH0*p_array, where=(r_optH0*p_array <= 1.0), facecolor=colourfit, alpha=0.7, interpolate=True, edgecolor = 'None')
    ax1.fill_between(Tau, 1.0, r_optH0*p_array, where=(r_optH0*p_array <= 1.0 + e_optH0) & (r_optH0*p_array >= 1.0), facecolor='olivedrab', alpha=0.5, interpolate=True, edgecolor = 'None')
    ax1.fill_between(Tau, 1.0, 1.0 + e_optH0, where=(r_optH0*p_array >= 1.0 + e_optH0), facecolor='olivedrab', alpha=0.5, interpolate=True, edgecolor = 'None')

    ax2.plot(Tau, r_optH0*p_unsorted, label='power profile', lw=0.1, color = 'k')
    ax2.set_xlabel('time ' + r'$t$' + ' (years)')
    ax2.set_ylabel(r'$p$')
    ax2.set_xlim(0.0,1.0)
    ax2.set_ylim(0.0, 4.0)

    ax1.plot(Tau, r_optH0*p_array_no_bat, label='power duration curve', color = 'k', lw=2) 
    ax1.plot(Tau, r_optH0*p_min_delta_d_array, label='power duration curve', color = 'darkorange', lw=2)
    ax2.plot(Tau, r_optH0*p_unsorted_no_bat, label='power profile', lw=0.1, color = 'k')
    ax2.plot(Tau, d_unsorted, label='power profile', lw=0.1, color = 'darkorange') 
    ax3 = ax2.twinx()
    ax3.tick_params(axis='y', colors='darkorange')
    ax3.set_yticks([0, 2, 4])
    ax3.yaxis.label.set_color('darkorange')
    ax3.spines['right'].set_color('darkorange')
    ax3.set_ylabel(r'$d$', color='darkorange')
    plt.savefig(folder + '\\power duration curve filled w demand' + scenario, dpi=600,bbox_inches='tight')
    
    plt.close('all')
    
    #save some relevant data
    with open(folder + '\\data_for_run.dat', 'wb') as f:
        pickle.dump([eps, rho, e_star, r_star, e_optH, cost_star, r_optH, f_star_hydro, r_star_hydro, e_star_hydro, cost_star_hydro, flevels, hlevels], f)    

    return Synergy, Vp, VH2, Cint, rho, eps, r_star, e_star,b_min, C_H2_sa_min, P_conv[b_min_arg], cost_star, r_star_hydro, e_star_hydro, cost_star_hydro, flevels, hlevels, e_optH, r_optH, epsX, rhoX, r_starX, e_starX, f_star_hydro 

