# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:56:05 2023

@author: Mary Jo
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gumbel_r,norm
from scipy.optimize import minimize
import csv
import pandas as pd
import math as m
import os

#Defining constants we will use in this code
GAMMA= 0.5772
PI= m.pi

def write_params_to_csv(file,loc,scale):
    with open(file,'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([str(loc),str(scale)])

def write_var_to_csv(file,var_mu,var_beta,mu_bound,beta_bound):
    with open(file,'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([str(var_mu),str(var_beta),str(mu_bound),str(beta_bound)])

def log_likelihood(params,data):
    return np.sum(gumbel_r.logpdf(data, loc=params[0], scale=params[1]))

def estimators(filename,loc,scale,N,tag):

    sample = np.random.gumbel(loc=loc,scale=scale,size=N)
    init = [0,1]
    result = minimize(lambda params: -log_likelihood(params,sample), init)

    # Extract the estimated parameters from the optimization result
    loc_k, scale_k = result.x
    
    if tag==0:
        write_params_to_csv('data/'+filename+'.csv',loc_k, scale_k)

    return loc_k, scale_k
    
def hist_estimators(filename,N):

    df = pd.read_csv('data/'+filename+'.csv')
    locs = np.sort(df.iloc[:, 0])
    scales = np.sort(df.iloc[:, 1])

    freq_locs = {}
    for l in locs:
        if round(l,2) not in freq_locs.keys():
            freq_locs[round(l,2) ] = 1
        else :
            freq_locs[round(l,2) ] += 1
        
    freq_scales = {}
    for l in scales:
        if round(l,2) not in freq_scales.keys():
            freq_scales[round(l,2) ] = 1
        else :
            freq_scales[round(l,2) ] += 1


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,4))
    fig.suptitle('Estimators values for size N='+str(N),fontsize=(15))
    ax1.set_xlabel('Mu value')
    ax1.set_ylabel('Frequency')
    ax1.axvline(x=1, color='r',label='True mu')
    ax1.legend()
    ax1.bar(freq_locs.keys(),freq_locs.values())
    ax2.set_xlabel('Beta value')
    ax2.set_ylabel('Frequency')
    ax2.axvline(x=2, color='r',label='True beta')
    ax2.legend()
    ax2.bar(freq_scales.keys(),freq_scales.values())
    fig.tight_layout()
    plt.savefig('plots/estimators_'+filename+'.png')

def plot_variances_convergence(filename,N,M):
    df = pd.read_csv('data/'+filename+'.csv')
    var_locs = df.iloc[:, 0]
    var_scales = df.iloc[:, 1]
    # conv_rates_loc = [np.abs(loc - var_loc) for loc in var_locs]
    # conv_rates_scale = [np.abs(scale - var_scale) for scale in var_scales]
    mu_bound = df.iloc[:, 2]
    beta_bound = df.iloc[:, 3]

    x = np.arange(1,M)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle('Log-Log graph of empirical variances for N = '+str(N),fontsize=(15))
    ax1.loglog(x,var_locs,color='g')
    ax1.loglog(x, mu_bound,label='Cramer-Rao bound',color='r')
    ax1.legend()
    ax1.set_xlabel('Iteration [log(i)]')
    ax1.set_ylabel('Loc variance [log(variance)]')
    ax2.loglog(x,var_scales,color='g')
    ax2.loglog(x, beta_bound,label='Cramer-Rao bound',color='r')
    ax2.legend()
    ax2.set_xlabel('Iteration [log(i)]')
    ax2.set_ylabel('Scale variance [log(variance)]')
    fig.tight_layout()
    plt.savefig('plots/'+filename+'.png')
   
def CramerRaoBounds(MU,B, N):
    #computing CRB for both parameters 
    Vmu=(6 * B**2) * (1 + (PI**2)/6+ GAMMA**2-2*GAMMA) / (N * PI**2)
    Vbeta=(6 * B**2)/(N * PI**2)
    
    return Vmu, Vbeta 

def GumbelMean(mu, beta):
    return mu + beta*GAMMA

def GumbelVar(mu, beta):
    return (PI**2)/6*beta**2

def plot_normal_gumbel(mu,beta):
    #values to plot the pdfs
    x = np.linspace(-10, 10, 1000)

    #computing the mean and variance of the gumbel 
    gumbel_mean = GumbelMean(mu,beta)
    gumbel_var = GumbelVar(mu,beta)

    #computing the pdfs
    pdf_gumb = gumbel_r.pdf(x, loc=mu, scale=beta)
    pdf_norm=norm.pdf(x, loc=gumbel_mean, scale=np.sqrt(gumbel_var))

    #plots
    plt.plot(x, pdf_gumb, label='Gumbel')
    plt.plot(x, pdf_norm, label='Normal')
    plt.xlabel('x')
    plt.ylabel('PDF, f(x)')
    plt.title('Comparing Gumbel and Normal Distributions')
    plt.legend()
    plt.savefig('Gumb_Norm_'+str(beta)+'.png')
    
plot_normal_gumbel(0,3.5)

# Question c

N1 = [50,100,200,500,1000]
M1 = 500

for size in N1:
    file = 'data_'+str(size)
    for i in range(M1):
        data_i = estimators(file,1,2,size,0)
    hist_estimators(file,size)

# Question d
N2 = [50,100,500,1000,3000]
M2 = 10**4
for size in N2:
    file='variances_'+str(size)
    mu_arr = []
    beta_arr = []
    mu_var = []
    beta_var = []
    for i in range(M2):
        
        data_i = estimators(file,1,2,size,1)

        mu_arr.append(data_i[0])
        beta_arr.append(data_i[1])
        var_mu_k = np.var(mu_arr)
        var_beta_k = np.var(beta_arr)
        mu_var.append(var_mu_k)
        beta_var.append(var_beta_k)

        bounds = CramerRaoBounds(1,2,size)
        write_var_to_csv('data/'+file+'.csv',var_mu_k,var_beta_k,bounds[0],bounds[1])
        
    plot_variances_convergence(file,size,M2)
    