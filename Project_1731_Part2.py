# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:56:05 2023

@author: Mary Jo
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gumbel_r
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

def write_var_to_csv(file,var_mu,var_beta):
    with open(file,'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([str(var_mu),str(var_beta)])

def log_likelihood(params,data):
    return np.sum(gumbel_r.logpdf(data, loc=params[0], scale=params[1]))

def estimators(filename,loc,scale,M):

    mu_arr = []
    beta_arr = []
    mu_var = []
    beta_var = []

    for i in range(M):
        sample = np.random.gumbel(loc,scale,size=i)
        init = [0,1]
        result = minimize(lambda params: -log_likelihood(params,sample), init)

        # Extract the estimated parameters from the optimization result
        loc_k, scale_k = result.x

        mu_arr.append(loc_k)
        beta_arr.append(scale_k)
        var_mu_k = np.var(mu_arr)
        var_beta_k = np.var(beta_arr)
        mu_var.append(var_mu_k)
        beta_var.append(var_beta_k)

        write_params_to_csv('data/'+filename,loc_k, scale_k)
        write_var_to_csv('var.csv',var_mu_k,var_beta_k)

    return loc_k, scale_k, var_mu_k, var_beta_k


def plot_estimators_convergence(filename,mu,beta):
    df = pd.read_csv('data/'+filename)
    locs = df.iloc[:, 0]
    scales = df.iloc[:, 1]
    conv_rates_loc = [np.abs(loc - mu) for loc in locs]
    conv_rates_scale = [np.abs(scale - beta) for scale in scales]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.plot(conv_rates_loc)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Relative Convergence Rate of Loc')
    ax2.plot(conv_rates_scale)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Relative Convergence Rate of Scale')
    fig.tight_layout()
    plt.savefig('estim_conv_rates.png')

# def variances_converge(file,N_rep):
#     i=1
#     mu_arr = []
#     beta_arr = []
#     mu_var = []
#     beta_var = []
#     while(i<N_rep):
#         f = 'estimators_'+str(i)+'.csv'
#         mu_k,beta_k = estimators(f,1,2,i)
#         mu_arr.append(mu_k)
#         beta_arr.append(beta_k)
#         var_mu_k = np.var(mu_arr)
#         var_beta_k = np.var(beta_arr)
#         mu_var.append(var_mu_k)
#         beta_var.append(var_beta_k)
#         # for filename in os.listdir('data'):
#         #     df = pd.read_csv('data/'+filename)
#         #     locs = df.iloc[:, 0]
#         #     scales = df.iloc[:, 1]
#         write_var_to_csv(file,var_mu_k,var_beta_k,i)
#         i+=200
    
#     return np.var(mu_arr),np.var(beta_arr)

def plot_variances_convergence(filename,var_loc,var_scale):
    df = pd.read_csv(filename)
    var_locs = df.iloc[:, 0]
    var_scales = df.iloc[:, 1]
    conv_rates_loc = [np.abs(loc - var_loc) for loc in var_locs]
    conv_rates_scale = [np.abs(scale - var_scale) for scale in var_scales]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.plot(conv_rates_loc)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Relative Convergence Rate of Variance Loc')
    ax2.plot(conv_rates_scale)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Relative Convergence Rate of Variance Scale')
    fig.tight_layout()
    plt.savefig('var_conv_rates.png')

    
def CramerRaoBounds(B, N):
    #computing CRB for both parameters 
    Vmu=(6 * B**2) * (1 + (PI**2)/6+ GAMMA**2-2*GAMMA)
    Vbeta=(6 * B**2)/(N * PI**2)
    
    return Vmu, Vbeta 

def GumbelMean(mu, beta):
    return mu + beta*GAMMA

def GumbelVar(mu, beta):
    return (PI**2)/6*beta**2

data = estimators('data.csv',1,2,1000)
plot_variances_convergence('var.csv',data[2],data[3])