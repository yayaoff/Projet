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

#Defining constants we will use in this code
GAMMA= 0.5772
PI= m.pi


def write_to_csv(file,loc,scale):
    with open(file,'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([str(loc),str(scale)])

def log_likelihood(params,data):
    return np.sum(gumbel_r.logpdf(data, loc=params[0], scale=params[1]))

def estimators(loc,scale,M):

    for i in range(M):
        sample = np.random.gumbel(loc,scale,size=i*10)
        init = [0,1]
        result = minimize(lambda params: -log_likelihood(params,sample), init)

        # Extract the estimated parameters from the optimization result
        loc_k, scale_k = result.x
        write_to_csv("params.csv",loc_k,scale_k)

    return loc_k, scale_k 


def plot_convergence(filename):
    df = pd.read_csv(filename)
    locs = df.iloc[:, 0]
    scales = df.iloc[:, 1]
    conv_rates_loc = [np.abs((loc - locs.iloc[0])/loc) for loc in locs]
    conv_rates_scale = [np.abs((scale - scales.iloc[0])/scale) for scale in scales]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.plot(conv_rates_loc)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Relative Convergence Rate of Loc')
    ax2.plot(conv_rates_scale)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Relative Convergence Rate of Scale')
    fig.tight_layout()
    plt.savefig('conv_rates.png')
    
    
def CramerRaoBounds(B, N):
    #computing CRB for both parameters 
    Vmu=(6 * B**2) * (1 + (PI**2)/6+ GAMMA**2-2*GAMMA)
    Vbeta=(6 * B**2)/(N * PI**2)
    
    return Vmu, Vbeta 

def GumbelMean(mu, beta):
    return mu + beta*GAMMA

def GumbelVar(mu, beta):
    return (PI**2)/6*beta**2
    

estimators(1,2,1000)
plot_convergence("params.csv")
