import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gumbel_r
from scipy.optimize import minimize
import csv

def write_to_csv(file,loc,scale):
    with open(file,'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([str(loc),str(scale)])

def log_likelihood(params,data):
    return np.sum(gumbel_r.logpdf(data, loc=params[0], scale=params[1]))

def estimators(M):

    for i in range(M):
        sample = np.random.gumbel(1,2,size=i*10)
        init = [0,1]
        result = minimize(lambda params: -log_likelihood(params,sample), init)

        # Extract the estimated parameters from the optimization result
        loc_k, scale_k = result.x
        write_to_csv("params.csv",loc_k,scale_k)

    return loc_k, scale_k 

def plot_estimators(file):
    

estimators(100)
