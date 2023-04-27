import numpy as np
from scipy.optimize import minimize
from scipy.stats import gumbel_r
import matplotlib.pyplot as plt
from math import pi


M=10000;
mu, beta = 1, 2 # Gumbel(1,2)
mu_hat=[]
beta_hat=[]
gamma = 0.5772
mu_cr = []
beta_cr = []
mu_var = []
beta_var = []

#Define de log-likelihood function
def log_likelihood(params, w):
    return np.sum(gumbel_r.logpdf(w, loc=params[0], scale=params[1]))

#Compute mu_hat and beta_hat using the methods explained in c.
for i in range (1,M):
    w = np.random.gumbel(loc=mu, scale=beta, size=i)  #generate samples

    
    init_guess = [0,1]
    
    res = minimize(lambda params : -log_likelihood(params, w), init_guess)
    
    mu_hat.append(res.x[0])
    beta_hat.append(res.x[1])
    mu_var.append(np.var(mu_hat))
    beta_var.append(np.var(beta_hat))
    mu_cr.append(6*beta**2*(1+(pi**2)/6 + gamma**2 -2*gamma)/(i*pi**2))
    beta_cr.append((6*beta**2)/(i*pi**2))


#Plot results
x = np.arange(1,M)
plt.loglog(x, mu_var, label="Variance of estimator")
plt.loglog(x, mu_cr, label="Cramér-Rao bound")
plt.legend()
plt.ylabel("Variance of the mu estimator")
plt.xlabel("Number of iterations")
plt.savefig("e_mu.png")
plt.show()

x = np.arange(1,M)
plt.plot(x, beta_var, label="Variance of estimator")
plt.plot(x, beta_cr, label="Cramér-Rao bound")
plt.legend()
plt.ylabel("Variance of the beta estimator")
plt.xlabel("Number of iterations")
plt.savefig("e_beta.png")
plt.show()