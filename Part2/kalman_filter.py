from get_data import *
import numpy as np

# -----------------------------------
# Kalman filter 
#       -> Estimate s = (x,y) : position of the boat
#------------------------------------

x_pred = np.zeros(n)
y_pred = np.zeros(n)

# Informations / Variables 
dt = 0.1                # [sec]
gamma = 0.5772          # Euler cste
mu=1                    # Mu parameter for random acceleration Gumbel distribution
beta=1                  # Beta parameter for random acceleration Gumbel distribution

# 1 : Initial conditions

# initial covariance matrix
P0_pos = 50
P0_vel = 10
P0 = np.diag([P0_pos,P0_pos,P0_vel,P0_vel])
# initial position and velocity for random normal distribution
s_0_pos = np.random.normal(10,50,(n,n))
s_0_vel = np.random.normal(10,10,(n,n))

# Define the state transition matrix
A = np.eye(4)
# Define the process noise covariance matrix
# Q = 

# 2 : Run the filter
#           --> X_pred_k = A*X_pred_k-1 + B*u_k + G (mu + beta*gamma, mu + beta*gamma)

for k in range(N):
    # Predict state and covariance matrix
    #Compute Kalman gain
    pass

