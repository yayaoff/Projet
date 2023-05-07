from get_data import *
import numpy as np
from math import pi

# ----------------------------------------------------------------------------------
# Kalman filter 
#       -> Estimate s = (x,y) : position of the boat
#-----------------------------------------------------------------------------------

#------------------------------------
# Informations / Variables 
#------------------------------------
dt = 0.1                # [sec]
gamma = 0.5772          # Euler cste
mu=1                    # Mu parameter for random acceleration Gumbel distribution
beta=1                  # Beta parameter for random acceleration Gumbel distribution
cov_x_pos = np.cov(x_true_pos)
cov_y_pos = np.cov(y_true_pos)
cov_x_vel = np.cov(x_true_vel)
cov_y_vel = np.cov(y_true_vel)

#------------------------------------
# Matrices
#------------------------------------
#       Define the state transition matrix
A = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
#       Define the control input matrix
B = np.array([[(dt**2)/2,0],[0,(dt**2)/2],[dt,0],[0,dt]])
#        Define G matrix
G = np.array([[(dt**2)/2,0],[0,(dt**2)/2],[dt,0],[0,dt]])
#        Define the process noise covariance matrix
Q = pi**2/6 * beta**2 * np.dot(G,G.T)
#        Define the observation matrix
C = np.array([1,0,0,1])
#        Define the measurement noise covariance matrix
R = np.diag([cov_x_pos,cov_y_pos,cov_x_vel,cov_y_vel])
#        Define the (initial) predicted error covariance matrix
P0_pos = 50
P0_vel = 10
P = np.diag([P0_pos,P0_pos,P0_vel,P0_vel])   # P0

#------------------------------------
# Useful functions
#------------------------------------

def kalman_gain():
    return np.dot(np.dot(P, C.T), np.linalg.inv(np.dot(np.dot(C, P), C.T) + R))

def predict(X,u_k):
    X_pred = np.empty_like(X)
    shift_term = [mu+beta*gamma,mu+beta*gamma]
    # X_pred_k = A*X_pred_k-1 + B*u_k + G (mu + beta*gamma, mu + beta*gamma)
    X_pred = np.dot(A,X) + np.dot(B, u_k) + np.dot(G,shift_term)
    P_k = np.dot(np.dot(A, P), A.T) + Q
    return X_pred,P_k

def update(X,y,P):
    K = kalman_gain()
    X += np.dot(K,y-np.dot(C,X))
    P = P - np.dot(np.dot(K,C),P)

#------------------------------------
# 1 : Initial state (initial position and velocity for random normal distribution)
#------------------------------------

s_0_pos = np.random.normal(10,50,(n,n))
s_0_vel = np.random.normal(10,10,(n,n))

#------------------------------------
# 2 : Run the filter
#------------------------------------

for k in range(N):
    X_k = [s_0_pos[0][k],s_0_pos[1][k],s_0_vel[0][k],s_0_vel[1][k]]
    u_k = np.array([u_x[k],u_y[k]])
    # Predict state and covariance matrix
    X_pred,P_pred = predict(X_k,u_k)
    #Compute Kalman gain
    K_k = kalman_gain()
    # Update state and error covariance matrix
    update(X_pred,P_pred,P)