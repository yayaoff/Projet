from get_data import *
import numpy as np
from math import pi
import matplotlib.pyplot as plt

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
X0_moy = 10
X0 = np.array([X0_moy,X0_moy,X0_moy,X0_moy])
X_k = X0
X = np.empty((n,4))
X[0] = X0

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
Q = (pi**2)* (beta**2)/6 * np.dot(G,G.T)
#        Define the observation matrix
C = np.array([[1,0,0,0],[0,1,0,0]])
#        Define the measurement noise covariance matrix
R = np.eye(2)

#------------------------------------
# 1 : Initial state 
#------------------------------------

P0_pos = 50
P0_vel = 10
#        Define the (initial) predicted error covariance matrix
P = np.diag([P0_pos,P0_pos,P0_vel,P0_vel])   # P0

#------------------------------------
# Useful functions
#------------------------------------

def kalman_gain():
    return np.dot( np.dot(P, C.T), np.linalg.inv( np.add(np.dot(np.dot(C, P), C.T),R)))

def predict(u_k):
    global P
    global X_k
    shift_term = np.array([mu+beta*gamma,mu+beta*gamma,mu+beta*gamma,mu+beta*gamma])
                # X_pred_k = A*X_pred_k-1 + B*u_k + G (mu + beta*gamma, mu + beta*gamma)
    X_k = np.dot(A,X_k) + np.dot(B, u_k) + np.dot(Q,shift_term)
    P = np.dot(np.dot(A, P), A.T) + Q 
    # P_prexd = A @ P @ A.T + Q - A @ P @ C.T @ np.linalg.inv(C @ P @ C.T + R) @ C @ P @ A.T

def update(y):
    global P
    global X_k
    # Compute Kalman gain
    K = kalman_gain()
    # Update state
    X_k = np.add(X_k , np.dot(K,y-np.dot(C,X_k)))
    P = np.subtract(P, np.dot(np.dot(K,C),P))

#------------------------------------
# Run the filter
#------------------------------------

for k in range(1,n):
    u_k = np.array([u_x[k],u_y[k]])
    # Predict state and covariance matrix
    predict(u_k)
    # Update state and error covariance matrix
    y_k = np.array([x_noisy_obs[k],y_noisy_obs[k]])
    update(y_k)
    X[k] = X_k

#------------------------------------
# 3 : Plots 
#------------------------------------

fig = plt.figure(figsize=(15,10))
fig = plt.title("Estimated positions - Kalman Filter",fontsize=20,fontweight='bold')
fig = plt.scatter(x_true_pos,y_true_pos,color='orange',label='True state',s=15)
fig = plt.scatter(X.T[0],X.T[1],color='green',label='Filtered state',s=15)
fig = plt.xlabel('x position',fontsize=15)
fig = plt.ylabel('y position',fontsize=15)
fig = plt.legend()
fig = plt.savefig("plots/kalman.png")
plt.show()