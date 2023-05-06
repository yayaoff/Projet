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
#       Define the state transition matrix
A = np.eye(4)
#       Define the ... matrix
B = np.eye(4)
#        Define the process noise covariance matrix
Q = np.eye(4)
#        Define the ... matrix
P = np.eye(4)
#        Define the ... matrix
G = np.eye(4)
#        Define the ... matrix
C = np.eye(4)
#        Define the ... matrix
R = np.eye(4)

# Useful onctions

def kalman_gain(innovation_covariance,predicted_error_covariance):
    return np.dot(np.dot(P, C.T), np.linalg.inv(np.dot(np.dot(C, P), C.T) + R))

def predict_state_cov(s,u):
    s[0] = np.dot(A,s[0]) - np.dot(B, u)
    s[1] = np.dot(A,s[1]) - np.dot(B, u)
    P = np.dot(np.dot(A, P), A.T) + Q
    return s,P

def update(K,s,y,P):
    s[0] += np.dot(K,y-np.dot(C, s[0]))
    s[1] += np.dot(K,y-np.dot(C, s[1]))
    P -= np.dot(np.dot(K,C),P)

# 1 : Initial conditions

# initial covariance matrix
P0_pos = 50
P0_vel = 10
P0 = np.diag([P0_pos,P0_pos,P0_vel,P0_vel])
# initial position and velocity for random normal distribution
s_0_pos = np.random.normal(10,50,(n,n))
s_0_vel = np.random.normal(10,10,(n,n))

# 2 : Run the filter
#           --> X_pred_k = A*X_pred_k-1 + B*u_k + G (mu + beta*gamma, mu + beta*gamma)

for k in range(N):
    # Predict state and covariance matrix
    #Compute Kalman gain
    pass