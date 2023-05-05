import numpy as np
import os

# ---------------------------------------------------
# Global variables
# ---------------------------------------------------
N = 100
DATA_DIR = 'data_files/'

# ---------------------------------------------------
# Functions to get data from given files
# ---------------------------------------------------

def load_file(filename):
    return np.loadtxt(filename,delimiter=' ')

def get_true_data():
    positions = load_file(DATA_DIR+'True_data.txt')
    n = len(positions)
    x_pos = np.zeros(n)
    y_pos = np.zeros(n)
    x_vel = np.zeros(n)
    y_vel = np.zeros(n)
    for i in range(n):
        x_pos[i] = positions[i][0]
        y_pos[i] = positions[i][1]
        x_vel[i] = positions[i][2]
        y_vel[i] = positions[i][3]
    return x_pos,y_pos,x_vel,y_vel

def get_noisy_obs():
    noisy_obs = load_file(DATA_DIR+'Observations.txt')
    n = len(noisy_obs)
    x_noisy_obs = np.zeros(n)
    y_noisy_obs = np.zeros(n)
    for i in range(n):
        x_noisy_obs[i] = noisy_obs[i][0]
        y_noisy_obs[i] = noisy_obs[i][1]
    return x_noisy_obs,y_noisy_obs

def get_white_noise_v(x_true,y_true,x_noisy_obs,y_noisy_obs):
    n = x_noisy_obs.size
    v_x = np.zeros(n)
    v_y = np.zeros(n)
    for i in range(n):
        v_x[i] = x_noisy_obs[i] - x_true[i]
        v_y[i] = y_noisy_obs[i] - y_true[i]
    return v_x,v_y
    
def get_acc_u():
    u_input = load_file(DATA_DIR+'Input.txt')
    n = len(u_input)
    u_x = np.zeros(n)
    u_y = np.zeros(n)
    for i in range(n):
        u_x[i] = u_input[i][0]
        u_y[i] = u_input[i][1]
    return u_x,u_y

# -------------------------------------------------------------------------------------------------
# Get informations
# -------------------------------------------------------------------------------------------------

x_true,y_true,x_true_vel,y_true_vel = get_true_data()
x_noisy_obs,y_noisy_obs = get_noisy_obs()
v_x,v_y = get_white_noise_v(x_true,y_true,x_noisy_obs,y_noisy_obs)
u_x,u_y = get_acc_u()