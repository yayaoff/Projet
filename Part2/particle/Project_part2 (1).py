#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:39:17 2023

@author: collinemma
"""
import numpy as np
import random  # for random.choices
import matplotlib.pyplot as plt

# ---- OPEN FILES ---- #

input_1 = [] # manoeuvring acceleration uk in 1
input_2 = [] # manoeuvring acceleration uk in 2

with open('Input.txt', 'r') as file:
    for line in file:
        line = line.strip().split(' ')
        input_1.append(float(line[0]))
        input_2.append(float(line[1]))

observation_1 = [] # noisy observations Y in 1
observation_2 = [] # noisy observations Y in 2

with open('Observations.txt', 'r') as file:
    for line in file:
        line = line.strip().split(' ')
        observation_1.append(float(line[0]))
        observation_2.append(float(line[1]))
        
true_data_u1 = []
true_data_u2 = []
true_data_1 = []
true_data_2 = []
        
with open('True_data.txt', 'r') as file:
    for line in file:
        line = line.strip().split(' ')
        true_data_u1.append(float(line[0]))
        true_data_u2.append(float(line[1])) 
        true_data_1.append(float(line[2]))
        true_data_2.append(float(line[3]))


# ---- Equation du système ----#
mu = 1
beta = 1
size = 2 # ? Car veut vecteur taille 2
w = np.random.gumbel(mu, beta, size) # random acceleration
v = np.random.normal(0,1,2) # Observation noise
dt = 0.1 # time step in sec
Np = 1000 # number of particules
N = 100 # number of iterations

#Observation_1 = xk + v
#Observation_2 = yk + v

#x{k+1} = xk + dt* x°{k+1} + 1/2*dt**2 * input_1 + 1/2*dt**2 * w
#y{k+1} = yk + dt* y°{k+1} + 1/2*dt**2 * input_2 + 1/2*dt**2 * w
#x°{k+1} = x°k + dt * input_1 + dt * w
#y°{k+1} = y°k + dt * input_2 + dt * w
#MSE = 1/N  Somme de k = 1 jusqu'à N de ||s_k - ^s_k||^2_2

# ---- PARTICULE FILTER ----

dt = 0.1 # time step in sec
Np = 1000 # number of particules
s_moy = 10
s_var = 50
sRond_moy = 10
sRond_var = 10
#Taille du cadre max ? -> max dans Y
#Trouver largeur max
np.random.seed(0)

def min_frame(array_wanted):
    minimum = 0
    for i in range(len(array_wanted)) : 
        if array_wanted[i] < minimum :
            minimum = array_wanted[i]
            #print(minimum)
    minimum_R = round(minimum)
    if minimum_R < minimum :
        minimum_R -= 1
        
    return minimum_R
    
    
def max_frame(array_wanted):
    maximum = 0
    for i in range(len(array_wanted)) : 
        if array_wanted[i] > maximum :
            maximum = array_wanted[i]
            #print(maximum)
    maximum_R = round(maximum)
    if maximum_R > maximum :
        maximum_R += 1
    
    if min_frame(array_wanted) < 0:
        maximum_R -= min_frame(array_wanted)
    return maximum_R

frame_width = max_frame(true_data_1)
frame_height = max_frame(true_data_2)


def initialize_particles():
    particles = np.random.rand(Np,4)
    for i in range(Np):
        particles[i,0] = np.random.normal(s_moy, s_var, 1)
        particles[i,1] = np.random.normal(s_moy, s_var, 1)
        particles[i,2] = np.random.normal(sRond_moy, sRond_var, 1)
        particles[i,3] = np.random.normal(sRond_moy, sRond_var, 1)
    #particles = particles * np.array((frame_width, frame_height, dt,dt)) 
    particles[:,2:4] -= dt/2.0
    return particles


#Plot particule sur le graph -> s'aider de la fct display sur site et voir comme fait pour Kalman

def apply_velocity(particles): #Appliquer vitesse aux particules
    particles[:,0] += particles[:,2]
    particles[:,1] += particles[:,3]

    return particles


def normal_dist(x , mean , sd):
    prob_density = 1/(sd*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mean)/sd)**2) 
    return prob_density

def weight(observation, particles, iteration):
    w = []
    for i in range(Np):
        x = observation[iteration] - particles[i, 0]
        weight = normal_dist(x, 0, 1)
        w.append(weight)
    return w

def normalized(vecteur) :
    somme = 0
    for i in range(len(vecteur)):
        somme += vecteur[i]
    normalized_vecteur = []
    for j in range(len(vecteur)):
        norme = vecteur[j]/somme
        normalized_vecteur.append(norme)
    return(normalized_vecteur)
    
#Apply function to the data.
#mean = 0
#sd = 1
#x = y_{i+1}−x ̃_{i+1} -> y = observation et x ̃= prediction particule avant rééchantillonnage
#pdf = normal_dist(x,mean,sd)
# ->> Rééchantillonage -> retourner une nouvelle liste de Np particules random avec densité de
# probilité donné par poids calculé avant

#
particule = initialize_particles()
print(len(observation_1))
print(len(particule[:,0]))
weight_1 = weight(observation_1, particule, 1)
#print(weight_1)
#normalized_weight = weight_1/np.linalg.norm(weight_1)
norme = normalized(weight_1)
print(norme)

somme = 0
somme_norme = 0
for i in range(len(weight_1)):
    somme += weight_1[i]
    somme_norme += norme[i]
    
print(somme) 
print(somme_norme) 



    
vitesse_part = apply_velocity(particule)
#print(vitesse_part)






