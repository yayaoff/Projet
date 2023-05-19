import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as scp
import time

# cov=[[1,0],[0,1]]
# pdf = lambda x: (1 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))) * np.exp(
#     -0.5 * np.dot(np.dot((x[0], x[1]), np.linalg.inv(cov)), np.array([x[0], x[1]]).T)
# )
#definir la pdf normale
pdf = lambda x : 1/np.sqrt((2*np.pi)) * np.exp(-0.5*(x @ x.T))

dt = 0.1
A = np.eye(4)
B=np.zeros((4,2))
G=np.zeros((4,2))

#defining A
A[0][2]=dt
A[1][3]=dt

#defining B 
B[0][0]=1/2*dt**2
B[1][1]=1/2*dt**2
B[2][0]=dt
B[3][1]=dt

#defining G
G[0][0]=1/2*dt**2
G[1][1]=1/2*dt**2
G[2][0]=dt
G[3][1]=dt

#Prediction using Gauss-Markov Model
def Prediction(x,u,w):
    x_k=x
    for i in range(len(x)):
        x_k[i]=A @ x[i] + B @ u + G @ w[i]
    return x_k

def compute_weights(particles, observations, N):
    weights = np.zeros(len(particles))

    for i in range(len(particles)):
        # Calculate the difference between predicted observation and actual observation

        diff = observations - particles[i,:2]
        weights[i] = pdf(diff)

        if(weights[i]==0):
            weights[i]+=1e-12

    norm = np.linalg.norm(weights)
    if norm ==0:
        norm += 1e-12
    weights /= norm
    return weights

def resample(particles, weights):
    indices = random.choices(range(len(particles)), weights=weights, k=len(particles))
    resampled_particles = particles[indices]
    return resampled_particles

def particle_filter(input, obs,dt,N,Np,mean_i,cov_i, mu, beta):
    w_k=np.random.gumbel(mu, beta, size=(Np, 2))

    #-------------------------------------------
    #Step 1: Generate N random samples
    #-------------------------------------------
    particles= np.random.multivariate_normal(mean_i, cov_i, size=Np)
    #We initialize initial state x_k
    
    #to store the different results we will find 
    res=np.empty((N,4))
    res[0] = np.mean(particles,axis=0)

    for t in range(N-1):
        #-------------------------------------------
        #Step 2: Prediction 
        #-------------------------------------------
        #faut mettre les bons vecteurs
        particles=Prediction(particles,input[t],w_k)

        #-------------------------------------------
        #Step 3: Update
        #-------------------------------------------
        #Compute the weights and normalize them 
        #Estimate theta t+1
        weights = compute_weights(particles, obs[t+1], len(obs))

        #Resample
        particles = resample(particles, weights)
        #-------------------------------------------

        #-------------------------------------------
        #Step 4 : set t=t+1 and restart from step 2
        #-------------------------------------------
        res[t+1] = np.mean(particles,axis=0)
        #this is why we are iteration in a for loop 
    return res


input_data = np.loadtxt('data_files/Input.txt')
observation_data = np.loadtxt('data_files/Observations.txt')
true_data = np.loadtxt('data_files/True_data.txt')

#-------------------------------------------
#Prep
#Creating the system 
#-------------------------------------------
N=100
Np=1000

#distribution of initial state particles
mean_i = np.array([10, 10, 10, 10])
cov_i = np.diag([50, 50, 10, 10])

# Informations / Variables 
dt = 0.1                # [sec]
gamma = 0.5772          # Euler cste
mu=1                    # Mu parameter for random acceleration Gumbel distribution
beta=1                  # Beta parameter for random acceleration Gumbel distribution
#       Define the state transition matrix

start_time = time.time()
sol=np.array(particle_filter(input_data,observation_data,dt,N,Np,mean_i,cov_i,mu,beta))
for i in range(500):
    sol+=np.array(particle_filter(input_data,observation_data,dt,N,Np,mean_i,cov_i,mu,beta))
elapsed_time = time.time() - start_time
sol/=501

mse = np.mean((true_data[:2] - sol[:2]) ** 2)
print("MSE="+str(mse))
print('Execution time : '+str(elapsed_time)+'seconds')

fig, ax = plt.subplots()

# Plot the true data
ax.scatter(true_data[:, 0], true_data[:, 1], label='True Data')

# Scatter plot of the calculated solution
ax.scatter(sol[:, 0], sol[:, 1],color='orange', label='Particle Filter Solution')

# Set labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Comparison of Particle Filter Solution and True Data')
# Afficher mse sur le plot
text = 'MSE = '+str(round(mse,2))
bbox_props = dict(boxstyle='round', facecolor='white', edgecolor='black',pad=0.2)
plt.text(0.75, 0.05, text, fontsize=10, ha='left', va='bottom', bbox=bbox_props, transform=plt.gca().transAxes)

# Add a legend
ax.legend()

#Save plot
plt.savefig('plots/particle_500_1.png')

# Show the plot
# plt.show()