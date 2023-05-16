import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as scp

#definir la pdf normale
pdf = lambda x : 1/(2*np.pi) * np.exp(-0.5*(x[0]**2+x[1]**2))

def system_mat(dt):
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

    return A, B, G

#Prediction using Gauss-Markov Model
def Prediction(x,u,w,A,B,G):
    x_k=x
    for i in range(len(x)):
        x_k[i]=A @ x[i] + B @ u[i]+ G @ w[i]
    return x_k

def compute_weights(particles, observations, N):
    weights = np.zeros(N)

    for i in range(len(particles)):
        # Calculate predicted observation using x
        # TODO
        # x = particles[i]
        # Calculate the difference between predicted observation and actual observation
        #diff = observations[i] - particles[i,:2]
        diff = observations - particles[i,:2]

        # Calculate the weight using the difference and a normal distribution
        #weight = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (diff**2))
        #print( scp.norm.pdf(diff,loc=0,scale=1))
        #weights[i]=scp.norm.pdf(diff,loc=0,scale=1).prod()
        weights[i] = pdf(diff)
        if(weights[i]==0):
            weights[i]+=1e-12

    #Normalize the weights
    if(np.sum(weights)==0):
        weights/=1e-12
        return weights
    weights /= np.sum(weights)

    return weights

def resample(particles, weights):
    indices = random.choices(range(len(particles)), weights=weights, k=len(particles))
    resampled_particles = particles[indices]
    return resampled_particles





def particle_filter(input, obs,dt,N,Np,mean_i,cov_i, mu, beta):
    w_k=np.random.gumbel(mu, beta, size=(Np, 2))
    A,B,G=system_mat(dt)
    #-------------------------------------------
    #Step 1: Generate N random samples
    #-------------------------------------------
    particles= np.random.multivariate_normal(mean_i, cov_i, size=Np)
    #We initialize initial state x_k
    
    #to store the different results we will find 
    res=[]
    res.append(np.mean(particles,axis=0))
    particles=np.array([np.mean(particles,axis=0) for _ in range(N)])

    for t in range(N-1):
        #-------------------------------------------
        #Step 2: Prediction 
        #-------------------------------------------
        #TODO
        #faut mettre les bons vecteurs
        particles=Prediction(particles,input,w_k,A,B,G)
        

        #-------------------------------------------
        #Step 3: Update
        #-------------------------------------------
        #                 #Compute the weights and normalize them 
        #TODO
        #weights = compute_weights(particles, obs, len(obs))
        weights = compute_weights(particles, obs[t+1], len(obs))
        #Estimate theta t+1
        #TODO

        #Resample
        #TODO
        particles = resample(particles, weights)
        #-------------------------------------------

        #-------------------------------------------
        #Step 4 : set t=t+1 and restart from step 2
        #-------------------------------------------
        res.append(np.mean(particles,axis=0))
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


sol=np.array(particle_filter(input_data,observation_data,dt,N,Np,mean_i,cov_i,mu,beta))
for i in range(100):
    sol+=np.array(particle_filter(input_data,observation_data,dt,N,Np,mean_i,cov_i,mu,beta))

sol/=101

mse = np.mean((true_data[:2] - sol[:2]) ** 2)
print("MSE="+str(mse))


fig, ax = plt.subplots()

# Plot the true data
ax.scatter(true_data[:, 0], true_data[:, 1], label='True Data')

# Scatter plot of the calculated solution
ax.scatter(sol[:, 0], sol[:, 1],color='orange', label='Particle Filter Solution')

# Set labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Comparison of Particle Filter Solution and True Data')

# Add a legend
ax.legend()

# Show the plot
plt.show()



