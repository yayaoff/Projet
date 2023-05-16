import numpy as np
input_data = np.loadtxt('data_files\Input.txt')
observation_data = np.loadtxt('data_files\Observations.txt')
true_data = np.loadtxt('data_files\True_data.txt')
#-------------------------------------------
#Prep
#Creating the system 
#-------------------------------------------
n_it=100
n_particle=1000

# Informations / Variables 
dt = 0.1                # [sec]
gamma = 0.5772          # Euler cste
mu=1                    # Mu parameter for random acceleration Gumbel distribution
beta=1                  # Beta parameter for random acceleration Gumbel distribution
#       Define the state transition matrix
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
A,B,G=system_mat(dt)

#We initialize initial state x_k
particles_s=np.random.multivariate_normal(10, 50, size=n_particle)
particles_ds=np.random.multivariate_normal(10, 10, size=n_particle)
        #-------------------------------------------
        #Step 1: Generate N random samples
        #-------------------------------------------
        #TODO
        


        #-------------------------------------------
        #Step 2: Prediction 
        #-------------------------------------------
        #TODO
        #faut mettre les bons vecteurs

        #-------------------------------------------
        #Step 3: Update
        #-------------------------------------------
            #Compute the weights and normalize them 
            #TODO

            #Estimate theta t+1
            #TODO

            #Resample
            #TODO

        #-------------------------------------------

        #-------------------------------------------
        #Step 4 : set t=t+1 and restart from step 2
        #-------------------------------------------
        #this is why we are iteration in a for loop 