import numpy as np
input_data = np.loadtxt('data_files\Input.txt')
observation_data = np.loadtxt('data_files\Observations.txt')
true_data = np.loadtxt('data_files\True_data.txt')
#-------------------------------------------
#Prep
#Creating the system 
#-------------------------------------------
N=100

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
for i in range(1000):
    x_k=np.concatenate(np.random.normal(10,50,2), np.random.normal(10,10,2))
    w_k=np.array([np.random.gumbel(mu, beta,100), np.random.gumbel(mu, beta,100)])
    for i in range(N):
        #-------------------------------------------
        #Step 1: Generate N random samples
        #-------------------------------------------
        #TODO
        samples = np.random.normal(0, 1, size=(N, 2))
        samples *= np.array([beta / np.sqrt(6), beta / np.sqrt(6)])  # Scale to Gumbel distribution


        #-------------------------------------------
        #Step 2: Prediction 
        #-------------------------------------------
        #TODO
        #faut mettre les bons vecteurs
        for i in range(len(observation_data)):
            x_k=A @ x_k + B @ input_data[i]+ G* w_k[i]

        #-------------------------------------------
        #Step 3: Update
            #Compute the weights and normalize them 
            #TODO
            weights = np.exp(-0.5 * ((x_pred - observation_data[:, 0]) ** 2 + (y_pred - observation_data[:, 1]) ** 2) / (gamma ** 2))
            weights /= np.sum(weights)

            #Estimate theta t+1
            #TODO
            theta = np.zeros(4)
            for i in range(N):
                theta += weights[i] * x[i]

            #Resample
            #TODO
            resample_idx = np.random.choice(N, size=N, p=weights)
            x = x[resample_idx]
            x_pred = x_pred[resample_idx]
            y_pred = y_pred[resample_idx]
        #-------------------------------------------

        #-------------------------------------------
        #Step 4 : set t=t+1 and restart from step 2
        #-------------------------------------------
        #this is why we are iteration in a for loop 