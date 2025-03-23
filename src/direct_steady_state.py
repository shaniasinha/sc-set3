import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

#Our parameters
radius = 2.0
h = 0.1  
D = 1.0  
N = int(2 * radius / h) + 1  

#The stability condition 
dt = 0.25 * h**2 / D  
tolerance = 1e-6  
max_steps = 10000  

#Creating our grid
x = np.linspace(-radius, radius, N)
y = np.linspace(-radius, radius, N)
X, Y = np.meshgrid(x, y)

#M and b creation
size = N * N
M = sp.lil_matrix((size, size), dtype=float)
b = np.zeros(size)

#Position source
i_source = np.argmin(np.abs(x - 0.6))
j_source = np.argmin(np.abs(y - 1.2))
index_source = i_source * N + j_source

#Filling M and b
for i in range(N):
    for j in range(N):
        index = i * N + j
        r = np.sqrt(X[i, j]**2 + Y[i, j]**2)

        if r > radius - h / 2:  #Dirichlet condition
            M[index, index] = 1
            b[index] = 0
        
        elif index == index_source:  
            M[index, index] = 1
            b[index] = 10 
        
        else:
            #Laplacian
            M[index, index] = -4 * D / h**2
            if i > 0:
                M[index, index - N] = D / h**2
            if i < N - 1:
                M[index, index + N] = D / h**2
            if j > 0:
                M[index, index - 1] = D / h**2
            if j < N - 1:
                M[index, index + 1] = D / h**2

#Csr
M = M.tocsr()

#Initial conditions
c = np.zeros(size)

#Iterative scheme
for step in range(max_steps):
    c_new = c + dt * (M @ c + b)
    
    #Convergence verification
    diff = np.linalg.norm(c_new - c) / np.linalg.norm(c) if np.linalg.norm(c) > 0 else np.linalg.norm(c_new)
    if diff < tolerance:
        print(f"Converged after {step + 1} steps with diff = {diff:.2e}")
        break
    c = c_new.copy()
else:
    print("Max steps reached without convergence.")

# Reshaping
C = c.reshape((N, N))

#To get a disk domain
C = np.ma.masked_where(np.sqrt(X**2 + Y**2) > radius - h / 2, C)

#Just a test ( uncommented)
# print("min value :", np.min(C))
# print("max value de C :", np.max(C))

#To get a correct colorbar
C_normalized = (C - np.min(C)) / (np.max(C) - np.min(C))

# Another verification
#print("min value :", np.min(C_normalized))
#print("max value :", np.max(C_normalized))


def plot_steady_state_concentration(savefig=False):
    #Plotting
    plt.figure(figsize=(6, 5), dpi=300)
    plt.imshow(C_normalized.T, extent=[-radius, radius, -radius, radius], origin="lower", cmap="viridis")
    plt.colorbar(label="Concentration")
    # plt.title("Steady-State Concentration in a disk")
    plt.xlabel("x", fontsize=16)
    plt.ylabel("y", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    #Saving
    if savefig:
        # plt.savefig("results/direct_steady_state/steady_state_concentration.png", bbox_inches='tight')
        plt.savefig("results/direct_steady_state/steady_state_concentration.pgf", bbox_inches='tight')
    
    # plt.savefig("results/direct_steady_state/steady_state_concentration.pgf", bbox_inches='tight')
    # plt.savefig("results/direct_steady_state/steady_state_concentration.png", bbox_inches='tight')
    
    plt.show()