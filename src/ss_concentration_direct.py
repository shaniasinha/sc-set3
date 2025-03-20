
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

#Our parameters
radius = 2.0
h = 0.1  
D = 1.0  
N = int(2 * radius / h) + 1  

# Stability condition ( cf report)
dt = 0.25 * h**2 / D  
tolerance = 1e-6  
max_steps = 10000  

# Creating the grid
x = np.linspace(-radius, radius, N)
y = np.linspace(-radius, radius, N)
X, Y = np.meshgrid(x, y)

# Initialize matrix M and vector b
size = N * N
M = sp.lil_matrix((size, size), dtype=float)
b = np.zeros(size)

#Source's position in the grid
i_source = np.argmin(np.abs(x - 0.6))
j_source = np.argmin(np.abs(y - 1.2))
index_source = i_source * N + j_source

# Filling matrix M and vector b based on the diffusion equation ( cf report)
for i in range(N):
    for j in range(N):
        index = i * N + j
        r = np.sqrt(X[i, j]**2 + Y[i, j]**2)

        if r > radius - h / 2:  # Dirichlet boundary condition at the boundary
            M[index, index] = 1
            b[index] = 0
        
        elif index == index_source:  # Source term
            M[index, index] = 1
            b[index] = 1
        
        else:
            # Diffusion equation (Laplacian) with five-point stencil
            M[index, index] = -4 * D / h**2
            if i > 0:
                M[index, index - N] = D / h**2
            if i < N - 1:
                M[index, index + N] = D / h**2
            if j > 0:
                M[index, index - 1] = D / h**2
            if j < N - 1:
                M[index, index + 1] = D / h**2

# Convert M to CSR format for efficiency 
M = M.tocsr()

# Initial guess for c (zero everywhere)
c = np.zeros(size)

# Iterative solver using explicit Euler updates
for step in range(max_steps):
    c_new = c + dt * (M @ c + b)  
    
    # Convergence check
    diff = np.linalg.norm(c_new - c) / np.linalg.norm(c)
    if diff < tolerance:
        print(f"Converged after {step + 1} steps with diff = {diff:.2e}")
        break
    
    c = c_new.copy()

else:
    print("Max steps reached without convergence.")

# Reshape solution back to 2D grid
C = c.reshape((N, N))

#Plot
plt.figure(figsize=(6, 5))
plt.imshow(C, extent=[-radius, radius, -radius, radius], origin="lower", cmap="viridis")
plt.colorbar(label="Concentration")
plt.title("Steady-State Concentration in a Disk")
plt.xlabel("x")
plt.ylabel("y")

# Save the figure as a PGF file for the overleaf doc
plt.savefig("steady_state_concentrationfinalresult.pgf")

plt.show()
