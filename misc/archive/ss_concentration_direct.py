
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

#Our parameters
radius = 2.0
h = 0.1   # Spatial step
D = 1.0   # Diffusion coefficient
N = int(2 * radius / h) + 1   # Number of grid points in each direction

# Create the grid
x = np.linspace(-radius, radius, N)
y = np.linspace(-radius, radius, N)
X, Y = np.meshgrid(x, y)

# Initialize the matrix M and vector b
size = N * N
M = sp.lil_matrix((size, size), dtype=float)
b = np.zeros(size)

# Source position
i_source = np.argmin(np.abs(x - 0.6))
j_source = np.argmin(np.abs(y - 1.2))
index_source = i_source * N + j_source

# Filling the matrix M and vector b (explained in the report)
for i in range(N):
    for j in range(N):
        index = i * N + j
        r = np.sqrt(X[i, j]**2 + Y[i, j]**2)

        if r > radius - h / 2:
            # Dirichlet boundary condition at the edge of the disk
            M[index, index] = 1
            b[index] = 0
        elif index == index_source:
            # Point source at (0.6, 1.2) with concentration set to 10
            M[index, index] = 1
            b[index] = 10
        else:
            # Discretized Laplacian operator
            M[index, index] = -4 * D / h**2
            if i > 0:
                M[index, index - N] = D / h**2
            if i < N - 1:
                M[index, index + N] = D / h**2
            if j > 0:
                M[index, index - 1] = D / h**2
            if j < N - 1:
                M[index, index + 1] = D / h**2

# Convert matrix to CSR format (for faster solving) ( seen in lecture)
M = M.tocsr()

# Solving the system directly
try:
    c = spla.spsolve(M, b)
    if np.all(np.isnan(c)):
        print("Solution failed.")
    else:
        print(" Solution successful!")
except Exception as e:
    print(f" Error during solution: {e}")
    exit()

C = c.reshape((N, N))

# Mask points outside the disk to only have the circle domain

C = np.ma.masked_where(np.sqrt(X**2 + Y**2) > radius - h / 2, C)

C_normalized = (C - np.min(C)) / (np.max(C) - np.min(C))

# Plotting the solution
plt.figure(figsize=(6, 5))
plt.imshow(C_normalized.T, extent=[-radius, radius, -radius, radius], origin="lower", cmap="viridis")
plt.colorbar(label="Concentration")
plt.title("Steady-State Concentration in a Disk")
plt.xlabel("x")
plt.ylabel("y")

# Saving the figure pgf for overleaf
plt.savefig("steady_state_concentration.pgf")
plt.show()

