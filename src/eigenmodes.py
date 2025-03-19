import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import eigh, eig
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh
import time

class EigenvalueProblem:
    def __init__(self, shape, dimensions, n=30):
        self.shape = shape
        self.dimensions = dimensions
        self.n = n
        self.matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.mask = None

    def create_matrix(self):
        n = self.n
        if self.shape in ['square', 'circle']:
            L = self.dimensions[0]
        elif self.shape == 'rectangle':
            L = self.dimensions[0]
            W = self.dimensions[1]
        else:
            raise ValueError("Shape must be 'square', 'rectangle', or 'circle'")

        h = L / (n + 1)

        diagonals = [2*np.ones(n), -np.ones(n-1), -np.ones(n-1)]
        lap_1d = diags(diagonals, [0, -1, 1], format='csr')

        if self.shape == 'square':
            I = eye(n, format='csr')
            self.matrix = kron(I, lap_1d) + kron(lap_1d, I)
            self.mask = np.ones(n*n)

        elif self.shape == 'rectangle':
            scale_y = (L / W)**2
            I = eye(n, format='csr')
            self.matrix = kron(I, lap_1d) + scale_y * kron(lap_1d, I)
            self.mask = np.ones(n*n)

        elif self.shape == 'circle':
            I = eye(n, format='csr')
            full_lap = kron(I, lap_1d) + kron(lap_1d, I)

            x, y = np.linspace(0, L, n), np.linspace(0, L, n)
            X, Y = np.meshgrid(x, y)
            mask_2d = ((X - L/2)**2 + (Y - L/2)**2 <= (L/2)**2)
            self.mask = mask_2d.flatten()

            for i in range(full_lap.shape[0]):
                if not self.mask[i]:
                    full_lap[i, i] = 1e10

            self.matrix = full_lap

        self.matrix /= h**2

    def solve_eigen_problem(self, solver='eigh', k=10):
        self.create_matrix()
        start_time = time.time()

        if solver == 'eigh':
            eigvals, eigvecs = eigh(self.matrix.toarray())
        elif solver == 'eig':
            eigvals, eigvecs = eig(self.matrix.toarray())
            eigvals, eigvecs = eigvals.real, eigvecs.real
        elif solver == 'eigsh':
            eigvals, eigvecs = eigsh(self.matrix, k=k, which='SM')
        else:
            raise ValueError("Solver must be 'eigh', 'eig', or 'eigsh'")

        idx = np.argsort(eigvals)
        self.eigenvalues = eigvals[idx]
        self.eigenvectors = eigvecs[:, idx]

        elapsed_time = time.time() - start_time
        print(f"Solver: {solver}, Time taken: {elapsed_time:.4f} s, First eigenvalue: {self.eigenvalues[0]:.4f}")

        return self.eigenvalues, self.eigenvectors, elapsed_time

    def plot_eigenvectors(self, num_modes=6):
        if self.eigenvectors is None:
            raise RuntimeError("Solve eigenproblem before plotting.")

        n = self.n
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()

        for i in range(num_modes):
            mode = self.eigenvectors[:, i].reshape(n, n)

            if self.shape == 'circle':
                mask = self.mask.reshape(n, n)
                mode = np.ma.masked_array(mode, mask=~mask)

            freq = np.sqrt(np.abs(self.eigenvalues[i])) / (2*np.pi)
            im = axs[i].imshow(mode, cmap='viridis', origin='lower')
            axs[i].set_title(f'Mode {i+1}, λ={self.eigenvalues[i]:.2f}, f={freq:.2f} Hz')
            fig.colorbar(im, ax=axs[i])

        plt.suptitle(f"Eigenmodes: {self.shape.capitalize()}, Dimension: {self.dimensions}")
        plt.tight_layout()
        plt.show()

    def animate_eigenmode(self, mode_number=0, frames=100, interval=50):
        n = self.n
        mode = self.eigenvectors[:, mode_number].reshape(n, n)
        frequency = np.sqrt(self.eigenvalues[mode_number])

        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.imshow(mode, cmap='viridis', origin='lower', animated=True)
        fig.colorbar(cax)

        def update(t):
            u_t = mode * np.cos(frequency * t)
            cax.set_array(u_t)
            ax.set_title(f"Mode {mode_number+1}, t={t:.2f}s")
            return [cax]

        ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, frames), interval=interval, blit=True)
        plt.show()
