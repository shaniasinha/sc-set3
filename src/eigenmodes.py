import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import eigh, eig
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh, eigs
import time
import os
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

class EigenvalueProblem:
    def __init__(self, shape, dimensions, n=30):
        self.shape = shape
        self.dimensions = dimensions
        self.n = n
        self.matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.mask = None
        self.simulated_grid = None
        self.solver = None

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

        # row zeroing approach (2)
        elif self.shape == 'circle':
            I = eye(n, format='csr')
            full_lap = kron(I, lap_1d) + kron(lap_1d, I)

            x, y = np.linspace(0, L, n), np.linspace(0, L, n)
            X, Y = np.meshgrid(x, y)
            mask_2d = ((X - L/2)**2 + (Y - L/2)**2 <= (L/2)**2)
            self.mask = mask_2d.flatten()

            for i in range(full_lap.shape[0]):
                if not self.mask[i]:
                    full_lap[i, :] = 0
                    full_lap[i, i] = 1

            self.matrix = full_lap

        self.matrix /= h**2

    def solve_eigen_problem(self, solver='eigh', k=9):
        self.create_matrix()
        start_time = time.time()
        self.solver = solver

        # Ensure correct matrix format
        if solver == 'eigh':
            eigvals, eigvecs = eigh(self.matrix.toarray())  # Dense symmetric solver
        elif solver == 'eig':
            eigvals, eigvecs = eig(self.matrix.toarray())  # General solver
            # if np.iscomplexobj(eigvecs):
            #     assert "complex eigenvalues!"
        elif solver == 'eigs':
            eigvals, eigvecs = eigs(self.matrix.asformat('csr'), k=min(k, self.matrix.shape[0]-1), which='SM')
        elif solver == 'eigsh':
            eigvals, eigvecs = eigsh(self.matrix.asformat('csr'), k=min(k, self.matrix.shape[0]-1), which='SM')
        else:
            raise ValueError("Solver must be 'eigh', 'eig', 'eigs' or 'eigsh'")

        # Sorting (eigsh already sorts)
        if solver in ['eigh', 'eig']:
            idx = np.argsort(eigvals)
            eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        # Normalize eigenvectors for consistency across solvers
        # eigvecs /= np.linalg.norm(eigvecs, axis=0)

        elapsed_time = time.time() - start_time
        print(f"Solver: {solver}, Time taken: {elapsed_time:.4f} s, First eigenvalue: {eigvals[0]:.4f}")

        # Store results
        self.eigenvalues, self.eigenvectors = eigvals, eigvecs
        return self.eigenvalues, self.eigenvectors, elapsed_time

    def plot_eigenvectors(self, num_modes=9):
        if self.eigenvectors is None:
            raise RuntimeError("Solve eigenproblem before plotting.")

        n = self.n
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        axs = axs.flatten()

        # Filter for real and positive eigenvalues
        valid_indices = np.isreal(self.eigenvalues) & (self.eigenvalues > 0)
        real_eigenvalues = self.eigenvalues[valid_indices]
        real_eigenvectors = self.eigenvectors[:, valid_indices]
        
        # Make sure we have enough modes to plot
        actual_modes = min(num_modes, real_eigenvalues.shape[0])
        if actual_modes < num_modes:
            print(f"Warning: Only {actual_modes} valid eigenvalues available.")
        
        # Use only valid eigenvalues/vectors
        plot_eigenvalues = real_eigenvalues[:actual_modes]
        plot_eigenvectors = real_eigenvectors[:, :actual_modes]

        # Compute global min and max for color normalization
        global_min = np.min(np.real(plot_eigenvectors))
        global_max = np.max(np.real(plot_eigenvectors))

        # Create a single colorbar axis
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

        for i in range(actual_modes):
            mode = np.real(plot_eigenvectors[:, i].reshape(n, n))

            if self.shape == 'circle':
                mask = self.mask.reshape(n, n)
                mode = np.ma.masked_array(mode, mask=~mask)
                
            freq = np.sqrt(np.abs(plot_eigenvalues[i])) / (2 * np.pi)
            
            # Fix color scale across all modes
            im = axs[i].imshow(mode, cmap='Spectral', origin='lower', vmin=global_min, vmax=global_max)
            # Use np.real to display only the real part of the eigenvalue
            axs[i].set_title(f'Mode {i+1}, λ={np.real(plot_eigenvalues[i]):.2f}, f={freq:.2f} Hz', fontsize=14)
            axs[i].axis('off')
            
        # Hide unused subplots
        for i in range(actual_modes, len(axs)):
            axs[i].axis('off')

        # Add a single colorbar to the right of the plot
        fig.colorbar(im, cax=cbar_ax, orientation='vertical')

        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make space for colorbar

        # Ensure the directory exists before saving
        save_dir = "results/membrane_eigenmodes/"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{self.shape}_{self.dimensions}_{self.solver}.png"))
        plt.show()

    def animate_eigenmodes(self, total_modes=9, frames=100, interval=100, save_as="animation"):
        if not save_as.endswith(('.gif', '.mp4')):
            save_as += ".gif"  

        n = self.n

        fig, axs = plt.subplots(3, 3, figsize=(12, 12))
        axs = axs.flatten()

        modes = []
        frequencies = []
        images = []

        # Filter for real and positive eigenvalues
        real_indices = np.isreal(self.eigenvalues) & (self.eigenvalues > 0)
        real_eigenvalues = self.eigenvalues[real_indices]
        real_eigenvectors = self.eigenvectors[:, real_indices]
        
        # Further filter out eigenvalues that are too small (non-physical modes)
        nonzero_indices = real_eigenvalues > 1e-6
        filtered_eigenvalues = real_eigenvalues[nonzero_indices]
        filtered_eigenvectors = real_eigenvectors[:, nonzero_indices]

        for mode_number in range(min(total_modes, filtered_eigenvalues.shape[0])):
            mode = np.real(filtered_eigenvectors[:, mode_number].reshape(n, n))
            frequency = np.sqrt(np.abs(filtered_eigenvalues[mode_number]))

            if self.shape == 'circle':
                mask = self.mask.reshape(n, n)
                mode = np.ma.masked_array(mode, mask=~mask)
            
            # Normalize mode for consistent visualization amplitude
            mode = mode / np.max(np.abs(mode))
                
            modes.append(mode)
            frequencies.append(frequency)

            cax = axs[mode_number].imshow(mode, cmap='Spectral', origin='lower', 
                                        vmin=-1, vmax=1)
            images.append(cax)
            
            axs[mode_number].set_title(f'Mode {mode_number+1}, λ={np.real(filtered_eigenvalues[mode_number]):.2f}')
            axs[mode_number].axis('off')

        plt.tight_layout()
        
        # Scale down the frequencies to make visualization slower
        frequencies = [freq/3 for freq in frequencies]
        
        def update(t):
            for mode_number in range(len(modes)):
                u_t = modes[mode_number] * np.cos(frequencies[mode_number] * t)
                images[mode_number].set_array(u_t)
            return images

        save_dir = "results/membrane_eigenmodes/"
        os.makedirs(save_dir, exist_ok=True)

        ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, frames), interval=interval, blit=True)
        save_path = os.path.join(save_dir, save_as)
        ani.save(save_path, writer=PillowWriter(fps=15))  # Reduced fps for slower playback
        print(f"Animation successfully saved as {save_path}")




