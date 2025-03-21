import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import eigh, eig
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh, eigs
import time
import os
from matplotlib.animation import FuncAnimation, PillowWriter


class EigenvalueProblem:
    """
    A class to solve eigenvalue problems for different 2D shapes (square, rectangle, circle)
    using finite difference methods. It supports solving, visualizing, and animating eigenmodes.

    Attributes:
        shape (str): The shape of the domain ('square', 'rectangle', or 'circle').
        dimensions (tuple): Dimensions of the domain (e.g., (L,) for square, (L, W) for rectangle).
        n (int): Number of grid points along one dimension.
        matrix (scipy.sparse.csr_matrix): The finite difference matrix for the problem.
        eigenvalues (np.ndarray): Computed eigenvalues.
        eigenvectors (np.ndarray): Computed eigenvectors.
        mask (np.ndarray): Mask for the domain (used for circular domains).
        simulated_grid (np.ndarray): Grid representation of the domain.
        solver (str): The solver used for the eigenvalue problem.
    """

    def __init__(self, shape, dimensions, n=30):
        """
        Initialize the EigenvalueProblem class.

        Args:
            shape (str): The shape of the domain ('square', 'rectangle', or 'circle').
            dimensions (tuple): Dimensions of the domain (e.g., (L,) for square, (L, W) for rectangle).
            n (int): Number of grid points along one dimension.
        """
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
        """
        Create the finite difference matrix for the eigenvalue problem based on the shape and dimensions.
        """
        n = self.n
        if self.shape in ['square', 'circle']:
            L = self.dimensions[0]
        elif self.shape == 'rectangle':
            L = self.dimensions[0]
            W = self.dimensions[1]
        else:
            raise ValueError("Shape must be 'square', 'rectangle', or 'circle'")

        h = L / (n + 1)

        # 1D Laplacian
        diagonals = [2 * np.ones(n), -np.ones(n - 1), -np.ones(n - 1)]
        lap_1d = diags(diagonals, [0, -1, 1], format='csr')

        if self.shape == 'square':
            I = eye(n, format='csr')
            self.matrix = kron(I, lap_1d) + kron(lap_1d, I)
            self.mask = np.ones(n * n)

        elif self.shape == 'rectangle':
            scale_y = (L / W) ** 2
            I = eye(n, format='csr')
            self.matrix = kron(I, lap_1d) + scale_y * kron(lap_1d, I)
            self.mask = np.ones(n * n)

        elif self.shape == 'circle':
            I = eye(n, format='csr')
            full_lap = kron(I, lap_1d) + kron(lap_1d, I)

            x, y = np.linspace(0, L, n), np.linspace(0, L, n)
            X, Y = np.meshgrid(x, y)
            mask_2d = ((X - L / 2) ** 2 + (Y - L / 2) ** 2 <= (L / 2) ** 2)
            self.mask = mask_2d.flatten()

            for i in range(full_lap.shape[0]):
                if not self.mask[i]:
                    full_lap[i, :] = 0
                    full_lap[i, i] = 1

            self.matrix = full_lap

        self.matrix /= h ** 2

    def solve_eigen_problem(self, solver='eigh', k=9):
        """
        Solve the eigenvalue problem using the specified solver.

        Args:
            solver (str): The solver to use ('eigh', 'eig', 'eigs', or 'eigsh').
            k (int): Number of eigenvalues to compute (for sparse solvers).

        Returns:
            tuple: Eigenvalues, eigenvectors, and elapsed time.
        """
        self.create_matrix()
        start_time = time.time()
        self.solver = solver

        if solver == 'eigh':
            eigvals, eigvecs = eigh(self.matrix.toarray())
        elif solver == 'eig':
            eigvals, eigvecs = eig(self.matrix.toarray())
        elif solver == 'eigs':
            eigvals, eigvecs = eigs(self.matrix.asformat('csr'), k=min(k, self.matrix.shape[0] - 1), which='SM')
        elif solver == 'eigsh':
            eigvals, eigvecs = eigsh(self.matrix.asformat('csr'), k=min(k, self.matrix.shape[0] - 1), which='SM')
        else:
            raise ValueError("Solver must be 'eigh', 'eig', 'eigs' or 'eigsh'")

        if solver in ['eigh', 'eig']:
            idx = np.argsort(eigvals)
            eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        elapsed_time = time.time() - start_time
        print(f"Solver: {solver}, Time taken: {elapsed_time:.4f} s, First eigenvalue: {eigvals[0]:.4f}")

        self.eigenvalues, self.eigenvectors = eigvals, eigvecs
        return self.eigenvalues, self.eigenvectors, elapsed_time

    def plot_eigenvectors(self, num_modes=6):
        """
        Plot the eigenvectors (modes) of the eigenvalue problem.

        Args:
            num_modes (int): Number of modes to plot.
        """
        if self.eigenvectors is None:
            raise RuntimeError("Solve eigenproblem before plotting.")

        n = self.n
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()

        valid_indices = np.isreal(self.eigenvalues) & (self.eigenvalues > 0)
        real_eigenvalues = self.eigenvalues[valid_indices]
        real_eigenvectors = self.eigenvectors[:, valid_indices]

        actual_modes = min(num_modes, real_eigenvalues.shape[0])
        if actual_modes < num_modes:
            print(f"Warning: Only {actual_modes} valid eigenvalues available.")

        plot_eigenvalues = real_eigenvalues[:actual_modes]
        plot_eigenvectors = real_eigenvectors[:, :actual_modes]

        global_min = np.min(np.real(plot_eigenvectors))
        global_max = np.max(np.real(plot_eigenvectors))

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])


        for i in range(actual_modes):
            mode = np.real(plot_eigenvectors[:, i].reshape(n, n))

            if self.shape == 'circle':
                mask = self.mask.reshape(n, n)
                mode = np.ma.masked_array(mode, mask=~mask)

            freq = np.sqrt(np.abs(plot_eigenvalues[i])) / (2 * np.pi)

            im = axs[i].imshow(mode, cmap='Spectral', origin='lower', vmin=global_min, vmax=global_max)
            axs[i].set_title(f'λ={np.real(plot_eigenvalues[i]):.2f}, f={freq:.2f} Hz', fontsize=18)
            axs[i].axis('off')

        for i in range(actual_modes, len(axs)):
            axs[i].axis('off')

        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        cbar.ax.tick_params(labelsize=18)

        plt.tight_layout(rect=[0, 0, 0.9, 1])

        save_dir = "results/membrane_eigenmodes/"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{self.shape}_{self.dimensions}_{self.solver}.png"))
        plt.show()

    def animate_eigenmodes(self, total_modes=6, frames=100, interval=100, save_as="animation"):
        """
        Animate the eigenmodes of the eigenvalue problem.

        Args:
            total_modes (int): Number of modes to animate.
            frames (int): Number of frames in the animation.
            interval (int): Interval between frames in milliseconds.
            save_as (str): Filename to save the animation (supports .gif or .mp4).
        """
        if not save_as.endswith(('.gif', '.mp4')):
            save_as += ".gif"

        n = self.n

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()

        modes = []
        frequencies = []
        images = []

        real_indices = np.isreal(self.eigenvalues) & (self.eigenvalues > 0)
        real_eigenvalues = self.eigenvalues[real_indices]
        real_eigenvectors = self.eigenvectors[:, real_indices]

        nonzero_indices = real_eigenvalues > 1e-6
        filtered_eigenvalues = real_eigenvalues[nonzero_indices]
        filtered_eigenvectors = real_eigenvectors[:, nonzero_indices]

        for mode_number in range(min(total_modes, filtered_eigenvalues.shape[0])):
            mode = np.real(filtered_eigenvectors[:, mode_number].reshape(n, n))
            frequency = np.sqrt(np.abs(filtered_eigenvalues[mode_number]))

            if self.shape == 'circle':
                mask = self.mask.reshape(n, n)
                mode = np.ma.masked_array(mode, mask=~mask)

            mode = mode / np.max(np.abs(mode))

            modes.append(mode)
            frequencies.append(frequency)

            cax = axs[mode_number].imshow(mode, cmap='Spectral', origin='lower', vmin=-1, vmax=1)
            images.append(cax)

            axs[mode_number].set_title(f'λ={np.real(filtered_eigenvalues[mode_number]):.2f}')
            axs[mode_number].axis('off')

        plt.tight_layout()

        frequencies = [freq / 3 for freq in frequencies]

        def update(t):
            for mode_number in range(len(modes)):
                u_t = modes[mode_number] * np.cos(frequencies[mode_number] * t)
                images[mode_number].set_array(u_t)
            return images

        save_dir = "results/membrane_eigenmodes/"
        os.makedirs(save_dir, exist_ok=True)

        ani = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, frames), interval=interval, blit=True)
        save_path = os.path.join(save_dir, save_as)
        ani.save(save_path, writer=PillowWriter(fps=15))
        print(f"Animation successfully saved as {save_path}")
