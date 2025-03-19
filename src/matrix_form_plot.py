import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_matrix_form():
    # Define the matrix M
    M = np.array([
        [-4,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 1, -4,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  1, -4,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  1, -4,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0, -4,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  1, -4,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  1, -4,  1,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  1, -4,  1,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0, -4,  1,  0,  1,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  1, -4,  1,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -4,  1,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -4,  1,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -4,  1,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -4,  1,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -4,  1],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -4]
    ])

    # Create heatmap
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(M, 
                cmap="viridis", 
                annot=True, 
                fmt=".0f", 
                linewidths=0.5, 
                cbar=True,
                xticklabels=False,
                yticklabels=False)
    # plt.title("Heatmap of Matrix M")
    plt.savefig("results/matrix_form_plot.png", bbox_inches="tight")
    plt.show()
