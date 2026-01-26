import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from pathlib import Path
import sys
import click

file_directory = Path(__file__).parent
data_directory = file_directory.parent.parent / 'data'

# Run this file to visualize the results from a gridSearchParallel simulation
# Set the p_value and t_value range as used in the simulation, then run this file using for example:
# 'python gridSearchVisualization.py grid_search_results_1000_samples_high_res.npy'

@click.command()
@click.option('--input', required=True, help='name of input data file, stored in data folder (ex: grid_search.npy)')
def main(input):
    simulation_results = np.load(data_directory / input)

    p_values = np.linspace(0, 1, simulation_results.shape[0])
    t_values = np.linspace(0, 1, simulation_results.shape[1])

    optimal_p_indices = np.argmax(simulation_results, axis=0)
    optimal_p = p_values[optimal_p_indices]
    optimal_values = simulation_results[optimal_p_indices, np.arange(len(t_values))]

    lin_regression = stats.linregress(t_values, optimal_p)
    print("linear regression p-value: ", lin_regression.pvalue)

    # Plot 1
    P, T = np.meshgrid(t_values, p_values)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(P, T, simulation_results, cmap='viridis', alpha=0.8, linewidth=0.5, edgecolor='k')
    ax.plot(t_values, optimal_p, optimal_values, color='red', linewidth=3)
    ax.contour(P, T, simulation_results, zdir='z', offset=simulation_results.min() - 0.1, cmap='viridis', alpha=0.5)

    ax.set_xlabel('t')
    ax.set_ylabel('p')
    ax.set_zlabel('Average Performance at t=20')

    fig.colorbar(surf, ax=ax)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


    # Plot 2
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    norm = plt.Normalize(t_values.min(), t_values.max())

    for i, t in enumerate(t_values):
        color = plt.cm.viridis(norm(t))

        ax2.plot(p_values, simulation_results[:, i], color=color, alpha=0.5, linewidth=1.5, label=f't={t:.2f}')
        opt_index = optimal_p_indices[i]
        ax2.scatter(p_values[opt_index], simulation_results[opt_index, i], color=color, s=80, marker='X', alpha=0.8, linewidth=2)
        print(f't={t:.2f}, optimal p={p_values[opt_index]:.2f}, performance={simulation_results[opt_index, i]:.4f}')

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = fig2.colorbar(sm, ax=ax2)
    cbar.set_label('turnover percentage', fontsize=12)

    ax2.set_xlabel('copy-collaborate ratio')
    ax2.set_ylabel('Average Performance')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()