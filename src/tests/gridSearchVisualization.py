import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from pathlib import Path
import sys
import click

file_directory = Path(__file__).parent
data_directory = file_directory.parent.parent / 'results'

# Run this file to visualize the results from a gridSearchParallel simulation
# Set the p_value and t_value range as used in the simulation, then run this file using for example:
# 'python gridSearchVisualization.py grid_search_results_1000_samples_high_res.npy'

@click.command()
@click.option('--input', required=True, help='name of input data file, stored in data folder (ex: grid_search.npy)')
def main(input):
    simulation_results = np.load(data_directory / input)
    n_p, n_t, n_samples = simulation_results.shape
    p_values = np.linspace(0, 1, n_p)
    t_values = np.linspace(0, 1, n_t)
    Q = 25
    block_size = n_samples // Q

    p_optimal_blocks = np.zeros(shape=(len(t_values), Q))
    for t_idx in range(len(t_values)):
        for block_idx in range(Q):
            start_sample = block_idx * block_size
            end_sample = (block_idx + 1) * block_size
            
            block_data = simulation_results[:, t_idx, start_sample:end_sample]
            block_avg = np.mean(block_data, axis=1)  # shape: (n_p_values,)
            
            optimal_p_idx = np.argmax(block_avg)
            p_optimal_blocks[t_idx, block_idx] = p_values[optimal_p_idx]
            
    for t_index in range(n_t):
        plt.scatter(np.full(Q, t_values[t_index]), p_optimal_blocks[t_index], alpha=0.1)
        plt.scatter(t_values[t_index], np.average(p_optimal_blocks[t_index]), color='red')
    
    plt.xlabel('turnover ratio')
    plt.ylabel('optimal copy-collaborate ratio')
    plt.title(f'optimal copy-collaborate per block of {block_size} samples')
    plt.show()

    # bootstrap: calculate slope of many random picks of points
    n_bootstrap = 10000
    slopes = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        block_indices = np.random.randint(Q, size=n_t)
        y_bootstrap = p_optimal_blocks[np.arange(n_t), block_indices]

        slope, _ = np.polyfit(t_values, y_bootstrap, 1)
        slopes[b] = slope

    plt.hist(slopes, bins=50, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel('p_opt - t slope')
    plt.title('Distribution of p_opt - t slope')
    plt.show()

    p_value_negative = np.mean(slopes >= 0)  # proportion of slopes ≥ 0 under null
    print(f"p-value for slope < 0: {p_value_negative}")

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