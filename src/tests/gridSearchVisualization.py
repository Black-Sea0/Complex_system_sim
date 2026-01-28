import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from pathlib import Path
import sys
import click

file_directory = Path(__file__).parent
data_directory = file_directory.parent.parent / 'data'
results_directory = file_directory.parent.parent / 'results'
import os
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Run this file to visualize the results from a gridSearchParallel simulation
# Set the p_value and t_value range as used in the simulation, then run this file using for example:
# python gridSearchVisualization.py grid_search_results_1000_samples_high_res_20_steps.npy

@click.command()
@click.option('--input', required=True, help='name of input data file, stored in data folder (ex: grid_search.npy)')
@click.option('--fignum', default=0, help='give an int value to so you do not overwrite other figures')
def main(input, fignum):
    """
    This function gets access to the data folder parse through it to make plots for visualizing the results.

    Parameters
    ----------
    input : str
        Name of the input data file that is stored in the data folder.
    fignum : int
        A number value that is added at the end of the names of the figures to make sure previous ones are not 
        overwritten if the user does not want that.
    """
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
    
    # plot 1: optimal copy-collaborate ratio for multiple blocks of samples, and for varying turnover ratio's
    for t_index in range(n_t):
        plt.scatter(np.full(Q, t_values[t_index]), p_optimal_blocks[t_index], alpha=0.1)
        plt.scatter(t_values[t_index], np.average(p_optimal_blocks[t_index]), color='red')
    
    plt.xlabel('turnover ratio')
    plt.ylabel('optimal copy-collaborate ratio')
    plt.title(f'optimal copy-collaborate per block of {block_size} samples')
    plt.savefig(f'{results_directory}/opt_p_vs_t{fignum}.png')
    plt.show()

    # plot 1b: optimal copy-collaborate ratio for multiple blocks of samples, and for varying turnover ratio's
    for t_index in range(n_t):
        plt.scatter(t_values[t_index], np.average(p_optimal_blocks[t_index]), color='red')
    
    plt.xlabel('turnover ratio')
    plt.ylabel('optimal copy-collaborate ratio')
    plt.title(f'optimal copy-collaborate {n_samples} samples')
    plt.savefig(f'{results_directory}/opt_p_vs_t_singular_{fignum}.png')
    plt.show()

    # bootstrap: calculate slope of many random picks of points
    n_bootstrap = 10000
    slopes = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        block_indices = np.random.randint(Q, size=n_t)
        y_bootstrap = p_optimal_blocks[np.arange(n_t), block_indices]

        slope, _ = np.polyfit(t_values, y_bootstrap, 1)
        slopes[b] = slope

    # plot 2: histogram of the slope fitting to random optimal-p t points
    plt.hist(slopes, bins=50, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel('p_opt - t slope')
    plt.title('Distribution of p_opt - t slope')
    plt.savefig(f'{results_directory}/dist_opt_p{fignum}.png')
    plt.show()


    # ==================
    single_avg_results = np.average(simulation_results, axis=2)

    optimal_p_indices = np.argmax(single_avg_results, axis=0)
    optimal_p = p_values[optimal_p_indices]
    optimal_values = single_avg_results[optimal_p_indices, np.arange(len(t_values))]

    linregress_output = stats.linregress(t_values, optimal_p, alternative='less')
    print("less: ", linregress_output.pvalue)
    linregress_output = stats.linregress(t_values, optimal_p, alternative='two-sided')
    print("two-sided: ", linregress_output.pvalue)

    n_bootstrap = 10000
    slopes = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        t_bootstrap = np.random.permutation(t_values)

        slope, _ = np.polyfit(t_bootstrap, optimal_p, 1)
        slopes[b] = slope

    # plot 3: histogram of the slope fitting to random optimal-p t points
    plt.hist(slopes, bins=50, edgecolor='black')
    plt.axvline(x=linregress_output.slope, color='red', linestyle='--')
    plt.xlabel('slope')
    plt.title('Permutation of data points')
    plt.savefig(f'{results_directory}/hist_permutation_{fignum}.png')
    plt.show()

    # Plot 3: average fitness across the whole p-t range
    P, T = np.meshgrid(t_values, p_values)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(P, T, single_avg_results, cmap='viridis', alpha=0.8, linewidth=0.5, edgecolor='k')
    ax.contour(P, T, single_avg_results, zdir='z', offset=single_avg_results.min() - 0.1, cmap='viridis', alpha=0.5)

    ax.set_xlabel('t')
    ax.set_ylabel('p')
    ax.set_zlabel('Average Performance at t=20')

    fig.colorbar(surf, ax=ax)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{results_directory}/avg_perform_t20{fignum}.png')
    plt.show()


    # Plot 4: average fitness - copy-collaborate ratio curve for varying turnover rates, as a 2D plot
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    norm = plt.Normalize(t_values.min(), t_values.max())

    for i, t in enumerate(t_values):
        color = plt.cm.viridis(norm(t))

        ax2.plot(p_values, single_avg_results[:, i], color=color, alpha=0.5, linewidth=1.5, label=f't={t:.2f}')
        opt_index = np.argmax(single_avg_results[:, i])
        ax2.scatter(p_values[opt_index], single_avg_results[opt_index, i], color=color, s=80, marker='X', alpha=0.8, linewidth=2)
        print(f't={t:.2f}, optimal p={p_values[opt_index]:.2f}, performance={single_avg_results[opt_index, i]:.4f}')

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = fig2.colorbar(sm, ax=ax2)
    cbar.set_label('turnover percentage', fontsize=12)

    ax2.set_xlabel('copy-collaborate ratio')
    ax2.set_ylabel('Average Performance')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{results_directory}/avg_perform{fignum}.png')
    plt.show()

if __name__ == "__main__":
    main()

