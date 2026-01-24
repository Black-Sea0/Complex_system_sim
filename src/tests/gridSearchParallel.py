import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithm import ComplexOptimizer
from landscape import mason_watts_landscape
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool

def run_simulation_wrapper(args):
    p, t, x, y = args

    start_time = time.time()
    
    alg = ComplexOptimizer(
        N=100,
        S=100,
        A=16,
        p=p,
        r=6,
        t=t,
    )
    
    multi_run_data = alg.run_multiple_simulations(num_runs=1000, timesteps=20)
    
    run_avgs_at_end = []
    for run_data in multi_run_data:
        run_avgs_at_end.append(np.average(run_data, axis=1)[-1])
    
    run_avgs_at_end = np.array(run_avgs_at_end)
    
    elapsed_time = time.time() - start_time
    print(f"Completed: p={p:.2f}, t={t:.2f} in {elapsed_time:.1f} seconds")
    
    return (x, y, np.average(run_avgs_at_end))


def main():
    start_time = time.time()

    p_values = np.linspace(0, 1, 31)
    t_values = np.linspace(0, 0.3, 11)
    
    simulation_results = np.zeros(shape=(len(p_values), len(t_values)))
    params_list = []
    for x, p in enumerate(p_values):
        for y, t in enumerate(t_values):
            params_list.append((p, t, x, y))
    
    print(f"Running {len(params_list)} simulations in parallel...")
    
    with Pool(processes=75) as pool: # hard-coded number of threads to use
        results = pool.map(run_simulation_wrapper, params_list)
    
    elapsed_time = time.time() - start_time
    print(f"Simulations finished in {elapsed_time:.1f} seconds")

    for x, y, value in results:
        simulation_results[x, y] = value
    
    print("Saving results!")
    np.save("grid_search_results.npy", simulation_results)

    P, T = np.meshgrid(t_values, p_values)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(P, T, simulation_results, cmap='viridis', alpha=0.8, linewidth=0.5, edgecolor='k')
    ax.contour(P, T, simulation_results, zdir='z', offset=simulation_results.min() - 0.1, cmap='viridis', alpha=0.5)

    ax.set_xlabel('t')
    ax.set_ylabel('p')
    ax.set_zlabel('Average Performance at t=500')

    fig.colorbar(surf, ax=ax)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()