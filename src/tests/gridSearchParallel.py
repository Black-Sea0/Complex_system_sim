import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithm import run_multiple_simulations
import numpy as np
from multiprocessing import Pool

def run_simulation_wrapper(args):
    p, t, x, y = args

    start_time = time.time()

    multi_run_data = run_multiple_simulations(
        N=100,
        S=100,
        A=16,
        p=p,
        r=6,
        t=t,
        num_runs=1000,
        timesteps=60
    )
    
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
    t_values = np.linspace(0, 1, 31)
    
    simulation_results = np.zeros(shape=(len(p_values), len(t_values)))
    params_list = []
    for x, p in enumerate(p_values):
        for y, t in enumerate(t_values):
            params_list.append((p, t, x, y))
    
    print(f"Running {len(params_list)} simulations in parallel...")
    
    with Pool(processes=75, maxtasksperchild=1) as pool: # hard-coded number of threads to use
        results = pool.map(run_simulation_wrapper, params_list)
    
    elapsed_time = time.time() - start_time
    print(f"Simulations finished in {elapsed_time:.1f} seconds")

    for x, y, value in results:
        simulation_results[x, y] = value
    
    print("Saving results!")
    np.save("grid_search_results.npy", simulation_results)

if __name__ == "__main__":
    main()