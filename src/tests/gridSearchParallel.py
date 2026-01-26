import click
import sys
import os
import time
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithm import run_multiple_simulations
import numpy as np
from multiprocessing import Pool

file_directory = Path(__file__).parent
data_directory = file_directory.parent.parent / 'Data'

def run_simulation_wrapper(args):
    p, t, n_samples, x, y = args

    start_time = time.time()

    multi_run_data = run_multiple_simulations(
        N=100,
        S=100,
        A=16,
        p=p,
        r=6,
        t=t,
        num_runs=n_samples,
        timesteps=20
    )
    
    run_avgs_at_end = []
    for run_data in multi_run_data:
        run_avgs_at_end.append(np.average(run_data, axis=1)[-1])
    
    run_avgs_at_end = np.array(run_avgs_at_end)
    
    elapsed_time = time.time() - start_time
    
    print(f"Completed: p={p:.2f}, t={t:.2f} in {elapsed_time:.1f} seconds")
    
    return (x, y, np.average(run_avgs_at_end))

@click.command()
@click.option('--num_threads', default=16, help='max number of threads to use for parallel simulations')
@click.option('--p_steps', default=11, help='number of divisions of copy-collaborate ratio range [0, 1]')
@click.option('--t_steps', default=11, help='number of divisions of turnover ratio range [0, 1]')
@click.option('--n_samples', default=1000, help='number of simulations per parameter combination')
@click.option('--output', required=True, help='name of output file, stored in data folder')
def main(num_threads, p_steps, t_steps, n_samples, output):
    start_time = time.time()

    p_values = np.linspace(0, 1, p_steps)
    t_values = np.linspace(0, 1, t_steps)
    
    simulation_results = np.zeros(shape=(len(p_values), len(t_values)))
    params_list = []
    for x, p in enumerate(p_values):
        for y, t in enumerate(t_values):
            params_list.append((p, t, n_samples, x, y))
    
    print(f"Running {len(params_list)} simulations in parallel...")
    
    with Pool(processes=num_threads, maxtasksperchild=1) as pool:
        results = pool.map(run_simulation_wrapper, params_list)
    
    elapsed_time = time.time() - start_time
    print(f"Simulations finished in {elapsed_time:.1f} seconds")

    for x, y, value in results:
        simulation_results[x, y] = value
    
    print("Saving results!")
    np.save(data_directory / output, simulation_results)

if __name__ == "__main__":
    main()