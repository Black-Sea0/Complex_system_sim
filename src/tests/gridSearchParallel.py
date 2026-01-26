import click
import sys
import os
import time
from pathlib import Path

#import functions from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithm import run_multiple_simulations
import numpy as np
from multiprocessing import Pool

#makes sure that the data is saved in the "data" folder
file_directory = Path(__file__).parent
data_directory = file_directory.parent.parent / 'data' 

def run_simulation_wrapper(args):
    """
    Wrapper function to run each parameter combination n_samples time.

    Parameters
    ----------
    args: tuple
        A tuple containing the parameters for the simulation:
        - p: float
            Copy-collaborate ratio.
        - t: float
            Turnover rate.
        - time_steps: int
            Number of time steps per simulation.
        - n_samples: int
            Number of simualations to run per parameter combination.
        - x: int 
            Index of p value in the results array.
        - y: int
            Index of t value in the results array.
    Returns
    -------
    tuple
        A tuple containing:
        - x: int
            Index of p value in the results array.
        - y: int
            Index of t value in the results array.
        - average fitness: float
            Average fitness at the end of the simulations for the given parameter combination.
    """
    p, t, time_steps, n_samples, x, y = args

    start_time = time.time()

    #run the multiple simulations for the given parameters
    multi_run_data = run_multiple_simulations(
        N=100,
        S=100,
        A=16,
        p=p,
        r=6,
        t=t,
        num_runs=n_samples,
        timesteps=time_steps
    )
    
    #calculate the average the fitness at the end of the run
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
@click.option('--time_steps', default=20, help='number of (time) steps/iterations taken per simulation')
@click.option('--n_samples', default=1000, help='number of simulations per parameter combination')
@click.option('--output', required=True, help='name of output file, stored in data folder')
def main(num_threads, p_steps, t_steps, time_steps, n_samples, output):
    """
    Run every t value agasint every p value to get all of the necessary values for gridsearch to find optimal
    p values against each t value. This function runs with parallelization for optimization purposes. The data
    gets saved in a folder called 'data'.

    Parameters
    ----------
    num_threads: int
        Max number of threads to use for parallel simulations
    p_steps: int
        Number of divisions of copy-collaborate ratio range [0, 1].
    t_steps: int
        Number of divisions of turnover ratio range [0, 1].
    time_steps: int
        Number of (time) steps/iterations taken per simulation.
    n_samples: int
        Number of simulations per parameter combination.
    output: string
        Name of output file, stored in data folder.
    """
    assert num_threads > 0, f"num_threads is expected to be an int above 0, got: {num_threads}"
    assert p_steps > 0, f"p_steps is expected to be an int above 0, got: {p_steps}"
    assert t_steps > 0, f"t_steps is expected to be an int above 0, got: {t_steps}"
    assert time_steps > 0, f"time_steps is expected to be an int above 0, got: {time_steps}"
    assert n_samples > 0, f"n_samples is expected to be an int above 0, got: {n_samples}"
    start_time = time.time()

    p_values = np.linspace(0, 1, p_steps)
    t_values = np.linspace(0, 1, t_steps)
    
    #set up and run the simulation with the given arguments
    simulation_results = np.zeros(shape=(len(p_values), len(t_values)))
    params_list = []
    for x, p in enumerate(p_values):
        for y, t in enumerate(t_values):
            params_list.append((p, t, time_steps, n_samples, x, y))
    
    print(f"Running {len(params_list)} simulations in parallel...")
    
    #assign each simulation an open thread
    with Pool(processes=num_threads, maxtasksperchild=1) as pool:
        results = pool.map(run_simulation_wrapper, params_list)
    
    #print out how long the simulations took
    elapsed_time = time.time() - start_time
    print(f"Simulations finished in {elapsed_time:.1f} seconds")

    #assign each result from the simulation to the correct position in the results array
    for x, y, value in results:
        simulation_results[x, y] = value
    
    #save the results to a .nyp file in the directory 'data'
    print("Saving results!")
    np.save(data_directory / output, simulation_results)

if __name__ == "__main__":
    main()