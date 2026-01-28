import sys
import os
from pathlib import Path
import click

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithm import run_simulation
import numpy as np
import csv
from pathlib import Path

file_directory = Path(__file__).parent
data_directory = file_directory.parent.parent / 'data' 

@click.command()
@click.option('-p', default=0.67, help='copy-collaborate ratio')
@click.option('-t', default=0, help='turnover ratio')
@click.option('--time_steps', default=20, help='number of (time) steps/iterations taken per simulation')
@click.option('--output', required=True, help='name of output file, stored in data folder')
def main(p, t, time_steps, output):
    """
    This function does one run of the simulation with the copy-collaborate ratio and turnover rate and saves the data
    into a .nyp file to be plotted later.

    Parameters
    ----------
    p : float
        Copy-Collaborate ratio.
    t : float
        Turnover rate.
    time_steps: int
        Number of (time) steps/iterations taken per simulation.
    output: string
        Name of output file, stored in data folder.
    """
    assert p >= 0, f"p is expected to be a float between [0,1], got: {p}"
    assert t >= 0, f"t is expected to be a float between [0,1], got: {t}"
    assert time_steps > 0, f"time_steps is expected to be an int above 0, got: {time_steps}"
    fitness_history = run_simulation(
        N=100,
        S=100,
        A=16,
        p=p,
        r=6,
        t=t,
        timesteps=time_steps
    )
    avg_fitness_history = np.average(fitness_history, axis=1)

    # storage shouldn't really be necessary, since this simulation is very short
    np.save(data_directory / output, avg_fitness_history)

if __name__ == "__main__":
    main()
