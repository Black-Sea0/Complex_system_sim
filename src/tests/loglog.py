import sys
import os
from pathlib import Path
import click

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithm import run_simulation
import numpy as np
import csv

file_directory = Path(__file__).parent
data_directory = file_directory.parent.parent / 'data' 

@click.command()
@click.option('-p', default=0.67, help='copy-collaborate ratio')
@click.option('-t', default=0, help='turnover ratio')
@click.option('--time_steps', default=20, help='number of (time) steps/iterations taken per simulation')
@click.option('--output', required=True, help='name of output file, stored in data folder')
def main(p, t, time_steps, output):
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