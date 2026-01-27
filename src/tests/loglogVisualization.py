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

@click.command()
@click.option('--input', required=True, help='name of input data file, stored in data folder (ex: grid_search.npy)')
@click.option('--fignum', default=0, help='give an int value to so you do not overwrite other figures')
def main(input, fignum):

    avg_data = np.load(data_directory / input)

    plt.figure(figsize=(6, 4))
    plt.loglog(np.arange(avg_data.shape[0]), avg_data, marker="o")
    plt.xlabel("Timestep (log)")
    plt.ylabel("Average Payoff (log)")
    plt.title("Average Payoff vs Time (Log–Log)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{results_directory}/avg_vs_time_loglog_{fignum}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()