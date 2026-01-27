import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from algorithm import run_simulation
import matplotlib.pyplot as plt
import numpy as np
import gc
from multiprocessing import Pool
import csv
from pathlib import Path

file_directory = Path(__file__).parent
data_directory = file_directory.parent.parent / 'data'

def loglog_plot(N = 100, S = 100, A = 16, p = 0.67, r = 6, t = 0, timesteps = 101):
    csv_path = f"{data_directory}/avg_vs_time_{p}_{t}.csv"

    # Reset CSV but keep header
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "avg_payoff"])

    # Now run the simulation
    run_simulation(N, S, A, p, r, t, timesteps, csv_path, save_to_csv=True)
        