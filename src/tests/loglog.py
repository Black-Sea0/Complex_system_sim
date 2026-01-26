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

def main(N = 100, S = 100, A = 16, p = 0.67, r = 6, t = 0.07, timesteps = 101):
    csv_path = "avg_vs_time.csv"

    # Reset CSV but keep header
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "avg_payoff"])

    # Now run the simulation
    run_simulation(N, S, A, p, r, t, timesteps, save_to_csv=True)
        

if __name__ == "__main__":
    main()