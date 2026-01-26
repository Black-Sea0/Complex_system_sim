import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithm import run_simulation
import matplotlib.pyplot as plt
import numpy as np
import gc
from multiprocessing import Pool

def main(N = 100, S = 100, A = 16, p = 0.67, r = 6, t = 0.07, timesteps = 101):
    open("avg_vs_time.csv", "w").close()
    run_simulation(N, S, A, p, r, t, timesteps, save_to_csv=False)
    

if __name__ == "__main__":
    main()