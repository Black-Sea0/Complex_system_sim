
# This code ties everything together.
# Run this file to run entire simulation pipeline and visualisation.
# You may look into other files for 'TODO' comments to see what to work on

# NOTE: download required packages: pip install -r requirements.txt


from algorithm import ComplexOptimizer
from landscape import mason_watts_landscape

import matplotlib.pyplot as plt
import numpy as np
import config

for p in config.p_list:
    board = mason_watts_landscape(config.N)
    alg = ComplexOptimizer(
        board=board,
        N=config.N,
        S=config.S,
        A=config.A,
        p=p,
        r=config.r,
        t=config.t,
    )
    multi_run_data = alg.run_multiple_simulations(num_runs=2, timesteps=20)

    for i, run_data in enumerate(multi_run_data):
        run_data_avg = np.average(run_data, axis=1)
        plt.plot(run_data_avg, linewidth=1, label=f"Run {i+1}")

    plt.show()