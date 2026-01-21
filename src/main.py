
# This code ties everything together.
# Run this file to run entire simulation pipeline and visualisation.
# You may look into other files for 'TODO' comments to see what to work on

# NOTE: download required packages: pip install -r requirements.txt


from algorithm import ComplexOptimizer
import matplotlib.pyplot as plt
import numpy as np
from config import *
from statistical import combine_fitness_metrics, clear_data_fitness


for p in p_list:
    alg = ComplexOptimizer(N=N, S=S, A=A, p=p, r=r, t=0)
    multi_run_data = alg.run_multiple_simulations(num_runs=N_runs, timesteps=N_steps)

for run_data in multi_run_data:
    run_data_avg = np.average(run_data, axis=1)
    plt.plot(run_data_avg, linewidth=1)


for p in p_list:
    combine_fitness_metrics(prefix=f"fitness_metrics_p_{p}", output_file_name=f"averaged_fitness_metrics_p_{p}.csv")


plt.legend()
plt.show()