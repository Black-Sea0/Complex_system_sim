
# This code ties everything together.
# Run this file to run entire simulation pipeline and visualisation.
# You may look into other files for 'TODO' comments to see what to work on

# NOTE: download required packages: pip install -r requirements.txt


from algorithm import ComplexOptimizer
import matplotlib.pyplot as plt
import numpy as np

alg = ComplexOptimizer(N=100, S=100, A=16, p=0.8, r=6, t=0)
alg.generate_initial_state()
multi_run_data = alg.run_multiple_simulations(num_runs=2, timesteps=20)

for run_data in multi_run_data:
    run_data_avg = np.average(run_data, axis=1)
    plt.plot(run_data_avg, linewidth=1)

plt.legend()
plt.show()