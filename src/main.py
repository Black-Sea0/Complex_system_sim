
# This code ties everything together.
# Run this file to run entire simulation pipeline and visualisation.
# You may look into other files for 'TODO' comments to see what to work on

# NOTE: download required packages: pip install -r requirements.txt


from algorithm import MyAlgorithm
import matplotlib.pyplot as plt

# Define a simulation runner for multiple steps

alg = MyAlgorithm(N=100, S=100, A=16, p=0.8, r=6, t=0)
single_run_data = alg.run_simulation()

for i in range(single_run_data.shape[1]):
    plt.plot(single_run_data[:, i], alpha=0.4)

plt.show()