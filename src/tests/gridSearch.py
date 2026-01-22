import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithm import ComplexOptimizer
from landscape import mason_watts_landscape

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import numpy as np

p_values = np.linspace(0, 1, 10)
t_values = np.linspace(0, 1, 10)

simulation_results = np.zeros(shape=(len(p_values), len(t_values)))
board = mason_watts_landscape(100)

for x, p in enumerate(p_values):
    for y, t in enumerate(t_values):

        alg = ComplexOptimizer(
            board=board,
            N=100,
            S=100,
            A=16,
            p=p,
            r=6,
            t=t,
        )

        multi_run_data = alg.run_multiple_simulations(num_runs=25, timesteps=20)

        run_avgs_at_end = []
        for run_data in multi_run_data:
            run_avgs_at_end.append(np.average(run_data, axis=1)[-1])
        
        run_avgs_at_end = np.array(run_avgs_at_end)
        simulation_results[x, y] = np.average(run_avgs_at_end)
    
        print(len(t_values) * x + y + 1, ":", len(t_values) * len(p_values))


P, T = np.meshgrid(t_values, p_values)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(P, T, simulation_results, cmap='viridis', alpha=0.8, linewidth=0.5, edgecolor='k')
ax.contour(P, T, simulation_results, zdir='z', offset=simulation_results.min() - 0.1, cmap='viridis', alpha=0.5)

ax.set_xlabel('t')
ax.set_ylabel('p')
ax.set_zlabel('Average Performance at t=20')

fig.colorbar(surf, ax=ax)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()