# contains all parameters for the simulation

import numpy as np

# PARAMETERS
N = 100  # Grid size (N x N cells)
S = 100  # Number of distinct skills (also used as colors)
A = 16   # Number of agents
p_list = np.linspace(0, 1, 11)  # Probability of collaboration (vs copying)
r = 6    # Euclidean cutoff radius for skill-based collaboration
t = 0.0

# Params by ChatGPT:
NOISE_OCTAVES = 5
NOISE_PERSISTENCE = 2
NOISE_LACUNARITY = 2
N_steps = 20  # Number of simulation steps to run
N_runs = 100 # Number of runs of simulation