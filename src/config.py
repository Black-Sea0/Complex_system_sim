# contains all parameters for the simulation



# PARAMETERS
N = 100  # Grid size (N x N cells)
S = 100  # Number of distinct skills (also used as colors)
A = 16   # Number of agents
p = 0.8  # Probability of collaboration (vs copying)
r = 6    # Euclidean cutoff radius for skill-based collaboration

p_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Params by ChatGPT:
NOISE_OCTAVES = 5
NOISE_PERSISTENCE = 2
NOISE_LACUNARITY = 2

N_steps = 20  # Number of simulation steps to run
N_runs = 10 # Number of runs of simulation