
# This code ties everything together.
# Run this file to run entire simulation pipeline and visualisation.
# You may look into other files for 'TODO' comments to see what to work on

# NOTE: download required packages: pip install -r requirements.txt


from landscape import generate_fitness_landscape, create_skill_map
from agents import initialize_agents
from simulation import step_simulation
from visualisation import setup_plot, update_plot, create_fitness_plot
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
from statistical import save_fitness_metrics, clear_data_fitness, combine_fitness_metrics
from config import *

# Generate fitness landscape and skill map
board = generate_fitness_landscape(N, NOISE_OCTAVES, NOISE_PERSISTENCE, NOISE_LACUNARITY)
skills = create_skill_map(N, S)

# Initialize agents
agents = initialize_agents(board, A, N, S)

# Set up the plot
fig, ax, scatters = setup_plot(board, agents)

# Define the button callback
def on_click(event):
    moved = step_simulation(N, r, skills, agents, board, p, A)  # Updates agent positions
    update_plot(scatters, agents)                 # Reflect changes visually
    print("Simulation step completed")
    if not moved:
        print("No agents moved")

def run_simulation(event):
    for i in range(N_runs):
        for _ in range(N_steps):
            moved = step_simulation(N, r, skills, agents, board, p, A)
            save_fitness_metrics(agents, csv_filename=f"fitness_metrics_{i}.csv")  # Save fitness metrics after each step
            update_plot(scatters, agents)
            if not moved:
                print("No agents moved")
        print(f"Simulation: {N_steps} steps completed")
    combine_fitness_metrics()
            

# Add the button to trigger a simulation step
# TODO: instead of a button, we can make automaticly run for some number of steps
button = Button(plt.axes([0.4, 0.1, 0.2, 0.05]), "Next Step")
button.on_clicked(run_simulation)


# Clear data when main.py is run
for i in range(N_runs):
    clear_data_fitness(csv_filename=f"fitness_metrics_{i}.csv")
clear_data_fitness(csv_filename="averaged_fitness_metrics.csv")

plt.show()

create_fitness_plot()