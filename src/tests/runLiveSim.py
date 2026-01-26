import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithm import run_simulation, step_simulation
import numpy as np
from landscape import generate_fitness_landscape, create_skill_map, mason_watts_landscape
from agents import initialize_agents, replace_agents, get_max_fitness

agents = []

def setup_plot(board_values, agents):
    """
    Initialize the matplotlib plot for the fitness landscape and agents.

    Displays the 2D fitness landscape as a background and places scatter
    points for each agent at their initial positions.

    Parameters
    ----------
    board_values : np.ndarray
        2D array representing the fitness landscape.
    agents : list of dict
        List of agents, each with a 'pos' key indicating [row, col].
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2)

    # Display the fitness landscape
    im = ax.imshow(board_values, cmap="terrain")

    # Add a colorbar for fitness values
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fitness Value", rotation=270, labelpad=15)

    # Scatter agents on top of the landscape
    scatters = [
        ax.scatter(agent["pos"][1], agent["pos"][0])
        for agent in agents
    ]

    return fig, ax, scatters


def update_plot(scatters, agents):
    """
    Update the positions of agents on the existing plot.

    Moves each scatter point to the agent's current position.

    Parameters
    ----------
    scatters : list of matplotlib.collections.PathCollection
        Scatter objects representing each agent.
    agents : list of dict
        List of agents, each with a 'pos' key indicating [row, col].
    """
    for i, agent in enumerate(agents):
        scatters[i].set_offsets([[agent["pos"][1], agent["pos"][0]]])
    plt.draw()

def main():
    global agents
    N = 100
    S = 100
    A = 16
    t = 0.2
    p = 0.7
    r = 6
    

    # Generate fitness landscape and skill map
    # board = generate_fitness_landscape(N, NOISE_OCTAVES, NOISE_PERSISTENCE, NOISE_LACUNARITY)
    board = mason_watts_landscape(N)
    skills = create_skill_map(N, S)
    # Initialize agents
    agents = initialize_agents(board, A, N, S)
    # Set up the plot
    fig, ax, scatters = setup_plot(board, agents)

    # Define the button callback
    def on_click(event):
        global agents
        agents = step_simulation(board, skills, agents, N, S, A, p, r)  # Updates agent positions
        agents = replace_agents(agents, board, A, N, S, t)
        update_plot(scatters, agents)                 # Reflect changes visually
        print("Simulation step completed")

    # Add the button to trigger a simulation step
    button = Button(plt.axes([0.4, 0.1, 0.2, 0.05]), "Next Step")
    button.on_clicked(on_click)
    plt.show()


if __name__ == "__main__":
    main()