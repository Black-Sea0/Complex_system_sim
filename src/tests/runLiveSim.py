import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import click
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


@click.command()
@click.option('--t', default=0.0, help='turnover rate for agents')
@click.option('--p', default=0.7, help='collaboration vs. copying rate. p = collaboration, 1-p = copying.')
def main(t, p):
    """
    Run a single instance of the simulation to visualize the movements of the agents on the fitness landscape.

    Parameters
    ----------
    t: float
        Turnover rate for the agents. Each agent has a chance t of being replaced each timestep
        with an agent that spawns in a random location with a random skill.
    p: float
        Collaborate-Copy ratio of the network. For each agent this is the rate that decide if they will
        copy or collaborate. p = probability of collaboration, 1 - p = probability of copying.
    """
    global agents
    assert t >= 0.0 and t <= 1.0, f"t is expected to be a float between 0 or 1, got: {t}"
    assert p >= 0.0 and p <= 1.0, f"p is expected to be a float between 0 or 1, got: {p}"
    N = 100
    S = 100
    A = 16
    r = 6
    

    # Generate fitness landscape and skill map
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
        agents = replace_agents(agents, board, A, N, S, t) #triggers a turnover
        update_plot(scatters, agents)                 # Reflect changes visually
        print("Simulation step completed")

    # Add the button to trigger a simulation step
    button = Button(plt.axes([0.4, 0.1, 0.2, 0.05]), "Next Step")
    button.on_clicked(on_click)
    plt.show()


if __name__ == "__main__":
    main()