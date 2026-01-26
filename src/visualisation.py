

# Code to create visualisations


import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path
import pandas as pd
import numpy as np

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

def create_fitness_plot(csv_filename="averaged_fitness_metrics.csv", data_folder="data"):
    """
    Load averaged fitness metrics from a CSV file and plot average and maximum fitness
    over simulation steps with shaded error regions (95% CI for average, 2.5-97.5 percentile for max).

    The CSV is expected to contain the columns:
    - 'average_fitness_mean', 'average_fitness_lower', 'average_fitness_upper'
    - 'max_fitness_mean', 'max_fitness_lower', 'max_fitness_upper'

    Each row corresponds to one simulation step.
    """
    data_path = Path(data_folder)
    csv_path = data_path / csv_filename

    # Read CSV
    df = pd.read_csv(csv_path)

    steps = df.index  # row number = simulation step

    # Average fitness
    avg_mean = df["average_fitness_mean"]
    avg_lower = df["average_fitness_lower"]
    avg_upper = df["average_fitness_upper"]

    # Max fitness
    max_mean = df["max_fitness_mean"]
    max_lower = df["max_fitness_lower"]
    max_upper = df["max_fitness_upper"]

    plt.figure(figsize=(8, 5))

    # Plot average fitness with shaded CI
    plt.plot(steps, avg_mean, 'o-', label="Average Fitness")
    plt.fill_between(steps, avg_lower, avg_upper, color='blue', alpha=0.2)

    # Plot max fitness with shaded percentile interval
    plt.plot(steps, max_mean, 's-', label="Max Fitness", color='orange')
    plt.fill_between(steps, max_lower, max_upper, color='orange', alpha=0.2)

    plt.xlabel("Simulation Step")
    plt.ylabel("Fitness")
    plt.title("Fitness Over Time with Confidence Intervals")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()