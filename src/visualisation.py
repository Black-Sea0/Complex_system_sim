

# Code to create visualisations


import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path
import pandas as pd

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
    over simulation steps with standard deviation as error bars.

    The CSV is expected to contain the columns:
    - 'average_fitness_mean', 'average_fitness_std'
    - 'max_fitness_mean', 'max_fitness_std'

    Each row corresponds to one simulation step.
    """
    data_path = Path(data_folder)
    csv_path = data_path / csv_filename

    # Read CSV
    df = pd.read_csv(csv_path)

    steps = df.index  # row number = simulation step
    avg_mean = df["average_fitness_mean"]
    avg_std = df["average_fitness_std"]
    max_mean = df["max_fitness_mean"]
    max_std = df["max_fitness_std"]

    # Plot with error bars
    plt.figure(figsize=(8, 5))

    plt.errorbar(steps, avg_mean, yerr=avg_std, fmt='o-', capsize=3, label="Average Fitness")
    plt.errorbar(steps, max_mean, yerr=max_std, fmt='s-', capsize=3, label="Max Fitness")

    plt.xlabel("Simulation Step")
    plt.ylabel("Fitness")
    plt.title("Fitness Over Time with Std Dev")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()