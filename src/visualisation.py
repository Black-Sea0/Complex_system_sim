

# Code to create visualisations


import matplotlib.pyplot as plt
from matplotlib.widgets import Button

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