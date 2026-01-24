# Code to initialize agents



import numpy as np


def initialize_agents(board_values, A, N, S):
    """
    Initialize a population of agents with random positions and skills.

    Each agent is placed uniformly at random on an N x N grid and is
    assigned a skill drawn uniformly from the available skill set.
    The agent's initial payoff is determined by the fitness value
    of the landscape at its starting position.

    Parameters
    ----------
    board_values : np.ndarray
        A 2D array of shape (N, N) representing the fitness landscape.
    A : int
        Number of agents to initialize.
    N : int
        Size of one dimension of the square grid.
    S : int
        Total number of distinct skills.

    Returns
    -------
    list of dict
        A list of agent dictionaries, each containing:
        - 'pos' : np.ndarray of shape (2,)
            The agent's current position on the grid.
        - 'skill' : int
            The agent's assigned skill identifier.
        - 'payoff' : float
            The fitness value at the agent's current position.
    """
        
    agents = [] #TODO: we can make these a class later
    for _ in range(A):
        pos = np.random.randint(0, N, 2)
        skill = np.random.randint(0, S)
        agents.append({
            "pos": pos,
            "skill": skill,
            "payoff": board_values[pos[0], pos[1]]
        })
    return agents

def replace_agents(agents, board_values, A, N, S, t):
    """
    Return a new list of agents with the 'dead' agents replaced.

    Each new agent old agent is replaced with a new one that is randomly 
    assigned a position and skill.

    Parameters
    ----------
    agents : list of dicts
        List of agents each with 'pos', 'skill', and 'payoff' keys
    board_values : np.ndarray
        A 2D array of shape (N, N) representing the fitness landscape.
    A : int
        Number of agents on the board
    N : int
        Size of one dimension of the square grid.
    S : int
        Total number of distinct skills.
    t : float
        A float from 0, 1 to represent the chance of each agent having turnover.
        
    Returns
    -------
    list of dict
        A list of agent dictionaries, each containing:
        - 'pos' : np.ndarray of shape (2,)
            The agent's current position on the grid.
        - 'skill' : int
            The agent's assigned skill identifier.
        - 'payoff' : float
            The fitness value at the agent's current position.
    """
    for agent_idx in range(A):
        agent = agents[agent_idx]
        if np.random.random() < t:
            pos = np.random.randint(0, N, 2)
            skill = np.random.randint(0, S)
            agent['pos'] = pos
            agent['skill'] = skill
            agent['payoff'] = board_values[pos[0], pos[1]]
    return agents

def get_average_fitness(agents):
    """
    Calculate the average fitness (payoff) of all agents.

    Parameters
    ----------
    agents : list of dict
        List of agents, each with a 'payoff' key.

    Returns
    -------
    float
        The average fitness of the agents.
    """
    total_fitness = sum(agent["payoff"] for agent in agents)
    return total_fitness / len(agents) if agents else 0.0


def get_max_fitness(agents):
    """
    Determine the maximum fitness (payoff) among all agents.

    Parameters
    ----------
    agents : list of dict
        List of agents, each with a 'payoff' key.

    Returns
    -------
    float
        The maximum fitness of the agents.
    """
    return max(agent["payoff"] for agent in agents) if agents else 0.0