# Code to define the functioning of the full algorithm,
# getting multiple samples, etc...

from landscape import mason_watts_landscape, create_skill_map
from landscape import get_skill_cells, get_adjacent_cells
from agents import initialize_agents, replace_agents
import numpy as np
import os
import csv


def step_simulation(board, skills, agents, N, S, A, p, r):
    """
    Update the simulation by running one step. All agents try to move once.

    Parameters
    ----------
    board : np.ndarray
        A 2D array of shape (N, N) representing the fitness landscape.
    skills : np.ndarray
        A 2D array of shape (N, N) representing the skill of each cell.
    agents : list of dicts
        An array of S dictionaries, each corresponding to
        an agent with a 'pos', 'skill' and 'payoff' attribute.
    N : int
        Size of one dimension of the square grid.
    S : int
        Number of total possible skills.
    A : int
        Number of agents on the board.
    p : float
        copy-collaborate ratio.
    r : int
        radius of cell search during collaboration.

    Returns
    -------
    list of dicts
        the updated agents after the simulation step has finished.
    """
    agent_order = np.random.permutation(A)

    for agent_idx in agent_order:
        agent = agents[agent_idx]
        current_pos = agent['pos']
        current_skill = agent['skill']
        current_payoff = agent['payoff']

        candidate_cells = []

        # local neighbourhood
        adjacent = get_adjacent_cells(N, current_pos)
        if len(adjacent) > 0:
            candidate_cells.extend(adjacent)

        # skill-based exploration
        skills_to_check = [current_skill]

        if np.random.random() < p:  # collaboration
            for neighbor_idx in range(A):
                skills_to_check.append(agents[neighbor_idx]['skill'])
        else:  # copying
            for neighbor_idx in range(A):
                if neighbor_idx != agent_idx:
                    candidate_cells.append(agents[neighbor_idx]['pos'])

        skills_to_check = np.unique(np.array(skills_to_check))
        skill_cells = get_skill_cells(
            skills,
            current_pos,
            skills_to_check,
            r,
            N
        )
        if len(skill_cells) > 0:
            candidate_cells.extend(skill_cells)

        # find best move
        if candidate_cells:
            candidate_cells = np.unique(np.array(candidate_cells), axis=0)
            payoffs = board[candidate_cells[:, 0], candidate_cells[:, 1]]
            best_idx = np.argmax(payoffs)
            best_payoff = payoffs[best_idx]

            # move if improvement
            if best_payoff > current_payoff:
                best_pos = candidate_cells[best_idx]
                agents[agent_idx]['pos'] = best_pos
                agents[agent_idx]['payoff'] = best_payoff

    return agents


    # initialize fresh state for each run
def run_simulation(N, S, A, p, r, t, timesteps, save_to_csv: bool = False):
    """
    Run the simulation for a given number of successive steps.
    Initializes a board, skills and agents given the given model parameters.

    Parameters
    ----------
    N : int
        Size of one dimension of the square grid.
    S : int
        Number of total possible skills.
    A : int
        Number of agents on the board.
    p : float
        copy-collaborate ratio.
    r : int
        radius of cell search during collaboration.
    t : float
        turnover ratio.
    timesteps : int
        number of steps to run the simulation for.
    save_to_csv : bool, default False
        a bool value indicating whether the data should be saved to a csv or not.
        
    Returns
    -------
    np.ndarray
        a 2D array of shape (timesteps x A), where element (i, j)
         contains the payoff/fitness of agent j at timestep i.
    """
    board = mason_watts_landscape(N)
    skills = create_skill_map(N, S)
    agents = initialize_agents(board, A, N, S)

    payoffs_history = np.zeros((timesteps, A))

    for i in range(timesteps):
        agents = step_simulation(board, skills, agents, N, S, A, p, r)
        agents = replace_agents(agents, board, A, N, S, t)

        # record payoffs
        payoffs_history[i] = [agent['payoff'] for agent in agents]
    # save to csv
    if save_to_csv:
        csv_path = f"avg_vs_time_{p}_{t}.csv"
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestep", "avg_payoff"])

            for i in range(timesteps):
                agents = step_simulation(board, skills, agents, N, S, A, p, r)
                agents = replace_agents(agents, board, A, N, S, t)

                payoffs = [agent["payoff"] for agent in agents]
                payoffs_history[i] = payoffs
                avg_payoff = np.mean(payoffs)

                writer.writerow([i, avg_payoff])

    return payoffs_history


def run_multiple_simulations(N, S, A, p, r, t, num_runs, timesteps):
    """
    Run a simulation multiple times with the same model parameters for a
    given number of successive steps.
    Initializes a board, skills and agents given the given model parameters.

    Parameters
    ----------
    N : int
        Size of one dimension of the square grid.
    S : int
        Number of total possible skills.
    A : int
        Number of agents on the board.
    p : float
        copy-collaborate ratio.
    r : int
        radius of cell search during collaboration.
    t : float
        turnover ratio.
    num_runs : int
        number of simulations to do.
    timesteps : int
        number of steps to run the simulation for.

    Returns
    -------
    list of np.ndarrays
        a list of 2D arrays of shape (timesteps x A). Each 2D array stores the
        payoff history for a single simulation run.
        Element (i, j) of each entry contains the payoff/fitness of agent j at
        timestep i.
    """
    all_results = []

    for _ in range(num_runs):
        result = run_simulation(N, S, A, p, r, t, timesteps)
        all_results.append(result)

    return all_results
