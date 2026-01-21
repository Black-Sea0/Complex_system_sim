

# Code to set up and run the simulation

from typing import *
import numpy as np
from agents import get_adjacent_cells
from landscape import get_skill_cells


def step_simulation(
    N,
    r,
    board_skills,
    agents: List[Dict],
    board_values: np.ndarray,
    p: float,
    A: int
) -> bool:
    """
    Perform a single simulation step for all agents in the environment.

    Each agent evaluates potential moves based on a combination of local exploration,
    skill-based exploration, and interaction with other agents. The agent then moves
    to the position with the highest payoff, if it improves upon its current payoff.

    The movement rules for each agent are:
    1. **Local exploration:** Consider all immediately adjacent cells.
    2. **Skill-based exploration:** Consider all cells within a radius that match the agent's skill.
    3. **Collaboration (with probability `p`):** Expand the search to include
       skill-matching cells from neighboring agents.
    4. **Copying (with probability `1 - p`):** Consider the current positions of other agents.

    Candidate positions are evaluated, and the agent moves to the position with
    the highest fitness (payoff) if it exceeds the agent's current payoff.

    Parameters
    ----------
    agents : list of dict
        A list of agents, each represented as a dictionary containing:
        - 'pos' : array-like, current [i, j] position
        - 'skill' : int, assigned skill identifier
        - 'payoff' : float, current fitness at the agent's position
    board_values : np.ndarray
        2D array of shape (N, N) representing the fitness landscape.
    p : float
        Probability that an agent will choose collaboration over copying.
    A : int
        Total number of agents.

    Returns
    -------
    bool
        True if at least one agent moved during this step; False otherwise.
    """
    moved_any = False

    # TODO: when the agents teleport to each other (copying), they might land on each other.
    # This is allowed for now, but we might want to prevent it later.
    # It also makes agents seem to 'disappear' in the visualisation.

    for agent_idx in range(A):
        agent = agents[agent_idx]
        current_pos = agent['pos']
        current_skill = agent['skill']
        current_payoff = agent['payoff']

        candidate_cells = []

        # Local exploration (adjacent cells)
        adjacent = get_adjacent_cells(N, current_pos)
        if len(adjacent) > 0:
            candidate_cells.extend(adjacent)

        # Skill-based exploration using own skill
        skill_cells = get_skill_cells(board_skills, current_pos, current_skill, r, N)
        if len(skill_cells) > 0:
            candidate_cells.extend(skill_cells)

        # Decide between collaboration and copying
        if np.random.random() < p:  # collaboration
            for neighbor_idx in range(A):
                if neighbor_idx != agent_idx:
                    neighbor_skill = agents[neighbor_idx]['skill']
                    collab_cells = get_skill_cells(board_skills, current_pos, neighbor_skill, r, N)
                    if len(collab_cells) > 0:
                        candidate_cells.extend(collab_cells)
        else:  # copying
            for neighbor_idx in range(A):
                if neighbor_idx != agent_idx:
                    candidate_cells.append(agents[neighbor_idx]['pos'])

        # Remove duplicate candidate cells
        candidate_cells = np.unique(np.array(candidate_cells), axis=0)

        # Evaluate payoffs and choose best move
        payoffs = [board_values[pos[0], pos[1]] for pos in candidate_cells]
        best_idx = np.argmax(payoffs)
        best_pos = candidate_cells[best_idx]
        best_payoff = payoffs[best_idx]

        # Move only if payoff improves
        if best_payoff > current_payoff:
            agent['pos'] = best_pos
            agent['payoff'] = best_payoff
            moved_any = True

    if not moved_any:
        print("No agents moved")

    # update_plot()
    return moved_any