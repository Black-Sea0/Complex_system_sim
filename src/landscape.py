# Code to generate landscape

import numpy as np
from noise import pnoise2

def generate_fitness_landscape(N, oct, pers, lac):
    """
    Generate a two-dimensional fitness landscape with a global optimum.

    The landscape is constructed by combining Perlin noise, which introduces
    ruggedness and local variation, with a Gaussian peak centered on the grid
    to ensure the presence of a well-defined global maximum. This structure
    mimics complex problem spaces with both local optima and an overarching
    optimal solution.

    Parameters
    ----------
    N : int
        Size of one dimension of the square grid.
    oct : int
        Number of Perlin noise octaves, controlling the level of detail
        in the landscape.
    pers : float
        Persistence parameter for Perlin noise, determining how much each
        octave contributes to the final noise.
    lac : float
        Lacunarity parameter for Perlin noise, controlling the frequency
        scaling between successive octaves.

    Returns
    -------
    np.ndarray
        A 2D array of shape (N, N) representing the fitness value of each
        position on the landscape.
    """

    board_values = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            x = i / N
            y = j / N
            
            # Perlin noise component (ruggedness)
            board_values[i, j] = pnoise2(
                x, y,
                octaves=oct,
                persistence=pers,
                lacunarity=lac
            )
            
            # Gaussian signal to enforce a global optimum
            signal = np.exp(-((i - N // 2) ** 2 + (j - N // 2) ** 2) / (2 * 3 ** 2))
            board_values[i, j] += signal
    return board_values

def create_skill_map(N, S):
    """ 
    Assign a random skill to each cell on the grid.
    """
    return np.random.randint(0, S, (N, N))


def get_skill_cells(board_skills, pos, skill, r, N):
    """
    Identify grid cells within a given radius that match a specified skill.

    This function searches the local neighborhood around an agent's position
    and returns all grid cells whose assigned skill matches the target skill.
    The search is limited to a circular area defined by a Euclidean radius.

    Parameters
    ----------
    board_skills : np.ndarray
        A 2D array of shape (N, N) assigning a skill identifier to each grid cell.
    pos : array-like
        Current position [i, j] of the agent on the grid.
    skill : int
        Skill identifier to search for.
    r : int or float
        Euclidean radius within which cells are considered.
    N : int
        Size of one dimension of the square grid.

    Returns
    -------
    np.ndarray
        An array of grid coordinates (shape: [k, 2]) corresponding to all
        cells within radius r of the agent's position whose skill matches
        the specified skill. If no such cells exist, an empty array is returned.
    """
    cells = []
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di == 0 and dj == 0:
                continue
            dist = np.sqrt(di ** 2 + dj ** 2)
            if dist <= r:
                ni, nj = pos[0] + di, pos[1] + dj
                if 0 <= ni < N and 0 <= nj < N:
                    if board_skills[ni, nj] == skill: 
                        cells.append([ni, nj])
    return np.array(cells)