# Code to generate landscape

import numpy as np
from noise import pnoise2
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline

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

    # Rescale fitness values to [0, 1]
    min_val = board_values.min()
    max_val = board_values.max()
    board_values = (board_values - min_val) / (max_val - min_val)
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

def mason_watts_landscape(L, seed=None, rho=0.7, omega_min=3, omega_max=7, center_mean=False):
    """
    Generate a 2D fitness landscape inspired by Mason & Watts (R version).

    The landscape combines a dominant unimodal Gaussian peak with 
    multi-scale smooth noise (Perlin-like) to create a discrete NxN grid 
    that is locally correlated and visually smooth. This is intended to 
    simulate complex problem spaces where neighboring solutions have 
    similar fitness values, but with some local variation.

    Parameters
    ----------
    L : int
        Size of one dimension of the square grid (NxN).
    seed : int or None, optional
        Seed for the random number generator (default: None).
    rho : float, optional
        Scaling factor for the amplitude of successive noise octaves
        (default: 0.7). Lower values reduce the contribution of higher-frequency noise.
    omega_min : int, optional
        Minimum octave index for generating smooth noise (default: 3).
    omega_max : int, optional
        Maximum octave index for generating smooth noise (default: 7).
    center_mean : bool, optional
        If True, the Gaussian peak is centered in the middle of the grid;
        otherwise, it is randomly positioned (default: False).

    Returns
    -------
    np.ndarray
        A 2D array of shape (L, L) representing the fitness of each cell.
        Fitness values are scaled such that the maximum value is 100.0.
        The grid is discrete but visually smooth due to the combination of
        Gaussian signal and interpolated multi-scale noise.
    """
    rng = np.random.default_rng(seed)

    # 1) Unimodal bivariate Gaussian "signal"
    R = 3 * (L / 100.0)
    sd = np.sqrt(R)

    xs = np.arange(1, L + 1)

    if center_mean:
        mu_x = (L + 1) / 2.0
        mu_y = (L + 1) / 2.0
    else:
        mu_x = rng.uniform(1, L)
        mu_y = rng.uniform(1, L)

    X = norm.pdf(xs, loc=mu_x, scale=sd)
    Y = norm.pdf(xs, loc=mu_y, scale=sd)

    fitness = np.outer(X, Y)
    fitness = fitness / np.max(fitness)

    # 2) "Perlin noise" (value noise + bicubic interpolation)
    fine = np.arange(1, L + 1)

    for omega in range(omega_min, omega_max + 1):
        octave = 2 ** omega
        coarse = rng.uniform(0.0, 1.0, size=(octave, octave))
        coarse_seq = np.linspace(1, L, num=octave)

        spline = RectBivariateSpline(coarse_seq, coarse_seq, coarse, kx=3, ky=3)
        octave_full = spline(fine, fine)  # (L, L)

        octave_full *= (rho ** omega)
        fitness += octave_full

    # 3) Scale to max=100 (like R)
    fitness = fitness * (100.0 / np.max(fitness))

    return fitness
