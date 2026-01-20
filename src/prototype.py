import numpy as np
from noise import pnoise2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

#NOTE: download required packages: pip install -r requirements.txt

# PARAMETERS
N = 100  # Grid size (N x N cells)
S = 100  # Number of distinct skills (also used as colors)
A = 16   # Number of agents
p = 0.8  # Probability of collaboration (vs copying)
r = 6    # Euclidean cutoff radius for skill-based collaboration


# Initialize the fitness landscape
# Combines Perlin noise with a Gaussian peak at the center
# to create a rugged landscape with a clear global maximum
board_values = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        x = i / N
        y = j / N
        
        # Perlin noise component (ruggedness)
        board_values[i, j] = pnoise2(
            x, y,
            octaves=5,
            persistence=2,
            lacunarity=2
        )
        
        # Gaussian signal to enforce a global optimum
        signal = np.exp(-((i - N // 2) ** 2 + (j - N // 2) ** 2) / (2 * 3 ** 2))
        board_values[i, j] += signal

# Assign a random skill to each cell on the grid
board_skills = np.random.randint(0, S, (N, N))


# Initialise agents; 
# Each agent has:
# - a position
# - a skill
# - a payoff (fitness at its current position)
agents = []
for _ in range(A):
    pos = np.random.randint(0, N, 2)
    skill = np.random.randint(0, S)
    agents.append({
        'pos': pos,
        'skill': skill,
        'payoff': board_values[pos[0], pos[1]]
    })


def get_adjacent_cells(pos):
    """
    Return all valid adjacent cells (Moore neighborhood)
    around a given position.

    Parameters
    ----------
    pos : array-like
        Current position [i, j] of the agent.

    Returns
    -------
    np.ndarray
        Array of neighboring cell coordinates.
    """
    cells = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue  # Skip the agent's own cell
            ni, nj = pos[0] + di, pos[1] + dj
            if 0 <= ni < N and 0 <= nj < N:
                cells.append([ni, nj])
    return np.array(cells)


def get_skill_cells(pos, skill):
    """
    Return all cells within radius r that match a given skill.

    This represents collaboration: an agent can search further
    if it has access to specific skills (its own or others').

    Parameters
    ----------
    pos : array-like
        Current position [i, j] of the agent.
    skill : int
        Skill identifier to search for.

    Returns
    -------
    np.ndarray
        Array of cell coordinates matching the skill.
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





# VISUALIZATION
# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.2)

# Plot the fitness landscape
ax.imshow(board_values, cmap='terrain')

# Plot agents as scatter points
scatters = []
for agent in agents:
    pos = agent['pos']
    scatter = ax.scatter(pos[1], pos[0])
    scatters.append(scatter)

def update_plot():
    """
    Update the scatter plot to reflect current agent positions.
    """
    for agent_idx, agent in enumerate(agents):
        pos = agent['pos']
        scatters[agent_idx].set_offsets([[pos[1], pos[0]]])
    plt.draw()


def step_simulation(event):
    """
    Perform a single simulation step.

    For each agent:
    - Construct a set of candidate cells
    - Choose between copying and collaborating
    - Move to the best available cell if it improves payoff
    """
    moved_any = False

    for agent_idx in range(A):
        agent = agents[agent_idx]
        current_pos = agent['pos']
        current_skill = agent['skill']
        current_payoff = agent['payoff']

        candidate_cells = []

        # Local exploration (adjacent cells)
        adjacent = get_adjacent_cells(current_pos)
        if len(adjacent) > 0:
            candidate_cells.extend(adjacent)

        # Skill-based exploration using own skill
        skill_cells = get_skill_cells(current_pos, current_skill)
        if len(skill_cells) > 0:
            candidate_cells.extend(skill_cells)

        # Decide between collaboration and copying
        if np.random.random() < p:  # collaboration
            for neighbor_idx in range(A):
                if neighbor_idx != agent_idx:
                    neighbor_skill = agents[neighbor_idx]['skill']
                    collab_cells = get_skill_cells(current_pos, neighbor_skill)
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

    update_plot()


# INTERACTIVE BUTTON
step_button = Button(
    plt.axes([0.4, 0.1, 0.2, 0.05]),
    'Next Step'
)
step_button.on_clicked(step_simulation)

plt.show()