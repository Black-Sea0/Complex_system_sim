import numpy as np
from noise import pnoise2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# params :)
N = 100 # grid size (N x N cells)
S = 100 # number of skills / colors
A = 16 # number of agents
p = 0.8 # probability of using 'copy' over 'collaborate'
r = 6 # euclidian cutoff radius for collaboration cells

# set up the board and agents, with a random skill for each cell and a fitness value for each, using some perlin noise and an exponential function for a well-defined global maximum
board_values = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        x = i / N
        y = j / N
        board_values[i, j] = pnoise2(x, y, octaves=5, persistence=2, lacunarity=2)
        signal = np.exp(-((i - N//2)**2 + (j - N//2)**2) / (2 * 3**2))
        board_values[i, j] += signal

board_skills = np.random.randint(0, S, (N, N))

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
    cells = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = pos[0] + di, pos[1] + dj
            if 0 <= ni < N and 0 <= nj < N:
                cells.append([ni, nj])
    return np.array(cells)

def get_skill_cells(pos, skill):
    cells = []
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di == 0 and dj == 0:
                continue
            dist = np.sqrt(di**2 + dj**2)
            if dist <= r:
                ni, nj = pos[0] + di, pos[1] + dj
                if 0 <= ni < N and 0 <= nj < N:
                    if board_skills[ni, nj] == skill:
                        cells.append([ni, nj])
    return np.array(cells)

# setting up the live plot
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.2)

ax.imshow(board_values, cmap='terrain')

scatters = []
for agent_idx, agent in enumerate(agents):
    pos = agent['pos']
    skill = agent['skill']
    scatter = ax.scatter(pos[1], pos[0])
    scatters.append(scatter)

def update_plot():
    for agent_idx, agent in enumerate(agents):
        pos = agent['pos']
        scatters[agent_idx].set_offsets([[pos[1], pos[0]]])
    plt.draw()


def step_simulation(event):
    moved_any = False
    
    for agent_idx in range(A):
        agent = agents[agent_idx]
        current_pos = agent['pos']
        current_skill = agent['skill']
        current_payoff = agent['payoff']
        
        candidate_cells = []
        
        adjacent = get_adjacent_cells(current_pos)
        if len(adjacent) > 0:
            candidate_cells.extend(adjacent)
        
        skill_cells = get_skill_cells(current_pos, current_skill)
        if len(skill_cells) > 0:
            candidate_cells.extend(skill_cells)
        
        if np.random.random() < p: # collaboration
            for neighbor_idx in range(A):
                if neighbor_idx != agent_idx:
                    neighbor_skill = agents[neighbor_idx]['skill']
                    collab_cells = get_skill_cells(current_pos, neighbor_skill)
                    if len(collab_cells) > 0:
                        candidate_cells.extend(collab_cells)        
        else: # copy
            for neighbor_idx in range(A):
                if neighbor_idx != agent_idx:
                    neighbor_pos = agents[neighbor_idx]['pos']
                    candidate_cells.append(neighbor_pos)

        candidate_cells = np.unique(np.array(candidate_cells), axis=0)
        payoffs = [board_values[pos[0], pos[1]] for pos in candidate_cells]
        best_idx = np.argmax(payoffs)
        best_pos = candidate_cells[best_idx]
        best_payoff = payoffs[best_idx]
        
        if best_payoff > current_payoff:
            agent['pos'] = best_pos
            agent['payoff'] = best_payoff
            moved_any = True

    if not moved_any:
        print(f"no agents moved")

    update_plot()

step_button = Button(plt.axes([0.4, 0.1, 0.2, 0.05]), 'Next Step')
step_button.on_clicked(step_simulation)
plt.show()