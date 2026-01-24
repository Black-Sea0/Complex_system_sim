from landscape import mason_watts_landscape, create_skill_map, get_skill_cells, get_adjacent_cells
from agents import initialize_agents, replace_agents
import numpy as np

def run_simulation(N, S, A, p, r, t, timesteps):
    """Run a complete simulation."""
    # Initialize fresh state for each run
    board = mason_watts_landscape(N)
    skills = create_skill_map(N, S)
    agents = initialize_agents(board, A, N, S)
    
    payoffs_history = np.zeros((timesteps, A))
    
    for i in range(timesteps):
        agents = step_simulation(board, skills, agents, N, S, A, p, r)
        agents = replace_agents(agents, board, A, N, S, t)
        
        # record payoffs
        payoffs_history[i] = [agent['payoff'] for agent in agents]
    
    return payoffs_history

def step_simulation(board, skills, agents, N, S, A, p, r):
    """Perform one simulation step for all agents."""
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
        skill_cells = get_skill_cells(skills, current_pos, skills_to_check, r, N)
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

def run_multiple_simulations(N, S, A, p, r, t, num_runs, timesteps):
    """Run multiple simulations with same parameters."""
    all_results = []
    
    for _ in range(num_runs):
        result = run_simulation(N, S, A, p, r, t, timesteps)
        all_results.append(result)
    
    return all_results