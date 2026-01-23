from landscape import mason_watts_landscape, create_skill_map
from agents import initialize_agents, replace_agents
from visualisation import setup_plot, update_plot
from agents import get_adjacent_cells
from landscape import get_skill_cells
from statistical import save_fitness_metrics

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import config

class ComplexOptimizer:
    def __init__(self, N, S, A, p, r, t):
        self.N = N
        self.S = S
        self.A = A
        self.p = p
        self.r = r
        self.t = t
        
    def interactive_simulation(self):

        self.board = mason_watts_landscape(self.N)
        self.skills = create_skill_map(self.N, self.S)
        self.agents = initialize_agents(self.board, self.A, self.N, self.S)

        fig, ax, scatters = setup_plot(self.board, self.agents)

        def on_click(event):
            moved = self.step_simulation()  # Updates agent positions
            self.agents = replace_agents(self.agents, self.board, self.A, self.N, self.S, self.t)
            update_plot(scatters, self.agents)                 # Reflect changes visually
            print("Simulation step completed")
            if not moved:
                print("No agents moved")

        button = Button(plt.axes([0.4, 0.1, 0.2, 0.05]), "Next Step")
        button.on_clicked(on_click)
        plt.show()
            
    def run_simulation(self, index, timesteps = config.N_steps):
        self.board = mason_watts_landscape(self.N)
        self.skills = create_skill_map(self.N, self.S)
        self.agents = initialize_agents(self.board, self.A, self.N, self.S)

        data = np.zeros(shape=(timesteps, self.A))

        for i in range(timesteps):
            self.step_simulation(index=i)
            self.agents = replace_agents(self.agents, self.board, self.A, self.N, self.S, self.t)
            
            data[i] = [agent['payoff'] for agent in self.agents]

        # Save final step
        save_fitness_metrics(agents=self.agents, csv_filename=f"fitness_metrics_p_{self.p}_{index}.csv")

        return data

    def run_multiple_simulations(self, num_runs, timesteps = config.N_steps, reset_initial_state = False):
        data = []

        for i in range(num_runs):
            data.append(self.run_simulation(index=i, timesteps=timesteps))

        return data

    def step_simulation(self, index, save_every_step: bool=False) -> bool:
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

        Returns
        -------
        bool
            True if at least one agent moved during this step; False otherwise.
        """
        moved_any = False

        # TODO: when the agents teleport to each other (copying), they might land on each other.
        # This is allowed for now, but we might want to prevent it later.
        # It also makes agents seem to 'disappear' in the visualisation.

        #Randomly shuffling the list of agents so that the movement order of the agents are not
        #same in each generation. 
        
        agent_order = np.random.permutation(self.A)
        for agent_idx in agent_order:
            self.agent = self.agents[agent_idx]
            current_pos = self.agent['pos']
            current_skill = self.agent['skill']
            current_payoff = self.agent['payoff']

            candidate_cells = []

            # Local exploration (adjacent cells)
            adjacent = get_adjacent_cells(self.N, current_pos)
            if len(adjacent) > 0:
                candidate_cells.extend(adjacent)

            # Skill-based exploration using own skill
            skill_cells = get_skill_cells(self.skills, current_pos, current_skill, self.r, self.N)
            if len(skill_cells) > 0:
                candidate_cells.extend(skill_cells)

            # Decide between collaboration and copying
            if np.random.random() < self.p:  # collaboration
                for neighbor_idx in range(self.A):
                    if neighbor_idx != agent_idx:
                        neighbor_skill = self.agents[neighbor_idx]['skill']
                        collab_cells = get_skill_cells(self.skills, current_pos, neighbor_skill, self.r, self.N)
                        if len(collab_cells) > 0:
                            candidate_cells.extend(collab_cells)
            else:  # copying
                for neighbor_idx in range(self.A):
                    if neighbor_idx != agent_idx:
                        candidate_cells.append(self.agents[neighbor_idx]['pos'])

            # Remove duplicate candidate cells
            candidate_cells = np.unique(np.array(candidate_cells), axis=0)

            # Evaluate payoffs and choose best move
            payoffs = [self.board[pos[0], pos[1]] for pos in candidate_cells]
            best_idx = np.argmax(payoffs)
            best_pos = candidate_cells[best_idx]
            best_payoff = payoffs[best_idx]

            # Move only if payoff improves
            if best_payoff > current_payoff:
                self.agent['pos'] = best_pos
                self.agent['payoff'] = best_payoff
                moved_any = True

        if save_every_step:
            save_fitness_metrics(agents=self.agents, csv_filename=f"fitness_metrics_p_{self.p}_{index}.csv")

        return moved_any