# Collective Problem Solving via Copying and Collaboration

This project explores how **collaboration and imitation (copying)** influence collective problem-solving performance in a dynamic multi-agent system. Inspired by real-world research environments, we model agents navigating an artificial fitness landscape while sharing and leveraging heterogeneous skills.

This project is inspired by and heavily based on the following research paper: 

Akçakır, G., Lang, J.C. & Lamberson, P.J. Copy or collaborate? How networks impact collective problem solving. npj Complex 2, 35 (2025). https://doi.org/10.1038/s44260-025-00058-8 
(GitHub: https://github.com/gulsahakcakir/Group-Problem-Solving)
  
---

## Project Description

People working together often solve problems more effectively than individuals working alone. This effect is especially pronounced in **multi-disciplinary research**, where individuals possess different knowledge and skills.

We model this phenomenon using a **2D agent-based system** consisting of a **100 × 100 lattice** on which **16 agents** move. Each position on the grid has:
- a **fitness value**, defined by a noisy fitness landscape with a clear global maximum
- a **skill type**, drawn from a finite set of skills

Agents aim to improve their individual payoff by moving across the landscape using two mechanisms:

- **Copying**: moving toward positions associated with higher fitness discovered by other agents
- **Collaborating**: expanding their search space by leveraging the skills of nearby agents

The model investigates how agents balance **exploration vs. exploitation** when searching for high-fitness regions using a **dynamic agent network** which includes turnover of agents.

---

## Research Motivation

Previous research has used similar models with **fixed agent networks** to study the optimal ratio between copying and collaboration.

In this project, we aim to extend the model by introducing **dynamic agent networks**, reflecting the fact that real-world systems are not stationary. Specifically, we propose:
- agents have a chance of being **removed** from the system (retirement)
- retired agents are **replaced** by new agents with new skills and positions

This allows us to study how **agent turnover** affects:
- the relative payoff of copying vs. collaborating
- overall collective performance

### Hypothesis

We hypothesize that increased turnover will have a statistically significant effect on the optimal copy-collaboration ratio.

---

## Current Model Overview

The current implementation includes:

- A **Perlin-noise-based fitness landscape** with a well-defined global maximum
- Agents characterized by:
  - position
  - skill
  - payoff (fitness at their location)
- Two movement mechanisms:
  - **Copying (exploitation)**: 
    - possibility of moving to adjacent cells; OR
    - moving to cells of own skill within radius r; OR
    - moving to cell occupied by a neighbouring agents.
  - **Collaboration (exploration)**:
    - possibility of moving to adjacent cells; OR
    - moving to cells of own skill within radius r; OR
    - moving to cells with neighouring agents' skills within radius r
- A probabilistic choice between:
  - copying other agents’ positions
  - collaborating via skill-sharing
- An **interactive visualization** using Matplotlib
- **Assert** statements where there are user inputs to make sure an unwanted input that would break the code is not included

---

## Parameters

The following paramaters are kept constant across all tests:

```python
N = 100  # grid size (N x N)
S = 100  # number of skills
A = 16   # number of agents
r = 6    # collaboration radius
```

## Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running Data / Getting Plots
For convenience, first set the tests folder as your working directory:
```bash
cd src/tests
```
Then collect the data by running the simulations:
```bash
python gridSearchParallel.py --num_threads=75 --p_steps=31 --t_steps=31 --n_samples=1000 --output=grid_search_results_1000_samples_20_steps.npy
python loglog.py --output=loglog_20_steps.npy
```
Because some of these tests can take up to an hour, you can instead run the reduced quality version below:
```bash
python gridSearchParallel.py --num_threads=4 --p_steps=3 --t_steps=3 --n_samples=20 --output=grid_search_results_20_samples_20_steps.npy
python loglog.py --output=loglog_20_steps.npy
```

Then plot the results with:
```bash
python gridSearchVisualization.py --input=grid_search_results_20_samples_20_steps.npy
python loglogVisualization.py --input=loglog_20_steps.npy
```

If you wish to create more plots but do not wish to overwrite the previous plots then(the value for fignum can be any int the user wishes to use, the first saved figures are saved with 0 unless fignum specified):
```bash
python gridSearchVisualization.py --input=grid_search_results_20_samples_20_steps.npy --fignum=1
```
If you wish to visualize a single run with agents moving on the grid then you can run (The values for t and p can be changed to any values between [0,1]):
```bash
python runLiveSim.py --t=0.2 --p=0.7
```
---

## Documentation
The code is written using the PEP8 codestyle. The documentation can be generated by running:
```bash
cd src
pdoc --html -o ../documentation --force . 
```
The documentation can then be viewed by opening `documentation/index.html`

## Authors
Sooriya Karunaharan, Alara Karadeniz, Ivo Blok, Gileesa McCormack

University of Amsterdam / Msc Compuational Science / Complex System Simulation Course
