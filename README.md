# Collective Problem Solving via Copying and Collaboration

This project explores how **collaboration and imitation (copying)** influence collective problem-solving performance in a dynamic multi-agent system. Inspired by real-world research environments, we model agents navigating an artificial fitness landscape while sharing and leveraging heterogeneous skills.

---

## Project Description

People working together often solve problems more effectively than individuals working alone. This effect is especially pronounced in **multi-disciplinary research**, where individuals possess different knowledge and skills.

We model this phenomenon using a **2D agent-based system** consisting of a **100 × 100 lattice** on which **16 agents** move. Each position on the grid has:
- a **fitness value**, defined by a noisy fitness landscape with a clear global maximum
- a **skill type**, drawn from a finite set of skills

Agents aim to improve their individual payoff by moving across the landscape using two mechanisms:

- **Copying**: moving toward positions associated with higher fitness discovered by other agents
- **Collaborating**: expanding their search space by leveraging the skills of nearby agents

The model investigates how agents balance **exploration vs. exploitation** when searching for high-fitness regions.

---

## Research Motivation

Previous research has used similar models with **fixed agent networks** to study the optimal ratio between copying and collaboration.

In this project, we aim to extend the model by introducing **dynamic agent networks**, reflecting the fact that real-world systems are not stationary. Specifically, we propose:
- agents have a **limited lifetime (retirement)**
- retired agents are replaced by **new hires** with new skills and positions

This allows us to study how **agent turnover** affects:
- the effectiveness of collaboration
- the relative payoff of copying vs. collaborating
- overall collective performance

### Hypothesis

We hypothesize that increased turnover will:
- reduce the effectiveness of collaboration
- increase the relative payoff of copying behavior  
because fewer experienced (“wise”) agents are available for meaningful collaboration.

---

## Current Model Overview

The current implementation includes:

- A **Perlin-noise-based fitness landscape** with a well-defined global maximum
- Agents characterized by:
  - position
  - skill
  - payoff (fitness at their location)
- Two movement mechanisms:
  - **Local exploration** (adjacent cells)
  - **Skill-based exploration** within a radius
- A probabilistic choice between:
  - copying other agents’ positions
  - collaborating via skill-sharing
- An **interactive visualization** using Matplotlib

---

## Parameters

Key parameters defined in the simulation:

```python
N = 100  # grid size (N x N)
S = 100  # number of skills
A = 16   # number of agents
p = 0.8  # probability of collaboration vs copying
r = 6    # collaboration radius
```

## Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

## Authors
Sooriya Karunaharan, Alara Karadeniz, Ivo Blok, Gileesa McCormack
University of Amsterdam / Msc Compuational Sciencde / Complex System Simulation Course