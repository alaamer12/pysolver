# PySolver Project

This project implements three popular nature-inspired optimization algorithms:

1. **Ant Colony Optimization (ACO)** - Inspired by the foraging behavior of ants
2. **Artificial Bee Colony (ABC)** - Based on the intelligent foraging behavior of honey bees
3. **Particle Swarm Optimization (PSO)** - Mimics the social behavior of bird flocking or fish schooling


## Project Structure

```
.
├── README.md               # Project documentation
├── main.py                 # Main script with benchmarks and comparisons
├── _aco.py                 # Ant Colony Optimization implementation
├── _problem.py             # Problem definition interface
├── __abc.py                # Artificial Bee Colony implementation 
├── _pso.py                 # Particle Swarm Optimization implementation
└── requirements.txt        # Project dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alaamer12/pysolver.git
   cd pysolver
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate 
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running a Specific Algorithm

```bash
python main.py --algorithm pso --kwargs

python main.py --algorithm abc 

# automatically detected best algorithm
python main.py go
```

## Algorithms Overview

### Ant Colony Optimization (ACO)

ACO is particularly effective for discrete optimization problems like the Traveling Salesman Problem (TSP). The algorithm works by simulating ants leaving pheromone trails as they search for efficient paths.

Key parameters:
- `alpha` - Pheromone importance
- `beta` - Heuristic information importance
- `evaporation_rate` - Controls pheromone decay
- `n_ants` - Number of ants in the colony

### Artificial Bee Colony (ABC)

ABC uses three types of bees (employed bees, onlooker bees, and scout bees) to search for the optimal solution, balancing exploration and exploitation.

Key parameters:
- `colony_size` - Number of food sources/employed bees
- `limit` - Maximum trials before abandonment
- `max_iterations` - Number of optimization cycles

### Particle Swarm Optimization (PSO)

PSO uses particles that move through the solution space, influenced by their own best known position and the swarm's best known position.

Key parameters:
- `w` - Inertia weight
- `c1` - Cognitive coefficient (personal best)
- `c2` - Social coefficient (global best)
- `n_particles` - Number of particles in the swarm


## Requirements

- Python 3.9+ or higher
- NumPy - For numerical operations
- Pandas - For data manipulation
- Matplotlib - For data visualization

## References

Real life implementaions:

### Ant Colony Optimization (ACO)
- [PyACO](https://github.com/ganyariya/PyACO) - A Python implementation of Ant Colony Optimization

### Particle Swarm Optimization (PSO)
- [PySwarms](https://github.com/ljvmiranda921/pyswarms) - A research toolkit for Particle Swarm Optimization in Python

### Artificial Bee Colony (ABC)
- [BeeColPy](https://github.com/renard162/BeeColPy) - A Python implementation of the Artificial Bee Colony algorithm
