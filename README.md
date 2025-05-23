# PySolver Project

This project implements three popular nature-inspired optimization algorithms:

1. **Ant Colony Optimization (ACO)** - Inspired by the foraging behavior of ants, excellent for discrete optimization problems
2. **Artificial Bee Colony (ABC)** - Based on the intelligent foraging behavior of honey bees, suitable for continuous optimization
3. **Particle Swarm Optimization (PSO)** - Mimics the social behavior of bird flocking or fish schooling, ideal for continuous optimization

## Project Structure

```
.
├── README.md               # Project documentation
├── main.py                 # Main script with benchmarks and comparisons
├── feedback.py             # Logging and progress tracking utilities
├── _aco.py                 # Ant Colony Optimization implementation
├── _problem.py             # Problem definition interface
├── __abc.py                # Artificial Bee Colony implementation 
├── _pso.py                 # Particle Swarm Optimization implementation
├── requirements.txt        # Project dependencies
├── assets/                 # Visualization and result assets
│   ├── aco/               # ACO-specific visualizations
│   ├── abc/               # ABC-specific visualizations
│   └── pso/               # PSO-specific visualizations
├── docs/                   # Documentation and presentations
└── examples/               # Example implementations and notebooks
    ├── continuous/        # Examples for continuous optimization
    └── static/            # Static problem examples
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
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/macOS
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running a Specific Algorithm

```bash
# Run PSO algorithm with custom parameters
python main.py --algorithm pso --kwargs

# Run ABC algorithm with default parameters
python main.py --algorithm abc 

# Auto-detect and use the best algorithm for your problem
python main.py go
```

### Defining Your Own Problem

To use these optimization algorithms with your own problem, implement the `ProblemProtocol` interface from `_problem.py`:

```python
from _problem import ProblemProtocol

class MyProblem(ProblemProtocol):
    @property
    def num_components(self) -> int:
        # Return the number of components in your problem
        
    def get_heuristic(self, i: int, j: int) -> float:
        # Return the heuristic value between components i and j
        
    def is_valid_solution(self, solution: List[int]) -> bool:
        # Check if a solution is valid
        
    def evaluate_solution(self, solution: List[int]) -> float:
        # Evaluate the quality of a solution
```

### Using the Feedback Module

The project includes a sophisticated feedback module for logging and progress tracking:

#### Logging

```python
from feedback import log

# Different logging levels
log.debug("Debug information")
log.info("General information")
log.success("Operation completed successfully")
log.warn("Warning message")
log.error("Error occurred")
log.critical("Critical error")
```

#### Progress Tracking

```python
from feedback import progress

# Main progress bar (remains visible after completion)
with progress.main("Optimization Progress", total=100) as main_pbar:
    for i in range(100):
        # Simple progress bar (vanishes after completion)
        with progress.simple(f"Iteration {i+1}", total=50) as pbar:
            for j in range(50):
                # Your optimization code here
                pbar.update(1)
        main_pbar.update(1)
```

Progress bars features:
- Main progress bars (`progress.main()`) - Remain visible after completion
- Simple progress bars (`progress.simple()`) - Vanish after completion
- Customizable colors, descriptions, and formats
- Nested progress tracking support
- Dynamic terminal width adaptation

## Algorithms Overview

### Ant Colony Optimization (ACO)

ACO is particularly effective for discrete optimization problems like the Traveling Salesman Problem (TSP). The algorithm works by simulating ants leaving pheromone trails as they search for efficient paths.

Key parameters:
- `alpha` - Pheromone importance (typical values: 1.0-3.0)
- `beta` - Heuristic information importance (typical values: 2.0-5.0)
- `evaporation_rate` - Controls pheromone decay (typical values: 0.1-0.5)
- `n_ants` - Number of ants in the colony (typical values: 10-100)
- `iterations` - Number of optimization cycles

ACO features:
- Probabilistic decision making based on pheromone levels and heuristic information
- Reinforcement learning through pheromone deposits
- Implicit parallelism through multiple ant agents
- Sophisticated visualization tools for solution paths and pheromone levels

### Artificial Bee Colony (ABC)

ABC simulates the foraging behavior of honey bees, using three types of bees (employed bees, onlooker bees, and scout bees) to efficiently search for optimal food sources. This algorithm is particularly effective for continuous function optimization problems.

Key parameters:
- `colony_size` - Total number of bees in the colony (typical values: 20-100)
- `employed_ratio` - Proportion of employed bees to total colony (typical value: 0.5)
- `limit` - Maximum trials before abandonment (typical values: 10-50)
- `mutation_rate` - Controls neighborhood search size (typical values: 0.5-2.0)
- `iterations` - Number of optimization cycles

ABC features:
- Neighborhood search through employed bees
- Selection of promising regions through onlooker bees via fitness-proportionate selection
- Exploration of new regions through scout bees when food sources are exhausted
- Balance between exploitation and exploration through the three bee types
- Effective for high-dimensional continuous optimization problems
- Robust against local optima in multimodal functions

Example usage:
```python
from pysolver import ABC, ABCConfig

# Define your objective function
def sphere_function(x):
    return sum(xi**2 for xi in x)

# Configure the ABC algorithm
config = ABCConfig(
    objective_function=sphere_function,
    bounds=[(-5, 5)] * 10,  # 10-dimensional problem with bounds [-5, 5]
    colony_size=30,
    limit=20,
    iterations=200,
    mutation_rate=1.0,
    minimize=True  # Set to False for maximization problems
)

# Create ABC instance and optimize
abc = ABC(config)
best_position, best_value, history = abc.optimize()

# Display results
print(f"Best solution: {best_position}")
print(f"Best value: {best_value}")
```

### Particle Swarm Optimization (PSO)

PSO uses particles that move through the solution space, influenced by their own best known position and the swarm's best known position.

Key parameters:
- `w` - Inertia weight (typical values: 0.4-0.9)
- `c1` - Cognitive coefficient - personal best (typical values: 1.0-2.5)
- `c2` - Social coefficient - global best (typical values: 1.0-2.5)
- `n_particles` - Number of particles in the swarm (typical values: 10-50)
- `iterations` - Number of optimization cycles

PSO features:
- Simple implementation with powerful optimization capabilities
- Excellent for continuous optimization problems
- Balance between exploration and exploitation through velocity update
- No gradient information required
- Effective for high-dimensional problems

## Experiment Results

For detailed experimental results, benchmarks, and visualizations, see the [EXPERIMENTS.md](EXPERIMENTS.md) file.

## Requirements

- Python 3.9+ or higher
- NumPy - For numerical operations
- Pandas - For data manipulation
- Matplotlib - For data visualization
- tqdm - For progress tracking
- colorama - For terminal output styling

## References

### Real-life implementations

#### Ant Colony Optimization (ACO)
- [PyACO](https://github.com/ganyariya/PyACO) - A Python implementation of Ant Colony Optimization
- [antsys](https://github.com/ganyariya/antsys) - Another Python library for Ant Colony Optimization, optimized to solve the Traveling Salesman Problem (TSP)

#### Particle Swarm Optimization (PSO)
- [PySwarms](https://github.com/ljvmiranda921/pyswarms) - A research toolkit for Particle Swarm Optimization in Python

#### Artificial Bee Colony (ABC)
- [BeeColPy](https://github.com/renard162/BeeColPy) - A Python implementation of the Artificial Bee Colony algorithm
- [pyABC](https://github.com/bischtob/pyabc) - A framework for Artificial Bee Colony optimization

## Documentation

A detailed presentation about the project and algorithms can be found in the `docs` folder or [view online](https://www.canva.com/design/DAGk6GZUskE/jvIR0tYTOLtgYwQGMYs0YA/view).
