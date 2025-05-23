# PySolver Experimental Results

This document provides detailed information about experiments performed with the PySolver algorithms, including visualizations, benchmark comparisons, and performance analysis.

## Ant Colony Optimization (ACO) Experiments

ACO was tested on a variety of discrete optimization problems, with the Traveling Salesman Problem (TSP) being the primary benchmark.

### Convergence Analysis

![ACO Convergence](assets/aco/convergence.png)

The convergence plot shows how the ACO algorithm improves solution quality over iterations. Key observations:

- **Initial configuration**: Starts with a solution quality around 1000 and quickly improves to ~700 within the first 10 iterations, but then shows high volatility with minimal improvement
- **Optimized configuration**: Shows dramatically faster convergence, reaching its best solution quality of 482.58 by iteration 30
- **Stability comparison**: Initial configuration shows continuous fluctuations between 600-800 throughout all 200 iterations, while the optimized version maintains stable performance around 500-520 after finding its best solution
- **Overall improvement**: The optimized algorithm achieves a 15.17% better solution quality (482.58 vs. 568.85)

### Solution Comparison

![Solution Comparison](assets/aco/sol_comparison.png)

The visualization compares the actual tours found by initial and optimized configurations:

- **Initial solution**: Total distance of 568.85 units
- **Optimized solution**: Total distance of 482.58 units
- **Improvement percentage**: 15.17% reduction in tour length
- **Path characteristics**: The optimized solution shows more logical connections between adjacent cities, fewer crossings, and more efficient overall routing

### Parameter Impact

Comparing initial and optimized configurations:

![Initial Configuration](assets/aco/initial_config.png)
![Optimized Configuration](assets/aco/optimized_config.png)

Detailed parameter analysis:

| Parameter | Initial Value | Optimized Value | Impact |
|-----------|---------------|----------------|--------|
| n_ants | 15 | 60 | 4× more exploration capability |
| alpha (pheromone influence) | 0.5 | 2.0 | 4× higher pheromone sensitivity |
| beta (heuristic influence) | 1.0 | 6.0 | 6× stronger preference for shorter edges |
| evaporation_rate | 0.8 | 0.1 | 8× slower pheromone decay |
| initial_pheromone | 0.1 | 1.0 | 10× higher initial pheromone levels |

Key findings:
- The initial configuration shows high volatility with solution quality fluctuating between 600-800
- The optimized configuration finds its best solution (482.58) much earlier (iteration 30 vs. 100+)
- The initial configuration's rolling average remains unstable throughout all iterations
- The optimized configuration's rolling average stabilizes around 502, maintaining consistent performance

### ACO vs. Neural Networks

![ACO vs NN](assets/aco/aco_vs_nn.png)

Performance comparison between ACO and neural network approaches:

- For small problems (20-30 cities): ACO outperforms neural networks by 15-25%
- For medium problems (30-100 cities): ACO maintains 5-15% advantage with proper tuning
- For large problems (100+ cities): Neural networks begin to scale better, especially with specialized architectures
- Computation time: ACO requires less training time but more inference time per problem instance

## Particle Swarm Optimization (PSO) Experiments

PSO was tested on standard continuous optimization benchmark functions, including Rosenbrock, Rastrigin, and Ackley functions.

### Convergence on Different Functions

![PSO on Quadratic Functions](assets/pso/Convergence_of_pso_on_quadratic_functions.png)
![PSO on Rosenbrock Function](assets/pso/Convergence_of_pso_on_rosenbrock_function.png)

Detailed convergence analysis:

**Quadratic Function:**
- Best fitness: Near-optimal (< 0.1) reached within 5 iterations
- Average fitness: Converges to near-optimal (< 0.1) within 15 iterations
- Initial fitness: Starting from approximately 25.0
- Convergence pattern: Smooth, exponential decay without plateaus

**Rosenbrock Function:**
- Best fitness: Begins around 10^0, reaches 10^-9 after 200 iterations
- Average fitness: Much slower convergence, still around 10^0 after 200 iterations
- Convergence pattern: Step-wise improvements on logarithmic scale with distinct plateaus
- Note: Log scale visualization necessary due to the dramatic range of improvement (9 orders of magnitude)

### Performance on Multimodal Functions

![Rastrigin Function](assets/pso/Rastrigin.png)
![Ackley Function](assets/pso/Ackly.png)

**Rastrigin Function:**
- Initial best fitness: ~36
- Final best fitness: Near 0 (< 0.1) after ~60 iterations
- Average fitness: Begins at ~80, drops to ~0 after 200-250 iterations
- Convergence pattern: Step-wise improvements as particles escape local optima
- Notable feature: Sharp drop in average fitness around iteration 50-60 indicating swarm convergence

**Ackley Function:**
- Initial best fitness: ~6.5
- Final best fitness: Near 0 (< 0.1) after ~20 iterations
- Average fitness: Begins at ~17.5, gradually decreases to ~0 by iteration 80
- Convergence pattern: Smooth for best fitness, more gradual for average fitness
- Performance note: Despite Ackley's deceptive landscape with many local minima, PSO finds the global minimum efficiently

### Parameter Impact Analysis

![Parameter Impact](assets/pso/diff_paramter_impact.png)

Systematic analysis of parameter influence on convergence:

| Parameter Configuration | w | c1 | c2 | Convergence Speed | Final Quality | Notable Characteristics |
|-------------------------|---|----|----|-------------------|---------------|------------------------|
| Low values | 0.4 | 1.0 | 1.0 | Fastest | Good | Rapid early improvement |
| Medium values | 0.6 | 1.5 | 1.5 | Moderate | Good | Balanced performance |
| High values | 0.8 | 2.0 | 2.0 | Slowest | Worst | Step-wise improvements with plateaus |
| Higher cognitive | 0.7 | 2.0 | 1.0 | Moderate-Fast | Good | Strong local search capability |
| Higher social | 0.7 | 1.0 | 2.0 | Moderate | Good | Faster global convergence |

Key observations:
- Low w values (0.4) show fastest initial convergence, reaching near-optimal solutions within 10 iterations
- High w values (0.8) result in slower convergence with dramatic step-wise improvements
- Balanced configurations reach optimal solutions by iteration 40 regardless of starting performance
- Higher cognitive coefficient (c1=2.0) improves individual particle efficiency
- Higher social coefficient (c2=2.0) accelerates swarm consensus but may lead to premature convergence

## Implementation Examples

### TSP Example with ACO

From `examples/static/tsp_example.py`, we demonstrate solving a 30-city TSP problem:

```python
# Generate a random TSP instance
num_cities = 30
cities = generate_random_cities(num_cities)
tsp = TSPProblem(cities)

# Initial configuration
initial_config = ACOConfig(
    problem=tsp,
    n_ants=15,             # Very few ants
    alpha=0.5,             # Lower pheromone influence
    beta=1.0,              # Low heuristic influence
    evaporation_rate=0.8,  # Very high evaporation
    iterations=100
)

# Optimized configuration
optimized_config = ACOConfig(
    problem=tsp,
    n_ants=60,             # Many more ants
    alpha=2.0,             # Higher pheromone influence
    beta=6.0,              # Much higher heuristic influence
    evaporation_rate=0.1,  # Much lower evaporation rate
    iterations=100
)

# Results: Initial solution quality: 568.85, Optimized: 482.58 (15.17% improvement)
```

The visualization results clearly show how the optimized configuration not only achieves a better final solution but also demonstrates more stable performance throughout the optimization process. The dramatic reduction in evaporation rate (from 0.8 to 0.1) proves particularly effective at maintaining good solutions once they are found.

### Continuous Optimization with PSO

The PSO algorithm was evaluated on several benchmark functions with the following specific results:

- **Sphere Function (Quadratic)**: 
  - Convergence to within 0.1 of global optimum within 5 iterations (best fitness)
  - Average swarm fitness reaches near-optimal within 15 iterations
  - Final fitness values: 10^-16 magnitude after 100 iterations

- **Rosenbrock Function**: 
  - Best fitness improves from 10^0 to 10^-9 over 200 iterations
  - Average fitness remains challenging (10^0 to 10^1 range)
  - Shows PSO's ability to handle difficult fitness landscapes with narrow valleys

- **Rastrigin Function**: 
  - Best fitness converges to near-global optimum in distinct steps
  - Notable plateau between iterations 10-40 around fitness value of 9
  - Step change at iteration ~50 indicates successful escaping of local minimum
  - Final best fitness < 0.1 indicating successful global optimization

- **Ackley Function**: 
  - Global optimum (< 0.1) reached within 20 iterations despite highly deceptive landscape
  - Average fitness shows consistent progress, reaching < 0.5 by iteration 80
  - Demonstrates PSO's effectiveness on functions with numerous local optima

## Benchmark Results Summary

| Algorithm | Problem Type      | Best For                           | Measured Improvement | Iterations to Best | Computational Cost |
|-----------|-------------------|-----------------------------------|---------------------|-------------------|-------------------|
| ACO       | Discrete (TSP)    | Problems with ~20-100 components  | 15.17% (30 cities)  | 30-100            | Medium-High       |
| PSO       | Continuous (Sphere) | Low-dimensional (2-30) problems | >99.99% (10^-16 magnitude) | 5-10      | Low              |
| PSO       | Continuous (Rosenbrock) | Hard optimization landscapes | >99.99% (10^-9 magnitude) | 150-200   | Medium            |
| PSO       | Continuous (Rastrigin) | Multimodal functions        | >99% (< 0.1 final) | 50-100           | Medium            |
| PSO       | Continuous (Ackley)   | Deceptive landscapes         | >99% (< 0.1 final) | 15-25            | Low-Medium        |
| ABC       | Continuous        | Multimodal functions              | 20-40% (estimated)  | 50-100           | Medium            |

## Future Work

1. **Hybrid Algorithms**: 
   - Combine ACO with local search techniques for post-optimization refinement
   - Integrate PSO with gradient-based methods for faster final convergence
   - Explore ACO-PSO hybrids for mixed discrete-continuous problems

2. **Parallel Implementations**: 
   - Utilize GPU acceleration for larger problems (especially for ACO with 100+ components)
   - Implement island model parallelization for PSO to maintain diversity
   - Explore distributed computing approaches for industrial-scale problems

3. **Dynamic Problems**:
   - Adapt ACO pheromone update mechanisms for changing environments
   - Implement memory mechanisms in PSO for tracking shifting optima
   - Develop adaptive parameter tuning based on detected environmental changes

4. **Advanced Visualization**: 
   - Create interactive tools for solution analysis and parameter tuning
   - Implement 3D visualizations for higher-dimensional problems
   - Develop real-time convergence monitoring and early stopping criteria

## Documentation Resources

For comprehensive documentation about the algorithms and their implementation, refer to:
- [Full PDF Documentation](docs/PySolver.pdf)
- [Online Presentation](https://www.canva.com/design/DAGk6GZUskE/jvIR0tYTOLtgYwQGMYs0YA/view)
- Jupyter notebooks in the `examples` directory with detailed step-by-step guidance
- Interactive examples in `examples/static/aco_example_run.ipynb` and `examples/continuous/pso_examples_run.ipynb` 