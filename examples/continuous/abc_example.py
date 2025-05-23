"""
Artificial Bee Colony (ABC) Algorithm Example

This example demonstrates how to use the ABC algorithm to solve continuous optimization problems.
"""
import math
import random
import numpy as np
from typing import List

from pysolver import ABC, ABCConfig, plot


def sphere_function(x: List[float]) -> float:
    """
    Sphere function - a simple minimization test function.
    f(x) = sum(x_i^2)
    Global minimum at x = [0, 0, ..., 0]
    """
    return sum(xi**2 for xi in x)


def rosenbrock_function(x: List[float]) -> float:
    """
    Rosenbrock function - a non-convex minimization test function.
    f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    Global minimum at x = [1, 1, ..., 1]
    """
    result = 0
    for i in range(len(x) - 1):
        result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return result


def rastrigin_function(x: List[float]) -> float:
    """
    Rastrigin function - a highly multimodal function with many local minima.
    f(x) = 10*n + sum(x_i^2 - 10*cos(2*π*x_i))
    Global minimum at x = [0, 0, ..., 0]
    """
    n = len(x)
    return 10 * n + sum(xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x)


def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Define problem parameters
    dimensions = 10
    bounds = [(-5.12, 5.12)] * dimensions  # Same bounds for all dimensions
    iterations = 200
    
    print("\n===== Artificial Bee Colony Optimization =====")
    print(f"Dimensions: {dimensions}")
    print(f"Iterations: {iterations}")
    print(f"Bounds: [{bounds[0][0]}, {bounds[0][1]}]^{dimensions}")
    
    # Optimize the sphere function with default ABC settings
    print("\n1. Optimizing Sphere Function (Default Settings)")
    
    config = ABCConfig(
        objective_function=sphere_function,
        bounds=bounds,
        iterations=iterations,
        colony_size=30,
        limit=20,
        employed_ratio=0.5,
        mutation_rate=1.0,
        verbose=True,
        minimize=True
    )
    
    abc = ABC(config)
    best_position, best_value, history = abc.optimize()
    
    print(f"\nBest solution found:")
    print(f"Position: {[round(x, 6) for x in best_position[:5]]}...")  # Show only first 5
    print(f"Value: {best_value:.10f}")
    print(f"Known global minimum: 0.0 at [0, 0, ..., 0]")
    
    # Visualize the results
    plot(history, title="ABC on Sphere Function", save_path="abc_sphere_convergence.png")
    
    # Now try with harder function - Rosenbrock
    print("\n2. Optimizing Rosenbrock Function (Improved Settings)")
    
    # Use different settings for this harder problem
    config_rosenbrock = ABCConfig(
        objective_function=rosenbrock_function,
        bounds=[(-5, 10)] * dimensions,
        iterations=iterations,
        colony_size=60,       # More bees
        limit=30,             # Higher limit before abandonment 
        employed_ratio=0.5,
        mutation_rate=2.0,    # Higher mutation rate for better exploration
        verbose=True,
        minimize=True
    )
    
    abc_rosenbrock = ABC(config_rosenbrock)
    best_position_rosenbrock, best_value_rosenbrock, history_rosenbrock = abc_rosenbrock.optimize()
    
    print(f"\nBest solution found for Rosenbrock function:")
    print(f"Position: {[round(x, 6) for x in best_position_rosenbrock[:5]]}...")
    print(f"Value: {best_value_rosenbrock:.10f}")
    print(f"Known global minimum: 0.0 at [1, 1, ..., 1]")
    
    # Visualize the results
    plot(history_rosenbrock, title="ABC on Rosenbrock Function", save_path="abc_rosenbrock_convergence.png")
    
    # Finally try with a multimodal function - Rastrigin
    print("\n3. Optimizing Rastrigin Function (Many Local Optima)")
    
    config_rastrigin = ABCConfig(
        objective_function=rastrigin_function,
        bounds=[(-5.12, 5.12)] * dimensions,
        iterations=iterations,
        colony_size=100,      # Even more bees for this complex landscape
        limit=20,
        employed_ratio=0.5,
        mutation_rate=1.5,    # Balance between exploitation and exploration
        verbose=True,
        minimize=True
    )
    
    abc_rastrigin = ABC(config_rastrigin)
    best_position_rastrigin, best_value_rastrigin, history_rastrigin = abc_rastrigin.optimize()
    
    print(f"\nBest solution found for Rastrigin function:")
    print(f"Position: {[round(x, 6) for x in best_position_rastrigin[:5]]}...")
    print(f"Value: {best_value_rastrigin:.10f}")
    print(f"Known global minimum: 0.0 at [0, 0, ..., 0]")
    
    # Visualize the results
    plot(history_rastrigin, title="ABC on Rastrigin Function", save_path="abc_rastrigin_convergence.png")


if __name__ == "__main__":
    main() 