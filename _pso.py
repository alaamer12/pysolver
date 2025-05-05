"""
Particle Swarm Optimization (PSO) Algorithm Implementation

This module implements the Particle Swarm Optimization algorithm for solving continuous optimization problems.
PSO is inspired by social behavior of bird flocking or fish schooling, where particles move 
in the search space based on their own experience and the experience of the swarm.

Key components to implement:
1. Particle representation (position, velocity)
2. Personal best memory
3. Global best memory
4. Velocity update rule
5. Position update rule

Usage:
    from _pso import ParticleSwarmOptimization
    
    # Define objective function to minimize
    def objective_function(x):
        return sum(x**2)  # Example: minimize sum of squares
        
    # Define bounds for each parameter
    bounds = [(-5, 5), (-5, 5)]  # Example: 2D problem with bounds [-5, 5]
    
    # Create PSO instance
    config = PSOConfig(objective_function=objective_function, bounds=bounds)
    pso = ParticleSwarmOptimization(config)
    
    # Run optimization
    best_position, best_fitness = pso.optimize()
"""
from dataclasses import dataclass
from typing import Tuple , List, Callable    
import numpy as np
import random
from  matplotlib import pyplot as plt
from _problem import AlgConfig

@dataclass
class AlgConfig:
    """Base class for algorithm configuration"""
    def __post_init__(self) -> None:
        pass

@dataclass
class PSOConfig(AlgConfig):
    objective_function: Callable[[List[float]], float]
    bounds: List[tuple[float, float]]
    n_dimensions: int = None
    n_particles: int = 30
    iterations: int = 100
    w: float = 0.5
    c1: float = 2.0
    c2: float = 2.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_dimensions is None:
            self.n_dimensions = len(self.bounds)


class Particle:
    """
    Particle in the PSO algorithm.
    
    Each particle has a position, velocity, and memory of its best position.
    """
    def __init__(self, bounds: List[Tuple[float, float]], n_dimensions: int):

        self.position = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(n_dimensions)]
        
        
        self.velocity = [random.uniform(-1, 1) for _ in range(n_dimensions)]
        
        
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')  
        self.current_fitness = float('inf')
        
    def update_velocity(self, global_best_position: List[float], w: float, c1: float, c2: float) -> None:
        """Update the velocity of the particle"""
        for i in range(len(self.velocity)):
            # Inertia component
            inertia = w * self.velocity[i]
            
            # Cognitive component (personal best)
            r1 = random.random()
            cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            
            # Social component (global best)
            r2 = random.random()
            social = c2 * r2 * (global_best_position[i] - self.position[i])
            
            # Update velocity
            self.velocity[i] = inertia + cognitive + social
            
    def update_position(self, bounds: List[Tuple[float, float]]) -> None:
        """Update the position of the particle and handle boundary conditions"""
        for i in range(len(self.position)):
            # Update position
            self.position[i] += self.velocity[i]
            
            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]
                self.velocity[i] *= -0.5  # Bounce back with reduced velocity
            elif self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]
                self.velocity[i] *= -0.5  # Bounce back with reduced velocity
                
    def evaluate(self, objective_function: Callable[[List[float]], float]) -> float:
        """Evaluate the particle's current position and update best if improved"""
        self.current_fitness = objective_function(self.position)
        
        # Update personal best if current position is better
        if self.current_fitness < self.best_fitness:
            self.best_fitness = self.current_fitness
            self.best_position = self.position.copy()
            
        return self.current_fitness


class ParticleSwarmOptimization:
    """
    Main PSO algorithm implementation.
    
    Manages the swarm of particles and the optimization process.
    """
    def __init__(self, config: PSOConfig):
        self.config = config
        self.particles = []
        self.global_best_position = None
        self.global_best_fitness = float('inf') 
        self.fitness_history = []
        self.avg_fitness_history = []
        
        
        for _ in range(config.n_particles):
            self.particles.append(Particle(config.bounds, config.n_dimensions))
            
    def optimize(self) -> Tuple[List[float], float, List[float], List[float]]:
        """Run the optimization algorithm"""
        
        for particle in self.particles:
            fitness = particle.evaluate(self.config.objective_function)
            
            # Update global best if this particle is better
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()
        
        # Main optimization loop
        for i in range(self.config.iterations):
            current_best_fitness = self.global_best_fitness
            total_fitness = 0
            
            
            for particle in self.particles:
                particle.update_velocity(
                    self.global_best_position, 
                    self.config.w, 
                    self.config.c1, 
                    self.config.c2
                )
                particle.update_position(self.config.bounds)
                
                fitness = particle.evaluate(self.config.objective_function)
                total_fitness += fitness
                
                
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
            
            
            self.fitness_history.append(self.global_best_fitness)
            self.avg_fitness_history.append(total_fitness / len(self.particles))
            
            
            
        return (
            self.global_best_position, 
            self.global_best_fitness, 
            self.fitness_history, 
            self.avg_fitness_history
        )


# Visualization functions
def plot(history: List[float], avg_history: List[float] = None) -> None:
    """
    Plot the convergence of the algorithm over iterations.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Best Fitness', color='blue')
    
    if avg_history:
        plt.plot(avg_history, label='Average Fitness', color='red', linestyle='--')
    
    plt.title('PSO Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('pso_convergence.png')
    plt.show()


def compare(before_history: List[float], after_history: List[float],
            before_solution: List[int] = None, after_solution: List[int] = None,
            coordinates: List[Tuple[float, float]] = None,
            before_pheromones: np.ndarray = None, after_pheromones: np.ndarray = None,
            title: str = "Optimization Comparison", save_path: str = "optimization_comparison.png") -> None:
    """
    Compare the results of two optimization runs.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot convergence histories
    plt.subplot(2, 1, 1)
    plt.plot(before_history, label='Before', color='blue')
    plt.plot(after_history, label='After', color='red')
    plt.title(f'{title} - Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.legend()
    plt.grid(True)
    
    # Plot solutions if provided and coordinates are available
    if all(x is not None for x in [before_solution, after_solution, coordinates]):
        plt.subplot(2, 1, 2)
        
        # Extract coordinates
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]
        
        # Plot points
        plt.scatter(x_coords, y_coords, color='gray', alpha=0.5)
        
        # Plot before solution path
        before_x = [coordinates[i][0] for i in before_solution]
        before_y = [coordinates[i][1] for i in before_solution]
        if before_solution[0] == before_solution[-1]:  # Complete the loop for TSP
            before_x.append(before_x[0])
            before_y.append(before_y[0])
        plt.plot(before_x, before_y, 'b-', label='Before')
        
        # Plot after solution path
        after_x = [coordinates[i][0] for i in after_solution]
        after_y = [coordinates[i][1] for i in after_solution]
        if after_solution[0] == after_solution[-1]:  # Complete the loop for TSP
            after_x.append(after_x[0])
            after_y.append(after_y[0])
        plt.plot(after_x, after_y, 'r-', label='After')
        
        plt.title(f'{title} - Solutions')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True)
    
    # Plot pheromone levels if provided (for ACO)
    elif all(x is not None for x in [before_pheromones, after_pheromones]):
        plt.subplot(2, 2, 3)
        plt.imshow(before_pheromones, cmap='viridis')
        plt.title('Before Pheromones')
        plt.colorbar()
        
        plt.subplot(2, 2, 4)
        plt.imshow(after_pheromones, cmap='viridis')
        plt.title('After Pheromones')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()