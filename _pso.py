from dataclasses import dataclass
from typing import Tuple, List, Callable    
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import animation, cm
from tqdm.auto import tqdm

@dataclass
class AlgConfig:
    """
    Base class for algorithm configuration.
    
    This class can be extended for specific algorithms to include common parameters.
    """
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
        
        
        for _ in tqdm(range(config.n_particles), desc="Initializing particles", leave=False):
            self.particles.append(Particle(config.bounds, config.n_dimensions))
            
    def optimize(self) -> Tuple[List[float], float, List[float], List[float]]:
        """Run the optimization algorithm"""
        
        for particle in tqdm(self.particles, desc="Initial evaluation", leave=False):
            fitness = particle.evaluate(self.config.objective_function)
            
            # Update global best if this particle is better
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()
        
        # Main optimization loop
        for i in tqdm(range(self.config.iterations), desc="PSO optimization", leave=True):
            current_best_fitness = self.global_best_fitness
            total_fitness = 0
            
            
            for particle in tqdm(self.particles, desc=f"Iteration {i+1}/{self.config.iterations}", leave=False):
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
def plot_convergence(fitness_history, avg_fitness_history=None, title="PSO Convergence"):
    """Plot the convergence of the PSO algorithm"""
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, 'b-', linewidth=2, label='Global Best Fitness')
    if avg_fitness_history:
        plt.plot(avg_fitness_history, 'r--', linewidth=1, label='Average Swarm Fitness')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Use log scale for better visualization
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


def plot_2d_function(func, bounds, resolution=100, title="Function Surface"):
    """Plot a 2D function surface"""
    x = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in tqdm(range(resolution), desc="Calculating function values", leave=False):
        for j in range(resolution):
            Z[i, j] = func([X[i, j], Y[i, j]])

    fig = plt.figure(figsize=(12, 5))

    # 3D Surface plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surface = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(X, Y)')
    ax1.set_title(f'{title} - 3D View')
    fig.colorbar(surface, ax=ax1, shrink=0.5, aspect=5)

    # Contour plot
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(X, Y, Z, 50, cmap=cm.viridis)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'{title} - Contour View')
    fig.colorbar(contour, ax=ax2)

    plt.tight_layout()
    plt.show()


def visualize_pso_2d(func, bounds, position_history, global_best_pos, title="PSO Optimization"):
    """Visualize PSO particles movement in 2D space"""
    def sphere_function(x: List[float]) -> float:
        return sum(xi**2 for xi in x)
    def himmelblau_function(x: List[float]) -> float:
        if len(x) != 2:
            raise ValueError("Himmelblau function requires exactly 2 dimensions")
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
    # Create function surface
    resolution = 100
    x = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in tqdm(range(resolution), desc="Generating visualization surface", leave=False):
        for j in range(resolution):
            Z[i, j] = func([X[i, j], Y[i, j]])

    # Plot settings
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, Z, 50, cmap=cm.viridis, alpha=0.8)
    fig.colorbar(contour, ax=ax)

    # Plot particle positions for each iteration
    colors = plt.cm.jet(np.linspace(0, 1, len(position_history)))

    for i in tqdm(range(len(position_history)), desc="Plotting iterations", leave=False):
        positions = position_history[i]
        x_pos = [p[0] for p in positions]
        y_pos = [p[1] for p in positions]

        # Use smaller markers and decreasing alpha for earlier iterations
        alpha = 0.3 + 0.7 * (i / len(position_history))
        size = 10 + 40 * (i / len(position_history))

        # Only plot some iterations to avoid cluttering
        if i % max(1, len(position_history) // 10) == 0 or i == len(position_history) - 1:
            ax.scatter(x_pos, y_pos, color=colors[i], s=size, alpha=alpha,
                       edgecolors='k', linewidths=0.5, label=f'Iteration {i}')

    # Mark the global best position
    ax.scatter(global_best_pos[0], global_best_pos[1], color='red', s=200, marker='*',
               edgecolors='k', linewidths=1.5, label='Global Best')

    # Mark the true optimum if using a test function
    if func == sphere_function:
        ax.scatter(0, 0, color='white', s=200, marker='x', linewidths=2, label='True Optimum')
    elif func == himmelblau_function:
        # Himmelblau has four optima
        optima = [(3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)]
        for opt in optima:
            ax.scatter(opt[0], opt[1], color='white', s=100, marker='+', linewidths=2)
        ax.scatter([], [], color='white', s=100, marker='+', linewidths=2, label='True Optima')  # For legend

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)

    # Show only a few iterations in the legend to avoid cluttering
    handles, labels = ax.get_legend_handles_labels()
    selected_indices = [i for i in range(len(labels)) if 'Iteration' not in labels[i] or
                        int(labels[i].split()[-1]) in [0, len(position_history)//2, len(position_history)-1]]
    ax.legend([handles[i] for i in selected_indices], [labels[i] for i in selected_indices],
              loc='upper right')

    plt.tight_layout()
    plt.show()
