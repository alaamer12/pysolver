"""
Artificial Bee Colony (ABC) Algorithm Implementation

This module implements the Artificial Bee Colony algorithm for solving optimization problems.
ABC is inspired by the foraging behavior of honey bees, consisting of employed bees, 
onlooker bees, and scout bees working together to find optimal food sources.

Key components:
1. Food source representation (candidate solutions)
2. Employed bee phase for neighborhood search
3. Onlooker bee phase for exploitation
4. Scout bee phase for exploration
5. Fitness evaluation and selection
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional
import copy
import random
import math

import numpy as np
import matplotlib.pyplot as plt

from ._problem import AlgConfig
from .feedback import Logger, progress

# Create a logger instance for this module
log = Logger(name="ABC")


@dataclass
class ABCConfig(AlgConfig):
    """Configuration parameters for the ABC algorithm."""
    objective_function: Callable[[List[float]], float]
    bounds: List[Tuple[float, float]]
    colony_size: int = 30
    limit: int = 20  # Max trials before abandoning a food source
    employed_ratio: float = 0.5  # Ratio of employed bees to total population
    mutation_rate: float = 1.0  # Rate controlling the neighborhood search size
    verbose: bool = True
    minimize: bool = True  # Whether to minimize or maximize the objective function
        
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.colony_size <= 0 or self.limit <= 0:
            raise ValueError("Colony size and limit must be positive integers.")
            
        # Calculate derived values
        self.dimensions = len(self.bounds)
        self.employed_count = max(1, int(self.colony_size * self.employed_ratio))
        self.onlooker_count = self.colony_size - self.employed_count


class FoodSource:
    """
    Represents a candidate solution (food source) in the ABC algorithm.
    """
    def __init__(self, dimensions: int, bounds: List[Tuple[float, float]]):
        """Initialize a food source with random position within bounds."""
        self.position = [random.uniform(low, high) for low, high in bounds]
        self.fitness = float('-inf')
        self.objective_value = 0.0
        self.trials = 0
        
    def evaluate(self, objective_function: Callable[[List[float]], float], minimize: bool) -> float:
        """Evaluate this food source using the objective function and calculate fitness."""
        self.objective_value = objective_function(self.position)
        self.fitness = self._calculate_fitness(self.objective_value, minimize)
        return self.fitness
    
    def _calculate_fitness(self, objective_value: float, minimize: bool = True) -> float:
        """Convert objective function value to fitness (higher is always better)."""
        if minimize:
            # For minimization: lower objective values get higher fitness
            if objective_value >= 0:
                return 1.0 / (1.0 + objective_value)
            else:
                return 1.0 + abs(objective_value)
        else:
            # For maximization: higher objective values get higher fitness
            if objective_value >= 0:
                return 1.0 + objective_value
            else:
                return 1.0 / (1.0 + abs(objective_value))
    
    def create_neighbor(self, bounds: List[Tuple[float, float]], 
                        other_sources: List['FoodSource'], 
                        mutation_rate: float = 1.0) -> 'FoodSource':
        """Create a neighboring food source by modifying one dimension of the position."""
        # Create a copy of this food source
        neighbor = copy.deepcopy(self)
        
        # Select a random dimension to modify
        dimension = random.randrange(len(self.position))
        
        # Select another random food source to use as reference
        other = random.choice(other_sources) if other_sources else self
        
        # Calculate the new position with a random factor
        phi = (random.random() * 2 - 1) * mutation_rate  # Random value between -1 and 1 * mutation_rate
        new_value = self.position[dimension] + phi * (self.position[dimension] - other.position[dimension])
        
        # Apply bounds
        lower, upper = bounds[dimension]
        new_value = max(lower, min(upper, new_value))
        
        # Update the neighbor's position
        neighbor.position[dimension] = new_value
        neighbor.trials = 0  # Reset trials for the new neighbor
        
        return neighbor
    
    def __repr__(self) -> str:
        """String representation of the food source."""
        pos_str = ', '.join(f"{p:.2f}" for p in self.position[:3])
        if len(self.position) > 3:
            pos_str += f", ... ({len(self.position) - 3} more)"
        return f"FoodSource(fit={self.fitness:.4f}, obj={self.objective_value:.4f}, trials={self.trials})"


class ABC:
    """
    Main ABC algorithm implementation.
    """
    def __init__(self, config: ABCConfig):
        """Initialize the ABC algorithm."""
        self.config = config
        self.food_sources = []
        self.best_source = None
        self.best_objective = float('inf') if config.minimize else float('-inf')
        self.objective_history = []
        
        # Initialize population of food sources
        self._initialize_population()
        
    def _initialize_population(self) -> None:
        """Initialize the population of food sources randomly within bounds."""
        log.info(f"Initializing {self.config.employed_count} food sources")
        
        with progress.simple("Initializing food sources", self.config.employed_count) as pbar:
            for _ in range(self.config.employed_count):
                food_source = FoodSource(self.config.dimensions, self.config.bounds)
                food_source.evaluate(self.config.objective_function, self.config.minimize)
                self.food_sources.append(food_source)
                
                # Update best solution if this one is better
                self._update_best_solution(food_source)
                
                pbar.update(1)
                
    def _update_best_solution(self, food_source: FoodSource) -> bool:
        """Update the best solution if the given food source is better."""
        is_better = False
        
        if self.config.minimize:
            is_better = food_source.objective_value < self.best_objective
        else:
            is_better = food_source.objective_value > self.best_objective
            
        if is_better:
            self.best_source = copy.deepcopy(food_source)
            self.best_objective = food_source.objective_value
            return True
            
        return False
    
    def optimize(self) -> Tuple[List[float], float, List[float]]:
        """Run the ABC optimization algorithm."""
        if self.config.verbose:
            log.info(f"Starting ABC optimization with {self.config.iterations} iterations")
        
        # Track initial best objective
        self.objective_history.append(self.best_objective)
        
        # Main optimization loop
        with progress.main("ABC Optimization", self.config.iterations) as pbar:
            for iteration in range(self.config.iterations):
                # Employed bee phase
                self._employed_bee_phase()
                
                # Onlooker bee phase
                self._onlooker_bee_phase()
                
                # Scout bee phase
                self._scout_bee_phase()
                
                # Record history and log progress
                self.objective_history.append(self.best_objective)
                
                if self.config.verbose and (iteration + 1) % 10 == 0:
                    objective_type = "min" if self.config.minimize else "max"
                    log.info(f"Iteration {iteration + 1}/{self.config.iterations} - "
                             f"Best {objective_type}: {self.best_objective:.4f}")
                
                pbar.update(1)
                
        if self.config.verbose:
            log.success(f"Optimization complete. Best value: {self.best_objective:.4f}")
        
        return self.best_source.position, self.best_objective, self.objective_history
                
    def _employed_bee_phase(self) -> None:
        """Employed bee phase: Each employed bee searches the neighborhood of its food source."""
        with progress.simple("Employed bee phase", len(self.food_sources), disable=not self.config.verbose) as pbar:
            for i, source in enumerate(self.food_sources):
                # Create a neighbor solution
                other_sources = [s for j, s in enumerate(self.food_sources) if j != i]
                neighbor = source.create_neighbor(self.config.bounds, other_sources, self.config.mutation_rate)
                
                # Evaluate the neighbor
                neighbor.evaluate(self.config.objective_function, self.config.minimize)
                
                # If neighbor is better, replace the current food source
                if neighbor.fitness > source.fitness:
                    self.food_sources[i] = neighbor
                    self._update_best_solution(neighbor)
                else:
                    # Increment trials counter for the current source
                    source.trials += 1
                
                pbar.update(1)
    
    def _onlooker_bee_phase(self) -> None:
        """Onlooker bee phase: Onlooker bees choose food sources based on their fitness."""
        # Calculate selection probabilities based on fitness values
        total_fitness = sum(source.fitness for source in self.food_sources)
        if total_fitness == 0:
            selection_probs = [1.0 / len(self.food_sources)] * len(self.food_sources)
        else:
            selection_probs = [source.fitness / total_fitness for source in self.food_sources]
        
        with progress.simple("Onlooker bee phase", self.config.onlooker_count, disable=not self.config.verbose) as pbar:
            for _ in range(self.config.onlooker_count):
                # Select a food source probabilistically
                selected_idx = random.choices(range(len(self.food_sources)), weights=selection_probs, k=1)[0]
                source = self.food_sources[selected_idx]
                
                # Create a neighbor solution
                other_sources = [s for j, s in enumerate(self.food_sources) if j != selected_idx]
                neighbor = source.create_neighbor(self.config.bounds, other_sources, self.config.mutation_rate)
                
                # Evaluate the neighbor
                neighbor.evaluate(self.config.objective_function, self.config.minimize)
                
                # If neighbor is better, replace the current food source
                if neighbor.fitness > source.fitness:
                    self.food_sources[selected_idx] = neighbor
                    self._update_best_solution(neighbor)
                else:
                    # Increment trials counter for the current source
                    source.trials += 1
                
                pbar.update(1)
    
    def _scout_bee_phase(self) -> None:
        """Scout bee phase: Replace abandoned food sources with new random ones."""
        with progress.simple("Scout bee phase", len(self.food_sources), disable=not self.config.verbose) as pbar:
            for i, source in enumerate(self.food_sources):
                # If this source has exceeded the trial limit, abandon it
                if source.trials >= self.config.limit:
                    # Create a new random food source
                    new_source = FoodSource(self.config.dimensions, self.config.bounds)
                    new_source.evaluate(self.config.objective_function, self.config.minimize)
                    
                    # Replace the abandoned source
                    self.food_sources[i] = new_source
                    
                    # Update best solution if needed
                    self._update_best_solution(new_source)
                    
                    if self.config.verbose:
                        log.info(f"Scout bee found new food source: {new_source.objective_value:.4f}")
                
                pbar.update(1)


def plot(history: List[float], title: str = "ABC Convergence", save_path: str = "abc_convergence.png", 
         minimize: bool = True) -> None:
    """Plot the convergence history of the ABC algorithm."""
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    iterations = range(len(history))
    plt.plot(iterations, history, 'b-', linewidth=2, label='Objective Value')
    
    # Add trend line (rolling average)
    window = min(10, len(history) // 5) if len(history) > 10 else 1
    if window > 1:
        trend = np.convolve(history, np.ones(window) / window, mode='valid')
        plt.plot(range(window-1, len(history)), trend, 'r--', linewidth=1.5, label=f'{window}-Iteration Moving Avg')
    
    # Mark best value
    if minimize:
        best_idx = np.argmin(history)
        best_val = min(history)
        ylabel = "Objective Value (lower is better)"
    else:
        best_idx = np.argmax(history)
        best_val = max(history)
        ylabel = "Objective Value (higher is better)"
    
    plt.scatter([best_idx], [best_val], color='green', s=100, 
                label=f'Best: {best_val:.4f}', zorder=5)
    
    # Styling
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save and show the plot
    plt.savefig(save_path, dpi=300)
    plt.show()


def compare(before_history: List[float], after_history: List[float],
            minimize: bool = True, title: str = "ABC Optimization Comparison", 
            save_path: str = "abc_comparison.png") -> None:
    """Compare the results of two ABC optimization runs."""
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot both histories
    before_iterations = range(len(before_history))
    after_iterations = range(len(after_history))
    plt.plot(before_iterations, before_history, 'r-', linewidth=2, alpha=0.7, label='Before')
    plt.plot(after_iterations, after_history, 'g-', linewidth=2, alpha=0.7, label='After')
    
    # Mark best values
    if minimize:
        best_before = min(before_history)
        best_before_idx = before_history.index(best_before)
        best_after = min(after_history)
        best_after_idx = after_history.index(best_after)
        improvement = (best_before - best_after) / best_before * 100 if best_before != 0 else 0
    else:
        best_before = max(before_history)
        best_before_idx = before_history.index(best_before)
        best_after = max(after_history)
        best_after_idx = after_history.index(best_after)
        improvement = (best_after - best_before) / best_before * 100 if best_before != 0 else 0
    
    plt.annotate(f"Improvement: {improvement:.2f}%",
                 xy=(0.5, 0.05), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
                 ha='center')
    
    # Styling
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Objective Value', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save and show the plot
    plt.savefig(save_path, dpi=300)
    plt.show()
