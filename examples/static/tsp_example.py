"""
TSP Example using Ant Colony Optimization

This example demonstrates how to use the ACO algorithm to solve the Traveling Salesman Problem.
"""
import random
from typing import List, Tuple

import numpy as np

from pysolver import ACO, ACOConfig, plot, plot_solution, plot_pheromone_heatmap
from pysolver import ProblemProtocol
from pysolver import log


class TSPProblem(ProblemProtocol):
    """Implementation of the Traveling Salesman Problem for ACO."""

    def __init__(self, cities: List[Tuple[float, float]]):
        """
        Initialize a TSP problem instance.
        
        Args:
            cities: List of (x, y) coordinates for each city
        """
        self.cities = cities
        self.distances = self._calculate_distances()

    def _calculate_distances(self) -> np.ndarray:
        """Calculate the distance matrix between all cities."""
        n = len(self.cities)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Euclidean distance between cities
                    x1, y1 = self.cities[i]
                    x2, y2 = self.cities[j]
                    distances[i, j] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        return distances

    @property
    def num_components(self) -> int:
        """Return the number of cities."""
        return len(self.cities)

    def get_heuristic(self, i: int, j: int) -> float:
        """
        Return the heuristic value between cities i and j.
        
        For TSP, the heuristic is typically the inverse of distance.
        """
        if i == j:
            return 0.0
        
        # Set a minimum distance to avoid division by zero
        # The original approach tried to use max(0.1, distance) but there may be
        # cases where distance calculation is still producing values too close to zero
        distance = float(self.distances[i, j])
        if distance < 1e-10:  # Use a smaller epsilon for numerical stability
            return 10.0  # High heuristic value for very close cities
        
        return 1.0 / distance  # No need for max() if we check explicitly

    def is_valid_solution(self, solution: List[int]) -> bool:
        """
        Check if a solution is valid.
        
        For TSP:
        - A complete solution visits each city exactly once
        - A partial solution (during construction) should not have duplicate cities
        """
        # For partial solutions during construction, just check for duplicates
        if len(solution) < self.num_components:
            return len(solution) == len(set(solution))  # No duplicate cities
            
        # For complete solutions, check if all cities are included exactly once
        return len(solution) == self.num_components and len(set(solution)) == self.num_components

    def evaluate_solution(self, solution: List[int]) -> float:
        """
        Evaluate the quality of a solution.
        
        For TSP, this is the total distance of the tour.
        """
        if not solution or len(solution) <= 1:
            return float('inf')  # Invalid solution
            
        total_distance = 0.0

        # Calculate the distance of the tour
        for i in range(len(solution) - 1):
            from_city = solution[i]
            to_city = solution[i + 1]
            total_distance += self.distances[from_city, to_city]

        # Add the distance back to the starting city for a complete tour
        if len(solution) == self.num_components:
            total_distance += self.distances[solution[-1], solution[0]]

        return total_distance


def generate_random_cities(n: int, max_coord: float = 100.0) -> List[Tuple[float, float]]:
    """Generate n random cities with coordinates between 0 and max_coord.
    
    Uses a better distribution to avoid clustering and very close cities.
    """
    # Set a minimum distance between cities to avoid numerical issues
    min_distance = max_coord / (n * 0.5)
    cities = []
    
    while len(cities) < n:
        # Generate a candidate city
        new_city = (random.uniform(0, max_coord), random.uniform(0, max_coord))
        
        # Check if it's far enough from existing cities
        valid = True
        for city in cities:
            dx = new_city[0] - city[0]
            dy = new_city[1] - city[1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < min_distance:
                valid = False
                break
        
        # Add if valid
        if valid or len(cities) == 0:  # Always add the first city
            cities.append(new_city)
    
    return cities


def main():
    # Generate a more challenging random TSP instance with 30 cities
    num_cities = 30  # Increased from 20 to create a more challenging problem
    cities = generate_random_cities(num_cities)

    # Create the TSP problem instance
    tsp = TSPProblem(cities)

    # Set random seed for the initial run 
    random.seed(42)
    np.random.seed(42)

    # Configure the ACO algorithm with initial parameters (deliberately less effective)
    initial_config = ACOConfig(
        problem=tsp,
        n_ants=15,             # Very few ants
        alpha=0.5,             # Lower pheromone influence
        beta=1.0,              # Low heuristic influence
        pheromone_deposit=0.5, # Low pheromone deposit
        evaporation_rate=0.8,  # Very high evaporation (pheromones disappear quickly)
        initial_pheromone=0.1, # Low initial pheromone
        iterations=100,        # Same iterations for fair comparison
        rand_start=True,
        verbose=True           # Show progress
    )

    log.info("Running ACO with initial configuration...")
    
    # Create and run the ACO algorithm with initial parameters
    initial_aco = ACO(initial_config)
    initial_solution, initial_quality, initial_history = initial_aco.optimize()
    
    # Print the initial solution
    print("\nInitial solution found:")
    print(f"Path: {initial_solution}")
    print(f"Total distance: {initial_quality:.2f}")
    
    # Reset random seed for the optimized run to ensure different initialization
    random.seed(123)
    np.random.seed(123)
    
    # Configure the ACO algorithm with highly optimized parameters
    optimized_config = ACOConfig(
        problem=tsp,
        n_ants=60,             # Many more ants for better exploration
        alpha=2.0,             # Higher pheromone influence
        beta=6.0,              # Much higher heuristic influence to favor shortest paths
        pheromone_deposit=2.0, # Higher pheromone deposit
        evaporation_rate=0.1,  # Much lower evaporation rate for better convergence
        initial_pheromone=1.0, # Higher initial pheromone
        iterations=100,        # Same iterations for fair comparison
        rand_start=True,
        verbose=True           # Show progress
    )

    log.info("Running ACO with optimized configuration...")
    
    # Create and run the ACO algorithm with optimized parameters
    optimized_aco = ACO(optimized_config)
    optimized_solution, optimized_quality, optimized_history = optimized_aco.optimize()
    
    # Print the optimized solution
    print("\nOptimized solution found:")
    print(f"Path: {optimized_solution}")
    print(f"Total distance: {optimized_quality:.2f}")
    
    # Calculate improvement percentage
    improvement = (initial_quality - optimized_quality) / initial_quality * 100
    print(f"\nImprovement: {improvement:.2f}%")
    
    # Visualize individual results
    plot(initial_history, title="Initial Configuration Convergence", save_path="initial_convergence.png")
    plot(optimized_history, title="Optimized Configuration Convergence", save_path="optimized_convergence.png")
    
    plot_solution(initial_solution, cities, title="Initial Solution", save_path="initial_solution.png") 
    plot_solution(optimized_solution, cities, title="Optimized Solution", save_path="optimized_solution.png")
    
    # Get pheromone matrices for comparison
    initial_pheromones = initial_aco.get_pheromone_matrix()
    optimized_pheromones = optimized_aco.get_pheromone_matrix()
    
    plot_pheromone_heatmap(initial_pheromones, title="Initial Pheromone Distribution", save_path="initial_pheromones.png")
    plot_pheromone_heatmap(optimized_pheromones, title="Optimized Pheromone Distribution", save_path="optimized_pheromones.png")
    
    # Comprehensive comparison of before and after
    compare(
        before_history=initial_history,
        after_history=optimized_history,
        before_solution=initial_solution,
        after_solution=optimized_solution,
        coordinates=cities,
        before_pheromones=initial_pheromones,
        after_pheromones=optimized_pheromones,
        title="TSP Optimization Comparison",
        save_path="tsp_optimization_comparison.png"
    )


if __name__ == "__main__":
    main()
