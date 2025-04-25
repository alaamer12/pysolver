"""
Ant Colony Optimization (ACO) Algorithm Implementation

This module implements the Ant Colony Optimization algorithm for solving optimization problems.
ACO is inspired by the foraging behavior of ants, where they deposit pheromones to mark favorable paths.

Key components:
1. Graph/Problem representation with Node and Edge classes
2. Ant class for solution construction
3. Pheromone management
4. Solution evaluation
5. Algorithm parameters (alpha, beta, evaporation rate)

Usage:
    from _aco import ACO
    from _problem import Problem, AlgConfig

    # Define your problem
    problem = Problem(...)
    
    # Create ACO instance
    config = AlgConfig(problem=problem)
    aco = ACO(config)
    
    # Run optimization
    best_solution = aco.solve(problem)
"""
import copy
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, TypeVar, Generic

import matplotlib.pyplot as plt
import numpy as np

from _problem import AlgConfig, ProblemProtocol
from feedback import log, progress

# Type variable for generic Node content
T = TypeVar('T')


class Node(Generic[T]):
    """
    Represents a node in the problem graph.
    
    Each node has a unique ID, optional content, and keeps track of connected edges.
    """

    def __init__(self, node_id: int, content: Optional[T] = None):
        """
        Initialize a node.
        
        Args:
            node_id: Unique identifier for the node
            content: Optional data associated with the node
        """
        self.id = node_id
        self.content = content
        self.outgoing_edges: Dict[int, 'Edge'] = {}  # Maps destination node IDs to edges

    def add_edge(self, edge: 'Edge') -> None:
        """Add an outgoing edge from this node."""
        self.outgoing_edges[edge.dest.id] = edge

    def __repr__(self) -> str:
        return f"Node({self.id}, edges={len(self.outgoing_edges)})"


class Edge:
    """
    Represents an edge (connection) between two nodes in the problem graph.
    
    Each edge has a source and destination node, a pheromone level, and a heuristic value.
    """

    def __init__(self,
                 source: Node,
                 dest: Node,
                 initial_pheromone: float = 0.1,
                 heuristic: Optional[float] = None):
        """
        Initialize an edge.
        
        Args:
            source: Source node
            dest: Destination node
            initial_pheromone: Initial pheromone level
            heuristic: Heuristic value (if known)
        """
        self.source = source
        self.dest = dest
        self.pheromone = initial_pheromone
        self._heuristic = heuristic

    @property
    def heuristic(self) -> float:
        """Get the heuristic value for this edge."""
        return self._heuristic if self._heuristic is not None else 1.0

    @heuristic.setter
    def heuristic(self, value: float) -> None:
        """Set the heuristic value for this edge."""
        self._heuristic = value

    def __repr__(self) -> str:
        return f"Edge({self.source.id}->{self.dest.id}, p={self.pheromone:.2f}, h={self.heuristic:.2f})"


class Graph:
    """
    Represents the problem as a graph structure.
    
    The graph contains nodes and directed edges with pheromone levels and heuristic values.
    """

    def __init__(self, problem: ProblemProtocol, initial_pheromone: float = 0.1):
        self.problem = problem
        self.nodes: Dict[int, Node] = {}
        self.num_nodes = problem.num_components
        self.initial_pheromone = initial_pheromone

        # Create nodes
        self._create_nodes()

        # Create edges with initial pheromone levels
        self._create_edges()

    def _create_nodes(self) -> None:
        """Create nodes based on the problem's constraints."""
        for i in range(self.num_nodes):
            self.nodes[i] = Node(i)

    def _create_edges(self) -> None:
        """Create edges between nodes based on the problem's constraints."""
        with progress.simple("Creating graph edges", self.num_nodes) as pbar:
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if i != j:  # Avoid self-loops
                        source_node = self.nodes[i]
                        dest_node = self.nodes[j]

                        # Get heuristic from problem
                        heuristic = self.problem.get_heuristic(i, j)

                        # Create and store edge
                        edge = Edge(source_node, dest_node, self.initial_pheromone, heuristic=heuristic)
                        source_node.add_edge(edge)
                pbar.update(1)

    def get_node(self, node_id: int) -> Node:
        """Get a node by its ID."""
        return self.nodes[node_id]

    def get_edge(self, source_id: int, dest_id: int) -> Edge:
        """Get an edge by source and destination node IDs."""
        return self.nodes[source_id].outgoing_edges[dest_id]

    def update_edge_pheromone(self, source_id: int, dest_id: int, delta: float) -> None:
        """Update the pheromone on an edge."""
        self.get_edge(source_id, dest_id).pheromone += delta

    def apply_pheromone_evaporation(self, rate: float) -> None:
        """Apply evaporation to all edges in the graph."""
        for node in self.nodes.values():
            for edge in node.outgoing_edges.values():
                edge.pheromone *= (1 - rate)

    def __repr__(self) -> str:
        return f"Graph(nodes={len(self.nodes)})"


@dataclass
class ACOConfig(AlgConfig):
    """Configuration parameters for the ACO algorithm."""
    n_ants: int = 10
    alpha: float = 1.0
    beta: float = 2.0
    rand_start: bool = True
    pheromone_deposit: float = 1.0
    evaporation_rate: float = 0.5
    initial_pheromone: float = 0.1
    verbose: bool = True

    def __init__(self, problem: ProblemProtocol, **kwargs):
        # Extract iterations from kwargs or use default
        iterations = kwargs.get('iterations', 100)

        # Initialize AlgConfig with required parameters
        super().__init__(iterations=iterations)

        # Store the problem directly
        self.problem = problem

        # Set all the parameters from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Validate parameters
        self.validate_parameters()

    def validate_parameters(self) -> None:
        """Validate configuration parameters."""
        if self.n_ants <= 1:
            raise ValueError("Number of ants must be greater than 1.")
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Alpha and beta values must be greater than 0.")
        if self.evaporation_rate < 0 or self.evaporation_rate > 1:
            raise ValueError("Evaporation rate must be between 0 and 1.")
        if self.initial_pheromone <= 0:
            raise ValueError("Initial pheromone must be greater than 0.")


@dataclass
class Solution:
    """
    Represents a solution constructed by an ant.
    
    Contains the path (node IDs), the solution quality, and methods to evaluate the solution.
    """
    path: List[int] = field(default_factory=list)
    quality: Optional[float] = None

    def evaluate(self, problem: ProblemProtocol) -> float:
        """
        Evaluate the solution quality using the problem's evaluation function.
        
        Args:
            problem: The problem to evaluate against
            
        Returns:
            Solution quality (lower is better for minimization problems)
        """
        if not self.path:
            raise ValueError("Cannot evaluate empty solution")

        self.quality = problem.evaluate_solution(self.path)
        return self.quality

    def is_valid(self, problem: ProblemProtocol) -> bool:
        """Check if the solution is valid according to problem constraints."""
        return problem.is_valid_solution(self.path)

    def __repr__(self) -> str:
        quality_str = f"{self.quality:.2f}" if self.quality is not None else "Not evaluated"
        return f"Solution(len={len(self.path)}, quality={quality_str})"


class Ant:
    """
    Ant agent that constructs solutions.
    
    Each ant builds a complete solution by making probabilistic decisions
    based on pheromone levels and heuristic information.
    """

    def __init__(self, graph: Graph, alpha: float, beta: float, start_node_id: Optional[int] = None):
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.problem = graph.problem

        # Initialize starting position
        start_node_id = self._init_start_node(start_node_id)

        self.current_node = self.graph.get_node(start_node_id)

        # Initialize solution
        self._init_solution()

    def _init_solution(self) -> None:
        """Initialize the solution."""
        self.solution = Solution()
        self.solution.path.append(self.current_node.id)
        self.visited: Set[int] = {self.current_node.id}

    def _init_start_node(self, start_node_id: Optional[int] = None) -> int:
        """Initialize the starting node."""
        if start_node_id is None:
            start_node_id = random.randint(0, self.graph.num_nodes - 1)
        return start_node_id

    def construct_solution(self) -> Solution:
        """
        Build a complete solution by sequentially selecting nodes.
        
        Returns:
            The constructed solution
        """
        # Track number of steps without adding a new node (for early termination)
        stagnation = 0
        max_stagnation = self.graph.num_nodes * 2  # Arbitrary limit

        with progress.simple(f"Ant {id(self) % 1000}", self.graph.num_nodes, disable=True) as pbar:
            while len(self.visited) < self.graph.num_nodes:
                next_node_id = self._select_next_node()

                # No valid next nodes or reached a completion signal
                if next_node_id == -1:
                    log.debug(f"No valid next node found, visited {len(self.visited)}/{self.graph.num_nodes} nodes")
                    break

                # Move to the next node
                self.current_node = self.graph.get_node(next_node_id)
                self.solution.path.append(next_node_id)
                self.visited.add(next_node_id)
                stagnation = 0  # Reset stagnation counter
                pbar.update(1)

                # Break if we've visited all nodes or stagnated
                if len(self.visited) >= self.graph.num_nodes or stagnation >= max_stagnation:
                    break

                stagnation += 1

        # Evaluate solution only if complete for TSP
        if len(self.solution.path) == self.graph.num_nodes:
            self.solution.evaluate(self.problem)
        else:
            # For incomplete solutions, set a poor quality
            self.solution.quality = float('inf')

        return self.solution

    def _select_next_node(self) -> int:
        """
        Select the next node based on pheromone levels and heuristic information.
        
        Returns:
            The ID of the next node to visit
        """
        unvisited_ids = self._get_unvisited_nodes()

        # If no valid nodes left, we're done
        if not unvisited_ids:
            log.debug(f"No valid nodes left for ant - ending with {len(self.solution.path)} nodes")
            return -1  # Signal that we're done

        probabilities = self._calculate_node_probabilities(unvisited_ids)

        # If no valid moves (denominator was zero), choose randomly
        if not probabilities:
            return random.choice(unvisited_ids)

        # Select next node using roulette wheel selection
        return self._roulette_wheel_selection(probabilities)

    def _get_unvisited_nodes(self) -> list[int]:
        """
        Get all nodes that haven't been visited yet and would form a valid solution.
        
        For problems like the knapsack problem, this filters out nodes that would
        create an invalid solution (e.g., exceeding capacity).
        
        Returns:
            List of valid unvisited node IDs
        """
        unvisited = [i for i in range(self.graph.num_nodes) if i not in self.visited]

        # Filter for validity - try adding each unvisited node and check if it would be valid
        valid_nodes = []
        for node_id in unvisited:
            # Create a test solution with this node added
            test_path = self.solution.path + [node_id]

            # Check if this would be valid
            if self.problem.is_valid_solution(test_path):
                valid_nodes.append(node_id)

        return valid_nodes

    def _calculate_node_probabilities(self, unvisited_ids: list[int]) -> Dict[int, float]:
        """
        Calculate selection probabilities for each unvisited node.
        
        Args:
            unvisited_ids: List of unvisited node IDs
            
        Returns:
            Dictionary mapping node IDs to normalized probabilities
        """
        attractiveness_map = self._calculate_node_attractiveness(unvisited_ids)
        total_attractiveness = sum(attractiveness_map.values())

        # If denominator is zero (no valid moves), return empty dict
        if total_attractiveness == 0:
            return {}

        # Normalize probabilities
        return {
            node_id: attractiveness / total_attractiveness
            for node_id, attractiveness in attractiveness_map.items()
        }

    def _calculate_node_attractiveness(self, unvisited_ids: list[int]) -> Dict[int, float]:
        """
        Calculate the attractiveness of each unvisited node based on pheromone and heuristic.
        
        Args:
            unvisited_ids: List of unvisited node IDs
            
        Returns:
            Dictionary mapping node IDs to attractiveness values
        """
        attractiveness_map = {}

        for node_id in unvisited_ids:
            edge = self.current_node.outgoing_edges[node_id]
            attractiveness = (edge.pheromone ** self.alpha) * (edge.heuristic ** self.beta)
            attractiveness_map[node_id] = attractiveness

        return attractiveness_map

    @staticmethod
    def _roulette_wheel_selection(probabilities: Dict[int, float]) -> int:
        """
        Selects a node based on the provided probability distribution.
        
        Args:
            probabilities: Dictionary mapping node IDs to probabilities
            
        Returns:
            The selected node ID
        """
        # Convert dictionary to lists for selection
        nodes = list(probabilities.keys())
        probs = list(probabilities.values())

        # Choose based on probabilities
        return random.choices(nodes, weights=probs, k=1)[0]


class ACO:
    """
    Main ACO algorithm implementation.
    
    Manages the colony of ants, graph with pheromones, and optimization process.
    """

    def __init__(self, config: ACOConfig):
        self.config = config
        self.problem = config.problem

        log.info(f"Initializing ACO for {self.problem.__class__.__name__}")

        # Create the problem graph
        self.graph = Graph(self.problem, config.initial_pheromone)

        # Initialize solution tracking
        self._init_solution_tracking()

    def _init_solution_tracking(self) -> None:
        """Initialize solution tracking."""
        self.best_solution = Solution()
        self.best_quality = float('inf')  # For minimization problems
        self.iteration_best_quality: List[float] = []

    def create_ants(self) -> List[Ant]:
        """
        Create a colony of ants for the current iteration.
        
        Returns:
            A list of initialized Ant objects
        """
        ants = []
        with progress.simple(f"Creating {self.config.n_ants} ants", self.config.n_ants) as pbar:
            for _ in range(self.config.n_ants):
                # Determine starting position
                start_node_id = None
                if not self.config.rand_start:
                    # If not random start, we can use a fixed starting point
                    start_node_id = 0

                # Create and add an ant to the colony
                ant = self._create_ant(start_node_id)
                ants.append(ant)
                pbar.update(1)

        return ants

    def _create_ant(self, start_node_id: Optional[int] = None) -> Ant:
        """Create an ant with the given starting node ID."""
        return Ant(
            graph=self.graph,
            alpha=self.config.alpha,
            beta=self.config.beta,
            start_node_id=start_node_id
        )

    def update_pheromones(self, solutions: List[Solution]) -> None:
        """
        Update pheromone trails based on ant solutions.
        
        Args:
            solutions: List of solutions constructed by ants
        """
        # First, apply evaporation to all pheromone trails
        self._evaporate_pheromones()

        # Then, add new pheromones based on solution quality
        for solution in solutions:
            # Skip invalid solutions (those with inf quality)
            if solution.quality is None or solution.quality == float('inf'):
                continue

            deposit = self._calculate_pheromone_deposit(solution)
            self._deposit_pheromones_on_path(solution.path, deposit)

    def _evaporate_pheromones(self) -> None:
        """Apply evaporation to all pheromone trails."""
        self.graph.apply_pheromone_evaporation(self.config.evaporation_rate)

    def _calculate_pheromone_deposit(self, solution: Solution) -> float:
        """
        Calculate pheromone deposit amount based on solution quality.
        
        Better solutions (lower quality for minimization) deposit more pheromone.
        
        Args:
            solution: The solution to calculate deposit for
            
        Returns:
            The amount of pheromone to deposit
        """
        # Avoid division by zero by using a small epsilon value
        # when solution quality is zero or very small
        epsilon = 1e-10
        return self.config.pheromone_deposit / max(solution.quality, epsilon)

    def _deposit_pheromones_on_path(self, path: List[int], deposit: float) -> None:
        """
        Deposit pheromones on all edges in the solution path.
        
        Args:
            path: The solution path
            deposit: Amount of pheromone to deposit
        """
        # Update pheromone on the path edges
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            self.graph.update_edge_pheromone(from_node, to_node, deposit)

        # For closed-loop problems (like TSP), connect the last and first nodes
        self._handle_closed_loop(path, deposit)

    def _handle_closed_loop(self, path: List[int], deposit: float) -> None:
        """
        Handle closed-loop problems by connecting last and first nodes if valid.
        
        Args:
            path: The solution path
            deposit: Amount of pheromone to deposit
        """
        if self.problem.is_valid_solution(path + [path[0]]):
            last_node = path[-1]
            first_node = path[0]
            self.graph.update_edge_pheromone(last_node, first_node, deposit)

    def optimize(self) -> Tuple[List[int], float, List[float]]:
        """
        Run the ACO optimization process.
        
        Returns:
            Tuple containing (best solution path, best solution quality, history of best qualities)
        """
        self._log_optimization_start()

        with progress.main("ACO Optimization", self.config.iterations) as pbar:
            for iteration in range(self.config.iterations):
                solutions = self._run_iteration(iteration)
                self._track_iteration_progress(solutions, iteration)
                pbar.update(1)

        self._log_optimization_complete()

        return self.best_solution.path, self.best_quality, self.iteration_best_quality

    def _log_optimization_start(self) -> None:
        """Log the start of the optimization process if verbose mode is enabled."""
        if self.config.verbose:
            log.info(f"Starting ACO optimization with {self.config.iterations} iterations...")
            log.info(f"Parameters: α={self.config.alpha}, β={self.config.beta}, ρ={self.config.evaporation_rate}")

    def _log_optimization_complete(self) -> None:
        """Log the completion of the optimization process if verbose mode is enabled."""
        if self.config.verbose:
            log.success(f"Optimization complete. Best solution quality: {self.best_quality:.4f}")

    def _run_iteration(self, iteration: int) -> List[Solution]:
        """
        Run a single iteration of the ACO algorithm.
        
        Args:
            iteration: The current iteration number
            
        Returns:
            List of solutions constructed in this iteration
        """
        ants = self.create_ants()
        solutions = self._construct_all_solutions(ants, iteration)
        self.update_pheromones(solutions)
        return solutions

    def _construct_all_solutions(self, ants: List[Ant], iteration: int) -> List[Solution]:
        """
        Have all ants construct solutions and update the best solution if needed.
        
        Args:
            ants: List of ant agents
            iteration: Current iteration number
            
        Returns:
            List of constructed solutions
        """
        solutions = []
        with progress.simple(f"Constructing solutions (iter {iteration + 1})", len(ants)) as pbar:
            for ant in ants:
                solution = ant.construct_solution()
                solutions.append(solution)
                self._update_best_solution(solution, iteration)
                pbar.update(1)
        return solutions

    def _update_best_solution(self, solution: Solution, iteration: int) -> None:
        """
        Update the best solution if the current solution is better.
        
        Args:
            solution: The current solution to evaluate
            iteration: Current iteration number
        """
        if solution.quality < self.best_quality:
            self.best_solution = copy.deepcopy(solution)
            self.best_quality = solution.quality
            self._log_new_best_solution(iteration)

    def _log_new_best_solution(self, iteration: int) -> None:
        """
        Log when a new best solution is found if verbose mode is enabled.
        
        Args:
            iteration: Current iteration number
        """
        if self.config.verbose:
            log.success(f"Iteration {iteration + 1}: New best solution found with quality {self.best_quality:.4f}")

    def _track_iteration_progress(self, solutions: List[Solution], iteration: int) -> None:
        """
        Track the progress of the current iteration and log if needed.
        
        Args:
            solutions: List of solutions from the current iteration
            iteration: Current iteration number
        """
        self._record_iteration_best_quality(solutions)
        self._log_iteration_progress(iteration)

    def _record_iteration_best_quality(self, solutions: List[Solution]) -> None:
        """
        Record the best solution quality from the current iteration.
        
        Args:
            solutions: List of solutions from the current iteration
        """
        iteration_best = min(solution.quality for solution in solutions)
        self.iteration_best_quality.append(iteration_best)

    def _log_iteration_progress(self, iteration: int) -> None:
        """
        Log the progress of iterations if verbose mode is enabled.
        
        Args:
            iteration: Current iteration number
        """
        if self.config.verbose and (iteration + 1) % 10 == 0:
            current_best = self.iteration_best_quality[-1]
            log.info(f"Iteration {iteration + 1}/{self.config.iterations} - Current best: {current_best:.4f}")

    def get_pheromone_matrix(self) -> np.ndarray:
        """
        Get the current pheromone levels as a matrix.
        
        Returns:
            A matrix of pheromone levels where element [i,j] is the pheromone on the edge from i to j
        """
        n = self.graph.num_nodes
        pheromone_matrix = np.zeros((n, n))

        for i in range(n):
            node = self.graph.get_node(i)
            for j, edge in node.outgoing_edges.items():
                pheromone_matrix[i, j] = edge.pheromone

        return pheromone_matrix


# Visualization functions
def plot(history: List[float], title: str = "ACO Convergence", save_path: str = 'aco_convergence.png') -> None:
    """
    Create an enhanced visualization of the optimization process.
    
    Args:
        history: List of best solution qualities per iteration
        title: Plot title
        save_path: Path to save the plot
    """
    log.info("Generating enhanced convergence plot...")
    plt.figure(figsize=(12, 7))

    # Better styling
    plt.style.use('seaborn-v0_8-darkgrid')

    # Main convergence line
    iterations = range(1, len(history) + 1)
    plt.plot(iterations, history, 'b-', linewidth=2.5, label='Solution Quality')

    # Add rolling average to show trend
    window = min(10, len(history) // 5) if len(history) > 10 else 1
    if window > 1:
        rolling_avg = np.convolve(history, np.ones(window) / window, mode='valid')
        plt.plot(range(window, len(history) + 1), rolling_avg, 'r--',
                 linewidth=1.5, label=f'{window}-Iteration Rolling Avg')

    # Highlight best solution
    best_idx = np.argmin(history)
    best_val = history[best_idx]
    plt.scatter([best_idx + 1], [best_val], color='green', s=100,
                label=f'Best Solution: {best_val:.2f}', zorder=5)

    # Annotate improvement points (when solution significantly improves)
    improvements = []
    for i in range(1, len(history)):
        if history[i] < history[i - 1] * 0.95:  # 5% improvement threshold
            improvements.append(i)

    # Limit the number of annotations to avoid cluttering
    if len(improvements) > 5:
        step = len(improvements) // 5
        improvements = improvements[::step]

    for idx in improvements:
        plt.annotate(f"{history[idx]:.2f}", xy=(idx + 1, history[idx]),
                     xytext=(10, -20), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

    # Better labels and styling
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Solution Quality (lower is better)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save with higher quality
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    log.info(f"Enhanced plot saved as '{save_path}'")


def plot_solution(solution: List[int], coordinates: List[Tuple[float, float]],
                  title: str = 'ACO Solution', save_path: str = 'aco_solution.png') -> None:
    """
    Create an enhanced visualization of a spatial solution like TSP.
    
    Args:
        solution: The solution path as a list of indices
        coordinates: List of (x, y) coordinates for each component
        title: Plot title
        save_path: Path to save the plot
    """
    log.info("Generating enhanced solution visualization...")
    plt.figure(figsize=(12, 10))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Extract coordinates for plotting
    x = [coordinates[i][0] for i in solution]
    y = [coordinates[i][1] for i in solution]

    # Add starting point at the end for closed tours
    x.append(x[0])
    y.append(y[0])

    # Calculate total distance for the title
    total_distance = 0
    for i in range(len(solution)):
        j = (i + 1) % len(solution)  # Wrap around to first city
        city1 = solution[i]
        city2 = solution[j]
        x1, y1 = coordinates[city1]
        x2, y2 = coordinates[city2]
        segment_dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_distance += segment_dist

    # Plot the tour with arrows to show direction
    for i in range(len(x) - 1):
        plt.arrow(x[i], y[i], x[i + 1] - x[i], y[i + 1] - y[i],
                  head_width=total_distance / 200, head_length=total_distance / 100,
                  fc='blue', ec='blue', length_includes_head=True, alpha=0.7)

    # Add different colors for start/end points
    plt.scatter(x[0], y[0], color='green', s=200, zorder=5, label='Start/End')
    plt.scatter(x[1:-1], y[1:-1], color='red', s=100, zorder=4, label='Cities')

    # Add a background scatter of all cities with lower opacity
    all_x = [coord[0] for coord in coordinates]
    all_y = [coord[1] for coord in coordinates]
    plt.scatter(all_x, all_y, color='gray', s=30, alpha=0.2, zorder=1, label='All Cities')

    # Add labels to cities
    for i, city_idx in enumerate(solution):
        if i < len(solution):  # Don't label the repeated start node
            plt.annotate(str(city_idx), (x[i], y[i]), fontsize=10,
                         xytext=(5, 5), textcoords='offset points')

    # Enhanced title and styling
    plt.title(f"{title}\nTotal Distance: {total_distance:.2f}", fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save with higher quality
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    log.info(f"Enhanced solution plot saved as '{save_path}'")


def plot_pheromone_heatmap(pheromone_matrix: np.ndarray,
                           title: str = 'Pheromone Levels',
                           save_path: str = 'pheromone_heatmap.png') -> None:
    """
    Create an enhanced visualization of pheromone levels.
    
    Args:
        pheromone_matrix: Matrix of pheromone levels
        title: Plot title
        save_path: Path to save the plot
    """
    log.info("Generating enhanced pheromone heatmap...")
    plt.figure(figsize=(12, 10))

    # Create a mask for the diagonal (self-edges) which are usually zeros
    mask = np.eye(pheromone_matrix.shape[0], dtype=bool)

    # Plot the heatmap with a better colormap and normalization
    cmap = plt.cm.get_cmap()

    # Create a logarithmic normalization to better visualize pheromone distribution
    from matplotlib.colors import LogNorm

    # Find min non-zero value for logarithmic scale
    min_nonzero = np.min(pheromone_matrix[pheromone_matrix > 0]) if np.any(pheromone_matrix > 0) else 0.001
    max_val = np.max(pheromone_matrix)

    # Create a custom norm that works with zeros
    class CustomLogNorm(LogNorm):
        def __call__(self, value, clip=None):
            masked_value = np.ma.array(value, mask=(value <= 0))
            return LogNorm.__call__(self, masked_value, clip)

    # Use the custom norm if we have non-zero values
    norm = CustomLogNorm(vmin=min_nonzero, vmax=max_val) if max_val > 0 else None

    # Plot the heatmap
    plt.imshow(pheromone_matrix, cmap=cmap, norm=norm, interpolation='nearest')
    plt.colorbar(label='Pheromone Level (log scale)')

    # Add a structured grid
    n = pheromone_matrix.shape[0]
    plt.xticks(range(n))
    plt.yticks(range(n))
    plt.grid(False)

    # Add edge annotations for strong connections
    threshold = np.percentile(pheromone_matrix[pheromone_matrix > 0], 90)  # Top 10% of edges
    for i in range(n):
        for j in range(n):
            if i != j and pheromone_matrix[i, j] > threshold:
                plt.text(j, i, f'{pheromone_matrix[i, j]:.2f}',
                         ha='center', va='center', color='white', fontsize=8)

    # Enhanced title and styling
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('To Node', fontsize=12)
    plt.ylabel('From Node', fontsize=12)
    plt.tight_layout()

    # Save with higher quality
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    log.info(f"Enhanced pheromone heatmap saved as '{save_path}'")


def compare(before_history: List[float], after_history: List[float],
            before_solution: List[int] = None, after_solution: List[int] = None,
            coordinates: List[Tuple[float, float]] = None,
            before_pheromones: np.ndarray = None, after_pheromones: np.ndarray = None,
            title: str = "Optimization Comparison", save_path: str = "aco_comparison.png") -> None:
    """
    Compare optimization results before and after algorithm improvements.
    
    Args:
        before_history: Convergence history from before improvements
        after_history: Convergence history from after improvements
        before_solution: Solution path from before improvements
        after_solution: Solution path from after improvements
        coordinates: List of (x, y) coordinates for each component (if comparing solutions)
        before_pheromones: Pheromone matrix from before improvements
        after_pheromones: Pheromone matrix from after improvements
        title: Plot title
        save_path: Path to save the plot
    """
    log.info("Generating optimization comparison...")

    # Determine how many plots we need
    plot_count = 1  # We always have convergence history
    if before_solution is not None and after_solution is not None and coordinates is not None:
        plot_count += 1
    if before_pheromones is not None and after_pheromones is not None:
        plot_count += 1

    # Create figure with appropriate subplots
    fig = plt.figure(figsize=(15, 5 * plot_count))
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Plot convergence history comparison
    ax1 = fig.add_subplot(plot_count, 1, 1)

    # Normalize iteration counts if they differ
    before_iterations = range(1, len(before_history) + 1)
    after_iterations = range(1, len(after_history) + 1)

    # Plot both histories
    ax1.plot(before_iterations, before_history, 'r-', linewidth=2, alpha=0.7, label='Before')
    ax1.plot(after_iterations, after_history, 'g-', linewidth=2, alpha=0.7, label='After')

    # Add best values
    best_before = min(before_history)
    best_before_idx = before_history.index(best_before)
    best_after = min(after_history)
    best_after_idx = after_history.index(best_after)

    ax1.scatter(best_before_idx + 1, best_before, color='darkred', s=100, zorder=5)
    ax1.scatter(best_after_idx + 1, best_after, color='darkgreen', s=100, zorder=5)

    ax1.annotate(f"Best: {best_before:.2f}", xy=(best_before_idx + 1, best_before),
                 xytext=(10, 10), textcoords='offset points', color='darkred')
    ax1.annotate(f"Best: {best_after:.2f}", xy=(best_after_idx + 1, best_after),
                 xytext=(10, -20), textcoords='offset points', color='darkgreen')

    # Calculate improvement percentage
    if best_before > 0:
        improvement = (best_before - best_after) / best_before * 100
        ax1.annotate(f"Improvement: {improvement:.2f}%",
                     xy=(0.5, 0.05), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
                     ha='center')

    ax1.set_title('Convergence Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Solution Quality (lower is better)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. If provided, plot solution comparison
    if before_solution is not None and after_solution is not None and coordinates is not None:
        ax2 = fig.add_subplot(plot_count, 1, 2)

        # Calculate total distances
        before_distance = calculate_tour_distance(before_solution, coordinates)
        after_distance = calculate_tour_distance(after_solution, coordinates)

        # Plot before solution
        plot_tour(ax2, before_solution, coordinates, 'red', 'Before')

        # Plot after solution
        plot_tour(ax2, after_solution, coordinates, 'green', 'After')

        # Add a background scatter of all cities
        all_x = [coord[0] for coord in coordinates]
        all_y = [coord[1] for coord in coordinates]
        ax2.scatter(all_x, all_y, color='gray', s=30, alpha=0.2, zorder=1, label='All Cities')

        # Add title with distance information
        distance_diff = before_distance - after_distance
        percent_improvement = (distance_diff / before_distance) * 100 if before_distance > 0 else 0
        ax2.set_title(f'Solution Comparison\nBefore: {before_distance:.2f} → After: {after_distance:.2f} '
                      f'(Improved by {percent_improvement:.2f}%)',
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('X Coordinate', fontsize=12)
        ax2.set_ylabel('Y Coordinate', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. If provided, plot pheromone comparison
    if before_pheromones is not None and after_pheromones is not None:
        ax3 = fig.add_subplot(plot_count, 1, plot_count)

        # Create a delta pheromone matrix
        delta_pheromones = after_pheromones - before_pheromones

        # Plot the delta heatmap
        cmap = plt.cm.ColormapRegistry['RdYlBu']  # Red for negative, blue for positive
        img = ax3.imshow(delta_pheromones, cmap=cmap, interpolation='nearest')
        plt.colorbar(img, ax=ax3, label='Pheromone Change (After - Before)')

        # Add title and labels
        ax3.set_title('Pheromone Distribution Change', fontsize=14, fontweight='bold')
        ax3.set_xlabel('To Node', fontsize=12)
        ax3.set_ylabel('From Node', fontsize=12)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    log.info(f"Comparison plot saved as '{save_path}'")


def calculate_tour_distance(solution: List[int], coordinates: List[Tuple[float, float]]) -> float:
    """Calculate the total distance of a tour."""
    total = 0
    for i in range(len(solution)):
        j = (i + 1) % len(solution)  # Wrap around to first city
        city1 = solution[i]
        city2 = solution[j]
        x1, y1 = coordinates[city1]
        x2, y2 = coordinates[city2]
        segment_dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total += segment_dist
    return total


def plot_tour(ax, solution: List[int], coordinates: List[Tuple[float, float]],
              color: str, label: str) -> None:
    """Plot a tour on the given axis with the specified color and label."""
    # Extract coordinates for plotting
    x = [coordinates[i][0] for i in solution]
    y = [coordinates[i][1] for i in solution]

    # Add starting point at the end for closed tours
    x.append(x[0])
    y.append(y[0])

    # Plot path with lower opacity to avoid visual clutter
    ax.plot(x, y, color=color, linestyle='-', linewidth=1.5, alpha=0.5)

    # Add arrows to show direction (but not too many)
    n_arrows = min(len(solution) // 5, 10)  # At most 10 arrows
    if n_arrows > 0:
        step = len(solution) // n_arrows
        for i in range(0, len(solution), step):
            j = (i + 1) % len(solution)
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            ax.arrow(x[i], y[i], dx * 0.8, dy * 0.8, head_width=2, head_length=3,
                     fc=color, ec=color, length_includes_head=True, alpha=0.7)

    # Plot points with the label only for the first point
    if len(solution) > 0:
        ax.scatter(x[0], y[0], color=color, s=100, zorder=5,
                   label=f"{label} Start/End")
        ax.scatter(x[1:-1], y[1:-1], color=color, s=50, zorder=4, alpha=0.7)
