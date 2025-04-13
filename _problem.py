"""
Problem definition interface.

Implement this for your specific optimization problem.
This class should follow the ProblemProtocol interface.

Example problem implementation (e.g., TSP):

class TSPProblem:

    @property
    def num_components(self) -> int:
        ...

    def get_heuristic(self, i: int, j: int) -> float:
        ...

    def is_valid_solution(self, solution: List[int]) -> bool:
        ...

    def evaluate_solution(self, solution: List[int]) -> float:
    ...
"""
from dataclasses import dataclass
from typing import List, Protocol


class ProblemProtocol(Protocol):
    """
    Protocol class defining the interface that any problem must implement for ACO.

    This ensures that any problem implementation provides the necessary methods
    for the ACO algorithm to work with it.
    """

    @property
    def num_components(self) -> int:
        """
        Returns the number of components in the problem (e.g., cities in TSP).
        """

    def get_heuristic(self, i: int, j: int) -> float:
        """
        Returns the heuristic value between components i and j.

        For TSP, this could be 1/distance between cities i and j.
        Higher values indicate more desirable connections.
        """

    def is_valid_solution(self, solution: List[int]) -> bool:
        """
        Checks if a solution is valid according to problem constraints.

        For TSP, a valid solution visits each city exactly once.
        """

    def evaluate_solution(self, solution: List[int]) -> float:
        """
        Evaluates the quality of a solution.

        Returns a fitness value where lower is better (for minimization problems).
        For TSP, this would be the total distance of the tour.
        """


# Example problem implementation (e.g., TSP)
class Problem:
    """
    Problem definition interface.

    Implement this for your specific optimization problem.
    This class should follow the ProblemProtocol interface.
    """
    # TODO: Implement problem-specific components:
    # - Define problem representation
    # - Implement heuristic information calculation
    # - Implement solution evaluation
    # - Implement any problem-specific constraints
    pass


@dataclass
class AlgConfig:
    iterations: int

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("Number of iterations must be positive.")
