"""
PySolver - Nature-inspired optimization algorithms

This package implements three optimization algorithms:
1. Ant Colony Optimization (ACO)
2. Artificial Bee Colony (ABC)
3. Particle Swarm Optimization (PSO)
"""

from ._aco import ACO, ACOConfig, plot, plot_solution, compare, plot_pheromone_heatmap
from ._problem import ProblemProtocol, AlgConfig
from .__abc import ABC, ABCConfig
from ._pso import ParticleSwarmOptimization, PSOConfig, plot_2d_function

__all__ = [
    'ACO', 'ACOConfig', 'plot', 'plot_solution', 'compare', 'plot_pheromone_heatmap',
    'ProblemProtocol', 'AlgConfig',
    'ABC', 'ABCConfig',
    'ParticleSwarmOptimization', 'PSOConfig', 'plot_2d_function'
] 