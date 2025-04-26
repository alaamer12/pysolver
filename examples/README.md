# Examples

This directory contains demonstration examples of using the PySolver optimization framework. The examples are organized into different categories based on the nature of the problems they solve.

## Directory Structure

- `static/`: Contains examples of solving static optimization problems
  - Problems with fixed parameters and well-defined objectives
  - Examples include:
    - Traveling Salesman Problem (TSP) using Ant Colony Optimization
    - Other discrete optimization problems

- `continuous/`: Contains examples of solving dynamic/continuous optimization problems
  - Problems involving hyperparameter optimization
  - Problems with changing landscapes or dynamic objectives
  - Examples of using different algorithms (PSO, ABC, etc.) for continuous optimization

## Running the Examples

Each example is self-contained and can be run independently. Python files (`.py`) contain complete runnable examples, while Jupyter notebooks (`.ipynb`) provide interactive tutorials with detailed explanations.

For example, to run the TSP example:
```bash
python static/tsp_example.py
```
