markdown# MOFARO Simulator

This repository contains the Python simulator code for the MOFARO framework, a multi-objective optimization model for task allocation in edge-fog-cloud mobile crowdsensing (MCS) systems. The associated paper is under preparation.

## Overview
MOFARO extends the Artificial Rabbit Optimization algorithm with fractal-based exploration, Lévy flights, Pareto optimization, and TOPSIS. It optimizes execution time, energy consumption, cost, and QoS violations.

Key features:
- Models tasks as Directed Acyclic Graphs (DAGs).
- Incorporates Markov chain-based mobility predictions.
- Simulated on real-world traces (e.g., Google Cluster Traces).

## Repository Structure
- **code/**: Contains the Python simulator script (`mofaro_simulation.py`).

## Requirements
- Python 3.12+ with libraries: numpy, networkx, scipy, matplotlib, etc. (listed in the code).

## Installation and Usage
1. Clone the repository:
git clone https://github.com/yourusername/MOFARO-simulator.git
cd MOFARO-simulator
text2. Run the simulator:
cd code
python mofaro_simulation.py
text- This will generate plots and metrics for different NRUN values.
- Outputs: Base64-encoded PNG plots for metrics vs. NRUN.

## Results
- Achieves ~0.85 hypervolume on Pareto front.
- 9.5%-14.3% improvements over baselines in key metrics.


## Contact
For questions, open a GitHub issue or contact [Ali.hosseini5422@yahoo.com].



