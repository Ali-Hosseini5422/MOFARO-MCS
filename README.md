MOFARO: Multi-Objective Optimization Framework for Dynamic Task Allocation in Mobile Crowdsensing
Overview

This repository contains the Python implementation of MOFARO (Multi-Objective Fractal-based Artificial Rabbit Optimization), a framework for dynamic task allocation in edge-fog-cloud Mobile Crowdsensing (MCS) systems. MOFARO optimizes four objectives—execution time (makespan), energy consumption, cost, and Quality of Service (QoS) violations—using fractal-based exploration, Lévy flights, Pareto optimization, and TOPSIS. The framework models tasks as Directed Acyclic Graphs (DAGs) and incorporates Markov chain-based mobility predictions to handle dynamic MCS environments.

The code includes a custom Python-based simulator to evaluate MOFARO against baseline algorithms (FCFS, EDF, GfE, Detour, PSG, PSG-M) using synthetic datasets inspired by real-world traces (e.g., Google Cluster Traces, Azure Public Datasets). It generates performance metrics and visualizations for makespan, number of satisfied tasks, percentage of deadline-satisfied tasks (PDST), energy consumption, cost, and total penalty.
Repository Structure

MOFARO/
├── src/
│   ├── MOFARO_Simulation.ipynb  # Jupyter notebook with MOFARO and baseline implementations
├── README.md                 # This file

Note: The current implementation is contained in MOFARO_Simulation.ipynb. You may refactor it into separate Python modules (e.g., mofaro.py, simulator.py) for better modularity, as suggested in the structure above.
Prerequisites

To run the code, ensure you have the following installed:

    Python: Version 3.8 or higher
    Jupyter Notebook: For running MOFARO_Simulation.ipynb
    Dependencies: Listed in requirements.txt. Install them using:

    pip install -r requirements.txt

    Key libraries include:
        numpy for numerical computations
        networkx for DAG-based task modeling
        scipy for Lévy flight distributions and statistical analysis
        matplotlib for visualization
        pandas (optional, for data handling if extended)

Installation
Clone the repository:

git clone https://github.com/Ali-Hosseini5422/Ali-Hosseini5422.git
cd MOFARO

Install dependencies:

pip install -r requirements.txt

Verify setup by opening the Jupyter notebook:

jupyter notebook src/MOFARO_Simulation.ipynb

Usage

The main implementation is in MOFARO_Simulation.ipynb, which contains:

    MOFARO Algorithm: Implements the multi-objective optimization with fractal-based exploration, Lévy flights, Pareto front generation, and TOPSIS for task allocation.
    Baseline Algorithms: FCFS, EDF, GfE, Detour, PSG, and PSG-M for comparison.
    Simulator: Generates tasks, nodes (edge, fog, cloud), and network parameters (bandwidth, latency, distance matrices) to simulate an MCS environment.
    Visualization: Generates bar and line plots for performance metrics across different run counts (NRUN).

Running the Simulation

    Open MOFARO_Simulation.ipynb in Jupyter Notebook.
    Run all cells to execute the simulation with default parameters:
        NUM_TASKS = 50
        NUM_EDGE_NODES = 30, NUM_FOG_NODES = 5, NUM_CLOUD_NODES = 1
        POP_SIZE = 5, MAX_IT = 5, NRUN = 10
        LEVY_BETA = 1.5 for Lévy flights
    The notebook outputs:
        A table of results comparing MOFARO and baselines for makespan, number of satisfied tasks (numSat), PDST, energy consumption (engCons), cost, and total penalty (totPenalty).
        Bar plots for each metric across algorithms.
        Line plots showing metric trends for varying NRUN values (1, 5, 10).
        Sample task allocations and updated node resources.

Customizing Parameters

Modify the constants at the top of MOFARO_Simulation.ipynb to adjust the simulation:

    NUM_TASKS: Number of tasks (e.g., 50, 100, 500)
    NUM_EDGE_NODES, NUM_FOG_NODES, NUM_CLOUD_NODES: Node counts per layer
    POP_SIZE, MAX_IT: Population size and iterations for MOFARO
    NRUN: Number of simulation runs for averaging
    LEVY_BETA, FRACTAL_DIM_MIN, FRACTAL_DIM_MAX: Parameters for Lévy flights and fractal exploration
    N_USERS: Number of mobile users for mobility modeling

Example: To run with 100 tasks and 20 edge nodes, modify:

NUM_TASKS = 100
NUM_EDGE_NODES = 20

Outputs

    Results: Printed tables show average metric values for each algorithm.
    Plots: Saved in the results/ directory as PNG files (bar plots for each metric, line plots for NRUN trends).
    Sample Allocation: Displays task-to-node assignments and updated node resources (e.g., available capacity and memory).

Datasets

The code generates synthetic datasets in generate_parameters() and generate_for_old_models():

    Tasks: Include computational load (s), memory (m), deadline (d), priority (p), QoS (q), input/output data sizes (in_, out).
    Nodes: Edge, fog, and cloud nodes with processing capacity (c), memory (m), bandwidth (b), power consumption (pmax, pmin), and cost rate (pc).
    DAG: A simple directed acyclic graph with a single edge for task dependencies (extendable for complex workflows).
    Network: Bandwidth, latency, and distance matrices for inter-node communication.

To use custom datasets, modify generate_parameters() to load external task DAGs or node configurations from CSV/JSON files in the data/ directory.
Key Features

    MOFARO Algorithm: Combines fractal-based exploration, Lévy flights, non-dominated sorting, and TOPSIS for multi-objective task allocation in edge-fog-cloud MCS systems.
    Baseline Comparisons: Evaluates against FCFS, EDF, GfE, Detour, PSG, and PSG-M, focusing on fog and cloud layers (no edge for baselines).
    Mobility Modeling: Uses Markov chains to predict user mobility, enhancing allocation efficiency.
    Metrics: Computes makespan, number of satisfied tasks, PDST, energy consumption, cost, and total penalty, with penalties for deadline violations and resource overuse.
    Visualization: Generates bar and line plots to compare algorithm performance and analyze scalability.

Results

Simulation results for 50 tasks demonstrate MOFARO's superior performance:

    Makespan: Lower than baselines due to edge-layer utilization.
    Energy Consumption: Reduced by leveraging low-latency edge nodes.
    Cost and QoS: Optimized through Pareto-based trade-offs.
    PDST: Higher percentage of deadline-satisfied tasks (up to ~90% in some runs).
    Scalability: Handles varying task counts and node configurations effectively.

Detailed results and plots are generated in the notebook and saved in the results/ directory.
Limitations

    Simplified DAG: The current implementation uses a minimal DAG (one edge). Extending to complex workflows requires modifying generate_parameters().
    Computational Overhead: The mobility constraint's double-sum scales with N_USERS, potentially slowing down large-scale simulations.
    Synthetic Data: While inspired by real traces, the generated data may not fully capture real-world variability.

Contributing

Contributions are welcome! Please submit issues or pull requests for bug fixes, optimizations, or enhancements (e.g., modularizing the notebook, adding complex DAGs). Ensure changes align with the MOFARO framework and include documentation.
License

This project is licensed under a custom license (see LICENSE file). It is not distributed under the MIT License. Review the license terms before using or modifying the code.
Contact

For questions or support, contact:

    Seyed Ali Hosseini: ali.hosseini1822@iau.ac.ir
    Omid Sojoodi Shijani: o_sojoodi@iau.ac.ir
    Vahid Khajehvand: vahidkhajehvand@iau.ac.ir

Acknowledgments

This work builds on concepts from the Artificial Rabbits Optimization (ARO), NSGA-II, and TOPSIS algorithms, as well as datasets like Google Cluster Traces and Azure Public Datasets. We thank the open-source community for providing libraries such as numpy, networkx, scipy, and matplotlib.
Notes

    Code Structure: The README assumes the Jupyter notebook is the primary implementation. If you plan to modularize the code (e.g., split into mofaro.py, simulator.py), update the repository structure section accordingly.
    License: Since you specified no MIT license, the README references a custom LICENSE file. You should create this file in the repository root with your preferred terms (e.g., a restrictive academic license or a simple non-commercial use clause).
    Plots: The code generates plots in-memory and prints base64-encoded images. To save them to results/, modify the code to write PNG files directly (e.g., plt.savefig('results/metric.png')).
    Extensibility: The README suggests potential improvements (e.g., complex DAGs, external datasets) to guide users who want to extend the code.

Let me know if you need help creating the requirements.txt, LICENSE file, or further refinements to the README!
