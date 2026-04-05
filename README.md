# RL-PS
RL-PS is a Deep Reinforcement Learning (DRL)-based optimization project tailored for power systems.It aims to solve key challenges in distribution network management, such as minimizing active power loss, regulating bus voltages within safe limits, and ensuring operational stability under high PV penetration. The project integrates state-of-the-art deep reinforcement learning algorithms—DDPG, TD3, and SAC—with power system constraints (e.g., voltage limits, line overload, and external grid power boundaries), incorporating a hierarchical reward function that prioritizes voltage safety before loss optimization. It provides tools for experiment execution across IEEE 33-, 69-, and 141-bus systems, hyperparameter tuning, and result visualization for comparative algorithm evaluation.
# Environmental DependenciesInstall required libraries using requirements.txt:
gymnasium==1.0.0 # Reinforcement learning environment

numpy==1.26.4 # Numerical computing

torch==2.5.1 # Deep learning framework

matplotlib==3.10.0 # Plotting

pandas==2.2.3 # Data processing

pandapower==2.14.11 # Power system simulation

simbench==1.5.3 # Power system benchmarks

scipy==1.13.1 # Scientific computing

numba==0.58.1 # Numerical acceleration
# Installation command:
pip install -r requirements.txt
# Core Features
## 1. DRL Algorithm Suite
REINFORCE: Basic policy gradient algorithm for discrete action spaces (drl/reinforce.py). Advanced Variants: Experimental implementations of targetless TD3/SAC and one-step DDPG for performance comparison. Replay Buffers: Multiple buffer types (standard, prioritized, multi-agent) to support diverse DRL architectures.
## 2. Experiment Workflow
Evaluation: Track training metrics (returns, steps, constraints) with sliding averages and real-time logging (evaluation.py). Visualization: Generate training curves with configurable rolling windows, mean/std displays, and step range filtering (plot_returns.py). Reproducibility: Seed management tools ensure consistent experiment results.
## 3. Hyperparameter Tuning
Automated hyperparameter evaluation with evaluate_hps.py. Supports discrete/continuous hyperparameters and optimal value selection. Generates box plots and statistical analysis for hyperparameter impact.
## 4. Power System Constraints
Implements critical grid constraints (voltage limits, line/transformer loading rates) using pandapower. Ensures optimization adheres to physical power system rules.
# Usage Examples
## 1.Run ExperimentsExecute power system optimization tasks:
bash python experiment.py Example code snippet from experiment.py: python 运行
### Render power network state
def render(self, kwargs): ax = pp.plotting.simple_plot(self.net, kwargs) return ax
## 2. Hyperparameter Evaluation
bash python drl/hp_tuning/evaluate_hps.py --directory 20240506_test_hp_tuning4 --which_hyperparam actor_fc_dims --metric average_return
## 3. Visualize Training Results
bash python drl/util/plot_returns.py --directory data/ --rolling-window 3 --mean --std
# Notes
Experimental algorithms (targetless.py, ddpg1step.py) are under development. Power system simulations require valid pandapower/simbench network data. Training generates test_returns.csv for offline analysis. Detailed comments and modular design facilitate extension.
