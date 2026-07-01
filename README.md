# AID — Agent Information Dissemination

A simulation framework for studying how information propagates across a team of robots. A subset of robots act as **leaders**, following planned routes to visit and inform the rest of the team. Leaders can optionally recruit **assistant** to help spread information faster, and the framework supports both fixed heuristic strategies and a **reinforcement-learning (PPO) policy** for deciding when and how to assign assistants.

The project includes tooling to run single simulations, batch experiments, benchmarks against reference datasets, and to visualize and report on the results.

## Features

- **Multi-agent simulation** of leaders and assistants spreading information across a 2D map
- **Multiple assignment strategies** for recruiting assistant agents:
  - `normal-case`, `limited`, `toggle`, `adaptive-limit`, `adaptive-freq`, `random`, `no-assistant`, `base-case`, and `rl` (learned policy)
- **Reinforcement learning agents**: PPO implementation for learning assignment policies
- **Experiment & benchmark runners**: for comparing strategies over many trials
- **Visualization tools**: including static plots and animated GIFs of simulation runs
- **Reporting utilities**: for logging and comparing per-run metrics (distance traveled, communication steps, assistants used, etc.)

## Repository structure

| File | Description |
|---|---|
| `main.py` | Entry point for running a single simulation, an experiment batch, or a benchmark |
| `info_dissemination.py` | Core simulation class (`InfoDiss`) — agent setup, stepping, evaluation |
| `agent.py` | Agent, `Leader`, and `IntelligentLeader` (RL-controlled leader) classes |
| `algorithms.py` | Routing/geometry helpers — nearest-neighbor TSP, MST, minimum enclosing circle, k-means mTSP |
| `rl_agents.py` | PPO reinforcement learning agent implementation |
| `ctde.py` | CTDE-PPO (Centralized Training, Decentralized Execution) agent implementation |
| `experiments.py` | Batch experiment and benchmark run orchestration |
| `evaluate.py` | Evaluation utilities for trained agents/policies |
| `train.py` | Training loop for RL agents |
| `info_dissemination.py` / `visualizations.py` | Simulation dynamics and plotting/animation |
| `plot.py`, `visualizations.py`, `visualization.ipynb` | Plotting and analysis notebooks/scripts |
| `report.py` | Logging simulation results to disk |
| `utils.py` | Shared helper functions |
| `models/` | Saved/trained RL model checkpoints |

## Requirements

- Python 3.x
- [PyTorch](https://pytorch.org/) (CPU or CUDA build)
- NumPy, SciPy, pandas, scikit-learn
- Matplotlib (for plotting/visualization)
- OpenCV (`opencv-python`)
- Shapely, sympy, deap, tqdm

Install everything with:

```bash
pip install -r requirements.txt
```

## Getting started

Clone the repository:

```bash
git clone https://github.com/hamza-k-abuzahra/aid.git
cd aid
```

### Running a simulation

`main.py` currently runs simulations by editing the `argparse.Namespace` configuration defined in `main()` (command-line argument parsing is present in the file but commented out). Open `main.py`, adjust the parameters you want (assigning strategy, number of agents/leaders, map size, seed, etc.), then run:

```bash
python main.py
```

Key configuration options include:

- `mode`: `"run"` (single simulation), `"custom"` (run with specific initial agent positions), `"experiment"` (batch run), or `"bench"` (benchmark against a dataset)
- `assigning_strategy`: one of `base-case`, `normal-case`, `limited`, `toggle`, `adaptive-limit`, `adaptive-freq`, `random`, `no-assistant`, or `rl`
- `experiment_name`: in `"experiment"` mode, the results will be saved under a folder with the specified name
- `n_agents`: total number of agents in the swarm, (not used in `"experiment"` mode)
- `k`: number of leader agents (not used in `"experiment"` mode)
- `map_size`: size of the simulation map
- `seed`: random seed for reproducibility 
- `rl_agent`: path to a trained model checkpoint (used when `assigning_strategy == "rl"`)
- `visualize`: enable step-by-step visualization/frame saving (boolean)

### Running experiments or benchmarks

Set `mode="experiment"` (with an `experiment_name`) to run a batch of trials via `experiments.py`, or `mode="bench"` with a `data_path` to benchmark a strategy against an external dataset.

### Training an RL agent

Use `train.py` to train a PPO policy for assistant assignment. Trained model checkpoints are saved under `models/` and can be loaded back in via the `rl_agent` argument.

### Visualizing results

`visualizations.py` and `visualization.ipynb` provide plotting utilities for agent trajectories, comparative metrics across strategies, and animated GIFs of a simulation run.

## Output

Each simulation run creates a timestamped `simulation_run_*` folder containing per-leader subfolders (used for saved visualization frames) and a report of run statistics (total/average distance traveled, communication steps, number of assistants used, etc.), generated via `report.py`.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Issues and pull requests are welcome if you'd like to extend the strategies, RL agents, or evaluation tooling.