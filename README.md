# FedTD(0)
## Collaborative Value Function Estimation Under Model Mismatch: A Federated Temporal Difference Analysis

This repository contains the code for our submission to **ECML PKDD 2025**, titled **"Collaborative Value Function Estimation Under Model Mismatch: A Federated Temporal Difference Analysis."** The code implements a Federated TD(0) algorithm for policy evaluation in environments with model mismatch. In our setting, each agent operates under a perturbed MDP (sampled from a distribution centered at the true transition model), and agents periodically share their value estimates to collaboratively learn the true value function.

---

## Repository Structure

```
├── FedTD-main.py          # Main simulation script (Federated TD(0)) with argparse support.
├── FedTD-plot.py          # Plotting routines for analyzing simulation results.
├── README.md              # This file.
├── requirements.txt       # List of required Python packages.
├── results/               # Directory for CSV output files (created automatically).
└── plots/                 # Directory for saving generated figures (created automatically).
```

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/AliBeikmohammadi/FedRL
cd FedRL
pip install -r requirements.txt
```
---

## Usage

### Running the Simulation

The main simulation script (`FedTD-main.py`) can be run with various arguments. To view all available options, use:

```bash
python FedTD-main.py -h
```

This will display:

```
usage: FedTD-main.py [-h] [--n N] [--N N] [--Delta DELTA] [--gamma GAMMA] 
                     [--alpha ALPHA] [--beta BETA] [--T T] [--K K] 
                     [--sampling {iid,markov}] [--iid_option {uniform,stationary}]
                     [--reward_type {uniform,gaussian}] [--seeds SEEDS [SEEDS ...]]
                     [--out_dir OUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --n N                 Number of states (default: 10)
  --N N                 Number of agents (default: 10)
  --Delta DELTA         Heterogeneity level (default: 0.01)
  --gamma GAMMA         Discount factor (default: 0.8)
  --alpha ALPHA         Local step size (default: 0.01)
  --beta BETA           Global step size (default: 0.1)
  --T T                 Total number of communication rounds (default: 100000)
  --K K                 Number of local steps per round (default: 5)
  --sampling {iid,markov}
                        Sampling mode: "iid" or "markov" (default: "iid")
  --iid_option {uniform,stationary}
                        IID option: "uniform" or "stationary" (default: "uniform")
  --reward_type {uniform,gaussian}
                        Reward type: "uniform" or "gaussian" (default: "uniform")
  --seeds SEEDS [SEEDS ...]
                        List of random seeds (default: [0,1,2,3,4])
  --out_dir OUT_DIR     Output directory for CSV results (default: "./results")
```

### Example Usage

Run with default settings:

```bash
python FedTD-main.py
```

Run with custom parameters:

```bash
python FedTD-main.py --n 10 --N 5 --Delta 0.1 --gamma 0.95 --alpha 0.01 --beta 0.5 --T 100000 --K 5 --sampling iid --iid_option uniform --reward_type uniform --seeds 0 1 2 3 4 --out_dir ./results
```

This will generate a CSV file (uniquely named based on the configuration) in the `./results` directory.

---

---

### Plotting Results

The plotting routines in `FedTD-plot.py` allow you to visualize and analyze simulation results. For example, to plot error metrics from one or more CSV files with customizable axes, legend, grid, and option to show or hide standard deviation bands:

```bash
python FedTD-plot.py
```

You can modify the script or use the provided functions to filter results and study the effect of individual parameters (e.g., \(\Delta\), \(K\), \(N\)). For instance, you might plot the effect of heterogeneity (\(\Delta\)) by loading all CSV files, filtering for fixed values (e.g., `n=10, N=5, K=5, sampling=iid, ...`), grouping by \(\Delta\), and then plotting RMSE over communication rounds.

---

## Experiments & Suggested Plots

Here are some suggested experiments and plots to gain insights into the performance of Federated TD(0):

- **Effect of Heterogeneity (\(\Delta\))**  
  Compare learning curves (or final RMSE values) for different \(\Delta\) values while keeping other parameters fixed. This will illustrate how sensitive FedTD(0) is to environment mismatch.

- **Effect of Local Steps (\(K\))**  
  Evaluate the impact of the number of local updates per communication round on convergence speed. Plot RMSE over communication rounds for different \(K\) values to assess the trade-off between local computation and global communication.

- **Effect of Number of Agents (\(N\))**  
  Analyze how increasing the number of agents affects the convergence and variance of the global value estimates. A plot of final error versus \(N\) will demonstrate the benefit of collaborative learning.

Our plotting code in `FedTD-plot.py` supports flexible filtering and grouping, so you can easily generate these plots by selecting appropriate CSV files and parameters.

---

## Citation

If you find this code useful in your research, please cite our paper:

```
TBD
```

## Contributing

Contributions, bug reports, and suggestions are welcome! Please open an issue or submit a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---
