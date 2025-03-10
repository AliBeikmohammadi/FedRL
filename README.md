# Collaborative Value Function Estimation Under Model Mismatch:  
## A Federated Temporal Difference Analysis

This repository contains the code for our submission to **ECML PKDD 2025**, titled **"Collaborative Value Function Estimation Under Model Mismatch: A Federated Temporal Difference Analysis."** The code implements a Federated TD(0) algorithm for policy evaluation in environments with model mismatch. In our setting, each agent operates under a perturbed MDP (sampled from a distribution centered at the true transition model), and agents periodically share their value estimates to collaboratively learn the true value function.

---

## Repository Structure

```
├── FedTD-main.py          # Main simulation script (Federated TD(0)) with argparse support.
├── FedTD-plot.py          # Plotting routines for analyzing simulation results.
├── submit_jobs.py         # (Optional) Code to generate sbatch scripts for parameter sweeps.
├── README.md              # This file.
├── requirements.txt       # List of required Python packages.
├── results/               # Directory for CSV output files (created automatically).
├── sbatch_scripts/        # Directory for generated sbatch scripts for full parameter sweeps.
├── sbatch_missing/        # Directory for sbatch scripts of missing configurations.
└── plots/                 # Directory for saving generated figures.
```

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/FedTD.git
cd FedTD
pip install -r requirements.txt
```

*requirements.txt* should include:
- numpy
- pandas
- matplotlib
- seaborn

---

## Usage

### Running the Simulation

You can run the Federated TD(0) simulation with default parameters:

```bash
python FedTD-main.py
```

Or, specify custom parameters via command-line arguments. For example:

```bash
python FedTD-main.py --n 10 --N 5 --Delta 0.1 --gamma 0.95 --alpha 0.01 --beta 0.5 --T 100000 --K 5 --sampling iid --iid_option uniform --reward_type uniform --seeds 0 1 2 3 4 --out_dir ./results
```

This will generate a CSV file (named uniquely based on the configuration) in the `./results` directory.

### Parameter Sweeps & SLURM Jobs

For extensive parameter sweeps, you can generate sbatch scripts for each configuration using the provided `submit_jobs.py` script. This will create sbatch files (stored under `sbatch_scripts/` or `sbatch_missing/`) that you can submit as an array job on a SLURM cluster:

```bash
python submit_jobs.py
```

Submit the generated sbatch array job as follows:

```bash
sbatch --array=1-<total_configs> sbatch_scripts/%a.sbatch
```

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

- **Combined Parameter Interactions**  
  Consider scatter or box plots of final error metrics as functions of multiple parameters (e.g., final RMSE vs. \(N\) for different \(\Delta\) values) to summarize overall performance.

Our plotting code in `FedTD-plot.py` supports flexible filtering and grouping, so you can easily generate these plots by selecting appropriate CSV files and parameters.

---

## Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@inproceedings{yourpaper2025,
  author = {Beikmohammadi, Ali and Collaborators},
  title = {Collaborative Value Function Estimation Under Model Mismatch: A Federated Temporal Difference Analysis},
  booktitle = {ECML PKDD 2025},
  year = {2025},
  publisher = {Springer},
  address = {Location}
}
```

Also, please cite related works as appropriate.

---

## Contributing

Contributions, bug reports, and suggestions are welcome! Please open an issue or submit a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

*This repository is maintained by Ali Beikmohammadi and colleagues.*
