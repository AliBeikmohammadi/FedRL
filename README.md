# Collaborative Value Function Estimation Under Model Mismatch:  
## A Federated Temporal Difference Analysis

This repository contains the code for our submission to ECML PKDD 2025, titled **"Collaborative Value Function Estimation Under Model Mismatch: A Federated Temporal Difference Analysis."** The code implements a Federated TD(0) algorithm for policy evaluation in environments with model mismatch. In our setting, each agent operates under a perturbed MDP (sampled from a distribution centered at the true transition model), and agents periodically share their value estimates to collaboratively learn the true value function.

---

## Repository Structure

├── FedTD-main.py # Main simulation script (Federated TD(0)) with argparse support. ├── FedTD-plot.py # Plotting routines for analyzing simulation results. ├── submit_jobs.py # (Optional) Code to generate sbatch scripts for parameter sweeps. ├── README.md # This file. ├── requirements.txt # List of required Python packages. ├── results/ # Directory for CSV output files (created automatically). ├── sbatch_scripts/ # Directory for generated sbatch scripts for full parameter sweeps. ├── sbatch_missing/ # Directory for sbatch scripts of missing configurations. └── plots/ # Directory for saving generated figures.

yaml
Copy

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/FedTD.git
cd FedTD
pip install -r requirements.txt
requirements.txt includes:

numpy
pandas
matplotlib
seaborn
Usage
Running the Simulation
You can run the Federated TD(0) simulation with default parameters:

bash
Copy
python FedTD-main.py
Or, specify custom parameters via command-line arguments. For example:

bash
Copy
python FedTD-main.py --n 10 --N 5 --Delta 0.1 --gamma 0.95 --alpha 0.01 --beta 0.5 --T 100000 --K 5 --sampling iid --iid_option uniform --reward_type uniform --seeds 0 1 2 3 4 --out_dir ./results
This will generate a CSV file (uniquely named based on the configuration) in the ./results directory.

Parameter Sweeps & SLURM Jobs
For extensive parameter sweeps, you can generate sbatch scripts for each configuration. Use the provided script (e.g., submit_jobs.py) to create sbatch scripts under the sbatch_scripts/ (or sbatch_missing/) folder. Then, submit them as an array job on a SLURM cluster:

bash
Copy
python submit_jobs.py
Submit the generated sbatch array job as follows:

bash
Copy
sbatch --array=1-<total_configs> sbatch_scripts/%a.sbatch
Plotting Results
The plotting routines in FedTD-plot.py allow you to visualize simulation results. For example, to plot error metrics from one or more CSV files with customizable axes, legend, and grid options:

bash
Copy
python FedTD-plot.py
You can also use the provided functions to filter results and study the effect of individual parameters (e.g., 
Δ
Δ, 
𝐾
K, 
𝑁
N). For instance, you may plot the effect of heterogeneity (
Δ
Δ) by loading all CSV files, filtering for fixed values (e.g., n=10, N=5, K=5, sampling=iid, ...), grouping by 
Δ
Δ, and then plotting RMSE over communication rounds.

Experiments & Suggested Plots
We recommend exploring several experiments and plots to understand the behavior of Federated TD(0):

Effect of Heterogeneity (
Δ
Δ)
Plot the final RMSE (or learning curves) versus different 
Δ
Δ values while keeping other parameters fixed.

Effect of Local Steps (
𝐾
K)
Compare learning curves for different numbers of local updates per communication round. Investigate the trade-off between local computation and global communication.

Effect of Number of Agents (
𝑁
N)
Analyze how increasing 
𝑁
N impacts convergence and variance, demonstrating the benefit of collaborative learning.

Combined Parameter Interactions
Generate scatter or box plots summarizing final error metrics as a function of multiple parameters (e.g., 
𝑁
N and 
Δ
Δ).

Our plotting code (in FedTD-plot.py) provides flexibility to filter CSV files and generate subplots for any error metric (e.g., 
∥
𝑉
𝑡
−
𝑉
∗
∥
2
∥V 
t
 −V 
∗
 ∥ 
2
​
 , MSE, RMSE) with options for logarithmic axes, grid styles, and saving figures.

Citation
If you use this code in your research, please cite our paper:

bibtex
Copy
@inproceedings{yourpaper2025,
  author = {Beikmohammadi, Ali and Collaborators},
  title = {Collaborative Value Function Estimation Under Model Mismatch: A Federated Temporal Difference Analysis},
  booktitle = {ECML PKDD 2025},
  year = {2025},
  publisher = {Springer},
  address = {Location}
}
Also cite related works as appropriate.

Contributing
Contributions, bug reports, and suggestions are welcome! Please open an issue or submit a pull request.

License
This project is licensed under the MIT License.

This repository is maintained by Ali Beikmohammadi and colleagues.

css
Copy

This README provides an overview of your project, clear instructions for inst
