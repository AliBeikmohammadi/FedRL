import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

###############################################################################
# Plotting Function
###############################################################################
def plot_multiple_csvs(csv_files, 
                       metrics=["error_l2", "error_l2sq", "mse", "rmse"],
                       legend_labels=None,
                       legend_title="",
                       xlim=None,
                       ylim=None,
                       grid=True,
                       grid_linestyle='-',
                       plot_std=True,
                       save_name=None,
                       log_x=True,
                       log_y=True,
                       plot_dir = "./plots"):
    """
    Plot error metrics from multiple CSV files, with options for custom legends, axis limits,
    logarithmic scales, and whether to plot standard deviation bands.
    
    Parameters:
      csv_files : list of str
          Paths to CSV files produced by fed_td_experiment(). Each file should contain
          columns such as "round", "error_l2", "error_l2sq", "mse", and "rmse".
      metrics : list of str, optional
          List of metric names (columns) to plot in separate subplots.
      legend_labels : list of str, optional
          List of labels for each CSV file. If not provided, the file names will be used.
      legend_title : str, optional
          Title for the legend.
      xlim : tuple (xmin, xmax), optional
          Limits for the x-axis.
      ylim : tuple (ymin, ymax), optional
          Limits for the y-axis.
      grid : bool, optional
          Whether to display grid lines.
      grid_linestyle : str, optional
          Line style for the grid (e.g., '-' for solid lines).
      plot_std : bool, optional
          If True, plot the standard deviation bands; otherwise, only plot the mean.
      save_name : str, optional
          If provided, the figure is saved as "./plots/{save_name}.png".
      log_x : bool, optional
          If True, set the x-axis to logarithmic scale.
      log_y : bool, optional
          If True, set the y-axis to logarithmic scale.
    """
    # Update matplotlib parameters.
    parameters = {
        'axes.labelsize': 14, 
        'axes.titlesize': 14, 
        'legend.fontsize': 14,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    }
    plt.rcParams.update(parameters)
    
    # Default legend labels to file basenames if not provided.
    if legend_labels is None:
        legend_labels = [os.path.basename(f) for f in csv_files]
    
    # Define a custom palette with distinct colors.
    palette = sns.color_palette("deep", len(csv_files))
    
    # LaTeX-style labels for each metric.
    metric_labels = {
        "error_l2": r"$\|\|V^t - V^*\|\|_2$",
        "error_l2sq": r"$\|\|V^t - V^*\|\|_2^2$",
        "mse": "MSE",
        "rmse": "RMSE"
    }
    
    n_metrics = len(metrics)
    fig, axs = plt.subplots(n_metrics, 1, figsize=(7, 5 * n_metrics))
    if n_metrics == 1:
        axs = [axs]
    
    for i, metric in enumerate(metrics):
        ax = axs[i]
        for j, csv_file in enumerate(csv_files):
            df = pd.read_csv(csv_file)
            group = df.groupby("round")
            rounds = group.mean().index.values
            metric_mean = group[metric].mean().values
            metric_std = group[metric].std().values
            
            ax.plot(rounds, metric_mean, label=legend_labels[j], color=palette[j])
            if plot_std:
                ax.fill_between(rounds, metric_mean - metric_std, metric_mean + metric_std, 
                                color=palette[j], alpha=0.2)
        
        ax.set_xlabel("Communication Round")
        ax.set_ylabel(metric_labels.get(metric, metric))
        #ax.set_title(f"{metric_labels.get(metric, metric)} vs. Communication Round")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if log_x:
            ax.set_xscale("log")
        else:
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        if log_y:
            ax.set_yscale("log")
        else:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        if grid:
            ax.grid(True, which="both", ls=grid_linestyle)
        
        # Custom legend settings.
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, legend_labels, loc='upper right', 
                  fontsize=10, title_fontsize=10, title=legend_title)
    
    fig.tight_layout()
    if save_name is not None:
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, f"{save_name}.png")
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Example CSV files (adjust paths as needed)
    csv_files = [
        "./results/fedtd_n_10_N_1_Delta_0.01_gamma_0.8_alpha_0.01_beta_0.5_T_100000_K_5_s_iid_iidopt_uniform_R_uniform.csv",
        "./results/fedtd_n_10_N_10_Delta_0.1_gamma_0.95_alpha_0.01_beta_0.5_T_100000_K_5_s_iid_iidopt_uniform_R_uniform.csv"
    ]
    metrics=["error_l2", "mse", "rmse"]
    legend_labels = ["N = 1", "N = 10"]
    
    # Plot selected metrics with custom settings.
    plot_multiple_csvs(
        csv_files,
        metrics=metrics,
        legend_labels=legend_labels,
        legend_title="N:",
        xlim=None,
        ylim=None,
        grid=True,
        grid_linestyle='-',
        plot_std=True,
        log_x=False,
        log_y=False,
        save_name="test",
        plot_dir = "./plots"
    )