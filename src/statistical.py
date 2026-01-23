import csv
import os
from pathlib import Path
from agents import get_average_fitness, get_max_fitness
import pandas as pd
from scipy.stats import t
import numpy as np

import config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data"

def clear_data_fitness(csv_filename="fitness_metrics.csv", data_folder="data"):
    """
    Clear existing fitness metrics data by deleting the CSV file if it exists.
    
    Parameters
    ----------
    csv_filename : str, optional
        Name of the CSV file (default: "fitness_metrics.csv").
    data_folder : str, optional
        Path to the folder where the CSV is saved (default: "data").
        
    Returns
    -------
    None
    """
    data_path = Path(data_folder)
    csv_path = data_path / csv_filename
    
    if csv_path.exists():
        csv_path.unlink()
        print(f"Cleared existing data at {csv_path}")
    else:
        print(f"No existing data to clear at {csv_path}")  


def save_fitness_metrics(agents, csv_filename="fitness_metrics.csv", data_folder="data"):
    """
    Calculate average and maximum fitness metrics and append them to a CSV file.
    
    Creates the CSV file with headers if it does not exist. Appends the current
    metrics (average fitness, max fitness) to the file.
    
    Parameters
    ----------
    agents : list of dict
        List of agents, each with a 'payoff' key.
    csv_filename : str, optional
        Name of the CSV file (default: "fitness_metrics.csv").
    data_folder : str, optional
        Path to the folder where the CSV will be saved (default: "data").
    clear_data : bool, optional
        If True, clears the existing data before writing (default: False).
        
    Returns
    -------
    None
    """

    print("Saving fitness metrics...")

    # Get metrics
    avg_fitness = get_average_fitness(agents)
    max_fitness = get_max_fitness(agents)
    
    # Ensure data folder exists
    data_path = Path(data_folder)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Full path to CSV file
    csv_path = data_path / csv_filename
    
    # Check if file exists to determine if we need to write headers
    file_exists = csv_path.exists()
    
    # Write to CSV
    with open(csv_path, mode='a', newline='') as csvfile:
        fieldnames = ['average_fitness', 'max_fitness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write headers if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the metrics
        writer.writerow({
            'average_fitness': avg_fitness,
            'max_fitness': max_fitness
        })


def combine_fitness_metrics(data_folder=DATA_PATH, prefix="fitness_metrics", output_file_name="averaged_fitness_metrics.csv", N=None, ci=0.95):
    """
    Load all fitness metric CSV files with a given prefix, compute the
    mean and confidence intervals for average and max fitness per simulation step,
    and save the averaged values to a CSV.

    Parameters
    ----------
    data_folder : str or Path
        Path to the folder containing CSV files.
    prefix : str
        Filename prefix to match (default: 'fitness_metrics').
    N : int or None
        Number of steps (rows) to include. If None, use all available rows.
    ci : float
        Confidence level for the error intervals (default: 0.95).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'average_fitness_mean', 'average_fitness_lower', 'average_fitness_upper'
        - 'max_fitness_mean', 'max_fitness_lower', 'max_fitness_upper'
        Each row corresponds to a simulation step.
    """
    data_path = Path(data_folder)

    if not data_path.exists():
        raise FileNotFoundError(f"Data folder does not exist: {data_path.resolve()}")

    csv_files = sorted(data_path.glob(f"{prefix}*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files starting with '{prefix}' found in {data_path.resolve()}")

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if N is not None:
            df = df.iloc[:N]
        dfs.append(df)

    n_runs = len(dfs)

    # Combine runs into arrays: steps x runs
    avg_arr = np.stack([df["average_fitness"].values for df in dfs], axis=1)
    max_arr = np.stack([df["max_fitness"].values for df in dfs], axis=1)

    # --- Average fitness: mean + t-based confidence interval ---
    avg_mean = avg_arr.mean(axis=1)
    avg_std = avg_arr.std(axis=1, ddof=1)
    t_val = t.ppf(1 - (1 - ci)/2, df=n_runs - 1)
    avg_lower = np.clip(avg_mean - t_val * (avg_std / np.sqrt(n_runs)), 0, 100)
    avg_upper = np.clip(avg_mean + t_val * (avg_std / np.sqrt(n_runs)), 0, 100)

    # --- Max fitness: percentile-based interval ---
    max_mean = max_arr.mean(axis=1)
    max_lower = np.clip(np.percentile(max_arr, 2.5, axis=1), 0, 100)
    max_upper = np.clip(np.percentile(max_arr, 97.5, axis=1), 0, 100)

    # Combine into DataFrame
    averaged = pd.DataFrame({
        "average_fitness_mean": avg_mean,
        "average_fitness_lower": avg_lower,
        "average_fitness_upper": avg_upper,
        "max_fitness_mean": max_mean,
        "max_fitness_lower": max_lower,
        "max_fitness_upper": max_upper
    })

    # Save to CSV
    averaged.to_csv(data_path / f"averaged_{prefix}.csv", index=False)


    return averaged

def save_fitness_metrics_timeseries(run_data, csv_filename="fitness_metrics.csv", data_folder="data", overwrite=True):
    """
    Save per-timestep average and max fitness to a CSV.

    Parameters
    ----------
    run_data : np.ndarray
        Array of shape (timesteps, num_agents) containing agent payoffs per timestep.
    csv_filename : str
        Output CSV filename.
    data_folder : str
        Folder to save CSV into.
    overwrite : bool
        If True, overwrite the file. If False, append (only safe if you know what you're doing).
    """
    run_data = np.asarray(run_data)

    avg_series = run_data.mean(axis=1)  # (timesteps,)
    max_series = run_data.max(axis=1)   # (timesteps,)

    data_path = Path(data_folder)
    data_path.mkdir(parents=True, exist_ok=True)
    csv_path = data_path / csv_filename

    df = pd.DataFrame({
        "average_fitness": avg_series,
        "max_fitness": max_series
    })

    mode = "w" if overwrite else "a"
    header = True if overwrite or not csv_path.exists() else False
    df.to_csv(csv_path, mode=mode, header=header, index=False)


