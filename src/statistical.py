import csv
import os
from pathlib import Path
from agents import get_average_fitness, get_max_fitness
import pandas as pd

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


def combine_fitness_metrics(data_folder=DATA_PATH, prefix="fitness_metrics", N=None):
    """
    Load all fitness metric CSV files with a given prefix, compute the
    average and standard deviation of fitness per simulation step,
    and save the averaged values to a CSV.

    Parameters
    ----------
    data_folder : str or Path
        Path to the folder containing CSV files.
    prefix : str
        Filename prefix to match (default: 'fitness_metrics').
    N : int or None
        Number of steps (rows) to include. If None, use all available rows.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'average_fitness_mean', 'average_fitness_std'
        - 'max_fitness_mean', 'max_fitness_std'
        Each row corresponds to a simulation step.
    """
    data_path = Path(data_folder)

    if not data_path.exists():
        raise FileNotFoundError(f"Data folder does not exist: {data_path.resolve()}")

    csv_files = sorted(data_path.glob(f"{prefix}*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files starting with '{prefix}' found in {data_path.resolve()}"
        )

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if N is not None:
            df = df.iloc[:N]
        dfs.append(df)

    combined = pd.concat(dfs, axis=0)

    # Compute mean and std per step
    mean_df = combined.groupby(combined.index).mean()
    std_df = combined.groupby(combined.index).std()

    # Rename columns to indicate mean and std
    averaged = pd.DataFrame({
        "average_fitness_mean": mean_df["average_fitness"],
        "average_fitness_std": std_df["average_fitness"],
        "max_fitness_mean": mean_df["max_fitness"],
        "max_fitness_std": std_df["max_fitness"]
    })

    # Save to CSV
    averaged.to_csv(data_path / "averaged_fitness_metrics.csv", index=False)

    return averaged

