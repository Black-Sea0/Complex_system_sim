import config
import numpy as np
from pathlib import Path
import pandas as pd
import statistical
import matplotlib.pyplot as plt
from scipy.stats import t

DATA_FOLDER = Path("data")
N_STEPS = config.N_steps      
CI = 0.95

ps = []
avg_per_p = []
avg_err = []

for p in config.p_list:
    p_str = f"{p:.2f}"
    statistical.combine_fitness_metrics(
    data_folder="data",
    prefix=f"fitness_metrics_p{p_str}_",
    N=N_STEPS,
    ci=CI
)
    run_files = sorted(DATA_FOLDER.glob(f"fitness_metrics_p{p_str}_run*.csv"))

    run_means = []
    for f in run_files:
        df = pd.read_csv(f).iloc[:N_STEPS]  # ensure 20 steps
        run_means.append(df["average_fitness"].mean())

    run_means = np.array(run_means, dtype=float)

    # mean across runs = one value per p
    m = run_means.mean()

    # 95% CI across runs
    n = len(run_means)
    s = run_means.std(ddof=1)
    t_val = t.ppf(1 - (1 - CI)/2, df=n - 1)
    err = t_val * s / np.sqrt(n)

    ps.append(p)
    avg_per_p.append(m)
    avg_err.append(err)

plt.figure()
plt.errorbar(ps, avg_per_p, yerr=avg_err, fmt='-o', capsize=4)
plt.xlabel("Probability of Collaborating (p)")
plt.ylabel("Average payoff")
plt.title("Average Payoff vs Collaboration Probability")
plt.tight_layout()
plt.show()

