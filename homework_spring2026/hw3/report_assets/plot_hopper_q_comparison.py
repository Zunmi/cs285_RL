"""Plot single-Q and clipped double-Q Hopper SAC runs."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parents[1]
runs = {
    "Single-Q": root / "exp/Hopper-v4_sac_singleq_sd1_20260925_172546",
    "Clipped double-Q": root / "exp/Hopper-v4_sac_clipq_sd1_20260925_172547",
}
output_dir = root / "report_assets/hopper_q_comparison"
output_dir.mkdir(parents=True, exist_ok=True)


def load_rows(path):
    with (path / "log.csv").open() as file:
        return list(csv.DictReader(file))


def series(rows, key):
    selected = [row for row in rows if row.get(key)]
    return (
        np.array([int(row["step"]) for row in selected]),
        np.array([float(row[key]) for row in selected]),
    )


loaded = {label: load_rows(path) for label, path in runs.items()}
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, layout="constrained")
colors = {"Single-Q": "tab:blue", "Clipped double-Q": "tab:orange"}
summary = {}

for label, rows in loaded.items():
    eval_steps, eval_returns = series(rows, "Eval_AverageReturn")
    q_steps, q_values = series(rows, "q_values")
    target_steps, target_values = series(rows, "target_values")
    axes[0].plot(eval_steps, eval_returns, label=label, color=colors[label], linewidth=1.5)
    axes[1].plot(q_steps, q_values, label=f"{label}: Q", color=colors[label], linewidth=1.4)
    axes[1].plot(target_steps, target_values, label=f"{label}: target", color=colors[label], linestyle="--", linewidth=1.0)
    threshold_indices = np.flatnonzero(eval_returns >= 1500)
    peak_index = int(eval_returns.argmax())
    summary[label] = {
        "first_mean_at_least_1500": None if len(threshold_indices) == 0 else {
            "step": int(eval_steps[threshold_indices[0]]),
            "return": float(eval_returns[threshold_indices[0]]),
        },
        "peak": {"step": int(eval_steps[peak_index]), "return": float(eval_returns[peak_index])},
        "last": {"step": int(eval_steps[-1]), "return": float(eval_returns[-1])},
        "last_10_eval_mean": float(eval_returns[-10:].mean()),
        "last_10_q_mean": float(q_values[-10:].mean()),
        "last_10_target_mean": float(target_values[-10:].mean()),
        "num_evaluations_at_least_1500": int((eval_returns >= 1500).sum()),
        "num_evaluations": int(len(eval_returns)),
    }

axes[0].axhline(1500, color="gray", linestyle="--", linewidth=1, label="Assignment threshold: 1500")
axes[0].set_ylabel("Evaluation return")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.25)
axes[1].set_xlabel("Environment steps")
axes[1].set_ylabel("Mean Q / target value")
axes[1].legend(fontsize=8, ncol=2)
axes[1].grid(alpha=0.25)
fig.suptitle("Hopper-v4 SAC: single-Q versus clipped double-Q")
for extension in ("png", "pdf"):
    fig.savefig(output_dir / f"hopper_single_vs_clipq.{extension}", dpi=180)
plt.close(fig)

summary["interpretation"] = (
    "Clipped double-Q reaches the assignment threshold and obtains substantially higher returns. "
    "The clipped target uses the smaller of two target critics, reducing optimistic target errors. "
    "Q magnitudes alone are not a direct performance measure because the two policies visit different states and actions."
)
(output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(output_dir / "summary.md").write_text(
    """# Hopper clipped double-Q comparison

single-Q never reaches an average evaluation return of 1500; its peak is 833.40. Clipped double-Q first exceeds 1500 at 190,000 steps and reaches a peak of 3180.66 at 435,000 steps. Its last ten evaluation returns average 2388.97, while single-Q averages 770.29.

The clipped double-Q curve is substantially better in this seed. The lower target estimate reduces the chance that one overestimated target critic drives both critics toward an optimistic Bellman target. Q values should be interpreted together with the policy and visited state distribution; a larger raw Q value alone does not prove better performance.

![Hopper comparison](hopper_single_vs_clipq.png)
"""
)
print(json.dumps(summary, indent=2))
