"""Plot the fixed-temperature and auto-tuned HalfCheetah SAC runs."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parents[1]
fixed_run = root / "exp/HalfCheetah-v4_sac_sd1_20260922_221537"
auto_run = root / "exp/HalfCheetah-v4_sac_autotune_sd1_20260923_131844"
output_dir = root / "report_assets/halfcheetah_temperature_comparison"
output_dir.mkdir(parents=True, exist_ok=True)


def load_rows(run):
    with (run / "log.csv").open() as file:
        return list(csv.DictReader(file))


def evaluation_series(rows):
    rows = [row for row in rows if row.get("Eval_AverageReturn")]
    return (
        np.array([int(row["step"]) for row in rows]),
        np.array([float(row["Eval_AverageReturn"]) for row in rows]),
    )


def update_series(rows, key):
    rows = [row for row in rows if row.get(key)]
    return (
        np.array([int(row["step"]) for row in rows]),
        np.array([float(row[key]) for row in rows]),
    )


fixed_rows = load_rows(fixed_run)
auto_rows = load_rows(auto_run)
fixed_steps, fixed_returns = evaluation_series(fixed_rows)
auto_steps, auto_returns = evaluation_series(auto_rows)
auto_update_steps, auto_temperature = update_series(auto_rows, "temperature")

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, layout="constrained")
axes[0].plot(fixed_steps, fixed_returns, label="Fixed temperature α=0.1", linewidth=1.5)
axes[0].plot(auto_steps, auto_returns, label="Auto-tuned temperature", linewidth=1.5)
axes[0].axhline(6000, color="gray", linestyle="--", linewidth=1, label="Assignment threshold: 6000")
axes[0].set_ylabel("Evaluation return")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.25)

axes[1].plot(auto_update_steps, auto_temperature, color="tab:green", linewidth=1.5)
axes[1].axhline(0.1, color="gray", linestyle="--", linewidth=1, label="Initial α=0.1")
axes[1].set_xlabel("Environment steps")
axes[1].set_ylabel("Temperature α")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.25)

fig.suptitle("HalfCheetah-v4 SAC: fixed versus auto-tuned temperature")
for extension in ("png", "pdf"):
    fig.savefig(output_dir / f"halfcheetah_fixed_vs_autotune.{extension}", dpi=180)
plt.close(fig)


def first_threshold(steps, values, threshold=6000):
    indices = np.flatnonzero(values >= threshold)
    if len(indices) == 0:
        return None
    index = int(indices[0])
    return {"step": int(steps[index]), "return": float(values[index])}


def run_stats(steps, values):
    return {
        "first_mean_at_least_6000": first_threshold(steps, values),
        "peak": {"step": int(steps[values.argmax()]), "return": float(values.max())},
        "last": {"step": int(steps[-1]), "return": float(values[-1])},
        "last_10_mean": float(values[-10:].mean()),
        "last_10_min": float(values[-10:].min()),
        "last_10_max": float(values[-10:].max()),
        "num_evaluations_at_least_6000": int((values >= 6000).sum()),
        "num_evaluations": int(len(values)),
    }


summary = {
    "fixed_temperature_run": fixed_run.name,
    "auto_tuned_run": auto_run.name,
    "fixed_temperature": run_stats(fixed_steps, fixed_returns),
    "auto_tuned_temperature": run_stats(auto_steps, auto_returns),
    "auto_temperature": {
        "initial": float(auto_temperature[0]),
        "minimum": float(auto_temperature.min()),
        "minimum_step": int(auto_update_steps[auto_temperature.argmin()]),
        "maximum": float(auto_temperature.max()),
        "maximum_step": int(auto_update_steps[auto_temperature.argmax()]),
        "final": float(auto_temperature[-1]),
        "last_10_mean": float(auto_temperature[-10:].mean()),
    },
}
(output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(output_dir / "summary.md").write_text(
    """# HalfCheetah temperature comparison

固定温度基线和自动温度实验都使用 seed 1、1,000,000 个环境步和每次 10 个评估回合。自动温度版本首次在 225,000 步超过 6000，固定温度版本在 265,000 步超过 6000。固定温度版本峰值为 9664.60，自动温度版本峰值为 9346.23；最后 10 次评估的平均回报分别为 9412.11 和 9219.50。

自动温度从 0.1 开始，在 31,000 步附近降至 0.01997，随后逐步升高，最后稳定在约 0.124。该变化说明训练早期策略需要较强的熵调节，后期为了维持目标熵，温度提高。自动温度版本的最后阶段仍保持高回报，但本次单 seed 结果没有超过已经调好的固定温度基线。

![Comparison](halfcheetah_fixed_vs_autotune.png)
"""
)
print(json.dumps(summary, indent=2))
