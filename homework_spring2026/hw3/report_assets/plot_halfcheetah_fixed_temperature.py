"""Plot and summarize the completed Section 3.4 HalfCheetah experiment."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


run = Path(__file__).resolve().parents[1] / "exp/HalfCheetah-v4_sac_sd1_20260922_221537"
with (run / "log.csv").open() as file:
    rows = list(csv.DictReader(file))
evals = [row for row in rows if row.get("Eval_AverageReturn")]
steps = np.array([int(row["step"]) for row in evals])
returns = np.array([float(row["Eval_AverageReturn"]) for row in evals])
peak_index = int(returns.argmax())
first_pass_index = int(np.flatnonzero(returns >= 6000)[0])

fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
ax.plot(steps, returns, linewidth=1.6, label="SAC, fixed temperature 0.1, seed 1")
ax.axhline(6000, color="tab:orange", linestyle="--", label="Assignment threshold: 6000")
ax.scatter(steps[peak_index], returns[peak_index], s=28, color="tab:blue", zorder=3)
ax.set(xlabel="Environment steps", ylabel="Evaluation return (mean of 10 episodes)",
       title="HalfCheetah-v4: fixed-temperature SAC", xlim=(0, 1_000_000))
ax.ticklabel_format(axis="x", style="plain")
ax.grid(alpha=0.25)
ax.legend(loc="lower right")
for extension in ("png", "pdf"):
    fig.savefig(run / f"halfcheetah_eval_return.{extension}", dpi=180)
plt.close(fig)

summary = {
    "run": run.name,
    "completed_environment_steps": 1_000_000,
    "completion_evidence": "Saved W&B output.log reports 1000000/1000000",
    "training_loop_seconds": 22191,
    "seed": 1,
    "temperature": 0.1,
    "auto_tune_temperature": False,
    "number_of_critics": 1,
    "evaluation_episodes_per_checkpoint": 10,
    "evaluation_count": len(evals),
    "first_mean_at_least_6000": {"step": int(steps[first_pass_index]), "return": float(returns[first_pass_index])},
    "peak_mean": {"step": int(steps[peak_index]), "return": float(returns[peak_index])},
    "last_evaluation": {"step": int(steps[-1]), "return": float(returns[-1]),
                        "episode_std": float(evals[-1]["Eval_StdReturn"])},
    "last_10_evaluations": {"step_range": [int(steps[-10]), int(steps[-1])],
                            "mean": float(returns[-10:].mean()),
                            "min": float(returns[-10:].min()),
                            "max": float(returns[-10:].max())},
    "assignment_return_threshold_met": True,
    "limitations": ["One training seed; ten evaluation episodes are not ten training seeds.",
                    "Last evaluation is at step 995000; final checkpoint was not separately evaluated."],
}
(run / "results_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
report = f"""# Section 3.4: HalfCheetah fixed-temperature SAC

使用默认配置、固定温度 0.1、单个 Critic 和训练 seed 1，完成 1,000,000 个环境步。每隔 5,000 步评估 10 个回合，并记录平均回报。

平均评估回报在 {steps[first_pass_index]:,} 步首次达到 {returns[first_pass_index]:.2f}，超过作业要求的 6000；在 {steps[peak_index]:,} 步达到最高值 {returns[peak_index]:.2f}。最后一次评估位于 {steps[-1]:,} 步，平均回报为 {returns[-1]:.2f}。最后 10 次评估（950,000–995,000 步）的平均回报为 {returns[-10:].mean():.2f}，范围为 {returns[-10:].min():.2f}–{returns[-10:].max():.2f}，均高于 6000。

曲线展示原始平均评估回报，未做平滑，横轴为环境步数。已记录的主要训练指标及最终模型参数均未出现 NaN/Inf；Q 值与目标值接近只能作为数值检查，不能单独证明价值估计准确。

本实验仅使用一个训练 seed，不能据此判断不同 seed 下的表现。最后一次评估在 995,000 步，最终 1,000,000 步 checkpoint 未另行评估。本轮保留为 Section 3.5 自动温度实验的固定温度基线。

![Evaluation return](halfcheetah_eval_return.png)
"""
(run / "results_summary.md").write_text(report)
print(json.dumps(summary, indent=2))
print("Saved evaluation plots and summaries to", run)
