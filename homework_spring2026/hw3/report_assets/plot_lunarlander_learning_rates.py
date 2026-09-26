"""Generate the HW3 Section 2.6 learning-rate comparison from saved logs."""

import csv
import json
import pickle
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report_assets" / "lunarlander_learning_rates"
OUTPUT.mkdir(parents=True, exist_ok=True)
RUNS = [
    ("1e-4", "LunarLander-v2_dqn_lr_1e-4_sd1_20260922_111712", "lunarlander_lr_1e-4.yaml"),
    ("3e-4", "LunarLander-v2_dqn_lr_3e-4_sd1_20260922_111712", "lunarlander_lr_3e-4.yaml"),
    ("1e-3", "LunarLander-v2_dqn_sd1_20260921_214108", "lunarlander.yaml"),
    ("3e-3", "LunarLander-v2_dqn_lr_3e-3_sd1_20260922_111712", "lunarlander_lr_3e-3.yaml"),
]
base = yaml.safe_load((ROOT / "experiments/dqn/lunarlander.yaml").read_text())
colors = ["#0072B2", "#009E73", "#E69F00", "#CC79A7"]
fig, ax = plt.subplots(figsize=(11, 7))
metrics = []
for (rate, directory, config_name), color in zip(RUNS, colors):
    config = yaml.safe_load((ROOT / "experiments/dqn" / config_name).read_text())
    assert config["learning_rate"] == float(rate)
    assert all(config[key] == value for key, value in base.items() if key not in {"learning_rate", "exp_name"})
    run = ROOT / "exp" / directory
    flags = json.loads((run / "flags.json").read_text())
    assert flags["seed"] == 1 and flags["eval_interval"] == 10000 and flags["num_eval_trajectories"] == 10
    with (run / "log.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    evals = [row for row in rows if row.get("Eval_AverageReturn")]
    steps = np.array([int(row["step"]) for row in evals])
    returns = np.array([float(row["Eval_AverageReturn"]) for row in evals])
    assert np.array_equal(steps, np.arange(0, 500000, 10000))
    for key in ["Eval_AverageReturn", "critic_loss", "q_values", "target_values", "grad_norm"]:
        assert all(np.isfinite(float(row[key])) for row in rows if row.get(key))
    with (run / "log.pkl").open("rb") as handle:
        saved = pickle.load(handle)
    assert max(row["step"] for row in saved["log"]) == 499000
    best = int(np.argmax(returns))
    late = returns[-10:]
    metrics.append({
        "learning_rate": rate,
        "first_eval_mean_ge_200_step": int(steps[returns >= 200][0]),
        "best_eval_mean": float(returns[best]),
        "best_eval_step": int(steps[best]),
        "last_logged_eval_mean": float(returns[-1]),
        "late_mean": float(late.mean()),
        "late_std_across_checkpoints": float(late.std()),
        "late_evals_ge_200": int(np.sum(late >= 200)),
        "all_evals_ge_200": int(np.sum(returns >= 200)),
        "saved_at": saved["time"],
        "run_directory": directory,
    })
    label = f"lr = {rate}" + (" (baseline)" if rate == "1e-3" else "")
    ax.plot(steps, returns, color=color, label=label, linewidth=1.8)

ax.axhline(200, color="#555555", linestyle="--", linewidth=1.1, label="Return = 200")
ax.axvspan(400000, 490000, color="#888888", alpha=0.07)
ax.set(title="LunarLander-v2: sensitivity to learning rate (Double DQN)", xlabel="Environment steps", ylabel="Evaluation return (mean of 10 episodes)", xlim=(0, 500000))
ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
ax.grid(alpha=0.2)
ax.legend(loc="lower right", fontsize=10)
caption = (
    "Learning rate was selected because it controls the scale of Adam updates and allows us to compare learning speed "
    "and stability of bootstrapped Q-learning. We compare the default 1e-3 with 1e-4, 3e-4, and 3e-3, keeping other "
    "training settings unchanged. Each run uses --seed 1 and 500,000 steps; each point averages 10 greedy evaluation "
    "episodes. Curves are unsmoothed. The shaded region marks the last 10 logged evaluations (400,000-490,000 steps). "
    "This is a single-run comparison per setting, not an average over training seeds."
)
fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.29)
fig.text(0.10, 0.035, textwrap.fill(caption, width=132), fontsize=9, va="bottom", linespacing=1.4)
for extension in ["png", "pdf"]:
    fig.savefig(OUTPUT / f"learning_rate_comparison.{extension}", dpi=200, bbox_inches="tight")
plt.close(fig)
with (OUTPUT / "metrics.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(metrics[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(metrics)

table = [
    "| 学习率 | 首次评估均值 ≥200 的步数 | 最高评估均值 | 最后一次评估 | 后期均值 | 后期波动标准差 | 后期达标次数 |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for item in metrics:
    table.append(f"| {item['learning_rate']} | {item['first_eval_mean_ge_200_step']:,} | {item['best_eval_mean']:.2f} | {item['last_logged_eval_mean']:.2f} | {item['late_mean']:.2f} | {item['late_std_across_checkpoints']:.2f} | {item['late_evals_ge_200']}/10 |")
report = """本实验完成 HW3 第 2.6 节要求：选择学习率作为超参数，将原始基线 1e-3 与另外三个取值 1e-4、3e-4、3e-3 在同一张图上比较。四次实验均完成 500,000 步，保存了最终检查点，记录的损失、Q 值和梯度范数均未出现 NaN 或无穷大。

![四个学习率的评估曲线](learning_rate_comparison.png)

图注：选择学习率，是因为它直接控制 Adam 参数更新的整体幅度，适合研究 Q-learning 学习速度与稳定性的关系。比较 1e-4、3e-4、默认值 1e-3 和 3e-3，其余训练设置一致；实验名称仅用于区分日志。每次训练使用 --seed 1，训练 500,000 个环境步，每隔 10,000 步评估 10 个回合，评估采用 ε=0 的贪心策略。横轴为环境步数，纵轴为评估平均回报；曲线未经平滑，浅灰区域为用于后期统计的最后 10 次评估。

""" + "\n".join(table) + """

统计口径：后期指第 400,000 至 490,000 步的 10 次评估。后期均值是这 10 个评估均值的平均数；后期波动标准差是这 10 个评估均值之间的总体标准差，不是单次评估的回合间标准差，也不是多训练种子之间的标准差。每次评估包含 10 个回合。最后一次评估位于 490,000 步；本文没有另行评估训练到 500,000 步时保存的最终 agent.pt。

1e-4 首次达到 200 分需要 380,000 步，学习最慢，但后期均值最高（244.22），波动最小（12.43），最后 10 次评估均超过 200，体现出较好的后期稳定性。

3e-4 在 190,000 步首次达标，明显早于 1e-4；其后期均值为 234.30，波动为 21.67，最后 10 次评估有 9 次达标。在本次实验中，它在学习速度和后期稳定性之间取得了较好的平衡。

默认 1e-3 在 170,000 步最早达标，并获得最高的峰值回报 280.34；但后期均值为 223.32、波动为 38.65，后期稳定性弱于两个较小学习率。峰值最好不等于后期平均表现最好。

3e-3 在 260,000 步首次达标，比默认值更晚；后期均值最低（167.19），波动最大（56.59），最后 10 次评估只有 3 次达标。这与较大学习率可能造成更新过大、学习不稳定的解释一致，但单次运行不能证明波动完全由学习率引起。

本次结果表明：降低学习率到 1e-4 有利于后期表现，但付出了更多环境交互步数；1e-3 学习最快且峰值最高；3e-4 提供了较好的速度与稳定性折中；增大到 3e-3 没有带来更快达标，反而表现更不稳定。

限制：每个设置只有一次训练，以上结论限于这些观测结果，不能视为跨随机种子的统计结论。当前脚本设置了 NumPy 和 PyTorch 的随机种子，但未显式设置 Gym 环境种子，因此相同 --seed 并不保证不同实验的环境随机过程完全一致。

数据来源（均位于 hw3/exp/）：

""" + "\n".join(f"- `{rate}`：`{directory}/log.csv`" for rate, directory, _ in RUNS) + "\n"
(OUTPUT / "summary.md").write_text(report, encoding="utf-8")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print(f"Artifacts: {OUTPUT}")
