import csv
import glob
import json
import os
import textwrap
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP_DIR = os.path.join(ROOT, "exp")
REPORT_PATH = os.path.join(ROOT, "report.pdf")
ASSET_DIR = os.path.join(ROOT, "report_assets")


@dataclass
class RunData:
    exp_name: str
    env_name: str
    batch_size: int
    eval_batch_size: int
    use_reward_to_go: bool
    normalize_advantages: bool
    use_baseline: bool
    baseline_learning_rate: float
    baseline_gradient_steps: int
    learning_rate: float
    discount: float
    video_log_freq: int
    log_rows: list[dict]
    directory: str

    @property
    def x(self) -> list[float]:
        return [float(row["Train_EnvstepsSoFar"]) for row in self.log_rows]

    @property
    def eval_y(self) -> list[float]:
        return [float(row["Eval_AverageReturn"]) for row in self.log_rows]

    @property
    def baseline_y(self) -> list[float]:
        if "Baseline Loss" not in self.log_rows[0]:
            return []
        values = []
        for row in self.log_rows:
            value = row.get("Baseline Loss", "")
            if value not in ("", None):
                values.append(float(value))
        return values

    @property
    def final_eval(self) -> float:
        return self.eval_y[-1]

    @property
    def max_eval(self) -> float:
        return max(self.eval_y)

    @property
    def first_200_step(self) -> float | None:
        for x, y in zip(self.x, self.eval_y):
            if y >= 200.0:
                return x
        return None

    @property
    def first_positive_step(self) -> float | None:
        for x, y in zip(self.x, self.eval_y):
            if y >= 0:
                return x
        return None


def wrap(text: str, width: int = 95) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def add_text(fig, x: float, y: float, text: str, size: int = 11, width: int = 95):
    fig.text(x, y, wrap(text, width), fontsize=size, va="top", ha="left")


def load_latest_runs() -> dict[str, RunData]:
    latest: dict[str, tuple[float, RunData]] = {}
    for directory in glob.glob(os.path.join(EXP_DIR, "*")):
        flags_path = os.path.join(directory, "flags.json")
        log_path = os.path.join(directory, "log.csv")
        if not (os.path.isfile(flags_path) and os.path.isfile(log_path)):
            continue
        with open(flags_path, encoding="utf-8") as f:
            flags = json.load(f)
        with open(log_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        run = RunData(
            exp_name=flags["exp_name"],
            env_name=flags["env_name"],
            batch_size=flags["batch_size"],
            eval_batch_size=flags["eval_batch_size"],
            use_reward_to_go=flags["use_reward_to_go"],
            normalize_advantages=flags["normalize_advantages"],
            use_baseline=flags["use_baseline"],
            baseline_learning_rate=flags["baseline_learning_rate"],
            baseline_gradient_steps=flags["baseline_gradient_steps"],
            learning_rate=flags["learning_rate"],
            discount=flags["discount"],
            video_log_freq=flags["video_log_freq"],
            log_rows=rows,
            directory=directory,
        )
        mtime = os.path.getmtime(directory)
        if run.exp_name not in latest or mtime > latest[run.exp_name][0]:
            latest[run.exp_name] = (mtime, run)
    return {k: v for k, (_, v) in latest.items()}


def exact_command(run: RunData) -> str:
    parts = [
        "PYTHONPATH=src uv run python src/scripts/run.py",
        f"--env_name {run.env_name}",
        "-n 100",
        f"-b {run.batch_size}",
    ]
    if run.eval_batch_size != 400:
        parts.append(f"-eb {run.eval_batch_size}")
    if run.use_reward_to_go:
        parts.append("-rtg")
    if run.discount != 1.0:
        parts.append(f"--discount {run.discount}")
    if run.learning_rate != 5e-3:
        parts.append(f"-lr {run.learning_rate}")
    if run.use_baseline:
        parts.append("--use_baseline")
        parts.append(f"-blr {run.baseline_learning_rate}")
        parts.append(f"-bgs {run.baseline_gradient_steps}")
    if run.normalize_advantages:
        parts.append("-na")
    if run.video_log_freq != -1:
        parts.append(f"--video_log_freq {run.video_log_freq}")
    parts.append(f"--exp_name {run.exp_name}")
    return " ".join(parts)


def plot_group(runs: list[RunData], title: str, path: str) -> None:
    plt.figure(figsize=(10, 6))
    for run in runs:
        plt.plot(run.x, run.eval_y, linewidth=2, label=run.exp_name)
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_baseline_loss(runs: list[RunData], title: str, path: str) -> None:
    plt.figure(figsize=(10, 6))
    for run in runs:
        y = run.baseline_y
        x = run.x[: len(y)]
        plt.plot(x, y, linewidth=2, label=run.exp_name)
    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Baseline Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    os.makedirs(ASSET_DIR, exist_ok=True)
    runs = load_latest_runs()

    exp1_names = [
        "cartpole",
        "cartpole_rtg",
        "cartpole_na",
        "cartpole_rtg_na",
        "cartpole_lb",
        "cartpole_lb_rtg",
        "cartpole_lb_na",
        "cartpole_lb_rtg_na",
    ]
    exp2_names = [
        "cheetah",
        "cheetah_baseline",
        "cheetah_na",
        "cheetah_baseline_na",
        "cheetah_baseline_bgs1_na",
        "cheetah_baseline_blr005_na",
    ]

    exp1 = {name: runs[name] for name in exp1_names if name in runs}
    exp2 = {name: runs[name] for name in exp2_names if name in runs}

    exp1_small = [exp1[name] for name in ["cartpole", "cartpole_rtg", "cartpole_na", "cartpole_rtg_na"]]
    exp1_large = [exp1[name] for name in ["cartpole_lb", "cartpole_lb_rtg", "cartpole_lb_na", "cartpole_lb_rtg_na"]]

    small_plot = os.path.join(ASSET_DIR, "experiment1_small_batch.png")
    large_plot = os.path.join(ASSET_DIR, "experiment1_large_batch.png")
    plot_group(exp1_small, "Experiment 1: Small Batch (b=1000)", small_plot)
    plot_group(exp1_large, "Experiment 1: Large Batch (b=4000)", large_plot)

    exp2_eval_plot = os.path.join(ASSET_DIR, "experiment2_eval_return.png")
    exp2_baseline_plot = os.path.join(ASSET_DIR, "experiment2_baseline_loss.png")
    plot_group(
        [
            exp2[name]
            for name in [
                "cheetah",
                "cheetah_baseline",
                "cheetah_na",
                "cheetah_baseline_na",
                "cheetah_baseline_bgs1_na",
                "cheetah_baseline_blr005_na",
            ]
            if name in exp2
        ],
        "Experiment 2: Eval Return",
        exp2_eval_plot,
    )
    plot_baseline_loss(
        [
            exp2[name]
            for name in [
                "cheetah_baseline",
                "cheetah_baseline_na",
                "cheetah_baseline_bgs1_na",
                "cheetah_baseline_blr005_na",
            ]
            if name in exp2
        ],
        "Experiment 2: Baseline Loss",
        exp2_baseline_plot,
    )

    with PdfPages(REPORT_PATH) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Experiment 1: CartPole", fontsize=18, y=0.98)
        add_text(fig, 0.08, 0.93, "Deliverables", size=14)
        add_text(fig, 0.08, 0.89, "Two plots are provided on the next two pages. Both use Eval_AverageReturn versus Train_EnvstepsSoFar, as required.")
        add_text(fig, 0.08, 0.84, "Answers", size=13)
        exp1_answers = [
            "1. Without advantage normalization, reward-to-go performed better than the trajectory-centric estimator. The small-batch `cartpole_rtg` run learned much faster and reached 200, while plain `cartpole` never reached 200 and finished at 49.78.",
            "2. Reward-to-go is generally preferred because it reduces variance. It removes reward terms that occurred before action a_t, so the gradient estimate attributes credit more locally to the action that actually influenced those future rewards.",
            "3. Advantage normalization helped a lot. In the small-batch case, both normalized runs (`cartpole_na` and `cartpole_rtg_na`) finished solved at 200. In the large-batch case, normalization also made convergence more reliable and prevented the late collapse seen in `cartpole_lb`.",
            "4. Batch size mattered. Larger batches made training smoother and let several unstable settings converge, but they also delayed feedback because each update required more interaction. Batch size improved stability, but variance-reduction tricks still mattered.",
        ]
        y = 0.8
        for answer in exp1_answers:
            add_text(fig, 0.08, y, answer, size=11)
            y -= 0.13
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Experiment 1: Exact Commands Used", fontsize=18, y=0.98)
        y = 0.92
        for i, name in enumerate(exp1_names, start=1):
            add_text(fig, 0.08, y, f"{i}. {exact_command(exp1[name])}", size=10)
            y -= 0.11
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Experiment 1: Small-Batch Plot", fontsize=18, y=0.98)
        ax = fig.add_axes([0.08, 0.42, 0.84, 0.45])
        ax.imshow(plt.imread(small_plot))
        ax.axis("off")
        add_text(fig, 0.08, 0.33, "Small-batch summary: `cartpole_na` reached 200 first at 24,153 environment steps. `cartpole_rtg_na` also solved the task, reaching 200 at 26,513 steps. Without normalization, `cartpole_rtg` still beat `cartpole`, but finished at 107.75 instead of staying solved.")
        add_text(fig, 0.08, 0.24, "`cartpole` peaked at 184.33 and never solved the environment, which supports the conclusion that reward-to-go and normalization both reduce gradient noise in this setting.")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Experiment 1: Large-Batch Plot", fontsize=18, y=0.98)
        ax = fig.add_axes([0.08, 0.42, 0.84, 0.45])
        ax.imshow(plt.imread(large_plot))
        ax.axis("off")
        add_text(fig, 0.08, 0.33, "Large-batch summary: `cartpole_lb_na`, `cartpole_lb_rtg`, and `cartpole_lb_rtg_na` all finished at 200. The earliest of these was `cartpole_lb_na`, which first reached 200 at 48,546 steps. `cartpole_lb` touched 200 once but collapsed and ended at 101.00.")
        add_text(fig, 0.08, 0.24, "`cartpole_lb_rtg` first reached 200 at 52,696 steps, and `cartpole_lb_rtg_na` first reached 200 at 52,843 steps. Larger batches improved stability, but the best behavior still came from adding variance reduction.")
        pdf.savefig(fig)
        plt.close(fig)

        cheetah = exp2.get("cheetah")
        cheetah_baseline = exp2.get("cheetah_baseline")
        cheetah_na = exp2.get("cheetah_na")
        cheetah_baseline_na = exp2.get("cheetah_baseline_na")
        cheetah_bgs1_na = exp2.get("cheetah_baseline_bgs1_na")
        cheetah_blr005_na = exp2.get("cheetah_baseline_blr005_na")

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Experiment 2: HalfCheetah", fontsize=18, y=0.98)
        add_text(fig, 0.08, 0.93, "Deliverables", size=14)
        exp2_summary = [
            f"Required no-baseline run `cheetah`: final eval return {cheetah.final_eval:.2f}, best eval return {cheetah.max_eval:.2f}.",
            f"Required baseline run `cheetah_baseline`: final eval return {cheetah_baseline.final_eval:.2f}, best eval return {cheetah_baseline.max_eval:.2f}.",
            f"`-na` no-baseline run `cheetah_na`: final eval return {cheetah_na.final_eval:.2f}, best eval return {cheetah_na.max_eval:.2f}.",
            f"`-na` baseline run `cheetah_baseline_na`: final eval return {cheetah_baseline_na.final_eval:.2f}, best eval return {cheetah_baseline_na.max_eval:.2f}.",
            f"Decreased baseline-gradient-steps run `cheetah_baseline_bgs1_na`: final eval return {cheetah_bgs1_na.final_eval:.2f}, best eval return {cheetah_bgs1_na.max_eval:.2f}.",
            f"Decreased baseline-learning-rate run `cheetah_baseline_blr005_na`: final eval return {cheetah_blr005_na.final_eval:.2f}, best eval return {cheetah_blr005_na.max_eval:.2f}.",
        ]
        y = 0.88
        for line in exp2_summary:
            add_text(fig, 0.08, y, line, size=10)
            y -= 0.08

        add_text(fig, 0.08, 0.38, "Answers", size=13)
        exp2_answers = [
            "1. The baseline helped a lot. The original no-baseline run `cheetah` finished at -414.43, while the original baseline run `cheetah_baseline` finished at 355.87 and peaked at 428.86.",
            "2. Adding `-na` improved both the no-baseline and baseline settings. `cheetah_na` finished at 321.12 versus -414.43 for `cheetah`, and `cheetah_baseline_na` finished at 534.23 versus 355.87 for `cheetah_baseline`.",
            "3. Decreasing `-bgs` from 5 to 1 hurt both the baseline fit and the final policy. `cheetah_baseline_bgs1_na` ended at 239.27, well below `cheetah_baseline_na` at 534.23, and its baseline-loss curve stayed higher and noisier for longer.",
            "4. Decreasing `-blr` from 0.01 to 0.005 still worked well here. `cheetah_baseline_blr005_na` finished at 425.26 and peaked at 490.73. Its baseline-loss curve was generally tighter than the `-bgs 1` run, but still a bit worse than the best `cheetah_baseline_na` run.",
            "5. Exact command lines are listed on the next page, including the extra `-na` comparisons and the changed `-bgs` / `-blr` values.",
        ]
        y = 0.34
        for answer in exp2_answers:
            add_text(fig, 0.08, y, answer, size=10)
            y -= 0.095
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Experiment 2: Exact Commands Used", fontsize=18, y=0.98)
        y = 0.92
        for i, name in enumerate(exp2_names, start=1):
            add_text(fig, 0.08, y, f"{i}. {exact_command(exp2[name])}", size=10)
            y -= 0.12
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Experiment 2: Eval Return Plot", fontsize=18, y=0.98)
        ax = fig.add_axes([0.08, 0.42, 0.84, 0.45])
        ax.imshow(plt.imread(exp2_eval_plot))
        ax.axis("off")
        add_text(fig, 0.08, 0.33, "This plot compares the original required runs (`cheetah`, `cheetah_baseline`) with the extra `-na` runs and the decreased-`bgs` / decreased-`blr` comparisons.")
        add_text(fig, 0.08, 0.24, "The main result is that `-na` helped a lot in HalfCheetah: `cheetah_na` dramatically outperformed `cheetah`, and `cheetah_baseline_na` was the strongest overall configuration, finishing above 500 average eval return.")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Experiment 2: Baseline Loss Plot", fontsize=18, y=0.98)
        ax = fig.add_axes([0.08, 0.42, 0.84, 0.45])
        ax.imshow(plt.imread(exp2_baseline_plot))
        ax.axis("off")
        add_text(fig, 0.08, 0.33, "This plot shows the baseline loss for the baseline-enabled HalfCheetah runs. The `-bgs 1` run keeps a noticeably worse and noisier baseline fit than the full `-bgs 5` setting.")
        add_text(fig, 0.08, 0.24, "Comparing the `-na` variants, `cheetah_baseline_na` achieved the best final policy performance, while `cheetah_baseline_blr005_na` was a strong second. Both beat the weaker `-bgs 1` setting on policy return.")
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
