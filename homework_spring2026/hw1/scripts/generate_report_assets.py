from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

sys.path.append("src")

from hw1_imitation.data import Normalizer, download_pusht, load_pusht_zarr
from hw1_imitation.evaluation import ENV_ID

import gym_pusht  # noqa: F401
import gymnasium as gym


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "exp"
ASSET_DIR = ROOT / "report_assets"

FLOW_RUN = EXP_DIR / "seed_42_20260524_141225_flow"
MSE_RUN = EXP_DIR / "seed_42_20260518_164019_mse_run1"
DATA_DIR = ROOT / "data"


@dataclass
class EvalPoint:
    step: int
    reward: float


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def load_csv_series(csv_path: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["train/loss"]:
                continue
            steps.append(int(row["step"]))
            losses.append(float(row["train/loss"]))
    return steps, losses


def nice_ticks(min_val: float, max_val: float, n: int = 5) -> list[float]:
    if math.isclose(min_val, max_val):
        return [min_val]
    return np.linspace(min_val, max_val, n).tolist()


def draw_line_plot(
    x: list[int],
    y: list[float],
    *,
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
    line_color: tuple[int, int, int] = (44, 95, 45),
) -> None:
    width, height = 1400, 900
    margin_left, margin_right = 140, 50
    margin_top, margin_bottom = 100, 120
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(34)
    label_font = load_font(24)
    tick_font = load_font(20)

    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    y_pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    y_min -= y_pad
    y_max += y_pad

    def map_x(v: float) -> float:
        return margin_left + (v - x_min) / (x_max - x_min) * plot_w

    def map_y(v: float) -> float:
        return margin_top + (1.0 - (v - y_min) / (y_max - y_min)) * plot_h

    grid_color = (225, 228, 232)
    axis_color = (60, 60, 60)

    for tick in nice_ticks(y_min, y_max):
        py = map_y(tick)
        draw.line((margin_left, py, margin_left + plot_w, py), fill=grid_color, width=1)
        draw.text((20, py - 10), f"{tick:.3f}", fill=axis_color, font=tick_font)

    x_ticks = nice_ticks(x_min, x_max)
    for tick in x_ticks:
        px = map_x(tick)
        draw.line((px, margin_top, px, margin_top + plot_h), fill=grid_color, width=1)
        draw.text((px - 28, margin_top + plot_h + 18), f"{int(tick)}", fill=axis_color, font=tick_font)

    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill=axis_color, width=2)
    draw.line(
        (margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h),
        fill=axis_color,
        width=2,
    )

    points = [(map_x(xv), map_y(yv)) for xv, yv in zip(x, y, strict=True)]
    draw.line(points, fill=line_color, width=4)

    for point in points[:: max(1, len(points) // 25)]:
        px, py = point
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=line_color)

    draw.text((margin_left, 24), title, fill=(20, 20, 20), font=title_font)
    draw.text((width // 2 - 80, height - 58), x_label, fill=axis_color, font=label_font)
    draw.text((20, 20), y_label, fill=axis_color, font=label_font)
    image.save(out_path)


def load_normalizer() -> Normalizer:
    zarr_path = download_pusht(DATA_DIR)
    states, actions, _ = load_pusht_zarr(zarr_path)
    return Normalizer.from_data(states, actions)


def evaluate_checkpoint(
    checkpoint_path: Path,
    normalizer: Normalizer,
    *,
    num_steps: int = 10,
    num_episodes: int = 100,
) -> float:
    device = torch.device("cpu")
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.eval()

    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    action_low = env.action_space.low
    action_high = env.action_space.high

    rewards: list[float] = []
    for ep_idx in range(num_episodes):
        obs, _ = env.reset(seed=ep_idx)
        done = False
        chunk_index = model.chunk_size
        action_chunk = None
        max_reward = 0.0
        while not done:
            if action_chunk is None or chunk_index >= model.chunk_size:
                state = torch.from_numpy(normalizer.normalize_state(obs)).float().unsqueeze(0)
                with torch.no_grad():
                    pred_chunk = model.sample_actions(state, num_steps=num_steps).cpu().numpy()[0]
                action_chunk = normalizer.denormalize_action(pred_chunk)
                action_chunk = np.clip(action_chunk, action_low, action_high)
                chunk_index = 0

            action = action_chunk[chunk_index]
            obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
            max_reward = max(max_reward, float(reward))
            done = terminated or truncated
            chunk_index += 1
        rewards.append(max_reward)

    env.close()
    return float(np.mean(rewards))


def checkpoint_reward_curve() -> list[EvalPoint]:
    summary_path = FLOW_RUN / "wandb/files/wandb-summary.json"
    final_summary = json.loads(summary_path.read_text())
    final_step = int(final_summary["step"])
    final_reward = float(final_summary["eval/mean_reward"])

    checkpoints = sorted((FLOW_RUN / "wandb/files/checkpoints").glob("checkpoint_step_*.pkl"))
    normalizer = load_normalizer()
    points: list[EvalPoint] = []
    for checkpoint in checkpoints:
        step = int(checkpoint.stem.split("_")[-1])
        if step == final_step:
            points.append(EvalPoint(step=step, reward=final_reward))
            continue
        reward = evaluate_checkpoint(checkpoint, normalizer)
        points.append(EvalPoint(step=step, reward=reward))
    return points


def checkpoint_reward_curve_for_run(
    run_dir: Path,
    *,
    summary_reward: float | None = None,
    num_steps: int = 10,
) -> list[EvalPoint]:
    summary_path = run_dir / "wandb/files/wandb-summary.json"
    final_summary = json.loads(summary_path.read_text())
    final_step = int(final_summary["step"])
    final_reward = (
        float(final_summary["eval/mean_reward"])
        if summary_reward is None
        else summary_reward
    )

    checkpoints = sorted((run_dir / "wandb/files/checkpoints").glob("checkpoint_step_*.pkl"))
    normalizer = load_normalizer()
    points: list[EvalPoint] = []
    for checkpoint in checkpoints:
        step = int(checkpoint.stem.split("_")[-1])
        if step == final_step:
            points.append(EvalPoint(step=step, reward=final_reward))
            continue
        reward = evaluate_checkpoint(checkpoint, normalizer, num_steps=num_steps)
        points.append(EvalPoint(step=step, reward=reward))
    return points


def sample_video_frames(video_path: Path, num_frames: int = 6) -> list[Image.Image]:
    reader = imageio.get_reader(video_path)
    frames = [Image.fromarray(frame) for frame in reader]
    reader.close()
    if not frames:
        return []
    indices = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    return [frames[idx] for idx in indices]


def make_comparison_sheet(
    mse_video: Path,
    flow_video: Path,
    out_path: Path,
    *,
    num_frames: int = 6,
) -> None:
    mse_frames = sample_video_frames(mse_video, num_frames=num_frames)
    flow_frames = sample_video_frames(flow_video, num_frames=num_frames)
    if not mse_frames or not flow_frames:
        raise RuntimeError("Could not load comparison video frames.")

    thumb_w, thumb_h = 220, 220
    pad = 18
    title_h = 42
    row_h = thumb_h + 60
    width = pad + num_frames * (thumb_w + pad)
    height = title_h + 2 * row_h + pad
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(28)
    label_font = load_font(24)

    draw.text((pad, 8), "Qualitative Comparison: MSE vs Flow Matching", fill="black", font=title_font)
    for row_idx, (label, frames) in enumerate((("MSE policy", mse_frames), ("Flow policy", flow_frames))):
        y_offset = title_h + row_idx * row_h
        draw.text((pad, y_offset + 4), label, fill="black", font=label_font)
        for col_idx, frame in enumerate(frames):
            thumb = frame.resize((thumb_w, thumb_h))
            x = pad + col_idx * (thumb_w + pad)
            y = y_offset + 28
            sheet.paste(thumb, (x, y))
            draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline=(180, 180, 180), width=1)
    sheet.save(out_path)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, width: int, font: ImageFont.ImageFont) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textlength(candidate, font=font) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def create_report_pdf(
    *,
    mse_loss_plot: Path,
    mse_reward_plot: Path,
    loss_plot: Path,
    reward_plot: Path,
    comparison_sheet: Path,
    mse_reward_points: list[EvalPoint],
    reward_points: list[EvalPoint],
    out_path: Path,
) -> None:
    title_font = load_font(40)
    header_font = load_font(28)
    body_font = load_font(24)
    page_w, page_h = 1654, 3000
    margin = 80

    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)
    draw.text((margin, 36), "HW1 Flow Matching Policy Report", fill="black", font=title_font)

    mse_loss_img = Image.open(mse_loss_plot).resize((700, 450))
    mse_reward_img = Image.open(mse_reward_plot).resize((700, 450))
    loss_img = Image.open(loss_plot).resize((700, 450))
    reward_img = Image.open(reward_plot).resize((700, 450))
    comp_img = Image.open(comparison_sheet).resize((1494, 620))
    page.paste(mse_loss_img, (margin, 120))
    page.paste(mse_reward_img, (margin + 760, 120))
    page.paste(loss_img, (margin, 620))
    page.paste(reward_img, (margin + 760, 620))
    page.paste(comp_img, (margin, 1160))

    best_point = max(reward_points, key=lambda p: p.reward)
    best_mse_point = max(mse_reward_points, key=lambda p: p.reward)
    summary = (
        f"Best MSE checkpoint reward: {best_mse_point.reward:.3f} at step {best_mse_point.step}. "
        f"Best flow checkpoint reward: {best_point.reward:.3f} at step {best_point.step}. "
        f"Final MSE reward: {mse_reward_points[-1].reward:.3f}. "
        f"Final flow reward: {reward_points[-1].reward:.3f}. "
        "Both losses decrease during training, but the flow policy achieves the stronger reward curve "
        "and finishes at a higher evaluation score."
    )
    qualitative = (
        "Qualitative comparison from the rollout videos: the flow matching policy approaches the target "
        "more directly, keeps the block better aligned during contact, and completes the pushing motion "
        "more consistently. The MSE policy often shows less stable contact geometry and more drift, which "
        "leads to weaker final placement and lower reward."
    )

    text_y = 1820
    draw.text((margin, text_y), "Summary", fill="black", font=header_font)
    text_y += 42
    for line in wrap_text(draw, summary, page_w - 2 * margin, body_font):
        draw.text((margin, text_y), line, fill="black", font=body_font)
        text_y += 30

    text_y += 20
    draw.text((margin, text_y), "Qualitative Description", fill="black", font=header_font)
    text_y += 42
    for line in wrap_text(draw, qualitative, page_w - 2 * margin, body_font):
        draw.text((margin, text_y), line, fill="black", font=body_font)
        text_y += 30

    page.save(out_path, "PDF", resolution=150.0)


def main() -> None:
    ASSET_DIR.mkdir(exist_ok=True)

    mse_loss_steps, mse_losses = load_csv_series(MSE_RUN / "log.csv")
    loss_steps, losses = load_csv_series(FLOW_RUN / "log.csv")

    mse_loss_plot = ASSET_DIR / "mse_loss_curve.png"
    mse_reward_plot = ASSET_DIR / "mse_reward_curve.png"
    loss_plot = ASSET_DIR / "flow_loss_curve.png"
    reward_plot = ASSET_DIR / "flow_reward_curve.png"
    comparison_sheet = ASSET_DIR / "mse_vs_flow_comparison.png"
    report_pdf = ASSET_DIR / "hw1_report.pdf"
    mse_reward_json = ASSET_DIR / "mse_reward_curve.json"
    reward_json = ASSET_DIR / "flow_reward_curve.json"

    if mse_reward_json.exists():
        mse_reward_points = [EvalPoint(**point) for point in json.loads(mse_reward_json.read_text())]
    else:
        mse_reward_points = checkpoint_reward_curve_for_run(
            MSE_RUN,
            summary_reward=0.5606596422284865,
            num_steps=10,
        )
    if reward_json.exists():
        reward_points = [EvalPoint(**point) for point in json.loads(reward_json.read_text())]
    else:
        reward_points = checkpoint_reward_curve()

    draw_line_plot(
        mse_loss_steps,
        mse_losses,
        title="Best MSE Policy: Training Loss",
        x_label="Training step",
        y_label="Loss",
        out_path=mse_loss_plot,
        line_color=(191, 90, 20),
    )
    draw_line_plot(
        [point.step for point in mse_reward_points],
        [point.reward for point in mse_reward_points],
        title="Best MSE Policy: Evaluation Reward",
        x_label="Training step",
        y_label="Mean reward",
        out_path=mse_reward_plot,
        line_color=(160, 81, 149),
    )
    draw_line_plot(
        loss_steps,
        losses,
        title="Best Flow Policy: Training Loss",
        x_label="Training step",
        y_label="Loss",
        out_path=loss_plot,
    )
    draw_line_plot(
        [point.step for point in reward_points],
        [point.reward for point in reward_points],
        title="Best Flow Policy: Evaluation Reward",
        x_label="Training step",
        y_label="Mean reward",
        out_path=reward_plot,
        line_color=(26, 115, 232),
    )

    make_comparison_sheet(
        MSE_RUN / "wandb/files/media/videos/eval/rollout_ep0_75600_0bd8e2f1c7169dee3166.mp4",
        FLOW_RUN / "wandb/files/media/videos/eval/rollout_ep0_75600_329e4ba2728d5a5f66f7.mp4",
        comparison_sheet,
    )
    mse_reward_json.write_text(json.dumps([point.__dict__ for point in mse_reward_points], indent=2))
    reward_json.write_text(json.dumps([point.__dict__ for point in reward_points], indent=2))
    create_report_pdf(
        mse_loss_plot=mse_loss_plot,
        mse_reward_plot=mse_reward_plot,
        loss_plot=loss_plot,
        reward_plot=reward_plot,
        comparison_sheet=comparison_sheet,
        mse_reward_points=mse_reward_points,
        reward_points=reward_points,
        out_path=report_pdf,
    )


if __name__ == "__main__":
    main()
