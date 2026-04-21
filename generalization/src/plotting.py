from __future__ import annotations

"""Shared PDF plotting utilities for delivery-grid experiments."""

from dataclasses import dataclass
import os
from typing import Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

from src.instance import GridDeliveryInstance, Position


@dataclass(frozen=True)
class RenderStyle:
    """Centralized plotting style so all figures share one visual language."""

    background: str = "#f6f4ee"
    free_cell: str = "#fbfaf6"
    obstacle: str = "#374151"
    grid_line: str = "#d6d3d1"
    border: str = "#cbd5e1"
    text_primary: str = "#111827"
    text_muted: str = "#6b7280"
    title_bar: str = "#1f2937"
    title_text: str = "#fffaf0"
    path_outer: str = "#ffffff"
    path_inner: str = "#ef4444"
    path_marker: str = "#fef2f2"
    panel_bg: str = "#fffdf8"
    panel_edge: str = "#d6d3d1"
    start_fill: str = "#2563eb"
    goal_fill: str = "#f59e0b"
    start_goal_edge: str = "#ffffff"
    start_shadow: str = "#93c5fd"
    goal_shadow: str = "#fde68a"
    start_inner: str = "#1d4ed8"
    goal_inner: str = "#fbbf24"
    task_colors: Sequence[tuple[str, str, str, str]] = (
        ("#dbeafe", "#1d4ed8", "#eff6ff", "#1e3a8a"),
        ("#dcfce7", "#15803d", "#f0fdf4", "#14532d"),
        ("#fce7f3", "#be185d", "#fdf2f8", "#831843"),
        ("#ede9fe", "#6d28d9", "#f5f3ff", "#4c1d95"),
        ("#ffedd5", "#c2410c", "#fff7ed", "#7c2d12"),
    )


STYLE = RenderStyle()


def save_figure_pdf(fig, output_path: str) -> Dict[str, str]:
    """Save figure as PDF only, regardless of the provided suffix."""

    root, _ = os.path.splitext(output_path)
    pdf_path = f"{root}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"pdf": pdf_path}


def _task_palette(task_idx: int) -> tuple[str, str, str, str]:
    return STYLE.task_colors[(task_idx - 1) % len(STYLE.task_colors)]


def _style_axes(ax, rows: int, cols: int) -> None:
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_facecolor(STYLE.free_cell)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color=STYLE.grid_line, linestyle="-", linewidth=1.15)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", which="major", labelsize=9, colors=STYLE.text_muted)
    ax.set_xlabel("Column", fontsize=10, color=STYLE.text_muted, labelpad=10)
    ax.set_ylabel("Row", fontsize=10, color=STYLE.text_muted, labelpad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_title_banner(fig, title: str, subtitle: str | None, *, content_right: float = 0.90) -> None:
    banner_x = 0.04
    banner_width = max(0.40, content_right - 0.06)
    banner_center_x = banner_x + banner_width / 2.0
    banner = FancyBboxPatch(
        (banner_x, 0.915),
        banner_width,
        0.075,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        transform=fig.transFigure,
        linewidth=0,
        facecolor=STYLE.title_bar,
        clip_on=False,
        zorder=15,
    )
    fig.add_artist(banner)
    fig.text(
        banner_center_x,
        0.967,
        title,
        ha="center",
        va="center",
        fontsize=13.2,
        fontweight="bold",
        color=STYLE.title_text,
        zorder=16,
    )
    if subtitle:
        fig.text(
            banner_center_x,
            0.936,
            subtitle,
            ha="center",
            va="center",
            fontsize=8.9,
            color="#d1d5db",
            zorder=16,
        )


def _draw_cells(ax, instance: GridDeliveryInstance) -> None:
    for row, col in instance.obstacles:
        obstacle_cell = Rectangle(
            (col - 0.5, row - 0.5),
            1.0,
            1.0,
            facecolor=STYLE.obstacle,
            edgecolor=STYLE.obstacle,
            linewidth=0.0,
            zorder=1,
        )
        ax.add_patch(obstacle_cell)


def _draw_tasks(ax, instance: GridDeliveryInstance) -> None:
    for idx, task in enumerate(instance.delivery_tasks, start=1):
        pickup_fill, pickup_text, drop_fill, drop_text = _task_palette(idx)
        pr, pc = task.pickup
        dr, dc = task.dropoff

        pickup_box = FancyBboxPatch(
            (pc - 0.44, pr - 0.44),
            0.88,
            0.88,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            linewidth=1.5,
            edgecolor=pickup_text,
            facecolor=pickup_fill,
            zorder=3,
        )
        drop_box = FancyBboxPatch(
            (dc - 0.44, dr - 0.44),
            0.88,
            0.88,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            linewidth=1.5,
            edgecolor=drop_text,
            facecolor=drop_fill,
            zorder=3,
        )
        ax.add_patch(pickup_box)
        ax.add_patch(drop_box)
        ax.text(
            pc,
            pr,
            f"P{idx}",
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color=pickup_text,
            zorder=9,
        )
        ax.text(
            dc,
            dr,
            f"D{idx}",
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color=drop_text,
            zorder=9,
        )


def _draw_path(ax, path: Sequence[Position], show_direction_arrows: bool) -> None:
    if not path:
        return

    xs = [col for _, col in path]
    ys = [row for row, _ in path]
    ax.plot(xs, ys, color=STYLE.path_outer, linewidth=5.2, alpha=0.95, solid_capstyle="round", zorder=5)
    ax.plot(
        xs,
        ys,
        color=STYLE.path_inner,
        linewidth=2.8,
        marker="o",
        markersize=4.8,
        markerfacecolor=STYLE.path_marker,
        markeredgecolor=STYLE.path_inner,
        markeredgewidth=1.0,
        solid_capstyle="round",
        zorder=6,
    )

    if show_direction_arrows:
        for current, nxt in zip(path[:-1], path[1:]):
            row0, col0 = current
            row1, col1 = nxt
            start_x = col0 + 0.34 * (col1 - col0)
            start_y = row0 + 0.34 * (row1 - row0)
            end_x = col0 + 0.66 * (col1 - col0)
            end_y = row0 + 0.66 * (row1 - row0)
            arrow = FancyArrowPatch(
                (start_x, start_y),
                (end_x, end_y),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.4,
                color="#7f1d1d",
                shrinkA=0,
                shrinkB=0,
                zorder=8,
            )
            ax.add_patch(arrow)


def _draw_route_stats_cards(
    ax,
    shortest_path_steps: int | None,
    agent_steps: int | None,
) -> None:
    if shortest_path_steps is None and agent_steps is None:
        return

    cards = [
        ("Shortest Path", shortest_path_steps, 0.41),
        ("Agent Rollout", agent_steps, 0.23),
    ]
    for title, value, y0 in cards:
        card = FancyBboxPatch(
            (1.045, y0),
            0.26,
            0.14,
            boxstyle="round,pad=0.02,rounding_size=0.024",
            transform=ax.transAxes,
            linewidth=1.0,
            edgecolor=STYLE.panel_edge,
            facecolor=STYLE.panel_bg,
            clip_on=False,
            zorder=10,
        )
        ax.add_patch(card)
        ax.text(
            1.075,
            y0 + 0.105,
            title,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=9.8,
            fontweight="bold",
            color=STYLE.text_primary,
            zorder=11,
        )
        display_value = "N/A" if value is None else str(int(value))
        ax.text(
            1.175,
            y0 + 0.062,
            display_value,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color=STYLE.text_primary,
            zorder=11,
        )
        ax.text(
            1.175,
            y0 + 0.026,
            "steps",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.4,
            color=STYLE.text_muted,
            zorder=11,
        )


def _draw_start_goal(ax, instance: GridDeliveryInstance) -> None:
    sr, sc = instance.start
    gr, gc = instance.goal
    for x, y, shadow_color, main_color, inner_color in (
        (sc, sr, STYLE.start_shadow, STYLE.start_fill, STYLE.start_inner),
        (gc, gr, STYLE.goal_shadow, STYLE.goal_fill, STYLE.goal_inner),
    ):
        ax.scatter([x], [y], s=560, c=shadow_color, edgecolors="none", alpha=0.55, zorder=7)
        ax.scatter([x], [y], s=410, c=main_color, edgecolors="none", zorder=8)
        ax.scatter([x], [y], s=255, c=inner_color, edgecolors="none", alpha=0.82, zorder=8)
        ax.scatter([x - 0.10], [y - 0.10], s=70, c="white", edgecolors="none", alpha=0.28, zorder=9)

    ax.text(sc, sr, "S", ha="center", va="center", fontsize=13.5, fontweight="bold", color="white", zorder=10)
    ax.text(gc, gr, "G", ha="center", va="center", fontsize=13.5, fontweight="bold", color="#111827", zorder=10)


def _build_legend_handles(instance: GridDeliveryInstance, path_label: str) -> List[object]:
    handles: List[object] = [
        Patch(facecolor=STYLE.obstacle, edgecolor=STYLE.obstacle, label="Obstacle"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=STYLE.start_fill,
            markeredgecolor=STYLE.start_fill,
            markeredgewidth=0.0,
            markersize=12,
            label="Start",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=STYLE.goal_fill,
            markeredgecolor=STYLE.goal_fill,
            markeredgewidth=0.0,
            markersize=12,
            label="Goal",
        ),
        Line2D(
            [0],
            [0],
            color=STYLE.path_inner,
            linewidth=2.8,
            marker="o",
            markerfacecolor=STYLE.path_marker,
            markeredgecolor=STYLE.path_inner,
            markersize=6,
            label=path_label,
        ),
    ]
    for idx, _ in enumerate(instance.delivery_tasks, start=1):
        pickup_fill, pickup_text, drop_fill, drop_text = _task_palette(idx)
        handles.append(Patch(facecolor=pickup_fill, edgecolor=pickup_text, label=f"Pickup {idx}"))
        handles.append(Patch(facecolor=drop_fill, edgecolor=drop_text, label=f"Dropoff {idx}"))
    return handles


def plot_delivery_map(
    instance: GridDeliveryInstance,
    path: Sequence[Position] | None,
    title: str,
    output_path: str,
    *,
    subtitle: str | None = None,
    path_label: str = "Oracle Path",
    show_direction_arrows: bool = True,
    shortest_path_steps: int | None = None,
    agent_steps: int | None = None,
) -> Dict[str, str]:
    """Render a delivery map with a styled path and compact side legend."""

    path_list = list(path or [])
    inferred_steps = max(0, len(path_list) - 1) if path_list else None
    if agent_steps is None:
        agent_steps = inferred_steps
    if shortest_path_steps is None and path_label.lower().startswith("oracle"):
        shortest_path_steps = inferred_steps

    fig_width = max(7.4, instance.cols * 1.22 + 1.4)
    fig_height = max(6.6, instance.rows * 1.08 + 1.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=STYLE.background)

    _draw_cells(ax, instance)
    _style_axes(ax, instance.rows, instance.cols)
    _draw_tasks(ax, instance)
    _draw_start_goal(ax, instance)
    _draw_path(ax, path_list, show_direction_arrows=show_direction_arrows)
    legend = ax.legend(
        handles=_build_legend_handles(instance, path_label),
        loc="upper left",
        bbox_to_anchor=(1.01, 0.98),
        frameon=True,
        fontsize=8.6,
        borderpad=0.65,
        labelspacing=0.55,
    )
    legend.get_frame().set_facecolor("#fffdf8")
    legend.get_frame().set_edgecolor(STYLE.panel_edge)
    legend.get_frame().set_linewidth(1.0)
    _draw_route_stats_cards(ax, shortest_path_steps=shortest_path_steps, agent_steps=agent_steps)

    fig.tight_layout(rect=[0.015, 0.02, 0.90, 0.89])
    _draw_title_banner(fig, title, subtitle, content_right=0.90)
    return save_figure_pdf(fig, output_path)


def plot_training_curves(
    episodes: Sequence[Mapping[str, object]],
    title: str,
    output_path: str,
    *,
    subtitle: str | None = None,
) -> Dict[str, str]:
    """Plot tabular-training curves as a PDF."""

    x = np.array([int(item["episode_index"]) for item in episodes], dtype=np.int32)
    returns = np.array([float(item["return_value"]) for item in episodes], dtype=np.float64)
    steps = np.array([float(item["steps"]) for item in episodes], dtype=np.float64)
    success = np.array([1.0 if bool(item["success"]) else 0.0 for item in episodes], dtype=np.float64)

    def moving_average(values: np.ndarray, window: int = 20) -> np.ndarray:
        if len(values) == 0:
            return values
        kernel = np.ones(window, dtype=np.float64) / float(window)
        padded = np.pad(values, (window - 1, 0), mode="edge")
        return np.convolve(padded, kernel, mode="valid")[: len(values)]

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.2), facecolor=STYLE.background, sharex=True)
    panels = [
        ("Episode Return", returns, "#2563eb"),
        ("Episode Steps", steps, "#ea580c"),
        ("Success (Moving Avg)", moving_average(success, window=20), "#059669"),
    ]
    for ax, (label, values, color) in zip(axes, panels):
        ax.set_facecolor(STYLE.free_cell)
        ax.plot(x, values, color=color, linewidth=2.0)
        ax.set_ylabel(label, color=STYLE.text_primary, fontsize=10)
        ax.grid(color=STYLE.grid_line, linewidth=0.9, alpha=0.9)
        ax.tick_params(axis="both", colors=STYLE.text_muted, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].set_xlabel("Episode", color=STYLE.text_primary, fontsize=10)
    fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.89])
    _draw_title_banner(fig, title, subtitle, content_right=0.98)
    return save_figure_pdf(fig, output_path)


def plot_evaluation_summary(
    rows: Sequence[Mapping[str, object]],
    title: str,
    output_path: str,
    *,
    subtitle: str | None = None,
) -> Dict[str, str]:
    """Plot evaluation metrics for multiple agents/splits."""

    labels = [f"{row['encoder_name']}:{row['split_name']}" for row in rows]
    loose_success_rate = [float(row["loose_success_rate"]) for row in rows]
    optimal_success_rate = [float(row["optimal_success_rate"]) for row in rows]
    mean_return = [float(row["mean_return"]) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.4), facecolor=STYLE.background)
    for ax in axes:
        ax.set_facecolor(STYLE.free_cell)
        ax.grid(axis="y", color=STYLE.grid_line, linewidth=0.9, alpha=0.9)
        ax.tick_params(axis="both", colors=STYLE.text_muted, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)

    x = np.arange(len(labels))
    axes[0].bar(x, loose_success_rate, color="#2563eb", width=0.62)
    axes[0].set_title("Loose Success Rate", fontsize=11, color=STYLE.text_primary)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xticks(x, labels, rotation=18, ha="right")

    axes[1].bar(x, optimal_success_rate, color="#059669", width=0.62)
    axes[1].set_title("Shortest-Path Success Rate", fontsize=11, color=STYLE.text_primary)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(x, labels, rotation=18, ha="right")

    axes[2].bar(x, mean_return, color="#f59e0b", width=0.62)
    axes[2].set_title("Mean Return", fontsize=11, color=STYLE.text_primary)
    axes[2].set_xticks(x, labels, rotation=18, ha="right")

    fig.tight_layout(rect=[0.03, 0.02, 0.98, 0.89])
    _draw_title_banner(fig, title, subtitle, content_right=0.98)
    return save_figure_pdf(fig, output_path)
