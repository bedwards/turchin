"""Animated visualizations for SDT model simulations.

This module provides animated visualizations for cliodynamics analysis,
bringing secular cycles and model dynamics to life for essays and presentations.
Animations show how societies evolve through demographic-structural pressures
over time.

IMPORTANT: After generating any animation, visually verify it looks correct
before committing. Check that:
1. Animation plays smoothly without artifacts
2. Labels and legends are readable
3. Trajectories are correctly rendered
4. Phase transitions (if shown) are accurate

Example:
    >>> from cliodynamics.viz import animations
    >>> from cliodynamics.simulation import Simulator, SimulationResult
    >>>
    >>> # Animate time series
    >>> anim = animations.animate_time_series(
    ...     results,
    ...     variables=['N', 'W', 'psi'],
    ...     title="Roman Empire Dynamics"
    ... )
    >>> animations.save_animation(anim, 'rome_timeseries.gif')
    >>>
    >>> # Animate phase space trajectory
    >>> anim = animations.animate_phase_space(results, x='W', y='psi')
    >>> animations.save_animation(anim, 'phase_trajectory.gif')
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection

from cliodynamics.viz.cycles import PHASE_COLORS, CycleDetectionResult, CyclePhase
from cliodynamics.viz.plots import (
    DEFAULT_COLORS,
    STYLE_CONFIG,
    VARIABLE_LABELS,
)

if TYPE_CHECKING:
    from pandas import DataFrame

    from cliodynamics.simulation import SimulationResult

logger = logging.getLogger(__name__)

# Verification reminder for animations
VERIFICATION_REMINDER = """
=====================================================================
  IMPORTANT: Visually verify all animations before committing!

  Check that:
  1. Animation plays smoothly without artifacts
  2. Labels and legends are readable throughout
  3. Trajectories are correctly rendered
  4. Phase transitions (if shown) are accurate
  5. Frame rate is appropriate for the content
=====================================================================
"""


def _get_label(variable: str, custom_labels: dict[str, str] | None = None) -> str:
    """Get display label for a variable.

    Args:
        variable: Variable name.
        custom_labels: Optional custom label mapping.

    Returns:
        Display label for the variable.
    """
    if custom_labels and variable in custom_labels:
        return custom_labels[variable]
    return VARIABLE_LABELS.get(variable, variable)


def _get_dataframe(results: SimulationResult | DataFrame) -> DataFrame:
    """Extract DataFrame from results.

    Args:
        results: SimulationResult object or DataFrame.

    Returns:
        DataFrame with time series data.
    """
    if hasattr(results, "df"):
        return results.df
    return results


def animate_time_series(
    results: SimulationResult | DataFrame,
    variables: list[str] | None = None,
    title: str = "SDT Model Evolution",
    fps: int = 30,
    duration_seconds: float = 10.0,
    time_column: str = "t",
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    figsize: tuple[float, float] = (10, 6),
    subplot_layout: bool = False,
    grid: bool = True,
) -> FuncAnimation:
    """Animate time series with progressive reveal.

    Creates an animation that progressively reveals time series data,
    showing how variables evolve over the simulation period.

    Args:
        results: SimulationResult object or DataFrame with time series data.
        variables: List of variable names to plot. Defaults to ['N', 'W', 'psi'].
        title: Plot title.
        fps: Frames per second for animation.
        duration_seconds: Total duration of animation in seconds.
        time_column: Name of time column in data.
        labels: Display labels for variables.
        colors: Custom color list for variables.
        figsize: Figure size (width, height) in inches.
        subplot_layout: If True, use separate subplots for each variable.
        grid: If True, show grid lines.

    Returns:
        FuncAnimation object that can be saved or displayed.

    Example:
        >>> anim = animate_time_series(
        ...     results,
        ...     variables=['N', 'W', 'psi'],
        ...     title="Roman Empire 500 BCE - 500 CE",
        ...     duration_seconds=15.0
        ... )
        >>> save_animation(anim, 'rome_evolution.gif')
    """
    df = _get_dataframe(results)

    # Default variables
    if variables is None:
        variables = ["N", "W", "psi"]
        variables = [v for v in variables if v in df.columns]

    # Validate variables
    for var in variables:
        if var not in df.columns:
            raise ValueError(
                f"Variable '{var}' not found in data columns: {df.columns.tolist()}"
            )

    # Default labels
    if labels is None:
        labels = [_get_label(v) for v in variables]

    # Colors
    if colors is None:
        colors = DEFAULT_COLORS[: len(variables)]

    # Get data
    t = df[time_column].values
    data = {var: df[var].values for var in variables}

    # Calculate frames
    n_frames = int(fps * duration_seconds)
    n_points = len(t)
    indices = np.linspace(0, n_points - 1, n_frames).astype(int)

    with plt.rc_context(STYLE_CONFIG):
        if subplot_layout:
            fig, axes = plt.subplots(
                len(variables),
                1,
                figsize=(figsize[0], figsize[1] * len(variables) / 2),
                sharex=True,
            )
            if len(variables) == 1:
                axes = [axes]

            lines = []
            for i, (var, label, color) in enumerate(zip(variables, labels, colors)):
                ax = axes[i]
                (line,) = ax.plot([], [], color=color, linewidth=1.5)
                lines.append(line)
                ax.set_ylabel(label)
                ax.set_xlim(t[0], t[-1])
                y_data = data[var]
                y_margin = 0.1 * (y_data.max() - y_data.min())
                ax.set_ylim(y_data.min() - y_margin, y_data.max() + y_margin)
                if grid:
                    ax.grid(True, alpha=0.3, linestyle="--")

            axes[-1].set_xlabel(_get_label(time_column))
            axes[0].set_title(title, fontsize=14)

            def update(frame: int) -> list:
                idx = indices[frame]
                for i, var in enumerate(variables):
                    lines[i].set_data(t[: idx + 1], data[var][: idx + 1])
                return lines

        else:
            fig, ax = plt.subplots(figsize=figsize)

            lines = []
            for var, label, color in zip(variables, labels, colors):
                (line,) = ax.plot([], [], color=color, label=label, linewidth=1.5)
                lines.append(line)

            ax.set_xlabel(_get_label(time_column))
            ax.set_ylabel("Value")
            ax.set_title(title, fontsize=14)
            ax.legend(loc="best")
            ax.set_xlim(t[0], t[-1])

            # Set y limits based on all variables
            all_values = np.concatenate([data[v] for v in variables])
            y_margin = 0.1 * (all_values.max() - all_values.min())
            ax.set_ylim(all_values.min() - y_margin, all_values.max() + y_margin)

            if grid:
                ax.grid(True, alpha=0.3, linestyle="--")

            def update(frame: int) -> list:
                idx = indices[frame]
                for i, var in enumerate(variables):
                    lines[i].set_data(t[: idx + 1], data[var][: idx + 1])
                return lines

        fig.tight_layout()

        anim = FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=1000 // fps,
            blit=True,
        )

    return anim


def animate_phase_space(
    results: SimulationResult | DataFrame,
    x: str,
    y: str,
    color_by: str = "t",
    trail_length: int = 50,
    fps: int = 30,
    duration_seconds: float = 10.0,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    colormap: str = "viridis",
    show_start: bool = True,
) -> FuncAnimation:
    """Animate phase space trajectory with trailing effect.

    Creates an animation showing the trajectory through phase space
    with a fading trail effect to visualize direction of motion.

    Args:
        results: SimulationResult object or DataFrame with time series data.
        x: Variable name for x-axis.
        y: Variable name for y-axis.
        color_by: Variable to color trajectory by (e.g., 't' for time).
        trail_length: Number of points in the trailing effect.
        fps: Frames per second for animation.
        duration_seconds: Total duration of animation in seconds.
        title: Plot title. Defaults to auto-generated title.
        figsize: Figure size (width, height) in inches.
        colormap: Matplotlib colormap name for trajectory color.
        show_start: If True, mark starting point.

    Returns:
        FuncAnimation object that can be saved or displayed.

    Example:
        >>> anim = animate_phase_space(
        ...     results,
        ...     x='W', y='psi',
        ...     trail_length=100,
        ...     duration_seconds=12.0
        ... )
        >>> save_animation(anim, 'phase_trajectory.gif')
    """
    df = _get_dataframe(results)

    # Validate variables
    for var in [x, y]:
        if var not in df.columns:
            raise ValueError(
                f"Variable '{var}' not in data columns: {df.columns.tolist()}"
            )

    if color_by not in df.columns:
        raise ValueError(
            f"Color variable '{color_by}' not in columns: {df.columns.tolist()}"
        )

    x_vals = df[x].values
    y_vals = df[y].values
    c_vals = df[color_by].values

    # Calculate frames
    n_frames = int(fps * duration_seconds)
    n_points = len(x_vals)
    indices = np.linspace(0, n_points - 1, n_frames).astype(int)

    with plt.rc_context(STYLE_CONFIG):
        fig, ax = plt.subplots(figsize=figsize)

        # Set up colormap
        cmap = plt.colormaps.get_cmap(colormap)
        norm = plt.Normalize(c_vals.min(), c_vals.max())

        # Set axis limits
        x_margin = 0.1 * (x_vals.max() - x_vals.min())
        y_margin = 0.1 * (y_vals.max() - y_vals.min())
        ax.set_xlim(x_vals.min() - x_margin, x_vals.max() + x_margin)
        ax.set_ylim(y_vals.min() - y_margin, y_vals.max() + y_margin)

        ax.set_xlabel(_get_label(x))
        ax.set_ylabel(_get_label(y))

        if title is None:
            title = f"Phase Space: {_get_label(x)} vs {_get_label(y)}"
        ax.set_title(title, fontsize=14)

        # Mark starting point
        if show_start:
            ax.scatter(
                [x_vals[0]],
                [y_vals[0]],
                color="green",
                s=100,
                marker="o",
                zorder=5,
                label="Start",
            )
            ax.legend(loc="best")

        # Create line segments for trail
        (trail_line,) = ax.plot([], [], color="gray", linewidth=0.5, alpha=0.3)
        (head_point,) = ax.plot([], [], "o", color="red", markersize=8)

        # Add colorbar
        from matplotlib.cm import ScalarMappable

        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax, label=_get_label(color_by))

        # Store trail segments
        trail_segments = []

        def init():
            trail_line.set_data([], [])
            head_point.set_data([], [])
            return [trail_line, head_point] + trail_segments

        def update(frame: int) -> list:
            nonlocal trail_segments

            idx = indices[frame]

            # Clear old trail segments
            for seg in trail_segments:
                seg.remove()
            trail_segments.clear()

            # Draw colored trail
            start_idx = max(0, idx - trail_length)
            for i in range(start_idx, idx):
                alpha = (i - start_idx) / max(1, trail_length)
                color = cmap(norm(c_vals[i]))
                (seg,) = ax.plot(
                    [x_vals[i], x_vals[i + 1]],
                    [y_vals[i], y_vals[i + 1]],
                    color=color,
                    linewidth=1.5,
                    alpha=alpha * 0.8 + 0.2,
                )
                trail_segments.append(seg)

            # Update head position
            head_point.set_data([x_vals[idx]], [y_vals[idx]])

            return [trail_line, head_point] + trail_segments

        fig.tight_layout()

        anim = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=n_frames,
            interval=1000 // fps,
            blit=False,  # Cannot use blit with dynamic artists
        )

    return anim


def animate_phase_space_3d(
    results: SimulationResult | DataFrame,
    x: str,
    y: str,
    z: str,
    rotate_view: bool = True,
    trail_length: int = 50,
    fps: int = 30,
    duration_seconds: float = 10.0,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    colormap: str = "viridis",
    elevation: float = 20,
    initial_azimuth: float = 45,
    rotation_speed: float = 1.0,
) -> FuncAnimation:
    """Animate 3D phase space with optional view rotation.

    Creates an animation showing the trajectory through 3D phase space
    with optional view rotation to reveal the full structure.

    Args:
        results: SimulationResult object or DataFrame with time series data.
        x: Variable name for x-axis.
        y: Variable name for y-axis.
        z: Variable name for z-axis.
        rotate_view: If True, rotate the view during animation.
        trail_length: Number of points in the trailing effect.
        fps: Frames per second for animation.
        duration_seconds: Total duration of animation in seconds.
        title: Plot title. Defaults to auto-generated title.
        figsize: Figure size (width, height) in inches.
        colormap: Matplotlib colormap name for trajectory color.
        elevation: Viewing elevation angle in degrees.
        initial_azimuth: Initial viewing azimuth angle in degrees.
        rotation_speed: Speed of view rotation (degrees per frame).

    Returns:
        FuncAnimation object that can be saved or displayed.

    Example:
        >>> anim = animate_phase_space_3d(
        ...     results,
        ...     x='N', y='W', z='psi',
        ...     rotate_view=True,
        ...     duration_seconds=15.0
        ... )
        >>> save_animation(anim, 'phase_3d.gif')
    """
    df = _get_dataframe(results)

    # Validate variables
    for var in [x, y, z]:
        if var not in df.columns:
            raise ValueError(
                f"Variable '{var}' not in data columns: {df.columns.tolist()}"
            )

    x_vals = df[x].values
    y_vals = df[y].values
    z_vals = df[z].values

    # Calculate frames
    n_frames = int(fps * duration_seconds)
    n_points = len(x_vals)
    indices = np.linspace(0, n_points - 1, n_frames).astype(int)

    with plt.rc_context(STYLE_CONFIG):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Set up colormap
        cmap = plt.colormaps.get_cmap(colormap)

        ax.set_xlabel(_get_label(x))
        ax.set_ylabel(_get_label(y))
        ax.set_zlabel(_get_label(z))

        if title is None:
            title = f"3D Phase Space: {x}, {y}, {z}"
        ax.set_title(title, fontsize=14)

        ax.view_init(elev=elevation, azim=initial_azimuth)

        # Mark starting point
        ax.scatter(
            [x_vals[0]],
            [y_vals[0]],
            [z_vals[0]],
            color="green",
            s=100,
            marker="o",
            label="Start",
        )

        # Create head marker
        (head_point,) = ax.plot([], [], [], "o", color="red", markersize=8)

        # Store trail segments
        trail_segments = []

        def update(frame: int) -> list:
            nonlocal trail_segments

            idx = indices[frame]

            # Clear old trail segments
            for seg in trail_segments:
                seg.remove()
            trail_segments.clear()

            # Draw colored trail
            start_idx = max(0, idx - trail_length)
            for i in range(start_idx, idx):
                alpha = (i - start_idx) / max(1, trail_length)
                color = cmap(i / n_points)
                (seg,) = ax.plot(
                    [x_vals[i], x_vals[i + 1]],
                    [y_vals[i], y_vals[i + 1]],
                    [z_vals[i], z_vals[i + 1]],
                    color=color,
                    linewidth=1.5,
                    alpha=alpha * 0.8 + 0.2,
                )
                trail_segments.append(seg)

            # Update head position
            head_point.set_data_3d([x_vals[idx]], [y_vals[idx]], [z_vals[idx]])

            # Rotate view if enabled
            if rotate_view:
                azimuth = initial_azimuth + frame * rotation_speed
                ax.view_init(elev=elevation, azim=azimuth)

            return [head_point] + trail_segments

        fig.tight_layout()

        anim = FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=1000 // fps,
            blit=False,
        )

    return anim


def animate_secular_cycles(
    results: SimulationResult | DataFrame,
    cycles: CycleDetectionResult,
    variable: str = "psi",
    show_phase_transitions: bool = True,
    fps: int = 30,
    duration_seconds: float = 10.0,
    time_column: str = "t",
    title: str = "Political Stress Index with Secular Cycles",
    figsize: tuple[float, float] = (12, 6),
) -> FuncAnimation:
    """Animate time series with cycle phase transitions.

    Creates an animation that progressively reveals the time series
    while highlighting secular cycle phases with color-coded backgrounds.

    Args:
        results: SimulationResult object or DataFrame with time series data.
        cycles: CycleDetectionResult from detect_secular_cycles.
        variable: Variable to plot (typically 'psi').
        show_phase_transitions: If True, show phase backgrounds.
        fps: Frames per second for animation.
        duration_seconds: Total duration of animation in seconds.
        time_column: Name of time column.
        title: Plot title.
        figsize: Figure size (width, height) in inches.

    Returns:
        FuncAnimation object that can be saved or displayed.

    Example:
        >>> from cliodynamics.viz.cycles import detect_secular_cycles
        >>> detected = detect_secular_cycles(results['psi'], results['t'])
        >>> anim = animate_secular_cycles(results, detected)
        >>> save_animation(anim, 'secular_cycles.gif')
    """
    df = _get_dataframe(results)

    if variable not in df.columns:
        raise ValueError(
            f"Variable '{variable}' not in data columns: {df.columns.tolist()}"
        )

    t = df[time_column].values
    y = df[variable].values

    # Calculate frames
    n_frames = int(fps * duration_seconds)
    n_points = len(t)
    indices = np.linspace(0, n_points - 1, n_frames).astype(int)

    with plt.rc_context(STYLE_CONFIG):
        fig, ax = plt.subplots(figsize=figsize)

        # Set limits
        ax.set_xlim(t[0], t[-1])
        y_margin = 0.1 * (y.max() - y.min())
        ax.set_ylim(y.min() - y_margin, y.max() + y_margin)

        ax.set_xlabel(_get_label(time_column))
        ax.set_ylabel(_get_label(variable))
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, linestyle="--", zorder=0)

        # Create main line
        (main_line,) = ax.plot([], [], color="#0072B2", linewidth=1.5, zorder=3)

        # Store phase rectangles
        phase_patches = []

        # Create legend for phases
        if show_phase_transitions:
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(
                    facecolor=PHASE_COLORS[CyclePhase.EXPANSION],
                    alpha=0.3,
                    label="Expansion",
                ),
                Patch(
                    facecolor=PHASE_COLORS[CyclePhase.STAGFLATION],
                    alpha=0.3,
                    label="Stagflation",
                ),
                Patch(
                    facecolor=PHASE_COLORS[CyclePhase.CRISIS],
                    alpha=0.3,
                    label="Crisis",
                ),
                Patch(
                    facecolor=PHASE_COLORS[CyclePhase.DEPRESSION],
                    alpha=0.3,
                    label="Depression",
                ),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

        def update(frame: int) -> list:
            nonlocal phase_patches

            idx = indices[frame]
            current_time = t[idx]

            # Update main line
            main_line.set_data(t[: idx + 1], y[: idx + 1])

            # Clear old phase patches
            for patch in phase_patches:
                patch.remove()
            phase_patches.clear()

            # Draw phase backgrounds up to current time
            if show_phase_transitions:
                from matplotlib.patches import Rectangle

                for cycle in cycles.cycles:
                    for start, end, phase in cycle.phases:
                        if start < current_time:
                            visible_end = min(end, current_time)
                            if visible_end > start:
                                rect = Rectangle(
                                    (start, y.min() - y_margin),
                                    visible_end - start,
                                    y.max() - y.min() + 2 * y_margin,
                                    facecolor=PHASE_COLORS[phase],
                                    alpha=0.3,
                                    edgecolor="none",
                                    zorder=1,
                                )
                                ax.add_patch(rect)
                                phase_patches.append(rect)

            return [main_line] + phase_patches

        fig.tight_layout()

        anim = FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=1000 // fps,
            blit=False,
        )

    return anim


def animate_comparison(
    baseline: SimulationResult | DataFrame,
    counterfactual: SimulationResult | DataFrame,
    variables: list[str],
    labels: tuple[str, str] = ("Historical", "Counterfactual"),
    layout: str = "overlay",
    fps: int = 30,
    duration_seconds: float = 10.0,
    time_column: str = "t",
    title: str = "Scenario Comparison",
    figsize: tuple[float, float] = (10, 6),
    colors: tuple[str, str] | None = None,
) -> FuncAnimation:
    """Animate comparison of two simulation scenarios.

    Creates an animation comparing baseline and counterfactual scenarios,
    useful for policy analysis and exploring alternative histories.

    Args:
        baseline: SimulationResult or DataFrame for baseline scenario.
        counterfactual: SimulationResult or DataFrame for counterfactual.
        variables: List of variable names to compare.
        labels: Display labels for (baseline, counterfactual).
        layout: 'overlay' for same axes, 'side_by_side' for separate panels.
        fps: Frames per second for animation.
        duration_seconds: Total duration of animation in seconds.
        time_column: Name of time column.
        title: Plot title.
        figsize: Figure size (width, height) in inches.
        colors: Custom colors for (baseline, counterfactual).

    Returns:
        FuncAnimation object that can be saved or displayed.

    Example:
        >>> anim = animate_comparison(
        ...     baseline=historical_results,
        ...     counterfactual=policy_results,
        ...     variables=['psi', 'S'],
        ...     labels=('Historical', 'With Reform'),
        ...     layout='side_by_side'
        ... )
        >>> save_animation(anim, 'comparison.gif')
    """
    df_base = _get_dataframe(baseline)
    df_counter = _get_dataframe(counterfactual)

    # Validate variables
    for var in variables:
        if var not in df_base.columns:
            raise ValueError(f"Variable '{var}' not in baseline data")
        if var not in df_counter.columns:
            raise ValueError(f"Variable '{var}' not in counterfactual data")

    # Default colors
    if colors is None:
        colors = (DEFAULT_COLORS[0], DEFAULT_COLORS[1])

    t_base = df_base[time_column].values
    t_counter = df_counter[time_column].values

    # Calculate frames
    n_frames = int(fps * duration_seconds)
    n_points_base = len(t_base)
    n_points_counter = len(t_counter)
    indices_base = np.linspace(0, n_points_base - 1, n_frames).astype(int)
    indices_counter = np.linspace(0, n_points_counter - 1, n_frames).astype(int)

    # Get combined time range
    t_min = min(t_base[0], t_counter[0])
    t_max = max(t_base[-1], t_counter[-1])

    with plt.rc_context(STYLE_CONFIG):
        if layout == "side_by_side":
            fig, axes = plt.subplots(
                len(variables),
                2,
                figsize=(figsize[0] * 1.5, figsize[1] * len(variables) / 2),
                sharex=True,
            )
            if len(variables) == 1:
                axes = axes.reshape(1, 2)

            lines_base = []
            lines_counter = []

            for i, var in enumerate(variables):
                ax_base = axes[i, 0]
                ax_counter = axes[i, 1]

                (line_base,) = ax_base.plot(
                    [], [], color=colors[0], linewidth=1.5, label=labels[0]
                )
                (line_counter,) = ax_counter.plot(
                    [], [], color=colors[1], linewidth=1.5, label=labels[1]
                )

                lines_base.append(line_base)
                lines_counter.append(line_counter)

                # Set limits
                all_vals = np.concatenate([df_base[var].values, df_counter[var].values])
                y_margin = 0.1 * (all_vals.max() - all_vals.min())
                y_min, y_max = all_vals.min() - y_margin, all_vals.max() + y_margin

                for ax in [ax_base, ax_counter]:
                    ax.set_xlim(t_min, t_max)
                    ax.set_ylim(y_min, y_max)
                    ax.grid(True, alpha=0.3, linestyle="--")
                    ax.legend(loc="best")

                ax_base.set_ylabel(_get_label(var))

            axes[-1, 0].set_xlabel(_get_label(time_column))
            axes[-1, 1].set_xlabel(_get_label(time_column))
            axes[0, 0].set_title(f"{title} - {labels[0]}", fontsize=12)
            axes[0, 1].set_title(f"{title} - {labels[1]}", fontsize=12)

            def update(frame: int) -> list:
                idx_base = indices_base[frame]
                idx_counter = indices_counter[frame]

                for i, var in enumerate(variables):
                    lines_base[i].set_data(
                        t_base[: idx_base + 1], df_base[var].values[: idx_base + 1]
                    )
                    lines_counter[i].set_data(
                        t_counter[: idx_counter + 1],
                        df_counter[var].values[: idx_counter + 1],
                    )

                return lines_base + lines_counter

        else:  # overlay layout
            fig, axes = plt.subplots(
                len(variables),
                1,
                figsize=(figsize[0], figsize[1] * len(variables) / 2),
                sharex=True,
            )
            if len(variables) == 1:
                axes = [axes]

            lines_base = []
            lines_counter = []

            for i, var in enumerate(variables):
                ax = axes[i]

                (line_base,) = ax.plot(
                    [], [], color=colors[0], linewidth=1.5, label=labels[0]
                )
                (line_counter,) = ax.plot(
                    [],
                    [],
                    color=colors[1],
                    linewidth=1.5,
                    linestyle="--",
                    label=labels[1],
                )

                lines_base.append(line_base)
                lines_counter.append(line_counter)

                # Set limits
                all_vals = np.concatenate([df_base[var].values, df_counter[var].values])
                y_margin = 0.1 * (all_vals.max() - all_vals.min())

                ax.set_xlim(t_min, t_max)
                ax.set_ylim(all_vals.min() - y_margin, all_vals.max() + y_margin)
                ax.set_ylabel(_get_label(var))
                ax.grid(True, alpha=0.3, linestyle="--")
                ax.legend(loc="best")

            axes[-1].set_xlabel(_get_label(time_column))
            axes[0].set_title(title, fontsize=14)

            def update(frame: int) -> list:
                idx_base = indices_base[frame]
                idx_counter = indices_counter[frame]

                for i, var in enumerate(variables):
                    lines_base[i].set_data(
                        t_base[: idx_base + 1], df_base[var].values[: idx_base + 1]
                    )
                    lines_counter[i].set_data(
                        t_counter[: idx_counter + 1],
                        df_counter[var].values[: idx_counter + 1],
                    )

                return lines_base + lines_counter

        fig.tight_layout()

        anim = FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=1000 // fps,
            blit=True,
        )

    return anim


def save_animation(
    anim: FuncAnimation,
    path: str | Path,
    format: str | None = None,
    fps: int = 30,
    dpi: int = 150,
    progress_callback: callable | None = None,
) -> Path:
    """Save animation to file.

    Supports GIF export via Pillow (no ffmpeg required) and MP4 export
    via ffmpeg when available.

    Args:
        anim: FuncAnimation object to save.
        path: Output file path.
        format: Output format ('gif' or 'mp4'). If None, inferred from path.
        fps: Frames per second for output.
        dpi: Resolution for output.
        progress_callback: Optional callback for progress updates.
            Called with (current_frame, total_frames).

    Returns:
        Path to saved animation file.

    Raises:
        ValueError: If format is not supported.
        RuntimeError: If MP4 requested but ffmpeg not available.

    Example:
        >>> save_animation(anim, 'output.gif')
        PosixPath('output.gif')
        >>> save_animation(anim, 'output.mp4', format='mp4')
        PosixPath('output.mp4')
    """
    path = Path(path)

    # Infer format from path if not specified
    if format is None:
        format = path.suffix.lstrip(".").lower()
        if not format:
            format = "gif"

    format = format.lower()

    if format not in ("gif", "mp4"):
        raise ValueError(f"Unsupported format: {format}. Use 'gif' or 'mp4'.")

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure correct extension
    if path.suffix.lower() != f".{format}":
        path = path.with_suffix(f".{format}")

    logger.info(f"Saving animation to {path} (format={format}, fps={fps}, dpi={dpi})")

    if format == "gif":
        # Use Pillow writer for GIF (no ffmpeg required)
        try:
            writer = "pillow"
            anim.save(
                str(path),
                writer=writer,
                fps=fps,
                dpi=dpi,
                progress_callback=progress_callback,
            )
        except Exception as e:
            logger.error(f"Failed to save GIF: {e}")
            raise

    elif format == "mp4":
        # Check for ffmpeg
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg not found. Install ffmpeg for MP4 export, "
                "or use GIF format instead."
            )

        try:
            from matplotlib.animation import FFMpegWriter

            writer = FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(
                str(path),
                writer=writer,
                dpi=dpi,
                progress_callback=progress_callback,
            )
        except Exception as e:
            logger.error(f"Failed to save MP4: {e}")
            raise

    logger.info(f"Animation saved to {path}")

    # Verification reminder
    print(VERIFICATION_REMINDER)

    return path


__all__ = [
    "animate_time_series",
    "animate_phase_space",
    "animate_phase_space_3d",
    "animate_secular_cycles",
    "animate_comparison",
    "save_animation",
    "VERIFICATION_REMINDER",
]
