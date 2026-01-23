"""Time series and phase space visualization for cliodynamics analysis.

This module provides publication-quality visualizations for SDT model output,
including time series plots, phase space diagrams (2D and 3D), and
model vs. observed data comparison plots.

IMPORTANT: After generating any plot, visually verify it looks correct
before committing. Check that labels are readable, axes are properly scaled,
and the plot accurately represents the data.

Example:
    >>> from cliodynamics.viz.plots import plot_time_series, plot_phase_space
    >>> from cliodynamics.simulation import SimulationResult
    >>>
    >>> # Plot time series of multiple variables
    >>> fig = plot_time_series(
    ...     results,
    ...     variables=['N', 'W', 'psi'],
    ...     labels=['Population', 'Real Wages', 'Instability'],
    ...     title="Roman Empire 500 BCE - 500 CE"
    ... )
    >>> fig.savefig('timeseries.png', dpi=150)
    >>>
    >>> # Plot phase space diagram
    >>> fig = plot_phase_space(results, x='W', y='psi', color_by='t')
    >>> fig.savefig('phase_space.png', dpi=150)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection

if TYPE_CHECKING:
    from pandas import DataFrame

    from cliodynamics.simulation import SimulationResult

logger = logging.getLogger(__name__)

# Standard style settings for publication quality
STYLE_CONFIG = {
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}

# Default color palette (colorblind-friendly)
DEFAULT_COLORS = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#009E73",  # Green
    "#CC79A7",  # Pink
    "#F0E442",  # Yellow
    "#56B4E9",  # Light blue
    "#E69F00",  # Orange
]

# Variable display names
VARIABLE_LABELS = {
    "N": "Population (N)",
    "E": "Elite Population (E)",
    "W": "Real Wages (W)",
    "S": "State Fiscal Health (S)",
    "psi": "Political Stress Index (\u03c8)",
    "t": "Time",
}


def _apply_style(fig: Figure) -> None:
    """Apply consistent styling to a figure."""
    with plt.rc_context(STYLE_CONFIG):
        fig.tight_layout()


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


def plot_time_series(
    results: SimulationResult | DataFrame,
    variables: list[str] | None = None,
    labels: list[str] | None = None,
    title: str = "SDT Model Time Series",
    time_column: str = "t",
    figsize: tuple[float, float] = (10, 6),
    colors: list[str] | None = None,
    subplot_layout: bool = False,
    share_x: bool = True,
    grid: bool = True,
) -> Figure:
    """Create time series plot of simulation results.

    Plots one or more variables over time, either on a single axis
    or as vertically stacked subplots.

    Args:
        results: SimulationResult object or DataFrame with time series data.
        variables: List of variable names to plot. Defaults to all SDT variables.
        labels: Display labels for variables (same order as variables).
        title: Plot title.
        time_column: Name of time column in data.
        figsize: Figure size (width, height) in inches.
        colors: Custom color list for variables.
        subplot_layout: If True, use separate subplots for each variable.
        share_x: If True and subplot_layout is True, share x-axis across subplots.
        grid: If True, show grid lines.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> fig = plot_time_series(
        ...     results,
        ...     variables=['N', 'W', 'psi'],
        ...     labels=['Population', 'Real Wages', 'Instability'],
        ...     title="Roman Empire Dynamics"
        ... )
        >>> fig.savefig('timeseries.png')
    """
    # Get DataFrame from results
    if hasattr(results, "df"):
        df = results.df
    else:
        df = results

    # Default variables
    if variables is None:
        variables = ["N", "E", "W", "S", "psi"]
        variables = [v for v in variables if v in df.columns]

    # Validate variables exist
    for var in variables:
        if var not in df.columns:
            raise ValueError(
                f"Variable '{var}' not found in data columns: {df.columns.tolist()}"
            )

    # Default labels
    if labels is None:
        labels = [_get_label(v) for v in variables]
    elif len(labels) != len(variables):
        raise ValueError(
            f"Length of labels ({len(labels)}) must match variables ({len(variables)})"
        )

    # Colors
    if colors is None:
        colors = DEFAULT_COLORS[: len(variables)]
    elif len(colors) < len(variables):
        colors = colors + DEFAULT_COLORS[len(colors) : len(variables)]

    # Get time values
    t = df[time_column].values

    with plt.rc_context(STYLE_CONFIG):
        if subplot_layout:
            # Create stacked subplots
            fig, axes = plt.subplots(
                len(variables),
                1,
                figsize=(figsize[0], figsize[1] * len(variables) / 2),
                sharex=share_x,
            )
            if len(variables) == 1:
                axes = [axes]

            for i, (var, label, color) in enumerate(zip(variables, labels, colors)):
                ax = axes[i]
                ax.plot(t, df[var].values, color=color, linewidth=1.5)
                ax.set_ylabel(label)
                if grid:
                    ax.grid(True, alpha=0.3, linestyle="--")
                ax.set_xlim(t[0], t[-1])

            axes[-1].set_xlabel(_get_label(time_column))
            axes[0].set_title(title, fontsize=14)
        else:
            # Single plot with all variables
            fig, ax = plt.subplots(figsize=figsize)

            for var, label, color in zip(variables, labels, colors):
                ax.plot(t, df[var].values, color=color, label=label, linewidth=1.5)

            ax.set_xlabel(_get_label(time_column))
            ax.set_ylabel("Value")
            ax.set_title(title, fontsize=14)
            ax.legend(loc="best")
            ax.set_xlim(t[0], t[-1])
            if grid:
                ax.grid(True, alpha=0.3, linestyle="--")

        fig.tight_layout()

    return fig


def plot_phase_space(
    results: SimulationResult | DataFrame,
    x: str,
    y: str,
    color_by: str | None = "t",
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    colormap: str = "viridis",
    show_start: bool = True,
    show_end: bool = True,
    arrow_interval: int | None = None,
) -> Figure:
    """Create 2D phase space diagram.

    Plots trajectory of two variables against each other to show
    dynamical behavior and attractors.

    Args:
        results: SimulationResult object or DataFrame with time series data.
        x: Variable name for x-axis.
        y: Variable name for y-axis.
        color_by: Variable to color trajectory by (e.g., 't' for time).
            If None, uses solid color.
        title: Plot title. Defaults to auto-generated title.
        figsize: Figure size (width, height) in inches.
        colormap: Matplotlib colormap name for trajectory color.
        show_start: If True, mark starting point with circle.
        show_end: If True, mark ending point with square.
        arrow_interval: If set, add direction arrows every N points.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> fig = plot_phase_space(results, x='W', y='psi', color_by='t')
        >>> fig.savefig('phase_space.png')
    """
    # Get DataFrame from results
    if hasattr(results, "df"):
        df = results.df
    else:
        df = results

    # Validate variables
    for var in [x, y]:
        if var not in df.columns:
            raise ValueError(
                f"Variable '{var}' not in data columns: {df.columns.tolist()}"
            )

    if color_by is not None and color_by not in df.columns:
        raise ValueError(
            f"Color variable '{color_by}' not in columns: {df.columns.tolist()}"
        )

    x_vals = df[x].values
    y_vals = df[y].values

    with plt.rc_context(STYLE_CONFIG):
        fig, ax = plt.subplots(figsize=figsize)

        if color_by is not None:
            # Color-coded trajectory
            c_vals = df[color_by].values
            points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            from matplotlib.collections import LineCollection

            norm = plt.Normalize(c_vals.min(), c_vals.max())
            lc = LineCollection(segments, cmap=colormap, norm=norm)
            lc.set_array(c_vals[:-1])
            lc.set_linewidth(1.5)
            line = ax.add_collection(lc)

            fig.colorbar(line, ax=ax, label=_get_label(color_by))
        else:
            # Solid color trajectory
            ax.plot(x_vals, y_vals, color=DEFAULT_COLORS[0], linewidth=1.5)

        # Mark start and end points
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
        if show_end:
            ax.scatter(
                [x_vals[-1]],
                [y_vals[-1]],
                color="red",
                s=100,
                marker="s",
                zorder=5,
                label="End",
            )

        # Add direction arrows
        if arrow_interval is not None and arrow_interval > 0:
            for i in range(arrow_interval, len(x_vals) - 1, arrow_interval):
                dx = x_vals[i + 1] - x_vals[i]
                dy = y_vals[i + 1] - y_vals[i]
                ax.annotate(
                    "",
                    xy=(x_vals[i] + dx * 0.5, y_vals[i] + dy * 0.5),
                    xytext=(x_vals[i], y_vals[i]),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1),
                )

        ax.set_xlabel(_get_label(x))
        ax.set_ylabel(_get_label(y))

        if title is None:
            title = f"Phase Space: {_get_label(x)} vs {_get_label(y)}"
        ax.set_title(title, fontsize=14)

        if show_start or show_end:
            ax.legend(loc="best")

        ax.autoscale()
        ax.set_aspect("auto")
        fig.tight_layout()

    return fig


def plot_phase_space_3d(
    results: SimulationResult | DataFrame,
    x: str,
    y: str,
    z: str,
    color_by: str | None = "t",
    title: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    colormap: str = "viridis",
    elevation: float = 20,
    azimuth: float = 45,
    show_start: bool = True,
    show_end: bool = True,
) -> Figure:
    """Create 3D phase space diagram.

    Plots trajectory of three variables to visualize higher-dimensional
    dynamical behavior.

    Args:
        results: SimulationResult object or DataFrame with time series data.
        x: Variable name for x-axis.
        y: Variable name for y-axis.
        z: Variable name for z-axis.
        color_by: Variable to color trajectory by. If None, uses solid color.
        title: Plot title. Defaults to auto-generated title.
        figsize: Figure size (width, height) in inches.
        colormap: Matplotlib colormap name for trajectory color.
        elevation: Viewing elevation angle in degrees.
        azimuth: Viewing azimuth angle in degrees.
        show_start: If True, mark starting point.
        show_end: If True, mark ending point.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> fig = plot_phase_space_3d(results, x='N', y='W', z='psi')
        >>> fig.savefig('phase_space_3d.png')
    """
    # Get DataFrame from results
    if hasattr(results, "df"):
        df = results.df
    else:
        df = results

    # Validate variables
    for var in [x, y, z]:
        if var not in df.columns:
            raise ValueError(
                f"Variable '{var}' not in data columns: {df.columns.tolist()}"
            )

    if color_by is not None and color_by not in df.columns:
        raise ValueError(
            f"Color variable '{color_by}' not in columns: {df.columns.tolist()}"
        )

    x_vals = df[x].values
    y_vals = df[y].values
    z_vals = df[z].values

    with plt.rc_context(STYLE_CONFIG):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        if color_by is not None:
            # Color-coded trajectory using scatter for 3D
            c_vals = df[color_by].values
            # Plot line segments with color gradient
            cmap = plt.colormaps.get_cmap(colormap)
            for i in range(len(x_vals) - 1):
                ax.plot(
                    x_vals[i : i + 2],
                    y_vals[i : i + 2],
                    z_vals[i : i + 2],
                    color=cmap(c_vals[i] / c_vals.max()),
                    linewidth=1.5,
                )

            # Add colorbar
            from matplotlib.cm import ScalarMappable

            mappable = ScalarMappable(
                norm=plt.Normalize(c_vals.min(), c_vals.max()),
                cmap=colormap,
            )
            mappable.set_array([])
            fig.colorbar(mappable, ax=ax, shrink=0.6, label=_get_label(color_by))
        else:
            # Solid color trajectory
            ax.plot(x_vals, y_vals, z_vals, color=DEFAULT_COLORS[0], linewidth=1.5)

        # Mark start and end points
        if show_start:
            ax.scatter(
                [x_vals[0]],
                [y_vals[0]],
                [z_vals[0]],
                color="green",
                s=100,
                marker="o",
                label="Start",
            )
        if show_end:
            ax.scatter(
                [x_vals[-1]],
                [y_vals[-1]],
                [z_vals[-1]],
                color="red",
                s=100,
                marker="s",
                label="End",
            )

        ax.set_xlabel(_get_label(x))
        ax.set_ylabel(_get_label(y))
        ax.set_zlabel(_get_label(z))

        if title is None:
            title = f"3D Phase Space: {x}, {y}, {z}"
        ax.set_title(title, fontsize=14)

        ax.view_init(elev=elevation, azim=azimuth)

        if show_start or show_end:
            ax.legend(loc="best")

        fig.tight_layout()

    return fig


def plot_comparison(
    model_results: SimulationResult | DataFrame,
    observed_data: DataFrame,
    variables: list[str],
    time_column: str = "t",
    labels: list[str] | None = None,
    title: str = "Model vs Observed Data",
    figsize: tuple[float, float] = (10, 6),
    confidence_bands: bool = False,
    confidence_columns: dict[str, tuple[str, str]] | None = None,
    observed_marker: str = "o",
    model_style: str = "-",
) -> Figure:
    """Create comparison plot of model results and observed data.

    Plots model predictions alongside historical observations to assess
    model fit and calibration quality.

    Args:
        model_results: SimulationResult or DataFrame with model output.
        observed_data: DataFrame with observed/historical data.
        variables: List of variable names to compare.
        time_column: Name of time column in both datasets.
        labels: Display labels for variables.
        title: Plot title.
        figsize: Figure size (width, height) in inches.
        confidence_bands: If True, show confidence bands around model.
        confidence_columns: Dict mapping variable to (lower, upper) column names
            for confidence bands. Required if confidence_bands is True.
        observed_marker: Marker style for observed data points.
        model_style: Line style for model output.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> fig = plot_comparison(
        ...     model_results=simulation,
        ...     observed_data=historical,
        ...     variables=['N', 'psi'],
        ...     confidence_bands=True
        ... )
        >>> fig.savefig('comparison.png')
    """
    # Get DataFrame from results
    if hasattr(model_results, "df"):
        model_df = model_results.df
    else:
        model_df = model_results

    # Default labels
    if labels is None:
        labels = [_get_label(v) for v in variables]

    # Validate confidence columns if needed
    if confidence_bands and confidence_columns is None:
        raise ValueError("confidence_columns required when confidence_bands is True")

    with plt.rc_context(STYLE_CONFIG):
        fig, axes = plt.subplots(
            len(variables),
            1,
            figsize=(figsize[0], figsize[1] * len(variables) / 2),
            sharex=True,
        )
        if len(variables) == 1:
            axes = [axes]

        for i, (var, label) in enumerate(zip(variables, labels)):
            ax = axes[i]
            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]

            # Validate variable exists
            if var not in model_df.columns:
                raise ValueError(f"Variable '{var}' not found in model data")

            # Plot model output
            model_t = model_df[time_column].values
            model_y = model_df[var].values
            ax.plot(
                model_t,
                model_y,
                model_style,
                color=color,
                label="Model",
                linewidth=1.5,
            )

            # Plot confidence bands if requested
            if confidence_bands and var in confidence_columns:
                lower_col, upper_col = confidence_columns[var]
                lower = model_df[lower_col].values
                upper = model_df[upper_col].values
                ax.fill_between(
                    model_t, lower, upper, color=color, alpha=0.2, label="95% CI"
                )

            # Plot observed data if variable exists
            if var in observed_data.columns:
                obs_t = observed_data[time_column].values
                obs_y = observed_data[var].values
                # Filter out NaN values
                mask = ~np.isnan(obs_y)
                ax.scatter(
                    obs_t[mask],
                    obs_y[mask],
                    marker=observed_marker,
                    color=color,
                    facecolors="none",
                    s=50,
                    label="Observed",
                    linewidths=1.5,
                )

            ax.set_ylabel(label)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3, linestyle="--")

        axes[-1].set_xlabel(_get_label(time_column))
        axes[0].set_title(title, fontsize=14)
        fig.tight_layout()

    return fig


def save_figure(
    fig: Figure,
    path: str | Path,
    dpi: int = 150,
    formats: list[str] | None = None,
) -> list[Path]:
    """Save figure to file(s) in specified format(s).

    Args:
        fig: Matplotlib Figure object to save.
        path: Output file path. Extension determines format if formats not specified.
        dpi: Resolution for raster formats (PNG).
        formats: List of formats to save. If None, uses path extension.
            Supported: 'png', 'svg', 'pdf'.

    Returns:
        List of saved file paths.

    Example:
        >>> save_figure(fig, 'output.png')
        [PosixPath('output.png')]
        >>> save_figure(fig, 'output', formats=['png', 'svg', 'pdf'])
        [PosixPath('output.png'), PosixPath('output.svg'), PosixPath('output.pdf')]
    """
    path = Path(path)

    if formats is None:
        formats = [path.suffix.lstrip(".") if path.suffix else "png"]

    saved_paths = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        if fmt not in ("png", "svg", "pdf"):
            raise ValueError(f"Unsupported format: {fmt}. Use png, svg, or pdf.")

        output_path = path.with_suffix(f".{fmt}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "png":
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(output_path, bbox_inches="tight")

        logger.info(f"Figure saved to {output_path}")
        saved_paths.append(output_path)

    # Verification reminder
    print(VERIFICATION_REMINDER)
    return saved_paths


# Verification reminder
VERIFICATION_REMINDER = """
=====================================================================
  IMPORTANT: Visually verify all plots before committing!

  Check that:
  1. All text is readable (not too small)
  2. Axes labels are correct and not cut off
  3. Legend is visible and correctly positioned
  4. Data is plotted correctly without artifacts
  5. Colors are distinguishable
=====================================================================
"""


__all__ = [
    "plot_time_series",
    "plot_phase_space",
    "plot_phase_space_3d",
    "plot_comparison",
    "save_figure",
    "VARIABLE_LABELS",
    "DEFAULT_COLORS",
]
