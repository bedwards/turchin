"""Time series and phase space visualization for cliodynamics analysis.

This module provides publication-quality visualizations for SDT model output,
including time series plots, phase space diagrams, and
model vs. observed data comparison plots.

All visualizations are built with Altair for consistency with the project's
viz stack. For 3D phase space plots, see the animations module (Plotly).

IMPORTANT: After generating any plot, visually verify it looks correct
before committing. Check that labels are readable, axes are properly scaled,
and the plot accurately represents the data.

Example:
    >>> from cliodynamics.viz.plots import plot_time_series, plot_phase_space
    >>> from cliodynamics.simulation import SimulationResult
    >>>
    >>> # Plot time series of multiple variables
    >>> chart = plot_time_series(
    ...     results,
    ...     variables=['N', 'W', 'psi'],
    ...     labels=['Population', 'Real Wages', 'Instability'],
    ...     title="Roman Empire 500 BCE - 500 CE"
    ... )
    >>> save_chart(chart, 'timeseries.png')
    >>>
    >>> # Plot phase space diagram
    >>> chart = plot_phase_space(results, x='W', y='psi', color_by='t')
    >>> save_chart(chart, 'phase_space.png')
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import altair as alt
import pandas as pd

from cliodynamics.viz.charts import (
    CHART_HEIGHT_MEDIUM,
    CHART_WIDTH,
    FONT_SIZE_AXIS_LABEL,
    FONT_SIZE_AXIS_TITLE,
    FONT_SIZE_LEGEND_LABEL,
    FONT_SIZE_LEGEND_TITLE,
    FONT_SIZE_TITLE,
    configure_chart,
    save_chart,
)

if TYPE_CHECKING:
    from cliodynamics.simulation import SimulationResult

logger = logging.getLogger(__name__)

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
    "psi": "Political Stress Index (ψ)",
    "t": "Time",
}


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


def _to_dataframe(
    results: "SimulationResult | pd.DataFrame",
) -> pd.DataFrame:
    """Convert SimulationResult or DataFrame to pandas DataFrame.

    Args:
        results: SimulationResult object or DataFrame.

    Returns:
        pandas DataFrame.
    """
    if hasattr(results, "df"):
        df = results.df
    else:
        df = results

    # Convert polars to pandas if needed
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()

    return df


def plot_time_series(
    results: "SimulationResult | pd.DataFrame",
    variables: list[str] | None = None,
    labels: list[str] | None = None,
    title: str = "SDT Model Time Series",
    time_column: str = "t",
    figsize: tuple[float, float] | None = None,
    colors: list[str] | None = None,
    subplot_layout: bool = False,
    share_x: bool = True,
    grid: bool = True,
) -> alt.Chart:
    """Create time series plot of simulation results.

    Plots one or more variables over time, either on a single axis
    or as vertically stacked subplots (faceted chart).

    Args:
        results: SimulationResult object or DataFrame with time series data.
        variables: List of variable names to plot. Defaults to all SDT variables.
        labels: Display labels for variables (same order as variables).
        title: Plot title.
        time_column: Name of time column in data.
        figsize: Figure size (width, height) in inches. Converted to pixels.
        colors: Custom color list for variables.
        subplot_layout: If True, use separate subplots (faceted) for each variable.
        share_x: Unused in Altair (faceted charts always share x-axis).
        grid: Unused in Altair (grid is controlled by theme).

    Returns:
        Altair Chart object.

    Example:
        >>> chart = plot_time_series(
        ...     results,
        ...     variables=['N', 'W', 'psi'],
        ...     labels=['Population', 'Real Wages', 'Instability'],
        ...     title="Roman Empire Dynamics"
        ... )
        >>> save_chart(chart, 'timeseries.png')
    """
    df = _to_dataframe(results)

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

    # Calculate dimensions
    if figsize is not None:
        width = int(figsize[0] * 100)  # Convert inches to approx pixels
        height = int(figsize[1] * 100)
    else:
        width = CHART_WIDTH
        height = CHART_HEIGHT_MEDIUM

    # Create label mapping for variables
    label_map = dict(zip(variables, labels))

    # Melt data for Altair
    df_melted = df[[time_column] + variables].melt(
        id_vars=[time_column], var_name="variable", value_name="value"
    )

    # Map variable names to labels
    df_melted["label"] = df_melted["variable"].map(label_map)

    # Create color scale
    color_scale = alt.Scale(domain=labels, range=colors)

    if subplot_layout:
        # Faceted chart with separate panels
        chart = (
            alt.Chart(df_melted)
            .mark_line(strokeWidth=1.5)
            .encode(
                x=alt.X(
                    f"{time_column}:Q",
                    title=_get_label(time_column),
                    scale=alt.Scale(
                        domain=[df[time_column].min(), df[time_column].max()]
                    ),
                ),
                y=alt.Y("value:Q", title="Value"),
                color=alt.Color(
                    "label:N",
                    scale=color_scale,
                    legend=None,  # Hide legend since facet labels show variable
                ),
            )
            .properties(
                width=width,
                height=height // len(variables),
            )
            .facet(
                row=alt.Row(
                    "label:N",
                    header=alt.Header(
                        labelFontSize=FONT_SIZE_AXIS_TITLE,
                        labelAngle=0,
                        labelAlign="left",
                    ),
                    sort=labels,  # Maintain order
                ),
            )
            .properties(
                title=alt.TitleParams(
                    text=title, fontSize=FONT_SIZE_TITLE, anchor="start"
                )
            )
            .configure_axis(
                labelFontSize=FONT_SIZE_AXIS_LABEL,
                titleFontSize=FONT_SIZE_AXIS_TITLE,
            )
        )
    else:
        # Single chart with all variables
        chart = (
            alt.Chart(df_melted)
            .mark_line(strokeWidth=1.5)
            .encode(
                x=alt.X(
                    f"{time_column}:Q",
                    title=_get_label(time_column),
                    scale=alt.Scale(
                        domain=[df[time_column].min(), df[time_column].max()]
                    ),
                ),
                y=alt.Y("value:Q", title="Value"),
                color=alt.Color(
                    "label:N",
                    title="Variable",
                    scale=color_scale,
                    legend=alt.Legend(
                        labelFontSize=FONT_SIZE_LEGEND_LABEL,
                        titleFontSize=FONT_SIZE_LEGEND_TITLE,
                    ),
                ),
            )
        )
        chart = configure_chart(chart, title, width=width, height=height)

    return chart


def plot_phase_space(
    results: "SimulationResult | pd.DataFrame",
    x: str,
    y: str,
    color_by: str | None = "t",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    colormap: str = "viridis",
    show_start: bool = True,
    show_end: bool = True,
    arrow_interval: int | None = None,
) -> alt.Chart:
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
        colormap: Color scheme name for trajectory color.
        show_start: If True, mark starting point with circle.
        show_end: If True, mark ending point with square.
        arrow_interval: Unused in Altair (arrows not supported natively).

    Returns:
        Altair Chart object.

    Example:
        >>> chart = plot_phase_space(results, x='W', y='psi', color_by='t')
        >>> save_chart(chart, 'phase_space.png')
    """
    df = _to_dataframe(results)

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

    # Calculate dimensions
    if figsize is not None:
        width = int(figsize[0] * 100)
        height = int(figsize[1] * 100)
    else:
        width = CHART_WIDTH
        height = CHART_HEIGHT_MEDIUM

    if title is None:
        title = f"Phase Space: {_get_label(x)} vs {_get_label(y)}"

    # Create line chart for trajectory
    if color_by is not None:
        # Color-coded trajectory using line with detail encoding
        # We need to use a trick: sort by color_by and use order
        line = (
            alt.Chart(df)
            .mark_trail(strokeWidth=2)
            .encode(
                x=alt.X(f"{x}:Q", title=_get_label(x)),
                y=alt.Y(f"{y}:Q", title=_get_label(y)),
                color=alt.Color(
                    f"{color_by}:Q",
                    title=_get_label(color_by),
                    scale=alt.Scale(scheme=colormap),
                    legend=alt.Legend(
                        labelFontSize=FONT_SIZE_LEGEND_LABEL,
                        titleFontSize=FONT_SIZE_LEGEND_TITLE,
                    ),
                ),
                order=alt.Order(f"{color_by}:Q"),
                size=alt.value(2),
            )
        )
    else:
        # Solid color trajectory
        line = (
            alt.Chart(df)
            .mark_line(strokeWidth=1.5, color=DEFAULT_COLORS[0])
            .encode(
                x=alt.X(f"{x}:Q", title=_get_label(x)),
                y=alt.Y(f"{y}:Q", title=_get_label(y)),
                order=alt.Order("index:O") if "index" in df.columns else alt.Order(),
            )
        )
        # Add index for ordering if not present
        if "index" not in df.columns:
            df = df.reset_index()
            line = (
                alt.Chart(df)
                .mark_line(strokeWidth=1.5, color=DEFAULT_COLORS[0])
                .encode(
                    x=alt.X(f"{x}:Q", title=_get_label(x)),
                    y=alt.Y(f"{y}:Q", title=_get_label(y)),
                    order=alt.Order("index:O"),
                )
            )

    layers = [line]

    # Add start point marker
    if show_start:
        start_df = df.iloc[[0]]
        start_point = (
            alt.Chart(start_df)
            .mark_point(size=100, color="green", filled=True)
            .encode(
                x=alt.X(f"{x}:Q"),
                y=alt.Y(f"{y}:Q"),
            )
        )
        layers.append(start_point)

    # Add end point marker
    if show_end:
        end_df = df.iloc[[-1]]
        end_point = (
            alt.Chart(end_df)
            .mark_point(size=100, color="red", shape="square", filled=True)
            .encode(
                x=alt.X(f"{x}:Q"),
                y=alt.Y(f"{y}:Q"),
            )
        )
        layers.append(end_point)

    chart = alt.layer(*layers)
    chart = configure_chart(chart, title, width=width, height=height)

    return chart


def plot_comparison(
    model_results: "SimulationResult | pd.DataFrame",
    observed_data: pd.DataFrame,
    variables: list[str],
    time_column: str = "t",
    labels: list[str] | None = None,
    title: str = "Model vs Observed Data",
    figsize: tuple[float, float] | None = None,
    confidence_bands: bool = False,
    confidence_columns: dict[str, tuple[str, str]] | None = None,
    observed_marker: str = "o",
    model_style: str = "-",
) -> alt.Chart:
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
        observed_marker: Unused in Altair (fixed marker style).
        model_style: Unused in Altair (fixed line style).

    Returns:
        Altair Chart object.

    Example:
        >>> chart = plot_comparison(
        ...     model_results=simulation,
        ...     observed_data=historical,
        ...     variables=['N', 'psi'],
        ...     confidence_bands=True
        ... )
        >>> save_chart(chart, 'comparison.png')
    """
    model_df = _to_dataframe(model_results)
    observed_df = _to_dataframe(observed_data)

    # Default labels
    if labels is None:
        labels = [_get_label(v) for v in variables]

    # Validate confidence columns if needed
    if confidence_bands and confidence_columns is None:
        raise ValueError("confidence_columns required when confidence_bands is True")

    # Calculate dimensions
    if figsize is not None:
        width = int(figsize[0] * 100)
        base_height = int(figsize[1] * 100)
    else:
        width = CHART_WIDTH
        base_height = CHART_HEIGHT_MEDIUM

    # Height per variable panel
    panel_height = max(200, base_height // len(variables))

    charts_list = []

    for i, (var, label) in enumerate(zip(variables, labels)):
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]

        # Validate variable exists in model data
        if var not in model_df.columns:
            raise ValueError(f"Variable '{var}' not found in model data")

        layers = []

        # Model line
        model_line = (
            alt.Chart(model_df)
            .mark_line(strokeWidth=1.5, color=color)
            .encode(
                x=alt.X(
                    f"{time_column}:Q",
                    title=_get_label(time_column) if i == len(variables) - 1 else "",
                ),
                y=alt.Y(f"{var}:Q", title=label),
            )
        )
        layers.append(model_line)

        # Confidence bands
        if confidence_bands and var in confidence_columns:
            lower_col, upper_col = confidence_columns[var]
            band = (
                alt.Chart(model_df)
                .mark_area(opacity=0.2, color=color)
                .encode(
                    x=alt.X(f"{time_column}:Q"),
                    y=alt.Y(f"{lower_col}:Q"),
                    y2=alt.Y2(f"{upper_col}:Q"),
                )
            )
            layers.insert(0, band)  # Put band behind line

        # Observed data points
        if var in observed_df.columns:
            # Filter NaN values
            obs_valid = observed_df[[time_column, var]].dropna()
            if len(obs_valid) > 0:
                obs_points = (
                    alt.Chart(obs_valid)
                    .mark_point(
                        size=50,
                        color=color,
                        filled=False,
                        strokeWidth=1.5,
                    )
                    .encode(
                        x=alt.X(f"{time_column}:Q"),
                        y=alt.Y(f"{var}:Q"),
                    )
                )
                layers.append(obs_points)

        panel = alt.layer(*layers).properties(
            width=width,
            height=panel_height,
        )
        charts_list.append(panel)

    # Stack vertically
    chart = (
        alt.vconcat(*charts_list)
        .properties(
            title=alt.TitleParams(text=title, fontSize=FONT_SIZE_TITLE, anchor="start")
        )
        .configure_axis(
            labelFontSize=FONT_SIZE_AXIS_LABEL,
            titleFontSize=FONT_SIZE_AXIS_TITLE,
        )
        .configure_legend(
            labelFontSize=FONT_SIZE_LEGEND_LABEL,
            titleFontSize=FONT_SIZE_LEGEND_TITLE,
        )
    )

    return chart


# Re-export save_chart from charts module for convenience
__all__ = [
    "plot_time_series",
    "plot_phase_space",
    "plot_comparison",
    "save_chart",
    "VARIABLE_LABELS",
    "DEFAULT_COLORS",
]

# Backward compatibility for animations module (still uses matplotlib)
# These are exported but not used by the Altair-based functions in this module
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
