"""Visualization tools for Monte Carlo simulation results.

This module provides specialized visualizations for probabilistic forecasting:
- Fan charts (forecast with widening confidence bands)
- Tornado plots (parameter sensitivity ranking)
- Probability heatmaps (P(crisis) over time and threshold)
- Distribution plots (histograms, density)

All visualizations use Altair for consistency with the project's viz stack.

Example:
    >>> from cliodynamics.viz.monte_carlo import (
    ...     plot_fan_chart,
    ...     plot_tornado,
    ...     plot_probability_heatmap,
    ... )
    >>> from cliodynamics.simulation import MonteCarloSimulator
    >>>
    >>> results = mc.run(initial_conditions, time_span)
    >>> chart = plot_fan_chart(results, variable='psi')
    >>> save_chart(chart, 'forecast_fan.png')

References:
    Wickham, H. (2016). ggplot2. Springer.
    Saltelli, A. et al. (2008). Global Sensitivity Analysis. Wiley.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import altair as alt
import numpy as np
import pandas as pd

from cliodynamics.viz.charts import (
    CHART_HEIGHT_MEDIUM,
    CHART_WIDTH,
    FONT_SIZE_AXIS_LABEL,
    FONT_SIZE_LEGEND_LABEL,
    FONT_SIZE_LEGEND_TITLE,
    configure_chart,
    save_chart,
)

if TYPE_CHECKING:
    from cliodynamics.analysis.sensitivity import SensitivityResults
    from cliodynamics.simulation.monte_carlo import MonteCarloResults


# Color palette for fan charts (gradient from light to dark)
FAN_COLORS = [
    "#e8f4f8",  # 5-95% band (lightest)
    "#c3e0e8",  # 10-90% band
    "#8ec8d8",  # 25-75% band
    "#4a9bb8",  # Median line
]

# Variable display names
VARIABLE_LABELS = {
    "N": "Population (N)",
    "E": "Elite Population (E)",
    "W": "Real Wages (W)",
    "S": "State Fiscal Health (S)",
    "psi": "Political Stress Index (PSI)",
    "t": "Time (years)",
}


def _get_label(variable: str, custom_labels: dict[str, str] | None = None) -> str:
    """Get display label for a variable."""
    if custom_labels and variable in custom_labels:
        return custom_labels[variable]
    return VARIABLE_LABELS.get(variable, variable)


def plot_fan_chart(
    results: "MonteCarloResults",
    variable: str = "psi",
    title: str | None = None,
    time_label: str = "Year",
    percentiles: list[float] | None = None,
    colors: list[str] | None = None,
    show_median: bool = True,
    show_mean: bool = False,
    height: int | None = None,
    width: int | None = None,
) -> alt.LayerChart:
    """Create a fan chart showing forecast with uncertainty bands.

    Fan charts show the probability distribution of future values using
    nested confidence bands that widen over time.

    Args:
        results: MonteCarloResults from simulation.
        variable: State variable to plot (default 'psi').
        title: Chart title. Defaults to variable name.
        time_label: Label for x-axis.
        percentiles: Percentile bands to show. Default [5, 10, 25, 75, 90, 95].
        colors: Colors for bands (from outer to inner).
        show_median: If True, show median line.
        show_mean: If True, show mean line.
        height: Chart height in pixels.
        width: Chart width in pixels.

    Returns:
        Altair LayerChart with nested uncertainty bands.

    Example:
        >>> chart = plot_fan_chart(results, variable='psi', title='PSI Forecast')
        >>> save_chart(chart, 'psi_fan.png')
    """
    if percentiles is None:
        percentiles = [5, 10, 25, 75, 90, 95]

    if colors is None:
        colors = FAN_COLORS

    height = height or CHART_HEIGHT_MEDIUM
    width = width or CHART_WIDTH

    # Get fan chart data
    df = results.to_fan_chart_data(variable, percentiles)

    if title is None:
        title = f"{_get_label(variable)} Forecast with Uncertainty"

    # Create band pairs (e.g., 5-95, 10-90, 25-75)
    sorted_pcts = sorted(percentiles)
    n_bands = len(sorted_pcts) // 2
    bands = [(sorted_pcts[i], sorted_pcts[-(i + 1)]) for i in range(n_bands)]

    # Build layers
    layers = []

    for i, (lower, upper) in enumerate(bands):
        # Area band
        band_df = df[["time", f"p{int(lower)}", f"p{int(upper)}"]].copy()
        band_df.columns = ["time", "lower", "upper"]

        color = colors[min(i, len(colors) - 1)]

        band = (
            alt.Chart(band_df)
            .mark_area(opacity=0.8)
            .encode(
                x=alt.X("time:Q", title=time_label),
                y=alt.Y("lower:Q", title=_get_label(variable)),
                y2=alt.Y2("upper:Q"),
                color=alt.value(color),
                tooltip=[
                    alt.Tooltip("time:Q", title="Time"),
                    alt.Tooltip("lower:Q", title=f"{lower}th percentile", format=".3f"),
                    alt.Tooltip("upper:Q", title=f"{upper}th percentile", format=".3f"),
                ],
            )
        )
        layers.append(band)

    # Add median line if requested
    if show_median and 50 in percentiles:
        median_df = df[["time", "p50"]].copy()
        median_df.columns = ["time", "median"]

        median_line = (
            alt.Chart(median_df)
            .mark_line(color=colors[-1], strokeWidth=2)
            .encode(
                x=alt.X("time:Q"),
                y=alt.Y("median:Q"),
                tooltip=[
                    alt.Tooltip("time:Q", title="Time"),
                    alt.Tooltip("median:Q", title="Median", format=".3f"),
                ],
            )
        )
        layers.append(median_line)

    # Add mean line if requested
    if show_mean:
        mean_df = results.mean()
        mean_df = mean_df[["time", variable]].copy()
        mean_df.columns = ["time", "mean"]

        mean_line = (
            alt.Chart(mean_df)
            .mark_line(color="#d62728", strokeWidth=2, strokeDash=[5, 3])
            .encode(
                x=alt.X("time:Q"),
                y=alt.Y("mean:Q"),
                tooltip=[
                    alt.Tooltip("time:Q", title="Time"),
                    alt.Tooltip("mean:Q", title="Mean", format=".3f"),
                ],
            )
        )
        layers.append(mean_line)

    # Combine layers
    chart = alt.layer(*layers).properties(
        title=title,
        width=width,
        height=height,
    )

    return configure_chart(chart)


def plot_tornado(
    sensitivity_results: "SensitivityResults",
    title: str = "Parameter Sensitivity Ranking",
    show_interactions: bool = True,
    max_params: int | None = None,
    height: int | None = None,
    width: int | None = None,
) -> alt.Chart:
    """Create a tornado plot showing parameter sensitivity ranking.

    Tornado plots display horizontal bars showing each parameter's influence
    on the output, ranked from most to least influential.

    Args:
        sensitivity_results: SensitivityResults from analysis.
        title: Chart title.
        show_interactions: If True, show interaction strength alongside S1.
        max_params: Maximum number of parameters to show.
        height: Chart height in pixels.
        width: Chart width in pixels.

    Returns:
        Altair Chart with tornado plot.

    Example:
        >>> results = analyzer.sobol_analysis(...)
        >>> chart = plot_tornado(results)
        >>> save_chart(chart, 'tornado.png')
    """
    df = sensitivity_results.to_dataframe()

    # Sort by total-order index
    df = df.sort_values("ST", ascending=False)

    if max_params is not None:
        df = df.head(max_params)

    # Calculate height based on number of parameters
    n_params = len(df)
    height = height or max(200, n_params * 30)
    width = width or CHART_WIDTH

    if show_interactions:
        # Create stacked bar showing S1 + interaction
        df["interaction"] = df["ST"] - df["S1"]
        df_melted = df.melt(
            id_vars=["parameter"],
            value_vars=["S1", "interaction"],
            var_name="index_type",
            value_name="value",
        )

        # Order parameters by total ST
        param_order = df["parameter"].tolist()

        chart = (
            alt.Chart(df_melted)
            .mark_bar()
            .encode(
                y=alt.Y(
                    "parameter:N",
                    sort=param_order,
                    title="Parameter",
                    axis=alt.Axis(labelFontSize=FONT_SIZE_AXIS_LABEL),
                ),
                x=alt.X(
                    "value:Q",
                    stack="zero",
                    title="Sensitivity Index",
                    scale=alt.Scale(domain=[0, max(1.0, df["ST"].max() * 1.1)]),
                ),
                color=alt.Color(
                    "index_type:N",
                    scale=alt.Scale(
                        domain=["S1", "interaction"],
                        range=["#2171b5", "#6baed6"],
                    ),
                    legend=alt.Legend(
                        title="Index Type",
                        orient="bottom",
                        labelFontSize=FONT_SIZE_LEGEND_LABEL,
                        titleFontSize=FONT_SIZE_LEGEND_TITLE,
                    ),
                ),
                tooltip=[
                    alt.Tooltip("parameter:N", title="Parameter"),
                    alt.Tooltip("index_type:N", title="Type"),
                    alt.Tooltip("value:Q", title="Value", format=".3f"),
                ],
            )
        )
    else:
        # Simple bar chart with ST only
        chart = (
            alt.Chart(df)
            .mark_bar(color="#2171b5")
            .encode(
                y=alt.Y(
                    "parameter:N",
                    sort="-x",
                    title="Parameter",
                    axis=alt.Axis(labelFontSize=FONT_SIZE_AXIS_LABEL),
                ),
                x=alt.X(
                    "ST:Q",
                    title="Total-Order Sensitivity Index (ST)",
                    scale=alt.Scale(domain=[0, max(1.0, df["ST"].max() * 1.1)]),
                ),
                tooltip=[
                    alt.Tooltip("parameter:N", title="Parameter"),
                    alt.Tooltip("S1:Q", title="First-order (S1)", format=".3f"),
                    alt.Tooltip("ST:Q", title="Total-order (ST)", format=".3f"),
                ],
            )
        )

    chart = chart.properties(
        title=title,
        width=width,
        height=height,
    )

    return configure_chart(chart)


def plot_probability_heatmap(
    results: "MonteCarloResults",
    variable: str = "psi",
    thresholds: list[float] | None = None,
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
) -> alt.Chart:
    """Create a heatmap of P(variable > threshold) over time and threshold.

    This shows how the probability of exceeding different thresholds evolves
    over the forecast horizon.

    Args:
        results: MonteCarloResults from simulation.
        variable: State variable to analyze.
        thresholds: List of threshold values. Default spans observed range.
        title: Chart title.
        height: Chart height in pixels.
        width: Chart width in pixels.

    Returns:
        Altair Chart with probability heatmap.

    Example:
        >>> chart = plot_probability_heatmap(results, variable='psi')
        >>> save_chart(chart, 'prob_heatmap.png')
    """
    var_data = results.get_variable(variable)

    if thresholds is None:
        # Generate thresholds spanning the observed range
        min_val = np.percentile(var_data, 5)
        max_val = np.percentile(var_data, 95)
        thresholds = np.linspace(min_val, max_val, 20).tolist()

    height = height or CHART_HEIGHT_MEDIUM
    width = width or CHART_WIDTH

    if title is None:
        title = f"Probability of {_get_label(variable)} Exceeding Threshold"

    # Compute probabilities
    data = []
    for thresh in thresholds:
        for t_idx, t in enumerate(results.time):
            values = var_data[:, t_idx]
            prob = np.sum(values > thresh) / len(values)
            data.append({"time": t, "threshold": thresh, "probability": prob})

    df = pd.DataFrame(data)

    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("time:O", title="Time", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(
                "threshold:O",
                title=f"{_get_label(variable)} Threshold",
                sort="descending",
            ),
            color=alt.Color(
                "probability:Q",
                title="Probability",
                scale=alt.Scale(scheme="blues", domain=[0, 1]),
                legend=alt.Legend(
                    orient="right",
                    labelFontSize=FONT_SIZE_LEGEND_LABEL,
                    titleFontSize=FONT_SIZE_LEGEND_TITLE,
                ),
            ),
            tooltip=[
                alt.Tooltip("time:Q", title="Time"),
                alt.Tooltip("threshold:Q", title="Threshold", format=".2f"),
                alt.Tooltip("probability:Q", title="P(exceed)", format=".1%"),
            ],
        )
        .properties(
            title=title,
            width=width,
            height=height,
        )
    )

    return configure_chart(chart)


def plot_probability_over_time(
    results: "MonteCarloResults",
    variable: str = "psi",
    threshold: float = 1.0,
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
) -> alt.Chart:
    """Plot probability of threshold exceedance over time.

    Simple line chart showing how P(variable > threshold) evolves.

    Args:
        results: MonteCarloResults from simulation.
        variable: State variable to analyze.
        threshold: Threshold value.
        title: Chart title.
        height: Chart height in pixels.
        width: Chart width in pixels.

    Returns:
        Altair Chart with probability line.
    """
    df = results.probability_by_year(variable, threshold, comparison="greater")

    height = height or CHART_HEIGHT_MEDIUM
    width = width or CHART_WIDTH

    if title is None:
        title = f"P({_get_label(variable)} > {threshold:.2f}) Over Time"

    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=2, color="#2171b5")
        .encode(
            x=alt.X("year:Q", title="Time"),
            y=alt.Y(
                "probability:Q",
                title="Probability",
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip("year:Q", title="Time"),
                alt.Tooltip("probability:Q", title="Probability", format=".1%"),
            ],
        )
        .properties(
            title=title,
            width=width,
            height=height,
        )
    )

    # Add threshold reference line at 50%
    rule = (
        alt.Chart(pd.DataFrame({"y": [0.5]}))
        .mark_rule(color="gray", strokeDash=[3, 3])
        .encode(y="y:Q")
    )

    return configure_chart(alt.layer(chart, rule))


def plot_timing_distribution(
    results: "MonteCarloResults",
    variable: str = "psi",
    threshold: float = 0.5,
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
    bins: int = 20,
) -> alt.Chart:
    """Plot distribution of when threshold is first crossed.

    Histogram showing the distribution of first-crossing times.

    Args:
        results: MonteCarloResults from simulation.
        variable: State variable to analyze.
        threshold: Threshold value.
        title: Chart title.
        height: Chart height in pixels.
        width: Chart width in pixels.
        bins: Number of histogram bins.

    Returns:
        Altair Chart with timing histogram.
    """
    crossing_times = results.first_crossing_distribution(variable, threshold)
    valid_times = crossing_times[~np.isnan(crossing_times)]

    height = height or CHART_HEIGHT_MEDIUM
    width = width or CHART_WIDTH

    if title is None:
        n_cross = len(valid_times)
        n_total = len(crossing_times)
        pct = n_cross / n_total * 100
        title = (
            f"First Crossing of {_get_label(variable)} > {threshold:.2f} "
            f"({pct:.0f}% cross)"
        )

    df = pd.DataFrame({"time": valid_times})

    chart = (
        alt.Chart(df)
        .mark_bar(color="#2171b5")
        .encode(
            x=alt.X(
                "time:Q",
                bin=alt.Bin(maxbins=bins),
                title="Time of First Crossing",
            ),
            y=alt.Y("count():Q", title="Number of Simulations"),
            tooltip=[
                alt.Tooltip("time:Q", bin=alt.Bin(maxbins=bins), title="Time Bin"),
                alt.Tooltip("count():Q", title="Count"),
            ],
        )
        .properties(
            title=title,
            width=width,
            height=height,
        )
    )

    return configure_chart(chart)


def plot_ensemble_trajectories(
    results: "MonteCarloResults",
    variable: str = "psi",
    n_trajectories: int = 100,
    title: str | None = None,
    highlight_percentiles: bool = True,
    height: int | None = None,
    width: int | None = None,
    seed: int | None = None,
) -> alt.LayerChart:
    """Plot individual ensemble trajectories with summary statistics.

    Shows a subset of individual simulation trajectories to illustrate
    the range of possible outcomes.

    Args:
        results: MonteCarloResults from simulation.
        variable: State variable to plot.
        n_trajectories: Number of trajectories to show.
        title: Chart title.
        highlight_percentiles: If True, overlay median and percentile lines.
        height: Chart height in pixels.
        width: Chart width in pixels.
        seed: Random seed for trajectory selection.

    Returns:
        Altair LayerChart with trajectories.
    """
    var_data = results.get_variable(variable)
    n_sims = var_data.shape[0]

    height = height or CHART_HEIGHT_MEDIUM
    width = width or CHART_WIDTH

    if title is None:
        title = (
            f"{_get_label(variable)} Ensemble "
            f"({n_trajectories} of {n_sims} trajectories)"
        )

    # Select random trajectories
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_sims, size=min(n_trajectories, n_sims), replace=False)

    # Build long-form dataframe
    data = []
    for i, idx in enumerate(indices):
        for t_idx, t in enumerate(results.time):
            data.append({"time": t, "value": var_data[idx, t_idx], "trajectory": i})

    df = pd.DataFrame(data)

    # Plot individual trajectories
    trajectories = (
        alt.Chart(df)
        .mark_line(opacity=0.3, strokeWidth=0.5)
        .encode(
            x=alt.X("time:Q", title="Time"),
            y=alt.Y("value:Q", title=_get_label(variable)),
            detail="trajectory:N",
            color=alt.value("#1f77b4"),
        )
    )

    layers = [trajectories]

    if highlight_percentiles:
        # Add percentile lines
        pct_data = results.to_fan_chart_data(variable, [10, 50, 90])

        # Median line
        median_df = pct_data[["time", "p50"]].copy()
        median_df.columns = ["time", "value"]
        median_line = (
            alt.Chart(median_df)
            .mark_line(color="#d62728", strokeWidth=2)
            .encode(x="time:Q", y="value:Q")
        )
        layers.append(median_line)

        # 10th percentile
        p10_df = pct_data[["time", "p10"]].copy()
        p10_df.columns = ["time", "value"]
        p10_line = (
            alt.Chart(p10_df)
            .mark_line(color="#d62728", strokeWidth=1, strokeDash=[3, 3])
            .encode(x="time:Q", y="value:Q")
        )
        layers.append(p10_line)

        # 90th percentile
        p90_df = pct_data[["time", "p90"]].copy()
        p90_df.columns = ["time", "value"]
        p90_line = (
            alt.Chart(p90_df)
            .mark_line(color="#d62728", strokeWidth=1, strokeDash=[3, 3])
            .encode(x="time:Q", y="value:Q")
        )
        layers.append(p90_line)

    chart = alt.layer(*layers).properties(
        title=title,
        width=width,
        height=height,
    )

    return configure_chart(chart)


def plot_parameter_scatter(
    results: "MonteCarloResults",
    parameter: str,
    variable: str = "psi",
    target_time: float | None = None,
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
) -> alt.Chart:
    """Scatter plot of parameter value vs. output variable.

    Visualizes the relationship between a parameter and the model output.

    Args:
        results: MonteCarloResults from simulation.
        parameter: Parameter name to plot on x-axis.
        variable: Output variable to plot on y-axis.
        target_time: Time at which to evaluate output. Defaults to end time.
        title: Chart title.
        height: Chart height in pixels.
        width: Chart width in pixels.

    Returns:
        Altair Chart with scatter plot.
    """
    if parameter not in results.parameter_names:
        raise ValueError(f"Parameter '{parameter}' not in results")

    if target_time is None:
        target_time = results.time[-1]

    # Get parameter values
    param_idx = results.parameter_names.index(parameter)
    param_values = results.parameter_samples[:, param_idx]

    # Get output values at target time
    time_idx = np.argmin(np.abs(results.time - target_time))
    var_idx = results.state_names.index(variable)
    output_values = results.ensemble[:, time_idx, var_idx]

    height = height or CHART_HEIGHT_MEDIUM
    width = width or CHART_WIDTH

    if title is None:
        title = f"{parameter} vs. {_get_label(variable)} at t={target_time:.0f}"

    df = pd.DataFrame({parameter: param_values, variable: output_values})

    # Compute correlation
    corr = np.corrcoef(param_values, output_values)[0, 1]

    chart = (
        alt.Chart(df)
        .mark_circle(opacity=0.5, size=30)
        .encode(
            x=alt.X(f"{parameter}:Q", title=parameter),
            y=alt.Y(f"{variable}:Q", title=_get_label(variable)),
            tooltip=[
                alt.Tooltip(f"{parameter}:Q", format=".3f"),
                alt.Tooltip(f"{variable}:Q", format=".3f"),
            ],
        )
        .properties(
            title=f"{title} (r = {corr:.3f})",
            width=width,
            height=height,
        )
    )

    return configure_chart(chart)


__all__ = [
    "plot_fan_chart",
    "plot_tornado",
    "plot_probability_heatmap",
    "plot_probability_over_time",
    "plot_timing_distribution",
    "plot_ensemble_trajectories",
    "plot_parameter_scatter",
    "save_chart",
]
