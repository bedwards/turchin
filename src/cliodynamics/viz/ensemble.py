"""Visualization tools for ensemble simulation results.

This module provides specialized visualizations for parameter space exploration:
- Phase diagrams (2D stability maps)
- Bifurcation diagrams (parameter vs. outcome)
- Stability boundary plots
- 3D parameter space surfaces
- Animated parameter sweeps

All visualizations use Altair for consistency with the project's viz stack.

Example:
    >>> from cliodynamics.viz.ensemble import (
    ...     plot_phase_diagram,
    ...     plot_bifurcation_diagram,
    ...     plot_stability_boundary,
    ... )
    >>> from cliodynamics.simulation import EnsembleSimulator, EnsembleResults
    >>>
    >>> results = ensemble.run(initial_conditions, time_span)
    >>> chart = plot_phase_diagram(results, x='alpha', y='gamma')
    >>> save_chart(chart, 'stability_map.png')

References:
    Strogatz, S. (2015). Nonlinear Dynamics and Chaos. Westview Press.
    Turchin, P. (2016). Ages of Discord. Beresta Books.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import altair as alt
import pandas as pd

from cliodynamics.viz.charts import (
    CHART_HEIGHT_MEDIUM,
    CHART_WIDTH,
    FONT_SIZE_AXIS_LABEL,
    FONT_SIZE_LEGEND_LABEL,
    FONT_SIZE_LEGEND_TITLE,
    FONT_SIZE_TITLE,
    save_chart,
)

if TYPE_CHECKING:
    from cliodynamics.simulation.ensemble import EnsembleResults


# Color schemes for stability classification
STABILITY_COLORS = {
    "stable": "#2ca02c",  # Green
    "unstable": "#ff7f0e",  # Orange
    "collapse": "#d62728",  # Red
    "oscillating": "#9467bd",  # Purple
    "unknown": "#7f7f7f",  # Gray
}

# Parameter display names
PARAMETER_LABELS = {
    "alpha": "Elite Recruitment Rate (alpha)",
    "beta": "Wage Sensitivity (beta)",
    "gamma": "Labor Supply Effect (gamma)",
    "delta_e": "Elite Decline Rate (delta_e)",
    "eta": "Elite Extraction Effect (eta)",
    "epsilon": "Elite Burden on State (epsilon)",
    "lambda_psi": "Instability Growth Rate (lambda)",
    "mu": "Elite Extraction Rate (mu)",
    "psi_decay": "Instability Decay Rate (psi_decay)",
    "r_max": "Max Population Growth (r_max)",
    "rho": "State Revenue Coefficient (rho)",
    "sigma": "State Expenditure Rate (sigma)",
    "theta_e": "Elite Instability Weight (theta_e)",
    "theta_s": "State Weakness Weight (theta_s)",
    "theta_w": "Wage Instability Weight (theta_w)",
}

# Metric display names
METRIC_LABELS = {
    "max_psi": "Maximum Political Stress (PSI)",
    "mean_psi": "Mean Political Stress (PSI)",
    "final_psi": "Final Political Stress (PSI)",
    "psi_std": "PSI Standard Deviation",
}


def _get_label(name: str, custom_labels: dict[str, str] | None = None) -> str:
    """Get display label for a parameter or metric."""
    if custom_labels and name in custom_labels:
        return custom_labels[name]
    if name in PARAMETER_LABELS:
        return PARAMETER_LABELS[name]
    if name in METRIC_LABELS:
        return METRIC_LABELS[name]
    return name


def _configure_chart(chart: alt.Chart | alt.LayerChart) -> alt.Chart | alt.LayerChart:
    """Apply standard formatting to a chart.

    This is a local version that doesn't require a title parameter,
    since titles are already set in the chart properties.
    """
    return (
        chart.configure_axis(
            labelFontSize=FONT_SIZE_AXIS_LABEL,
            titleFontSize=FONT_SIZE_AXIS_LABEL,
        )
        .configure_legend(
            labelFontSize=FONT_SIZE_LEGEND_LABEL,
            titleFontSize=FONT_SIZE_LEGEND_TITLE,
        )
        .configure_title(
            fontSize=FONT_SIZE_TITLE,
        )
    )


def plot_phase_diagram(
    results: "EnsembleResults",
    x_param: str,
    y_param: str,
    metric: str = "max_psi",
    title: str | None = None,
    colormap: str = "viridis",
    show_boundary: bool = True,
    psi_threshold: float = 1.0,
    height: int | None = None,
    width: int | None = None,
    custom_labels: dict[str, str] | None = None,
) -> alt.LayerChart:
    """Create a 2D phase diagram showing stability across parameter space.

    Phase diagrams show how system behavior varies across two parameters,
    with color indicating the outcome metric (e.g., max instability).

    Args:
        results: EnsembleResults from simulation.
        x_param: Parameter for x-axis.
        y_param: Parameter for y-axis.
        metric: Metric to color by ('max_psi', 'mean_psi', 'classification').
        title: Chart title.
        colormap: Altair color scheme name.
        show_boundary: If True, overlay stability boundary line.
        psi_threshold: Threshold for stability boundary.
        height: Chart height in pixels.
        width: Chart width in pixels.
        custom_labels: Custom labels for parameters/metrics.

    Returns:
        Altair LayerChart with phase diagram.

    Example:
        >>> chart = plot_phase_diagram(results, x='alpha', y='gamma')
        >>> save_chart(chart, 'phase_diagram.png')
    """
    height = height or CHART_HEIGHT_MEDIUM
    width = width or CHART_WIDTH

    # Get data
    df = results.get_phase_diagram_data(x_param, y_param, metric)

    if title is None:
        title = f"Phase Diagram: {_get_label(metric, custom_labels)}"

    # Main heatmap
    heatmap = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X(
                f"{x_param}:O",
                title=_get_label(x_param, custom_labels),
                axis=alt.Axis(labelFontSize=FONT_SIZE_AXIS_LABEL, labelAngle=-45),
            ),
            y=alt.Y(
                f"{y_param}:O",
                title=_get_label(y_param, custom_labels),
                axis=alt.Axis(labelFontSize=FONT_SIZE_AXIS_LABEL),
                sort="descending",
            ),
            color=alt.Color(
                f"{metric}:Q",
                title=_get_label(metric, custom_labels),
                scale=alt.Scale(scheme=colormap),
                legend=alt.Legend(
                    orient="right",
                    labelFontSize=FONT_SIZE_LEGEND_LABEL,
                    titleFontSize=FONT_SIZE_LEGEND_TITLE,
                ),
            ),
            tooltip=[
                alt.Tooltip(f"{x_param}:Q", title=x_param, format=".4f"),
                alt.Tooltip(f"{y_param}:Q", title=y_param, format=".4f"),
                alt.Tooltip(f"{metric}:Q", title=metric, format=".3f"),
            ],
        )
    )

    layers = [heatmap]

    # Add stability boundary if requested
    if show_boundary:
        boundary_df = results.find_stability_boundary(psi_threshold)
        if (
            len(boundary_df) > 0
            and x_param in boundary_df.columns
            and y_param in boundary_df.columns
        ):
            stable_pts = boundary_df[boundary_df["classification"] == "stable"]
            if len(stable_pts) > 0:
                boundary_line = (
                    alt.Chart(stable_pts)
                    .mark_point(
                        shape="circle",
                        size=50,
                        color="white",
                        strokeWidth=2,
                        filled=False,
                    )
                    .encode(
                        x=alt.X(f"{x_param}:O"),
                        y=alt.Y(f"{y_param}:O", sort="descending"),
                    )
                )
                layers.append(boundary_line)

    chart = alt.layer(*layers).properties(
        title=title,
        width=width,
        height=height,
    )

    return _configure_chart(chart)


def plot_stability_map(
    results: "EnsembleResults",
    x_param: str,
    y_param: str,
    psi_threshold: float = 1.0,
    collapse_threshold: float = 10.0,
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
    custom_labels: dict[str, str] | None = None,
) -> alt.Chart:
    """Create a categorical stability map showing stable/unstable/collapse regions.

    Unlike phase diagrams which show continuous metrics, stability maps show
    discrete classifications of system behavior.

    Args:
        results: EnsembleResults from simulation.
        x_param: Parameter for x-axis.
        y_param: Parameter for y-axis.
        psi_threshold: PSI threshold for stable/unstable boundary.
        collapse_threshold: PSI threshold for collapse.
        title: Chart title.
        height: Chart height in pixels.
        width: Chart width in pixels.
        custom_labels: Custom labels for parameters.

    Returns:
        Altair Chart with stability map.
    """
    height = height or CHART_HEIGHT_MEDIUM
    width = width or CHART_WIDTH

    # Get classification data
    df = results.get_phase_diagram_data(x_param, y_param, "max_psi")

    # Add classification
    df["classification"] = df["max_psi"].apply(
        lambda x: "collapse"
        if x >= collapse_threshold
        else ("unstable" if x >= psi_threshold else "stable")
    )

    if title is None:
        x_label = _get_label(x_param, custom_labels)
        y_label = _get_label(y_param, custom_labels)
        title = f"Stability Map: {x_label} vs {y_label}"

    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X(
                f"{x_param}:O",
                title=_get_label(x_param, custom_labels),
                axis=alt.Axis(labelFontSize=FONT_SIZE_AXIS_LABEL, labelAngle=-45),
            ),
            y=alt.Y(
                f"{y_param}:O",
                title=_get_label(y_param, custom_labels),
                axis=alt.Axis(labelFontSize=FONT_SIZE_AXIS_LABEL),
                sort="descending",
            ),
            color=alt.Color(
                "classification:N",
                title="Stability",
                scale=alt.Scale(
                    domain=["stable", "unstable", "collapse"],
                    range=[
                        STABILITY_COLORS["stable"],
                        STABILITY_COLORS["unstable"],
                        STABILITY_COLORS["collapse"],
                    ],
                ),
                legend=alt.Legend(
                    orient="right",
                    labelFontSize=FONT_SIZE_LEGEND_LABEL,
                    titleFontSize=FONT_SIZE_LEGEND_TITLE,
                ),
            ),
            tooltip=[
                alt.Tooltip(f"{x_param}:Q", title=x_param, format=".4f"),
                alt.Tooltip(f"{y_param}:Q", title=y_param, format=".4f"),
                alt.Tooltip("classification:N", title="Stability"),
                alt.Tooltip("max_psi:Q", title="Max PSI", format=".3f"),
            ],
        )
        .properties(
            title=title,
            width=width,
            height=height,
        )
    )

    return _configure_chart(chart)


def plot_bifurcation_diagram(
    results: "EnsembleResults",
    parameter: str,
    metric: str = "max_psi",
    fixed_params: dict[str, float] | None = None,
    show_threshold: bool = True,
    psi_threshold: float = 1.0,
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
    custom_labels: dict[str, str] | None = None,
) -> alt.LayerChart:
    """Create a bifurcation diagram showing parameter vs. outcome.

    Bifurcation diagrams reveal where smooth parameter changes cause
    sudden transitions in system behavior.

    Args:
        results: EnsembleResults from simulation.
        parameter: Parameter to vary on x-axis.
        metric: Output metric for y-axis.
        fixed_params: Values for other parameters.
        show_threshold: If True, show stability threshold line.
        psi_threshold: Threshold value to show.
        title: Chart title.
        height: Chart height in pixels.
        width: Chart width in pixels.
        custom_labels: Custom labels for parameters/metrics.

    Returns:
        Altair LayerChart with bifurcation diagram.

    Example:
        >>> chart = plot_bifurcation_diagram(
        ...     results, parameter='alpha',
        ...     fixed_params={'gamma': 2.0}
        ... )
        >>> save_chart(chart, 'bifurcation.png')
    """
    height = height or CHART_HEIGHT_MEDIUM
    width = width or CHART_WIDTH

    df = results.get_bifurcation_diagram_data(parameter, fixed_params, metric)

    if title is None:
        title = f"Bifurcation Diagram: {_get_label(parameter, custom_labels)}"
        if fixed_params:
            fixed_str = ", ".join(f"{k}={v:.3f}" for k, v in fixed_params.items())
            title += f" (fixed: {fixed_str})"

    # Main line
    line = (
        alt.Chart(df)
        .mark_line(strokeWidth=2, color="#1f77b4")
        .encode(
            x=alt.X(
                f"{parameter}:Q",
                title=_get_label(parameter, custom_labels),
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                f"{metric}:Q",
                title=_get_label(metric, custom_labels),
            ),
        )
    )

    # Points
    points = (
        alt.Chart(df)
        .mark_circle(size=40, color="#1f77b4")
        .encode(
            x=alt.X(f"{parameter}:Q"),
            y=alt.Y(f"{metric}:Q"),
            tooltip=[
                alt.Tooltip(f"{parameter}:Q", title=parameter, format=".4f"),
                alt.Tooltip(f"{metric}:Q", title=metric, format=".3f"),
            ],
        )
    )

    layers = [line, points]

    # Threshold line
    if show_threshold:
        threshold_df = pd.DataFrame({"y": [psi_threshold]})
        threshold_line = (
            alt.Chart(threshold_df)
            .mark_rule(color="#d62728", strokeDash=[5, 5], strokeWidth=2)
            .encode(y="y:Q")
        )
        layers.append(threshold_line)

        # Add annotation
        threshold_text = (
            alt.Chart(pd.DataFrame({"x": [df[parameter].max()], "y": [psi_threshold]}))
            .mark_text(
                align="right",
                dx=-5,
                dy=-10,
                fontSize=FONT_SIZE_AXIS_LABEL,
                color="#d62728",
            )
            .encode(
                x="x:Q",
                y="y:Q",
                text=alt.value(f"Threshold = {psi_threshold}"),
            )
        )
        layers.append(threshold_text)

    # Find and mark bifurcation points
    bifurcations = results.find_bifurcation(parameter, fixed_params, psi_threshold)
    if bifurcations:
        bif_data = []
        for bif in bifurcations:
            # Find closest metric value
            closest_idx = (df[parameter] - bif.value).abs().idxmin()
            metric_val = df.loc[closest_idx, metric]
            bif_data.append(
                {
                    parameter: bif.value,
                    metric: metric_val,
                    "direction": bif.direction,
                }
            )

        bif_df = pd.DataFrame(bif_data)

        bif_markers = (
            alt.Chart(bif_df)
            .mark_point(
                shape="diamond",
                size=150,
                color="#d62728",
                filled=True,
                strokeWidth=2,
            )
            .encode(
                x=alt.X(f"{parameter}:Q"),
                y=alt.Y(f"{metric}:Q"),
                tooltip=[
                    alt.Tooltip(f"{parameter}:Q", title="Bifurcation at", format=".4f"),
                    alt.Tooltip("direction:N", title="Type"),
                ],
            )
        )
        layers.append(bif_markers)

    chart = alt.layer(*layers).properties(
        title=title,
        width=width,
        height=height,
    )

    return _configure_chart(chart)


def plot_stability_boundary(
    results: "EnsembleResults",
    x_param: str,
    y_param: str,
    psi_threshold: float = 1.0,
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
    custom_labels: dict[str, str] | None = None,
) -> alt.LayerChart:
    """Plot the stability boundary in parameter space.

    Shows the dividing line between stable and unstable regions,
    useful for understanding where transitions occur.

    Args:
        results: EnsembleResults from simulation.
        x_param: Parameter for x-axis.
        y_param: Parameter for y-axis.
        psi_threshold: Threshold for stability.
        title: Chart title.
        height: Chart height in pixels.
        width: Chart width in pixels.
        custom_labels: Custom labels for parameters.

    Returns:
        Altair LayerChart with boundary visualization.
    """
    height = height or CHART_HEIGHT_MEDIUM
    width = width or CHART_WIDTH

    if title is None:
        title = f"Stability Boundary (PSI = {psi_threshold})"

    # Get full data for context
    df = results.get_phase_diagram_data(x_param, y_param, "max_psi")
    df["stable"] = df["max_psi"] < psi_threshold

    # Background scatter showing all points
    background = (
        alt.Chart(df)
        .mark_point(size=30, opacity=0.3)
        .encode(
            x=alt.X(
                f"{x_param}:Q",
                title=_get_label(x_param, custom_labels),
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                f"{y_param}:Q",
                title=_get_label(y_param, custom_labels),
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "stable:N",
                scale=alt.Scale(
                    domain=[True, False],
                    range=[STABILITY_COLORS["stable"], STABILITY_COLORS["unstable"]],
                ),
                legend=alt.Legend(title="Stable"),
            ),
        )
    )

    layers = [background]

    # Find and plot boundary points
    boundary_df = results.find_stability_boundary(psi_threshold)
    if len(boundary_df) > 0 and x_param in boundary_df.columns:
        boundary_pts = (
            alt.Chart(boundary_df)
            .mark_point(
                shape="circle",
                size=80,
                color="black",
                filled=True,
                strokeWidth=2,
            )
            .encode(
                x=alt.X(f"{x_param}:Q"),
                y=alt.Y(f"{y_param}:Q"),
                tooltip=[
                    alt.Tooltip(f"{x_param}:Q", format=".4f"),
                    alt.Tooltip(f"{y_param}:Q", format=".4f"),
                    alt.Tooltip("classification:N"),
                ],
            )
        )
        layers.append(boundary_pts)

    chart = alt.layer(*layers).properties(
        title=title,
        width=width,
        height=height,
    )

    return _configure_chart(chart)


def plot_parameter_sensitivity_grid(
    results: "EnsembleResults",
    metric: str = "max_psi",
    title: str = "Parameter Sensitivity Grid",
    height: int | None = None,
    width: int | None = None,
) -> alt.Chart:
    """Create a small multiples grid showing metric variation across parameters.

    For grids with 2+ parameters, shows how each parameter affects the metric.

    Args:
        results: EnsembleResults from simulation.
        metric: Metric to analyze.
        title: Chart title.
        height: Height per facet.
        width: Width per facet.

    Returns:
        Altair Chart with faceted visualization.
    """
    height = height or 150
    width = width or 200

    df = results.to_dataframe()
    param_names = results.parameter_names

    # Melt to long form for faceting
    melted = df.melt(
        id_vars=[metric],
        value_vars=param_names,
        var_name="parameter",
        value_name="value",
    )

    chart = (
        alt.Chart(melted)
        .mark_point(opacity=0.5, size=20)
        .encode(
            x=alt.X("value:Q", title="Parameter Value"),
            y=alt.Y(f"{metric}:Q", title=_get_label(metric)),
            facet=alt.Facet(
                "parameter:N",
                columns=min(3, len(param_names)),
                title="",
            ),
        )
        .properties(
            title=title,
            width=width,
            height=height,
        )
    )

    return _configure_chart(chart)


def plot_stability_fraction(
    results: "EnsembleResults",
    parameter: str,
    psi_threshold: float = 1.0,
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
    custom_labels: dict[str, str] | None = None,
) -> alt.LayerChart:
    """Plot fraction of stable outcomes as a function of one parameter.

    Shows how the probability of stability changes as a parameter varies.

    Args:
        results: EnsembleResults from simulation.
        parameter: Parameter to vary.
        psi_threshold: Threshold for stability.
        title: Chart title.
        height: Chart height in pixels.
        width: Chart width in pixels.
        custom_labels: Custom labels for parameters.

    Returns:
        Altair LayerChart with stability fraction.
    """
    height = height or CHART_HEIGHT_MEDIUM
    width = width or CHART_WIDTH

    df = results.to_dataframe()
    df["stable"] = df["max_psi"] < psi_threshold

    # Group by parameter
    grouped = df.groupby(parameter)["stable"].mean().reset_index()
    grouped.columns = [parameter, "fraction_stable"]

    if title is None:
        title = f"Stability Probability vs {_get_label(parameter, custom_labels)}"

    chart = (
        alt.Chart(grouped)
        .mark_line(strokeWidth=2, point=True)
        .encode(
            x=alt.X(
                f"{parameter}:Q",
                title=_get_label(parameter, custom_labels),
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "fraction_stable:Q",
                title="Fraction Stable",
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip(f"{parameter}:Q", format=".4f"),
                alt.Tooltip("fraction_stable:Q", title="P(stable)", format=".1%"),
            ],
        )
        .properties(
            title=title,
            width=width,
            height=height,
        )
    )

    # Add 50% reference line
    ref_line = (
        alt.Chart(pd.DataFrame({"y": [0.5]}))
        .mark_rule(color="gray", strokeDash=[3, 3])
        .encode(y="y:Q")
    )

    return _configure_chart(alt.layer(chart, ref_line))


def plot_outcome_distribution(
    results: "EnsembleResults",
    metric: str = "max_psi",
    title: str | None = None,
    bins: int = 30,
    height: int | None = None,
    width: int | None = None,
) -> alt.Chart:
    """Plot histogram of outcome metric across all simulations.

    Shows the distribution of outcomes across the entire parameter space.

    Args:
        results: EnsembleResults from simulation.
        metric: Metric to plot.
        title: Chart title.
        bins: Number of histogram bins.
        height: Chart height in pixels.
        width: Chart width in pixels.

    Returns:
        Altair Chart with histogram.
    """
    height = height or CHART_HEIGHT_MEDIUM
    width = width or CHART_WIDTH

    df = results.to_dataframe()

    if title is None:
        title = f"Distribution of {_get_label(metric)}"

    chart = (
        alt.Chart(df)
        .mark_bar(color="#1f77b4")
        .encode(
            x=alt.X(
                f"{metric}:Q",
                bin=alt.Bin(maxbins=bins),
                title=_get_label(metric),
            ),
            y=alt.Y("count():Q", title="Number of Simulations"),
            tooltip=[
                alt.Tooltip(f"{metric}:Q", bin=alt.Bin(maxbins=bins), title="Range"),
                alt.Tooltip("count():Q", title="Count"),
            ],
        )
        .properties(
            title=title,
            width=width,
            height=height,
        )
    )

    return _configure_chart(chart)


def create_ensemble_report(
    results: "EnsembleResults",
    output_dir: str,
    psi_threshold: float = 1.0,
) -> dict[str, str]:
    """Generate a full set of ensemble visualizations.

    Creates multiple charts and saves them to the output directory.

    Args:
        results: EnsembleResults from simulation.
        output_dir: Directory to save charts.
        psi_threshold: Threshold for stability.

    Returns:
        Dictionary mapping chart names to file paths.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    files = {}

    param_names = results.parameter_names

    # Outcome distribution
    chart = plot_outcome_distribution(results)
    path = os.path.join(output_dir, "outcome_distribution.png")
    save_chart(chart, path)
    files["outcome_distribution"] = path

    # Phase diagrams for each pair of parameters
    if len(param_names) >= 2:
        for i, x_param in enumerate(param_names):
            for y_param in param_names[i + 1 :]:
                # Phase diagram
                chart = plot_phase_diagram(
                    results, x_param, y_param, psi_threshold=psi_threshold
                )
                name = f"phase_{x_param}_{y_param}"
                path = os.path.join(output_dir, f"{name}.png")
                save_chart(chart, path)
                files[name] = path

                # Stability map
                chart = plot_stability_map(
                    results, x_param, y_param, psi_threshold=psi_threshold
                )
                name = f"stability_{x_param}_{y_param}"
                path = os.path.join(output_dir, f"{name}.png")
                save_chart(chart, path)
                files[name] = path

    # Bifurcation diagrams for each parameter
    for param in param_names:
        chart = plot_bifurcation_diagram(results, param, psi_threshold=psi_threshold)
        name = f"bifurcation_{param}"
        path = os.path.join(output_dir, f"{name}.png")
        save_chart(chart, path)
        files[name] = path

        # Stability fraction
        chart = plot_stability_fraction(results, param, psi_threshold=psi_threshold)
        name = f"stability_fraction_{param}"
        path = os.path.join(output_dir, f"{name}.png")
        save_chart(chart, path)
        files[name] = path

    return files


__all__ = [
    "plot_phase_diagram",
    "plot_stability_map",
    "plot_bifurcation_diagram",
    "plot_stability_boundary",
    "plot_parameter_sensitivity_grid",
    "plot_stability_fraction",
    "plot_outcome_distribution",
    "create_ensemble_report",
    "save_chart",
    "STABILITY_COLORS",
    "PARAMETER_LABELS",
    "METRIC_LABELS",
]
