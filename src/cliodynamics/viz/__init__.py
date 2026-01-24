"""
Visualization module for cliodynamics.

Provides standardized chart, image, plot, and animation generation
for essays and analysis.

All static visualizations use Altair for consistency. Animated visualizations
use Plotly (in the plotly_animations module).

Submodules:
    charts: Altair-based charts (timelines, bar charts, heatmaps)
    images: Gemini-generated illustrations
    plots: Altair-based scientific plots (time series, phase space)
    cycles: Altair-based secular cycle detection and visualization
    animations: Matplotlib-based animated visualizations (legacy)
    plotly_animations: Plotly-based interactive animations (recommended)
    monte_carlo: Monte Carlo visualization (fan charts, tornado plots)
    ensemble: Ensemble visualization (phase diagrams, bifurcation diagrams)

Usage:
    from cliodynamics.viz import charts, images, plots, cycles, plotly_animations

    # Create a chart
    chart = charts.create_timeline_chart(df, 'nga', 'start', 'end', 'region')
    charts.save_chart(chart, 'output.png')

    # Generate an image
    images.generate_image("A detailed prompt...", "output.png")

    # Create time series plot
    chart = plots.plot_time_series(results, variables=['N', 'W', 'psi'])
    plots.save_chart(chart, 'timeseries.png')

    # Detect and plot secular cycles
    detected = cycles.detect_secular_cycles(results['psi'])
    chart = cycles.plot_with_cycles(results, detected)
    cycles.save_chart(chart, 'cycles.png')

    # Create interactive Plotly animations (recommended)
    fig = plotly_animations.animate_time_series(results, variables=['N', 'W', 'psi'])
    fig.write_html('evolution.html')

    # 3D phase space with camera orbit
    fig = plotly_animations.animate_phase_space_3d(
        results, x='N', y='W', z='psi', camera_orbit=True
    )
    fig.write_html('phase_3d.html')

    # Monte Carlo visualizations
    from cliodynamics.viz import monte_carlo as mc_viz
    chart = mc_viz.plot_fan_chart(mc_results, variable='psi')
    mc_viz.save_chart(chart, 'forecast.png')

    # Ensemble visualizations
    from cliodynamics.viz import ensemble as ens_viz
    chart = ens_viz.plot_phase_diagram(results, x='alpha', y='gamma')
    ens_viz.save_chart(chart, 'phase_diagram.png')
"""

from cliodynamics.viz import (
    animations,
    charts,
    cycles,
    ensemble,
    images,
    monte_carlo,
    plots,
    plotly_animations,
)

__all__ = [
    "charts",
    "images",
    "plots",
    "cycles",
    "animations",
    "plotly_animations",
    "monte_carlo",
    "ensemble",
]
