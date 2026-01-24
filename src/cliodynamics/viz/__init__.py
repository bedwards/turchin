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
"""

from cliodynamics.viz import (
    animations,
    charts,
    cycles,
    images,
    plots,
    plotly_animations,
)

__all__ = ["charts", "images", "plots", "cycles", "animations", "plotly_animations"]
