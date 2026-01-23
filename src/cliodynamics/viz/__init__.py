"""
Visualization module for cliodynamics.

Provides standardized chart, image, plot, and animation generation
for essays and analysis.

All static visualizations use Altair for consistency. 3D and animated
visualizations use Plotly (in the animations module).

Submodules:
    charts: Altair-based charts (timelines, bar charts, heatmaps)
    images: Gemini-generated illustrations
    plots: Altair-based scientific plots (time series, phase space)
    cycles: Altair-based secular cycle detection and visualization
    animations: Plotly-based animated visualizations for SDT simulations

Usage:
    from cliodynamics.viz import charts, images, plots, cycles, animations

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

    # Create animations
    anim = animations.animate_time_series(results, variables=['N', 'W', 'psi'])
    animations.save_animation(anim, 'evolution.gif')
"""

from cliodynamics.viz import animations, charts, cycles, images, plots

__all__ = ["charts", "images", "plots", "cycles", "animations"]
