"""
Visualization module for cliodynamics.

Provides standardized chart, image, and plot generation for essays and analysis.

Submodules:
    charts: Altair-based charts (timelines, bar charts, heatmaps)
    images: Gemini-generated illustrations
    plots: Matplotlib-based scientific plots (time series, phase space)
    cycles: Secular cycle detection and visualization

Usage:
    from cliodynamics.viz import charts, images, plots, cycles

    # Create a chart
    chart = charts.create_timeline_chart(df, 'nga', 'start', 'end', 'region')
    charts.save_chart(chart, 'output.png')

    # Generate an image
    images.generate_image("A detailed prompt...", "output.png")

    # Create time series plot
    fig = plots.plot_time_series(results, variables=['N', 'W', 'psi'])
    plots.save_figure(fig, 'timeseries.png')

    # Detect and plot secular cycles
    detected = cycles.detect_secular_cycles(results['psi'])
    fig = cycles.plot_with_cycles(results, detected)
"""

from cliodynamics.viz import charts, cycles, images, plots

__all__ = ["charts", "images", "plots", "cycles"]
