"""
Visualization module for cliodynamics.

Provides standardized chart and image generation for essays.

Usage:
    from cliodynamics.viz import charts, images

    # Create a chart
    chart = charts.create_timeline_chart(df, 'nga', 'start', 'end', 'region')
    charts.save_chart(chart, 'output.png')

    # Generate an image
    images.generate_image("A detailed prompt...", "output.png")
"""

from cliodynamics.viz import charts, images

__all__ = ['charts', 'images']
