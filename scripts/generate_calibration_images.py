#!/usr/bin/env python3
"""
Generate images for Essay #16: Parameter Calibration.

This script generates all Gemini illustrations and charts needed for the essay.
"""

import sys
import os

# Add src to path
sys.path.insert(0, '/Users/bedwards/turchin/.worktrees/issue-16/src')

# Load environment variables
from dotenv import load_dotenv
load_dotenv('/Users/bedwards/turchin/.worktrees/issue-16/.env')

from pathlib import Path
from cliodynamics.viz.images import generate_image

# Output directory
IMAGES_DIR = Path('/Users/bedwards/turchin/.worktrees/issue-16/docs/assets/images')
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Define all images needed for the essay
IMAGES = {
    'detective-clues.png': """
        A scholarly detective figure in a Victorian study, examining scattered historical
        documents and artifacts with a magnifying glass. The figure wears a tweed jacket and
        is surrounded by ancient manuscripts, census records, and economic charts.
        Warm lighting from an oil lamp illuminates the scene. The mood is thoughtful and
        investigative. Style: detailed academic illustration with rich textures.
    """,

    'loss-landscape.png': """
        A dramatic 3D visualization of an optimization loss landscape. The surface shows
        multiple peaks and valleys rendered as a mountainous terrain, with deep valleys
        (minima) colored in blue and peaks (high loss) in red/orange. Grid lines on the
        surface show the parameter space. A glowing point marks the global minimum in the
        deepest valley. Style: scientific visualization with vibrant colors, like a
        topographic map rendered in 3D.
    """,

    'optimization-trajectory.png': """
        A bird's-eye view of an optimization trajectory across a contour plot showing
        a loss landscape. The trajectory is a bold line with arrows showing direction,
        starting from a circle marker and ending at a star marker in the global minimum.
        The path shows exploration: some wandering, some direct descent. Contour lines
        show the shape of the loss function. Colors transition from red (high loss) to
        blue (low loss). Style: clean scientific diagram.
    """,

    'model-vs-observed.png': """
        A scientific chart showing model calibration results. Two overlapping time series:
        a solid smooth line (model predictions) and scattered points with error bars
        (historical observations). The x-axis shows time in years, the y-axis shows
        population or another variable. The model line generally follows the data points
        but shows some discrepancies. Background has a subtle grid. Style: professional
        scientific visualization.
    """,

    'confidence-intervals.png': """
        A horizontal bar chart showing parameter estimates with confidence intervals.
        Each parameter (labeled on the y-axis) has a point estimate shown as a circle
        and a horizontal line extending in both directions showing the 95% confidence
        interval. Some parameters have narrow intervals (well-determined) while others
        have wide intervals (uncertain). Colors differentiate parameter types.
        Style: clean statistical visualization.
    """,

    'uncertainty-fog.png': """
        A conceptual illustration showing uncertainty as atmospheric conditions.
        A mountainous landscape (representing the parameter space) with parts clearly
        visible in sunlight and other parts obscured by fog and clouds. The clear
        areas represent well-constrained parameters while the foggy regions represent
        uncertain parameter values. A small figure stands on a clear peak, looking
        toward the misty regions. Style: evocative landscape illustration with
        metaphorical meaning.
    """,

    'residual-analysis.png': """
        A diagnostic residual plot for model calibration. The main panel shows residuals
        (differences between model and data) plotted against time, scattered around a
        horizontal zero line. Most points cluster near zero but some outliers exist.
        A secondary panel at the top shows a histogram of residuals following a roughly
        normal distribution. The plot demonstrates good model fit with randomly
        distributed errors. Style: clean statistical diagnostic plot.
    """,

    'parameter-convergence.png': """
        A multi-line chart showing how parameter values converge during optimization.
        The x-axis shows iteration number, the y-axis shows parameter value (normalized).
        Multiple lines, each representing a different parameter, start from different
        initial values and gradually converge to their final estimates. Some parameters
        converge quickly while others oscillate before settling. A vertical dashed line
        marks convergence. Style: clean scientific line chart with legend.
    """,
}


def main():
    """Generate all images for the calibration essay."""
    print("Generating images for Essay #16: Parameter Calibration")
    print("=" * 60)

    for filename, prompt in IMAGES.items():
        output_path = IMAGES_DIR / filename

        if output_path.exists():
            print(f"SKIP: {filename} already exists")
            continue

        print(f"\nGenerating: {filename}")
        print("-" * 40)

        try:
            generate_image(prompt.strip(), output_path)
            print(f"SUCCESS: {filename}")
        except Exception as e:
            print(f"ERROR generating {filename}: {e}")
            continue

    print("\n" + "=" * 60)
    print("Image generation complete!")
    print("\nIMPORTANT: Visually verify each image before committing.")


if __name__ == '__main__':
    main()
