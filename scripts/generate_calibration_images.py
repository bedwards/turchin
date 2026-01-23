#!/usr/bin/env python3
"""
Generate images for Essay #16: Parameter Calibration.

This script generates all Gemini illustrations and charts needed for the essay.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load environment variables
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from cliodynamics.viz.images import generate_image

# Output directory
IMAGES_DIR = PROJECT_ROOT / "docs/assets/images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Gemini illustrations (conceptual images) go in docs/assets/images/
ILLUSTRATIONS = {
    "detective-clues.png": """
        A scholarly detective figure in a Victorian study, examining scattered
        historical documents and artifacts with a magnifying glass. The figure
        wears a tweed jacket and is surrounded by ancient manuscripts, census
        records, and economic charts. Warm lighting from an oil lamp illuminates
        the scene. The mood is thoughtful and investigative. Style: detailed
        academic illustration with rich textures.
    """,
    "uncertainty-fog.png": """
        A conceptual illustration showing uncertainty as atmospheric conditions.
        A mountainous landscape (representing the parameter space) with parts
        clearly visible in sunlight and other parts obscured by fog and clouds.
        The clear areas represent well-constrained parameters while the foggy
        regions represent uncertain parameter values. A small figure stands on
        a clear peak, looking toward the misty regions. Style: evocative
        landscape illustration with metaphorical meaning.
    """,
}

# Note: The following data charts should be generated using cliodynamics.viz.charts
# (Altair) and saved to docs/assets/charts/:
# - loss-landscape.png: 3D optimization loss landscape
# - optimization-trajectory.png: Optimization trajectory on contour plot
# - model-vs-observed.png: Model predictions vs observed data time series
# - confidence-intervals.png: Parameter estimates with confidence intervals
# - residual-analysis.png: Residual diagnostic plots
# - parameter-convergence.png: Parameter convergence during optimization
#
# For now, these are generated as Gemini illustrations as placeholders.
# Future work should replace them with proper Altair visualizations.


def main():
    """Generate all illustrations for the calibration essay."""
    print("Generating illustrations for Essay #16: Parameter Calibration")
    print("=" * 60)

    for filename, prompt in ILLUSTRATIONS.items():
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
    print("Illustration generation complete!")
    print("\nIMPORTANT: Visually verify each image before committing.")
    print("\nNote: Data charts should be generated using cliodynamics.viz.charts:")
    print("  - loss-landscape.png, optimization-trajectory.png")
    print("  - model-vs-observed.png, confidence-intervals.png")
    print("  - residual-analysis.png, parameter-convergence.png")


if __name__ == "__main__":
    main()
