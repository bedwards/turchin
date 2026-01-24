#!/usr/bin/env python3
"""Generate Gemini illustrations for Essay 070: Monte Carlo Methods.

This script creates conceptual illustrations to accompany the essay.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from cliodynamics.viz.images import generate_image

OUTPUT_DIR = "docs/assets/images/essay-070"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Generating Gemini illustrations for Essay 070...")
    print("=" * 60)
    
    # 1. Weather forecaster comparison
    print("\n1. Weather forecaster comparison...")
    generate_image(
        """A split illustration comparing two approaches to prediction.

LEFT SIDE: A weatherman with a single arrow pointing to "72°F" - labeled "Point Prediction"
The weatherman looks confident but slightly foolish.

RIGHT SIDE: A weatherman showing a probability distribution with a bell curve, range from 65°F to 79°F,
with different colored zones showing likelihood - labeled "Probability Forecast"
The weatherman looks more professional and thoughtful.

Clean, professional infographic style with blue and orange colors.
Educational illustration showing the difference between deterministic and probabilistic forecasting.
Title at top: "The Evolution of Prediction"
""",
        os.path.join(OUTPUT_DIR, "weather-forecaster-comparison.png")
    )
    
    # 2. Monte Carlo concept - many dice
    print("\n2. Monte Carlo concept - dice simulation...")
    generate_image(
        """An educational illustration showing the Monte Carlo method concept.

A large funnel at the top with hundreds of small dice falling into it.
The dice are colorful (red, blue, white) and tumbling through the air.
At the bottom of the funnel, a smooth probability distribution curve emerges.

The concept: individual random outcomes (dice) combine to reveal underlying patterns.

Clean, minimalist scientific illustration style.
Light background with vibrant dice colors.
Title: "From Randomness to Probability"
""",
        os.path.join(OUTPUT_DIR, "monte-carlo-dice.png")
    )
    
    # 3. Uncertainty sources diagram
    print("\n3. Uncertainty sources diagram...")
    generate_image(
        """A circular diagram showing four sources of uncertainty in forecasting.

Center: A crystal ball with question marks, labeled "Forecast Uncertainty"

Four connected circles around it:
1. TOP: "Data Uncertainty" - with icons of historical documents and question marks
2. RIGHT: "Parameter Uncertainty" - with mathematical symbols and error bars
3. BOTTOM: "Model Uncertainty" - with different equation symbols
4. LEFT: "Initial Conditions" - with a compass and starting point marker

Arrows connect all four to the center, showing they all contribute to forecast uncertainty.

Clean infographic style, professional colors (blues, grays).
Educational diagram for explaining uncertainty in mathematical modeling.
""",
        os.path.join(OUTPUT_DIR, "uncertainty-sources.png")
    )
    
    # 4. Fan chart concept
    print("\n4. Fan chart concept illustration...")
    generate_image(
        """An artistic illustration of a forecast fan chart concept.

A single path starting from NOW (left side) that expands into a widening cone of possibilities.

The cone shows:
- Dark blue core (most likely outcomes)
- Medium blue middle band
- Light blue outer edges (less likely but possible)

The path starts narrow and certain, then fans out wider as time progresses to the right.
Some faint individual trajectory lines visible within the cone.

Title at bottom: "The Cone of Possibility"

Artistic, slightly abstract visualization. Clean modern style.
Could be mistaken for a piece of data art in a museum.
""",
        os.path.join(OUTPUT_DIR, "fan-chart-concept.png")
    )
    
    # 5. Turkey problem / Black swan
    print("\n5. Turkey problem illustration...")
    generate_image(
        """A timeline illustration showing "The Turkey Problem" concept.

LEFT SIDE (Most of the image): A happy, confident turkey standing on a rising chart line.
The chart shows 1000 days of being fed, going up steadily.
The turkey has a thought bubble: "Life is predictable!"

RIGHT SIDE: Day 1001 marked with a dramatic drop in the line.
A small Thanksgiving turkey dinner plate appears at the end.

This illustrates how past patterns can break suddenly - the turkey's model of "I get fed every day"
fails catastrophically when the pattern changes.

Educational illustration, slightly humorous but making a serious point about prediction.
Include text: "1000 days of data... one day of surprise"
""",
        os.path.join(OUTPUT_DIR, "turkey-problem.png")
    )
    
    print("\n" + "=" * 60)
    print(f"All images saved to: {OUTPUT_DIR}")
    print("=" * 60)
    print("\n⚠️  IMPORTANT: Visually verify all images before committing!")


if __name__ == "__main__":
    main()
