#!/usr/bin/env python3
"""Generate Gemini illustrations for Essay 019: Seeing the Invisible.

This script creates conceptual illustrations about visualization concepts
for the cliodynamics visualization essay.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from cliodynamics.viz.images import generate_image

OUTPUT_DIR = "docs/assets/images/essay-019"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating Gemini illustrations for Essay 019...")
    print("=" * 60)

    # 1. Data transforming into insight
    print("\n1. Data transforming into insight...")
    generate_image(
        """An artistic illustration showing raw data transforming into understanding.

LEFT SIDE: A chaotic cloud of numbers, statistics, and data points swirling in confusion.
Numbers like population figures, dates, percentages float randomly.
Dark, murky colors (grays, browns).

MIDDLE: A large magnifying glass or prism through which the data passes.
Light rays show the transformation happening.

RIGHT SIDE: Clear, beautiful patterns emerge - smooth curves, cycles,
recognizable wave patterns. Bright, illuminated colors (blues, golds).
The patterns look like sine waves, secular cycles, phase space loops.

Title at bottom: "Visualization: From Chaos to Clarity"

Scientific illustration style, educational, inspiring.
High contrast between the messy left and clear right sides.
""",
        os.path.join(OUTPUT_DIR, "data-to-insight.png")
    )

    # 2. Viewer contemplating visualization
    print("\n2. Viewer contemplating complex visualization...")
    generate_image(
        """A scientist or researcher standing before a large wall display showing
complex data visualizations.

The person is viewed from behind, silhouetted against the glowing screens.
They have a thoughtful, contemplative posture - hand on chin.

The screens show:
- Time series plots with multiple lines
- A colorful phase space spiral
- A heatmap of correlations
- An animated trajectory

The lighting is dramatic - the visualizations glow and illuminate the room.
The scene conveys discovery, understanding, the "aha moment".

Colors: Deep blues and purples for the room, vibrant data colors on screens.
Style: Cinematic, slightly futuristic but grounded in real science.

Caption space at bottom: "Seeing What Numbers Cannot Tell"
""",
        os.path.join(OUTPUT_DIR, "contemplating-data.png")
    )

    # 3. Before/after - raw data vs pattern
    print("\n3. Before/after - raw data vs revealed pattern...")
    generate_image(
        """A split illustration showing the power of visualization.

LEFT SIDE labeled "Raw Data":
- A spreadsheet or table with hundreds of tiny numbers
- Dense, overwhelming, impossible to read
- Gray, drab colors
- A confused face emoji in the corner

RIGHT SIDE labeled "Visualized":
- The same data transformed into a beautiful wave pattern
- Clear secular cycle visible - expansion, crisis, recovery
- Vibrant colors with clear meaning (green for growth, red for crisis)
- A lightbulb emoji or "insight" symbol in the corner

A dividing line between them with an arrow pointing right.
Text: "Same data, different story"

Clean infographic style. Educational.
Shows that visualization reveals patterns hidden in raw numbers.
""",
        os.path.join(OUTPUT_DIR, "before-after-pattern.png")
    )

    # 4. Multi-dimensional thinking
    print("\n4. Multi-dimensional thinking concept...")
    generate_image(
        """An illustration showing the progression from 1D to 2D to 3D visualization.

THREE PANELS arranged left to right:

PANEL 1 (1D): A simple line graph on a flat surface.
Label: "One Dimension: Time Series"
A single wavy line going left to right.

PANEL 2 (2D): A phase space diagram showing loops and spirals.
Label: "Two Dimensions: Phase Space"
Trajectories spiral inward or outward.

PANEL 3 (3D): A beautiful 3D spiral or helix structure.
Label: "Three Dimensions: Full Dynamics"
The trajectory loops through 3D space, colorful gradient showing time.

An arrow connects all three, showing progression.
Title: "Adding Dimensions, Revealing Dynamics"

Clean, technical illustration style with academic feel.
Blue, orange, and green color scheme.
""",
        os.path.join(OUTPUT_DIR, "dimensional-progression.png")
    )

    # 5. Colorblind accessibility concept
    print("\n5. Colorblind accessibility illustration...")
    generate_image(
        """An educational illustration about colorblind-friendly visualization.

TOP HALF labeled "Problematic":
- A pie chart or bar chart using red and green
- Through colorblind simulation glasses, the colors look nearly identical
- Confused viewer icon

BOTTOM HALF labeled "Accessible":
- Same chart redesigned with blue, orange, and patterns/textures
- The differences remain clear even through colorblind simulation
- Happy viewer icon with checkmark

Side-by-side vision comparison showing:
- Normal vision view
- Deuteranopia (red-green colorblind) simulation view

Text: "8% of men are colorblind - design for everyone"

Clean infographic style. Educational.
Shows real color palette comparison (bad: red/green vs good: blue/orange).
""",
        os.path.join(OUTPUT_DIR, "colorblind-accessibility.png")
    )

    # 6. Animation concept - motion reveals pattern
    print("\n6. Animation revealing temporal patterns...")
    generate_image(
        """An illustration showing how animation reveals what static images cannot.

LEFT SIDE: A static phase space diagram that looks like tangled spaghetti.
The trajectories overlap and create visual chaos.
Label: "Static: Confusion"

RIGHT SIDE: The same diagram but with a glowing dot moving along a path.
Only a portion of the trajectory is visible (trailing behind the dot).
A motion blur effect shows direction.
Label: "Animated: Clarity"

The animated version makes the order of traversal clear.
You can see where the system started and where it's going.

Include film strip or play button icons to suggest animation.
Title: "Motion Reveals What Stasis Hides"

Dynamic illustration style with motion effects.
Blue color scheme with glowing accents.
""",
        os.path.join(OUTPUT_DIR, "animation-reveals-pattern.png")
    )

    print("\n" + "=" * 60)
    print(f"All images saved to: {OUTPUT_DIR}")
    print("=" * 60)
    print("\nIMPORTANT: Visually verify all images before committing!")


if __name__ == "__main__":
    main()
