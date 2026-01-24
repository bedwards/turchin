#!/usr/bin/env python3
"""
Generate Gemini illustrations for Essay 010 - Explorer Guide.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from cliodynamics.viz.images import generate_image

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "assets" / "images" / "essay-010"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGES = [
    {
        "filename": "interactive-exploration.png",
        "prompt": """A dramatic photorealistic scene of a person standing at a futuristic holographic
control panel, manipulating sliders and dials that control a vast three-dimensional visualization
of historical data floating in the air before them. The visualization shows interconnected nodes
representing population, elites, wages, and political stress, with glowing lines showing
feedback loops between them. The person's hands are reaching into the hologram, physically
adjusting parameters as the visualization morphs and shifts in response. The scene is lit
with blue and orange light, conveying both analytical precision and the drama of discovery.
A sense of wonder and power over understanding complex systems.""",
    },
    {
        "filename": "what-if-scenarios.png",
        "prompt": """A photorealistic artistic visualization showing multiple branching timelines
diverging from a single point, like a river splitting into many streams. Each branch represents
a different possible future for a civilization - one path leads to prosperity with green
flourishing colors, another to collapse shown in reds and grays, others showing various
intermediate states. At the branching point, a small figure stands at a control panel,
their hands on sliders that determine which path the civilization will take. The image
conveys the profound idea that small parameter changes can lead to dramatically different
outcomes. Use a style reminiscent of 19th century scientific illustrations combined with
modern data visualization aesthetics.""",
    },
    {
        "filename": "feedback-loops.png",
        "prompt": """An intricate mechanical visualization showing the interconnected feedback loops
of Structural-Demographic Theory as an elegant clockwork mechanism. At the center is a large
gear labeled 'Population', connected to smaller gears labeled 'Elites', 'Wages', 'State',
and 'Political Stress'. Arrows and belts show how each component drives the others -
population growth affects wages, elite overproduction strains the state, declining wages
increase stress. Some connections are reinforcing (positive feedback, shown in gold)
while others are balancing (negative feedback, shown in silver). The entire mechanism
is enclosed in a glass dome like a Victorian orrery, with a person peering inside
with scientific curiosity. Photorealistic with dramatic lighting.""",
    },
    {
        "filename": "crisis-threshold.png",
        "prompt": """A dramatic photorealistic scene of a pressure gauge or dial showing the
Political Stress Index approaching a critical red zone. The gauge is styled like a vintage
industrial meter, with brass fittings and glass face, but displaying modern data.
The needle is approaching a red danger zone marked 'CRISIS THRESHOLD' at 0.6.
Behind the gauge, through a window, we see a city skyline that appears to be
experiencing social unrest - there are fires, crowds in the streets, signs of turmoil.
The scene conveys the tension of a society approaching the breaking point, and the
value of being able to monitor and anticipate such crises. Photorealistic with
cinematic lighting and composition.""",
    },
    {
        "filename": "simulation-sandbox.png",
        "prompt": """A photorealistic scene of a scientist's workbench containing a miniature
simulation of a civilization in a large glass terrarium or snow globe. Inside, tiny
figures representing population, elite towers, factories for wages, and government
buildings for state health are all interconnected with thin glowing threads showing
their relationships. The scientist's hand is reaching toward a control panel on the
side of the terrarium with sliders labeled with Greek letters (alpha, gamma, lambda).
The scene conveys the power and responsibility of being able to experiment with
civilizational dynamics in a safe sandbox environment. Warm golden light from
a desk lamp illuminates the scene, creating a sense of intimate scientific inquiry.""",
    },
]


def main():
    """Generate all images."""
    print(f"Generating {len(IMAGES)} images to {OUTPUT_DIR}")

    for img in IMAGES:
        output_path = OUTPUT_DIR / img["filename"]
        print(f"\nGenerating: {img['filename']}")
        print(f"Prompt: {img['prompt'][:100]}...")

        try:
            generate_image(img["prompt"], output_path)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"ERROR generating {img['filename']}: {e}")
            raise

    print(f"\n{'='*60}")
    print("IMPORTANT: Visually verify all generated images!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
