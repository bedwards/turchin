#!/usr/bin/env python3
"""Generate Gemini illustrations for Essay 3: The Mathematics of Societal Collapse."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API key
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    print("ERROR: GEMINI_API_KEY not found in environment")
    sys.exit(1)

from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

# Output directory
OUTPUT_DIR = Path('docs/assets/images')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Gemini client
client = genai.Client(api_key=api_key)
MODEL = 'imagen-4.0-generate-001'


def generate_image(prompt: str, output_path: str) -> bool:
    """Generate an image using Gemini Imagen."""
    print(f"Generating: {output_path}")
    print(f"  Prompt: {prompt[:100]}...")

    try:
        response = client.models.generate_images(
            model=MODEL,
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=1)
        )

        if not response.generated_images:
            print("  ERROR: No images generated")
            return False

        generated = response.generated_images[0]
        if generated.image is None or generated.image.image_bytes is None:
            print("  ERROR: Generated image has no data")
            return False

        image_bytes = generated.image.image_bytes
        image = Image.open(BytesIO(image_bytes))
        image.save(output_path)

        print(f"  SUCCESS: Saved to {output_path}")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    print("Generating Gemini illustrations for Essay 3...")
    print()

    # Image 1: The SDT Cauldron
    generate_image(
        """A dramatic illustration of an alchemist's bronze cauldron with mathematical symbols
        and Greek letters (N, E, W, S, psi) swirling together inside like magical ingredients.
        The cauldron sits on an ancient stone pedestal. Inside, five glowing streams of different
        colors (representing population, elites, wages, state health, and political stress)
        spiral and interweave, creating complex patterns. Steam rises with differential equations
        floating in it. The style is detailed scientific illustration with a touch of mysticism.
        Dark moody background with warm lighting from the cauldron. No text labels, let the
        symbols speak for themselves.""",
        str(OUTPUT_DIR / 'sdt-cauldron.png')
    )

    # Image 2: Feedback Arrows
    generate_image(
        """An elegant scientific diagram showing interconnected nodes representing societal
        variables. Five glowing orbs arranged in a pentagon formation, each a different color:
        blue (population), gold (elites), green (wages), red (state), purple (instability).
        Curved arrows flow between them showing causal connections - some arrows thin
        (weak effects), some thick (strong effects), some with + signs (positive feedback),
        some with - signs (negative feedback). The arrows form a complex web of
        interdependencies. Clean white background, modern infographic style with subtle
        gradients. Professional scientific visualization aesthetic.""",
        str(OUTPUT_DIR / 'sdt-feedback-arrows.png')
    )

    # Image 3: Phase Space Journey
    generate_image(
        """An abstract artistic visualization of a mathematical phase space trajectory.
        A luminous golden path spirals through a dark three-dimensional space filled with
        subtle grid lines and coordinate axes. The path traces out a limit cycle - a
        recurring loop that represents the secular cycle of history. Faint ghostly
        previous trajectories visible in the background, all converging toward the same
        attractor. Small glowing markers along the path indicate key states: prosperity
        (bright), crisis (dark red), recovery (emerging blue). The style is somewhere
        between scientific visualization and abstract art. Deep blue-black background
        with stars suggesting cosmic scale.""",
        str(OUTPUT_DIR / 'sdt-phase-space.png')
    )

    print()
    print("Image generation complete!")
    print()
    print("IMPORTANT: Please visually verify all generated images before committing!")


if __name__ == '__main__':
    main()
