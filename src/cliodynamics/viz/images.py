"""
Standardized Gemini image generation for cliodynamics essays.

This module provides wrapper functions for generating illustrations
using the Gemini API with consistent quality standards.

IMPORTANT: After generating any image, visually verify it looks correct
before committing. Check that labels are placed correctly, the image
matches the prompt, and there are no obvious errors.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Check for API key at module load
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Standard image settings
DEFAULT_MODEL = 'imagen-4.0-generate-001'
DEFAULT_NUM_IMAGES = 1


def generate_image(
    prompt: str,
    output_path: str | Path,
    detailed_placement: bool = True,
) -> Path:
    """
    Generate an image using Gemini and save it.

    Args:
        prompt: Description of the image to generate
        output_path: Where to save the generated image
        detailed_placement: If True, remind about geographic/spatial accuracy

    Returns:
        Path to saved image

    Raises:
        ValueError: If API key not set
        RuntimeError: If image generation fails

    Example:
        >>> generate_image(
        ...     "A world map showing ancient civilizations with labels "
        ...     "placed directly on their geographic locations: "
        ...     "Rome on the Italian peninsula, Egypt on the Nile delta...",
        ...     "docs/assets/images/map.png"
        ... )
    """
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not set. Add it to .env file."
        )

    # Import here to avoid requiring google-genai for non-image work
    try:
        from google import genai
        from google.genai import types
        from PIL import Image
        from io import BytesIO
    except ImportError as e:
        raise ImportError(
            "Required packages not installed. Run: "
            "pip install google-genai pillow"
        ) from e

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add placement reminder for maps
    if detailed_placement and any(word in prompt.lower() for word in ['map', 'geographic', 'region', 'location']):
        logger.warning(
            "⚠️  MAP/GEOGRAPHIC IMAGE DETECTED\n"
            "Ensure your prompt specifies EXACT label placement:\n"
            "- 'Rome label MUST be on the Italian peninsula'\n"
            "- 'Egypt label MUST be on the Nile delta in northeast Africa'\n"
            "- etc."
        )

    logger.info(f"Generating image with prompt: {prompt[:100]}...")

    client = genai.Client(api_key=GEMINI_API_KEY)

    try:
        response = client.models.generate_images(
            model=DEFAULT_MODEL,
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=DEFAULT_NUM_IMAGES)
        )

        if not response.generated_images:
            raise RuntimeError("No images generated")

        generated = response.generated_images[0]
        if generated.image is None or generated.image.image_bytes is None:
            raise RuntimeError("Generated image has no data")

        image_bytes = generated.image.image_bytes
        image = Image.open(BytesIO(image_bytes))
        image.save(output_path)

        logger.info(f"Image saved to {output_path}")
        print(VERIFICATION_REMINDER)

        return output_path

    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise RuntimeError(f"Failed to generate image: {e}") from e


def generate_map_image(
    title: str,
    regions: dict[str, str],
    output_path: str | Path,
    style: str = "antique historical map with decorative border",
) -> Path:
    """
    Generate a map image with correctly placed labels.

    This function builds a detailed prompt that explicitly specifies
    where each label should be placed geographically.

    Args:
        title: Map title
        regions: Dict mapping region names to their geographic descriptions
                 e.g., {"Rome": "Italian peninsula in southern Europe",
                        "Egypt": "Nile River valley in northeast Africa"}
        output_path: Where to save the image
        style: Visual style description

    Returns:
        Path to saved image

    Example:
        >>> generate_map_image(
        ...     "Seshat Database Coverage",
        ...     {
        ...         "Rome": "Italian peninsula, southern Europe, Mediterranean coast",
        ...         "Egypt": "Nile River delta, northeast Africa",
        ...         "Mesopotamia": "between Tigris and Euphrates rivers, modern Iraq",
        ...         "China": "eastern Asia, Yellow and Yangtze river valleys",
        ...     },
        ...     "docs/assets/images/seshat-map.png"
        ... )
    """
    # Build explicit placement instructions
    placement_instructions = []
    for region, location in regions.items():
        placement_instructions.append(
            f"The label '{region}' MUST be placed directly on {location}"
        )

    prompt = f"""
{style}

A world map titled "{title}" showing the following regions with labels.

CRITICAL LABEL PLACEMENT REQUIREMENTS:
{chr(10).join(placement_instructions)}

Each label must be clearly readable and positioned EXACTLY on its geographic location.
Do NOT place labels in the ocean or on incorrect continents.
Use a clean, readable font for all labels.
"""

    return generate_image(prompt.strip(), output_path, detailed_placement=False)


# Verification reminder
VERIFICATION_REMINDER = """
╔══════════════════════════════════════════════════════════════════╗
║  IMPORTANT: Visually verify all generated images!                ║
║                                                                  ║
║  For MAPS, check that:                                          ║
║  1. Labels are on correct geographic locations                  ║
║  2. Rome is on Italy, not the ocean                            ║
║  3. Egypt is in Africa, not elsewhere                          ║
║  4. All text is readable                                        ║
║                                                                  ║
║  For ILLUSTRATIONS, check that:                                 ║
║  1. Image matches the prompt intent                             ║
║  2. No obvious artifacts or errors                              ║
║  3. Text/labels (if any) are correct                           ║
╚══════════════════════════════════════════════════════════════════╝
"""


def verify_image_exists(path: str | Path) -> bool:
    """Check if an image file exists and has reasonable size."""
    path = Path(path)
    if not path.exists():
        logger.error(f"Image not found: {path}")
        return False

    size = path.stat().st_size
    if size < 5000:  # Less than 5KB is suspicious for an image
        logger.warning(f"Image file suspiciously small ({size} bytes): {path}")
        return False

    logger.info(f"Image exists: {path} ({size:,} bytes)")
    print(VERIFICATION_REMINDER)
    return True
