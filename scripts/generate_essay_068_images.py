"""Generate Gemini illustrations for Essay 068: The Claude Code Experiment.

This script creates conceptual illustrations showing the human-AI collaboration
workflow, architecture diagrams, and process visualizations.
"""

import sys
sys.path.insert(0, '/Users/bedwards/turchin/src')

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/Users/bedwards/turchin/.env')

from cliodynamics.viz.images import generate_image

OUTPUT_DIR = Path('/Users/bedwards/turchin/docs/assets/images/essay-068')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check for API key
if not os.environ.get('GEMINI_API_KEY'):
    print("ERROR: GEMINI_API_KEY not found in environment")
    print("Please ensure .env file exists with GEMINI_API_KEY=...")
    sys.exit(1)

print("Generating Gemini illustrations for Essay 068...")
print("="*60)

# Image 1: Human-AI Collaboration Concept
print("\n1. Human-AI Collaboration...")
prompt1 = """
A professional illustration showing human-AI collaboration in software development.

The scene shows a modern workspace with two distinct zones:
- On the left: A human developer at a standing desk with a large monitor, 
  looking thoughtful with hand on chin, reviewing output
- On the right: An abstract representation of Claude Code as flowing streams of 
  code and text emanating from a stylized brain or neural network pattern

Between them: arrows and connecting lines showing bidirectional communication,
with symbols representing code, documents, and images flowing both ways.

Color palette: Professional blues, purples, and warm accents.
Style: Clean, modern tech illustration, not cartoonish.
No text labels on the image.
"""
try:
    generate_image(prompt1, OUTPUT_DIR / 'human-ai-collaboration.png')
    print(f"  Saved: {OUTPUT_DIR / 'human-ai-collaboration.png'}")
except Exception as e:
    print(f"  ERROR: {e}")

# Image 2: Manager-Worker Architecture
print("\n2. Manager-Worker Architecture...")
prompt2 = """
A technical diagram-style illustration showing an orchestration architecture.

At the top: A central node labeled as "Manager" with a conductor's baton motif,
connected by lines downward to multiple smaller nodes.

Below: 4-5 identical "Worker" nodes arranged in a row, each showing a small
gear or processing symbol. Each worker connects to a GitHub-style branch icon.

At the bottom: A merge point where all branches combine back together.

The overall flow shows: Manager -> spawns Workers -> Workers create branches 
-> Branches merge back.

Color palette: Technical blues and grays with accent colors for each worker.
Style: Clean architectural diagram, professional.
No text labels on the image.
"""
try:
    generate_image(prompt2, OUTPUT_DIR / 'manager-worker-architecture.png')
    print(f"  Saved: {OUTPUT_DIR / 'manager-worker-architecture.png'}")
except Exception as e:
    print(f"  ERROR: {e}")

# Image 3: Context Window Visualization
print("\n3. Context Window Visualization...")
prompt3 = """
An abstract visualization of a context window in AI conversation.

Show a large rectangular frame (the context window) containing:
- At the top: Older messages fading into transparency, becoming ghostly
- In the middle: Active conversation messages clearly visible
- At the bottom: The current message being processed, highlighted

Around the frame: Floating document icons, code snippets, and conversation 
bubbles trying to fit inside the limited space.

A meter or gauge on the side showing the window is 75% full.

The overall effect should convey the concept of limited memory/attention
that must be managed carefully.

Color palette: Deep blues and purples with golden highlights for active content.
Style: Technical yet artistic, showing the abstract concept visually.
No text labels on the image.
"""
try:
    generate_image(prompt3, OUTPUT_DIR / 'context-window.png')
    print(f"  Saved: {OUTPUT_DIR / 'context-window.png'}")
except Exception as e:
    print(f"  ERROR: {e}")

# Image 4: Session Handoff
print("\n4. Session Handoff Illustration...")
prompt4 = """
An illustration showing the concept of session handoff between AI instances.

On the left: A completed session represented as a closed book or folder with
a ribbon bookmark, glowing with accomplished work (checkmarks, documents).

In the center: A handoff moment - hands or abstract shapes passing a glowing
orb of knowledge/context from left to right.

On the right: A new session beginning, represented as an open book or fresh
canvas, ready to receive the transferred context.

Include visual elements of:
- Status documents being passed
- Git commit history
- Issue tracking state

Color palette: Warm oranges for the old session, fresh blues for the new.
Style: Elegant transition illustration, professional.
No text labels on the image.
"""
try:
    generate_image(prompt4, OUTPUT_DIR / 'session-handoff.png')
    print(f"  Saved: {OUTPUT_DIR / 'session-handoff.png'}")
except Exception as e:
    print(f"  ERROR: {e}")

# Image 5: Vibe Coding Workflow
print("\n5. Vibe Coding Workflow...")
prompt5 = """
An illustration representing the concept of "vibe coding" - high-level direction
leading to detailed implementation.

Show a flow from left to right:
- Left side: A human figure with thought bubbles containing rough sketches,
  ideas, and high-level concepts (represented as simple shapes and arrows)
- Middle: These ideas transform into increasingly detailed specifications,
  shown as the rough shapes becoming more defined
- Right side: Fully formed code, documentation, and visualizations emerging
  as polished outputs

The transformation should feel magical but technical - ideas becoming reality
through AI collaboration.

Include floating elements: requirement documents, code blocks, charts, essays.

Color palette: Warm creative colors on the left, transitioning to cool
technical blues on the right.
Style: Modern tech illustration with slight artistic flair.
No text labels on the image.
"""
try:
    generate_image(prompt5, OUTPUT_DIR / 'vibe-coding-workflow.png')
    print(f"  Saved: {OUTPUT_DIR / 'vibe-coding-workflow.png'}")
except Exception as e:
    print(f"  ERROR: {e}")

# Image 6: GitHub Integration Ecosystem
print("\n6. GitHub Integration Ecosystem...")
prompt6 = """
A technical illustration showing the GitHub-centered workflow.

In the center: A large GitHub octocat-inspired logo (without using the actual
trademark) - show it as a stylized repository icon.

Connected to the center by lines radiating outward:
- Issues (represented as tickets/cards)
- Pull Requests (represented as merging arrows)
- Code Review (represented as magnifying glass over code)
- Actions/CI (represented as gears or automation symbols)
- Branches (represented as tree branches)

Around the edges: Small icons representing the outputs: essays, charts,
animations, the website.

The overall effect should show GitHub as the central coordination hub
with all project artifacts flowing through it.

Color palette: GitHub's dark mode colors (dark background, green for success,
blue for information).
Style: Clean developer tools aesthetic.
No text labels on the image.
"""
try:
    generate_image(prompt6, OUTPUT_DIR / 'github-ecosystem.png')
    print(f"  Saved: {OUTPUT_DIR / 'github-ecosystem.png'}")
except Exception as e:
    print(f"  ERROR: {e}")

# Image 7: Visual Verification Process
print("\n7. Visual Verification Process...")
prompt7 = """
An illustration showing the concept of visual verification for generated content.

Split into two halves:
- Left half: A chart or map with obvious errors - labels in wrong places,
  distorted proportions, misaligned elements. Show a red X or warning symbol.
- Right half: The same chart/map but corrected - everything properly aligned
  and labeled. Show a green checkmark.

Between them: A human eye or magnifying glass symbol examining the content,
with arrows showing the feedback loop: generate -> verify -> fix -> verify again.

The message should convey: AI-generated visuals need human verification.

Color palette: Muted colors with red highlights for errors, green for success.
Style: Technical illustration showing quality control process.
No text labels on the image.
"""
try:
    generate_image(prompt7, OUTPUT_DIR / 'visual-verification.png')
    print(f"  Saved: {OUTPUT_DIR / 'visual-verification.png'}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "="*60)
print("Image generation complete!")
print(f"Output directory: {OUTPUT_DIR}")
print("="*60)
print("\nIMPORTANT: Visually verify all generated images before committing!")
