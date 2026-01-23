#!/usr/bin/env python3
"""Generate animation for Essay 3: Phase space trajectory animation."""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint
from pathlib import Path

from cliodynamics.models.sdt import SDTModel
from cliodynamics.models.params import SDTParams

# Output directory
OUTPUT_DIR = Path('docs/assets/animations')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_phase_space_animation():
    """Generate an animated GIF showing phase space trajectory."""
    print("Generating phase space animation...")

    # Run simulation
    t = np.linspace(0, 300, 600)
    params = SDTParams()
    model = SDTModel(params)
    y0 = [0.5, 0.05, 1.1, 1.0, 0.0]

    # Use try/except to handle potential convergence issues
    try:
        solution = odeint(model.system, y0, t)
    except Exception as e:
        print(f"  Warning: {e}")
        return

    # Extract elite and wage data
    elites = solution[:, 1]
    wages = solution[:, 2]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0.5, 1.3)
    ax.set_ylim(0, 0.25)
    ax.set_xlabel('Wages (W)', fontsize=14)
    ax.set_ylabel('Elite Population (E)', fontsize=14)
    ax.set_title('Phase Space Trajectory: The Secular Cycle', fontsize=16)
    ax.grid(True, alpha=0.3)

    # Initialize line and point
    line, = ax.plot([], [], 'b-', alpha=0.5, linewidth=1, label='Trajectory')
    point, = ax.plot([], [], 'ro', markersize=10, label='Current State')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

    ax.legend(loc='upper right')

    def init():
        line.set_data([], [])
        point.set_data([], [])
        time_text.set_text('')
        return line, point, time_text

    def animate(frame):
        # Show trajectory up to current frame
        idx = min(frame + 1, len(wages))
        line.set_data(wages[:idx], elites[:idx])
        point.set_data([wages[frame]], [elites[frame]])
        time_text.set_text(f'Time: {t[frame]:.0f} years')
        return line, point, time_text

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=range(0, len(t), 3),  # Skip frames for smaller file
        interval=50, blit=True
    )

    # Save as GIF
    output_path = OUTPUT_DIR / 'phase-space-trajectory.gif'
    anim.save(str(output_path), writer='pillow', fps=20, dpi=100)

    plt.close()
    print(f"  Saved to {output_path}")


def main():
    print("Generating animations for Essay 3...")
    print()

    generate_phase_space_animation()

    print()
    print("Animation generation complete!")


if __name__ == '__main__':
    main()
