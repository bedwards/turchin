#!/usr/bin/env python3
"""Example usage of Plotly animations for SDT model visualization.

This script demonstrates the various animation capabilities of the
cliodynamics.viz.plotly_animations module, showing how to create
interactive visualizations for structural-demographic theory analysis.

Examples include:
1. Time series animation with progressive reveal
2. 2D phase space trajectory animation
3. 3D phase space with camera orbit
4. Secular cycle animation with phase highlighting
5. Scenario comparison animation
6. Parameter sensitivity animation

All outputs are saved as interactive HTML files that can be viewed
in any web browser and embedded in GitHub Pages essays.

Usage:
    python scripts/examples/plotly_animation_examples.py

Output files are saved to: docs/assets/animations/examples/
"""

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd

from cliodynamics.viz import cycles, plotly_animations


def create_sample_simulation() -> pd.DataFrame:
    """Create sample SDT simulation data.

    Simulates a society going through demographic-structural cycles
    over a 300-year period with realistic dynamics.
    """
    t = np.linspace(0, 300, 601)

    # Population dynamics - secular cycle with ~150 year period
    N = 0.5 + 0.3 * np.sin(2 * np.pi * t / 150) + 0.05 * np.sin(2 * np.pi * t / 30)
    N = np.clip(N, 0.1, 1.0)

    # Elite population - follows population with lag
    E = 0.1 + 0.08 * np.sin(2 * np.pi * (t - 20) / 150)
    E = np.clip(E, 0.02, 0.25)

    # Real wages - inverse relationship with population
    W = 1.0 - 0.4 * np.sin(2 * np.pi * t / 150)
    W = np.clip(W, 0.3, 1.5)

    # State fiscal health - declines during crises
    S = 1.0 - 0.3 * np.sin(2 * np.pi * (t + 30) / 150)
    S = np.clip(S, 0.4, 1.2)

    # Political stress index - peaks during crises
    psi = 0.5 + 0.4 * np.sin(2 * np.pi * (t + 75) / 150) + 0.1 * np.random.randn(len(t))
    psi = np.clip(psi, 0.0, 1.0)

    return pd.DataFrame(
        {
            "t": t,
            "N": N,
            "E": E,
            "W": W,
            "S": S,
            "psi": psi,
        }
    )


def create_counterfactual_simulation() -> pd.DataFrame:
    """Create counterfactual scenario with policy intervention.

    Simulates what happens if reforms reduce elite overproduction
    and improve state fiscal health.
    """
    t = np.linspace(0, 300, 601)

    # Dampened oscillations due to policy intervention
    N = 0.5 + 0.2 * np.sin(2 * np.pi * t / 150)
    N = np.clip(N, 0.2, 0.9)

    E = 0.08 + 0.04 * np.sin(2 * np.pi * (t - 20) / 150)
    E = np.clip(E, 0.04, 0.15)

    W = 1.0 - 0.25 * np.sin(2 * np.pi * t / 150)
    W = np.clip(W, 0.5, 1.3)

    S = 1.0 - 0.15 * np.sin(2 * np.pi * (t + 30) / 150)
    S = np.clip(S, 0.7, 1.1)

    noise = 0.05 * np.random.randn(len(t))
    psi = 0.3 + 0.2 * np.sin(2 * np.pi * (t + 75) / 150) + noise
    psi = np.clip(psi, 0.0, 0.7)

    return pd.DataFrame(
        {
            "t": t,
            "N": N,
            "E": E,
            "W": W,
            "S": S,
            "psi": psi,
        }
    )


def main():
    """Generate all example animations."""
    # Create output directory
    output_dir = Path("docs/assets/animations/examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating example Plotly animations...")
    print(f"Output directory: {output_dir.absolute()}")
    print()

    # Generate sample data
    print("1. Creating sample simulation data...")
    df = create_sample_simulation()
    df_counter = create_counterfactual_simulation()

    # Example 1: Time series animation
    print("2. Creating time series animation...")
    fig = plotly_animations.animate_time_series(
        df,
        variables=["N", "W", "psi"],
        title="SDT Model Evolution - 300 Year Secular Cycle",
        duration_ms=12000,
        width=1100,
        height=600,
    )
    output_path = output_dir / "time_series.html"
    plotly_animations.save_animation(fig, output_path)
    print(f"   Saved: {output_path}")

    # Example 2: Time series with subplot layout
    print("3. Creating time series with subplots...")
    fig = plotly_animations.animate_time_series(
        df,
        variables=["N", "E", "W", "S", "psi"],
        title="All SDT Variables",
        subplot_layout=True,
        duration_ms=15000,
        width=1000,
        height=800,
    )
    output_path = output_dir / "time_series_subplots.html"
    plotly_animations.save_animation(fig, output_path)
    print(f"   Saved: {output_path}")

    # Example 3: 2D phase space
    print("4. Creating 2D phase space animation...")
    fig = plotly_animations.animate_phase_space(
        df,
        x="W",
        y="psi",
        trail_length=80,
        duration_ms=12000,
        title="Phase Space: Wages vs Political Stress",
        width=900,
        height=700,
        colorscale="Viridis",
    )
    output_path = output_dir / "phase_space_2d.html"
    plotly_animations.save_animation(fig, output_path)
    print(f"   Saved: {output_path}")

    # Example 4: 3D phase space with camera orbit
    print("5. Creating 3D phase space with camera orbit...")
    fig = plotly_animations.animate_phase_space_3d(
        df,
        x="N",
        y="W",
        z="psi",
        camera_orbit=True,
        trail_length=60,
        duration_ms=15000,
        title="3D Phase Space: Population, Wages, Stress",
        width=1000,
        height=800,
        colorscale="Plasma",
    )
    output_path = output_dir / "phase_space_3d.html"
    plotly_animations.save_animation(fig, output_path)
    print(f"   Saved: {output_path}")

    # Example 5: 3D phase space without orbit
    print("6. Creating 3D phase space (static camera)...")
    fig = plotly_animations.animate_phase_space_3d(
        df,
        x="E",
        y="S",
        z="psi",
        camera_orbit=False,
        trail_length=100,
        duration_ms=12000,
        title="Elite-State-Stress Phase Space",
        width=900,
        height=700,
    )
    output_path = output_dir / "phase_space_3d_static.html"
    plotly_animations.save_animation(fig, output_path)
    print(f"   Saved: {output_path}")

    # Example 6: Secular cycles animation
    print("7. Creating secular cycles animation...")
    detected = cycles.detect_secular_cycles(df["psi"], df["t"])
    fig = plotly_animations.animate_secular_cycles(
        df,
        detected,
        variable="psi",
        show_phase_transitions=True,
        duration_ms=15000,
        title="Political Stress Index with Secular Cycle Phases",
        width=1100,
        height=500,
    )
    output_path = output_dir / "secular_cycles.html"
    plotly_animations.save_animation(fig, output_path)
    print(f"   Saved: {output_path}")

    # Example 7: Scenario comparison (overlay)
    print("8. Creating scenario comparison (overlay)...")
    fig = plotly_animations.animate_comparison(
        df,
        df_counter,
        variables=["psi", "S"],
        labels=("Historical", "With Reform"),
        layout="overlay",
        duration_ms=12000,
        title="Policy Impact: Historical vs Reform Scenario",
        width=1000,
        height=600,
        colors=("#0072B2", "#D55E00"),
    )
    output_path = output_dir / "comparison_overlay.html"
    plotly_animations.save_animation(fig, output_path)
    print(f"   Saved: {output_path}")

    # Example 8: Scenario comparison (side by side)
    print("9. Creating scenario comparison (side-by-side)...")
    fig = plotly_animations.animate_comparison(
        df,
        df_counter,
        variables=["N", "psi"],
        labels=("Historical", "With Reform"),
        layout="side_by_side",
        duration_ms=12000,
        title="Side-by-Side Comparison",
        width=1200,
        height=500,
    )
    output_path = output_dir / "comparison_side_by_side.html"
    plotly_animations.save_animation(fig, output_path)
    print(f"   Saved: {output_path}")

    # Example 9: Parameter sensitivity
    print("10. Creating parameter sensitivity animation...")
    sensitivity_results = {}
    for r in [0.01, 0.015, 0.02, 0.025, 0.03]:
        t = np.linspace(0, 200, 401)
        # Faster/slower cycles based on r
        psi = 0.5 + (0.3 + r * 5) * np.sin(2 * np.pi * t / (100 / r / 0.02))
        psi = np.clip(psi, 0.0, 1.0)
        sensitivity_results[f"r={r:.3f}"] = pd.DataFrame({"t": t, "psi": psi})

    fig = plotly_animations.animate_parameter_sensitivity(
        sensitivity_results,
        variable="psi",
        parameter_name="Growth Rate (r)",
        duration_ms=10000,
        title="Parameter Sensitivity: Effect of Growth Rate on Instability",
        width=1000,
        height=600,
        colorscale="Viridis",
    )
    output_path = output_dir / "parameter_sensitivity.html"
    plotly_animations.save_animation(fig, output_path)
    print(f"   Saved: {output_path}")

    print()
    print("=" * 60)
    print("All animations generated successfully!")
    print("Open HTML files in a web browser to view interactive animations.")
    print("Files can be embedded in GitHub Pages essays.")
    print("=" * 60)


if __name__ == "__main__":
    main()
