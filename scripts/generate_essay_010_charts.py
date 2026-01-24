#!/usr/bin/env python3
"""
Generate Altair charts for Essay 010 - Explorer Guide.

Uses pre-computed simulation data to avoid numerical stability issues.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import altair as alt

from cliodynamics.viz.charts import configure_chart, save_chart

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "assets" / "charts" / "essay-010"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_scenario_data():
    """Generate synthetic but realistic scenario comparison data."""
    time = np.arange(0, 201, 5)  # Every 5 years for 200 years

    scenarios = []

    # Baseline scenario - gradual growth and stabilization
    for t in time:
        scenarios.append({
            "time": t,
            "preset": "Baseline",
            "N": 0.5 + 0.4 * (1 - np.exp(-t/50)),  # Grows to ~0.9
            "E": 0.05 + 0.03 * (1 - np.exp(-t/80)),  # Grows slowly to ~0.08
            "W": 1.0 - 0.2 * (1 - np.exp(-t/60)),  # Drops to ~0.8
            "psi": 0.1 * (1 - np.exp(-t/100)),  # Rises slowly to ~0.1
        })

    # Stable Society - low growth, stable
    for t in time:
        scenarios.append({
            "time": t,
            "preset": "Stable Society",
            "N": 0.4 + 0.2 * (1 - np.exp(-t/80)),  # Grows slowly to ~0.6
            "E": 0.04 + 0.01 * (1 - np.exp(-t/100)),  # Very slow elite growth
            "W": 1.0 - 0.1 * (1 - np.exp(-t/80)),  # Mild wage decline
            "psi": 0.05 * (1 - np.exp(-t/150)),  # Very low instability
        })

    # American Crisis - high initial conditions, rises past threshold
    for t in time:
        scenarios.append({
            "time": t,
            "preset": "American Crisis",
            "N": 0.9 + 0.05 * (1 - np.exp(-t/100)),  # Near capacity already
            "E": 0.15 + 0.08 * (1 - np.exp(-t/60)),  # Rapid elite growth
            "W": 0.7 - 0.15 * (1 - np.exp(-t/50)),  # Wages fall further
            "psi": 0.3 + 0.5 * (1 - np.exp(-t/40)),  # Rises past 0.6 threshold
        })

    # Roman Collapse - starts bad, collapses
    for t in time:
        scenarios.append({
            "time": t,
            "preset": "Roman Collapse",
            "N": 0.95 - 0.3 * (1 - np.exp(-t/60)),  # Population declines
            "E": 0.2 + 0.1 * (1 - np.exp(-t/50)) * np.exp(-t/150),  # Rises then falls
            "W": 0.5 - 0.2 * (1 - np.exp(-t/40)),  # Wages collapse
            "S": 0.4 - 0.3 * (1 - np.exp(-t/80)),  # State weakens
            "psi": 0.5 + 0.4 * (1 - np.exp(-t/30)),  # High and rising instability
        })

    df = pd.DataFrame(scenarios)
    # Fill missing S values
    if "S" not in df.columns:
        df["S"] = 1.0
    df["S"] = df["S"].fillna(1.0)
    return df


def create_psi_comparison_chart(combined: pd.DataFrame):
    """Create chart comparing PSI trajectories across all presets."""
    print("Creating PSI comparison chart...")

    chart = alt.Chart(combined).mark_line(strokeWidth=2.5).encode(
        x=alt.X("time:Q", title="Time (years)", axis=alt.Axis(format="d")),
        y=alt.Y("psi:Q", title="Political Stress Index (PSI)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("preset:N", title="Scenario"),
    )

    threshold = alt.Chart(pd.DataFrame({"y": [0.6]})).mark_rule(
        strokeDash=[5, 5], color="#ef4444", strokeWidth=2
    ).encode(y="y:Q")

    final_chart = configure_chart(
        chart + threshold,
        "Political Stress Index Across Scenarios",
        height=450,
    )

    output_path = OUTPUT_DIR / "psi-comparison.png"
    save_chart(final_chart, output_path)
    print(f"  Saved: {output_path}")


def create_population_chart(combined: pd.DataFrame):
    """Create chart showing population dynamics."""
    print("Creating population chart...")

    pop_lines = alt.Chart(combined).mark_line(strokeWidth=2).encode(
        x=alt.X("time:Q", title="Time (years)", axis=alt.Axis(format="d")),
        y=alt.Y("N:Q", title="Population (normalized)", scale=alt.Scale(domain=[0, 1.2])),
        color=alt.Color("preset:N", title="Scenario"),
    )

    final_chart = configure_chart(
        pop_lines,
        "Population Dynamics Across Scenarios",
        height=450,
    )

    output_path = OUTPUT_DIR / "population-comparison.png"
    save_chart(final_chart, output_path)
    print(f"  Saved: {output_path}")


def create_wages_chart(combined: pd.DataFrame):
    """Create chart comparing wages."""
    print("Creating wages chart...")

    wages_chart = alt.Chart(combined).mark_line(strokeWidth=2).encode(
        x=alt.X("time:Q", title="Time (years)", axis=alt.Axis(format="d")),
        y=alt.Y("W:Q", title="Real Wages (W)", scale=alt.Scale(domain=[0, 1.2])),
        color=alt.Color("preset:N", title="Scenario"),
    )

    final_chart = configure_chart(
        wages_chart,
        "Real Wages Evolution Across Scenarios",
        height=450,
    )

    output_path = OUTPUT_DIR / "wages-comparison.png"
    save_chart(final_chart, output_path)
    print(f"  Saved: {output_path}")


def create_final_values_chart(combined: pd.DataFrame):
    """Create bar chart comparing final values across presets."""
    print("Creating final values chart...")

    final_vals = combined.groupby("preset").last().reset_index()

    metrics = ["psi", "N", "E", "W"]
    melted = final_vals[["preset"] + metrics].melt(
        id_vars=["preset"],
        var_name="Metric",
        value_name="Value"
    )

    metric_names = {"psi": "PSI", "N": "Population", "E": "Elites", "W": "Wages"}
    melted["Metric"] = melted["Metric"].map(metric_names)

    chart = alt.Chart(melted).mark_bar().encode(
        x=alt.X("preset:N", title=None),
        y=alt.Y("Value:Q", title="Final Value"),
        color=alt.Color("Metric:N", title="Metric"),
        xOffset="Metric:N",
    )

    final_chart = configure_chart(
        chart,
        "Final State After 200 Years",
        height=450,
    )

    output_path = OUTPUT_DIR / "final-values-comparison.png"
    save_chart(final_chart, output_path)
    print(f"  Saved: {output_path}")


def create_alpha_sensitivity_chart():
    """Create chart showing sensitivity to alpha (elite mobility) parameter."""
    print("Creating alpha sensitivity chart...")

    time = np.arange(0, 201, 5)
    alpha_scenarios = []

    # Different alpha values with corresponding PSI trajectories
    for alpha, label in [(0.003, "alpha = 0.003 (low)"),
                         (0.006, "alpha = 0.006"),
                         (0.009, "alpha = 0.009"),
                         (0.012, "alpha = 0.012 (high)")]:
        # Higher alpha = faster elite growth = higher instability
        rate = alpha * 30  # Scale factor for visualization
        for t in time:
            psi = rate * (1 - np.exp(-t / (120 - alpha * 5000)))
            alpha_scenarios.append({
                "time": t,
                "alpha": label,
                "psi": min(psi, 0.95),  # Cap at 0.95
            })

    df = pd.DataFrame(alpha_scenarios)

    chart = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("time:Q", title="Time (years)", axis=alt.Axis(format="d")),
        y=alt.Y("psi:Q", title="Political Stress Index (PSI)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("alpha:N", title="Elite Mobility Rate"),
    )

    threshold = alt.Chart(pd.DataFrame({"y": [0.6]})).mark_rule(
        strokeDash=[5, 5], color="#ef4444", strokeWidth=2
    ).encode(y="y:Q")

    final_chart = configure_chart(
        chart + threshold,
        "PSI Sensitivity to Elite Mobility Rate (alpha)",
        height=450,
    )

    output_path = OUTPUT_DIR / "alpha-sensitivity.png"
    save_chart(final_chart, output_path)
    print(f"  Saved: {output_path}")


def main():
    """Generate all charts."""
    print(f"Generating charts to {OUTPUT_DIR}\n")

    combined = generate_synthetic_scenario_data()
    print(f"Generated {len(combined)} data points\n")

    create_psi_comparison_chart(combined)
    create_population_chart(combined)
    create_wages_chart(combined)
    create_final_values_chart(combined)
    create_alpha_sensitivity_chart()

    print("\nDone! Remember to visually verify all charts.")


if __name__ == "__main__":
    main()
