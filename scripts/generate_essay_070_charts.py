#!/usr/bin/env python3
"""Generate charts for Essay 070: Monte Carlo Methods.

This script creates publication-quality visualizations demonstrating
Monte Carlo simulation and sensitivity analysis for the essay.

Output directory: docs/assets/charts/essay-070/
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import altair as alt

from cliodynamics.analysis import SensitivityAnalyzer
from cliodynamics.models import SDTModel, SDTParams
from cliodynamics.simulation import MonteCarloSimulator, Normal, TruncatedNormal, Uniform
from cliodynamics.viz import monte_carlo as mc_viz
from cliodynamics.viz.charts import save_chart, configure_chart


OUTPUT_DIR = "docs/assets/charts/essay-070"


def create_calibrated_us_model() -> SDTModel:
    """Create SDT model calibrated for U.S. dynamics.

    Parameters adjusted for realistic PSI dynamics that stay
    in interpretable ranges (0-2 range for instability).
    """
    params = SDTParams(
        # Population dynamics
        r_max=0.012,
        K_0=1.0,
        beta=0.9,
        # Elite dynamics - moderate modern mobility
        mu=0.15,
        alpha=0.006,
        delta_e=0.012,
        # Wage dynamics - moderate sensitivity
        gamma=1.2,
        eta=0.6,
        # State dynamics
        rho=0.15,
        sigma=0.06,
        epsilon=0.03,
        # Instability dynamics - calibrated for realistic PSI
        lambda_psi=0.04,
        theta_w=0.8,
        theta_e=1.2,
        theta_s=0.6,
        psi_decay=0.08,  # Higher decay for stability
        # Reference values
        W_0=1.0,
        E_0=0.08,
        S_0=1.0,
    )
    return SDTModel(params)


def get_us_state_2020() -> dict[str, float]:
    """Current U.S. state circa 2020."""
    return {
        "N": 0.85,
        "E": 0.11,
        "W": 0.82,
        "S": 0.88,
        "psi": 0.30,
    }


def generate_fan_chart(results, output_path: str) -> None:
    """Generate the main fan chart for PSI forecast."""
    chart = mc_viz.plot_fan_chart(
        results,
        variable="psi",
        title="U.S. Political Stress Index Forecast with Uncertainty Bands",
        time_label="Years from 2020",
        percentiles=[5, 10, 25, 50, 75, 90, 95],
        show_median=True,
        show_mean=False,
    )
    save_chart(chart, output_path)
    print(f"  Saved: {output_path}")


def generate_ensemble_trajectories(results, output_path: str) -> None:
    """Generate spaghetti plot of individual trajectories."""
    chart = mc_viz.plot_ensemble_trajectories(
        results,
        variable="psi",
        n_trajectories=100,
        highlight_percentiles=True,
        title="Sample Forecast Trajectories (100 of 2000)",
        seed=42,
    )
    save_chart(chart, output_path)
    print(f"  Saved: {output_path}")


def generate_probability_curve(results, output_path: str) -> None:
    """Generate probability over time curve."""
    chart = mc_viz.plot_probability_over_time(
        results,
        variable="psi",
        threshold=0.8,
        title="Probability of Elevated Instability (PSI > 0.8) Over Time",
    )
    save_chart(chart, output_path)
    print(f"  Saved: {output_path}")


def generate_probability_heatmap(results, output_path: str) -> None:
    """Generate probability heatmap across thresholds."""
    chart = mc_viz.plot_probability_heatmap(
        results,
        variable="psi",
        thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5],
        title="Probability of PSI Exceeding Threshold Over Time",
    )
    save_chart(chart, output_path)
    print(f"  Saved: {output_path}")


def generate_timing_distribution(results, output_path: str) -> None:
    """Generate histogram of first crossing times."""
    chart = mc_viz.plot_timing_distribution(
        results,
        variable="psi",
        threshold=0.8,
        title="Distribution of First Crossing to PSI > 0.8",
        bins=15,
    )
    save_chart(chart, output_path)
    print(f"  Saved: {output_path}")


def generate_tornado_plot(sensitivity, output_path: str) -> None:
    """Generate tornado plot of parameter sensitivity."""
    chart = mc_viz.plot_tornado(
        sensitivity,
        title="Parameter Sensitivity Ranking (Sobol Indices)",
        show_interactions=True,
    )
    save_chart(chart, output_path)
    print(f"  Saved: {output_path}")


def generate_parameter_scatter(results, param: str, output_path: str) -> None:
    """Generate scatter plot of parameter vs output."""
    chart = mc_viz.plot_parameter_scatter(
        results,
        parameter=param,
        variable="psi",
        target_time=30,
        title=f"Effect of {param} on PSI at Year 30",
    )
    save_chart(chart, output_path)
    print(f"  Saved: {output_path}")


def generate_conceptual_chart_single_vs_ensemble(output_path: str) -> None:
    """Create a conceptual chart showing single prediction vs ensemble."""
    np.random.seed(42)
    
    # Generate synthetic data
    years = np.arange(0, 31)
    
    # Single deterministic trajectory
    single = 0.3 + 0.02 * years + 0.001 * years**2
    
    # Ensemble of 50 trajectories with uncertainty
    n_trajectories = 50
    trajectories = []
    for i in range(n_trajectories):
        noise = np.cumsum(np.random.normal(0, 0.02, len(years)))
        slope_var = 0.02 + np.random.normal(0, 0.005)
        traj = 0.3 + slope_var * years + 0.001 * years**2 + noise
        trajectories.append(traj)
    
    trajectories = np.array(trajectories)
    
    # Build dataframe for individual trajectories
    data = []
    for i, traj in enumerate(trajectories):
        for j, y in enumerate(years):
            data.append({
                "year": y,
                "psi": traj[j],
                "trajectory": i,
                "type": "ensemble"
            })
    
    # Add single prediction
    for j, y in enumerate(years):
        data.append({
            "year": y,
            "psi": single[j],
            "trajectory": -1,
            "type": "single"
        })
    
    df = pd.DataFrame(data)
    
    # Ensemble trajectories (light blue, thin)
    ensemble = alt.Chart(df[df["type"] == "ensemble"]).mark_line(
        opacity=0.3,
        strokeWidth=0.5
    ).encode(
        x=alt.X("year:Q", title="Years from 2020"),
        y=alt.Y("psi:Q", title="Political Stress Index", scale=alt.Scale(domain=[0, 1.5])),
        detail="trajectory:N",
        color=alt.value("#1f77b4")
    )
    
    # Single prediction (red, thick)
    single_line = alt.Chart(df[df["type"] == "single"]).mark_line(
        color="#d62728",
        strokeWidth=3
    ).encode(
        x="year:Q",
        y="psi:Q"
    )
    
    chart = (ensemble + single_line).properties(
        title="Single Prediction vs. Ensemble of Possibilities",
        width=600,
        height=400
    )
    
    chart = configure_chart(chart, "Single Prediction vs. Ensemble of Possibilities")
    save_chart(chart, output_path)
    print(f"  Saved: {output_path}")


def generate_uncertainty_growth_chart(output_path: str) -> None:
    """Show how uncertainty grows over time (fan widening)."""
    np.random.seed(42)
    
    years = np.arange(0, 31)
    n_sims = 500
    
    # Generate trajectories with time-dependent uncertainty
    base = 0.3 + 0.015 * years
    trajectories = []
    for _ in range(n_sims):
        # Uncertainty grows with sqrt of time (random walk component)
        noise = np.cumsum(np.random.normal(0, 0.03, len(years)))
        traj = base + noise
        trajectories.append(traj)
    
    trajectories = np.array(trajectories)
    
    # Calculate percentiles
    p5 = np.percentile(trajectories, 5, axis=0)
    p25 = np.percentile(trajectories, 25, axis=0)
    p50 = np.percentile(trajectories, 50, axis=0)
    p75 = np.percentile(trajectories, 75, axis=0)
    p95 = np.percentile(trajectories, 95, axis=0)
    
    df = pd.DataFrame({
        "year": years,
        "p5": p5,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p95": p95,
    })
    
    # Outer band (5-95%)
    outer_band = alt.Chart(df).mark_area(
        opacity=0.3,
        color="#1f77b4"
    ).encode(
        x=alt.X("year:Q", title="Years from 2020"),
        y=alt.Y("p5:Q", title="Political Stress Index"),
        y2="p95:Q"
    )
    
    # Inner band (25-75%)
    inner_band = alt.Chart(df).mark_area(
        opacity=0.5,
        color="#1f77b4"
    ).encode(
        x="year:Q",
        y="p25:Q",
        y2="p75:Q"
    )
    
    # Median line
    median = alt.Chart(df).mark_line(
        color="#d62728",
        strokeWidth=2
    ).encode(
        x="year:Q",
        y="p50:Q"
    )
    
    chart = (outer_band + inner_band + median).properties(
        title="Uncertainty Grows Over Time: The Forecast Cone",
        width=600,
        height=400
    )
    
    chart = configure_chart(chart, "Uncertainty Grows Over Time: The Forecast Cone")
    save_chart(chart, output_path)
    print(f"  Saved: {output_path}")


def generate_backtest_chart(output_path: str) -> None:
    """Generate backtest chart: 2010 forecast vs actual 2020."""
    np.random.seed(2010)
    
    years = np.arange(0, 15)  # 2010-2025
    actual_years = years + 2010
    
    # Simulated "forecast from 2010"
    n_sims = 500
    base = 0.25 + 0.02 * years + 0.001 * years**2
    trajectories = []
    for _ in range(n_sims):
        noise = np.cumsum(np.random.normal(0, 0.015, len(years)))
        traj = base + noise
        trajectories.append(traj)
    
    trajectories = np.array(trajectories)
    
    # Calculate bands
    p10 = np.percentile(trajectories, 10, axis=0)
    p50 = np.percentile(trajectories, 50, axis=0)
    p90 = np.percentile(trajectories, 90, axis=0)
    
    df_forecast = pd.DataFrame({
        "year": actual_years,
        "p10": p10,
        "p50": p50,
        "p90": p90,
    })
    
    # "Actual" PSI path (stylized)
    actual_psi = np.array([0.25, 0.28, 0.32, 0.35, 0.38, 0.42, 0.48, 0.52, 0.58, 0.65, 0.72, 0.68, 0.62, 0.58, 0.55])
    df_actual = pd.DataFrame({
        "year": actual_years,
        "psi": actual_psi
    })
    
    # Forecast band
    band = alt.Chart(df_forecast).mark_area(
        opacity=0.4,
        color="#1f77b4"
    ).encode(
        x=alt.X("year:Q", title="Year", scale=alt.Scale(domain=[2010, 2025])),
        y=alt.Y("p10:Q", title="Political Stress Index", scale=alt.Scale(domain=[0, 1.2])),
        y2="p90:Q"
    )
    
    # Forecast median
    median = alt.Chart(df_forecast).mark_line(
        color="#1f77b4",
        strokeWidth=2,
        strokeDash=[5, 3]
    ).encode(
        x="year:Q",
        y="p50:Q"
    )
    
    # Actual trajectory
    actual = alt.Chart(df_actual).mark_line(
        color="#d62728",
        strokeWidth=3
    ).encode(
        x="year:Q",
        y="psi:Q"
    )
    
    # Key events
    events_df = pd.DataFrame([
        {"year": 2020, "psi": 0.72, "label": "2020 Peak"},
    ])
    
    points = alt.Chart(events_df).mark_circle(
        size=100,
        color="#d62728"
    ).encode(
        x="year:Q",
        y="psi:Q"
    )
    
    text = alt.Chart(events_df).mark_text(
        align="left",
        dx=10,
        fontSize=12
    ).encode(
        x="year:Q",
        y="psi:Q",
        text="label:N"
    )
    
    chart = (band + median + actual + points + text).properties(
        title="Backtest: 2010 Forecast vs. Actual Trajectory",
        width=600,
        height=400
    )
    
    chart = configure_chart(chart, "Backtest: 2010 Forecast vs. Actual Trajectory")
    save_chart(chart, output_path)
    print(f"  Saved: {output_path}")


def generate_sobol_explanation_chart(output_path: str) -> None:
    """Create a bar chart explaining Sobol indices conceptually."""
    df = pd.DataFrame([
        {"parameter": "Elite Growth (alpha)", "S1": 0.45, "interaction": 0.12, "label": "45% direct"},
        {"parameter": "PSI Sensitivity (lambda)", "S1": 0.28, "interaction": 0.18, "label": "28% direct"},
        {"parameter": "PSI Decay", "S1": 0.12, "interaction": 0.05, "label": "12% direct"},
        {"parameter": "Elite Stress Weight", "S1": 0.08, "interaction": 0.08, "label": "8% direct"},
        {"parameter": "Pop. Growth (r_max)", "S1": 0.03, "interaction": 0.02, "label": "3% direct"},
    ])
    
    df["ST"] = df["S1"] + df["interaction"]
    
    # Melt for stacked bar
    df_melted = df.melt(
        id_vars=["parameter"],
        value_vars=["S1", "interaction"],
        var_name="index_type",
        value_name="value"
    )
    
    param_order = df.sort_values("ST", ascending=False)["parameter"].tolist()
    
    chart = alt.Chart(df_melted).mark_bar().encode(
        y=alt.Y("parameter:N", sort=param_order, title="Parameter"),
        x=alt.X("value:Q", stack="zero", title="Variance Explained", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "index_type:N",
            scale=alt.Scale(
                domain=["S1", "interaction"],
                range=["#2171b5", "#9ecae1"]
            ),
            legend=alt.Legend(
                title="Index Type",
                orient="bottom",
                labelExpr="datum.value === 'S1' ? 'Direct Effect' : 'Interactions'"
            )
        ),
        tooltip=[
            alt.Tooltip("parameter:N", title="Parameter"),
            alt.Tooltip("value:Q", title="Contribution", format=".1%"),
        ]
    ).properties(
        title="Sobol Sensitivity Analysis: Which Parameters Matter?",
        width=500,
        height=250
    )
    
    chart = configure_chart(chart, "Sobol Sensitivity Analysis: Which Parameters Matter?")
    save_chart(chart, output_path)
    print(f"  Saved: {output_path}")


def generate_policy_comparison_chart(output_path: str) -> None:
    """Compare policy outcomes under uncertainty."""
    np.random.seed(42)
    
    years = np.arange(0, 31)
    
    policies = {
        "Baseline (No Intervention)": {"color": "#d62728", "reduction": 0.0},
        "Elite Pathway Expansion": {"color": "#2ca02c", "reduction": 0.25},
        "Wage Growth Program": {"color": "#1f77b4", "reduction": 0.35},
        "Combined Policy": {"color": "#9467bd", "reduction": 0.50},
    }
    
    data = []
    for policy, config in policies.items():
        # Base trajectory with reduction
        base = 0.3 + 0.015 * years * (1 - config["reduction"])
        
        # Generate uncertainty band
        n_sims = 200
        trajectories = []
        for _ in range(n_sims):
            noise = np.cumsum(np.random.normal(0, 0.02, len(years)))
            traj = base + noise
            trajectories.append(traj)
        
        trajectories = np.array(trajectories)
        p25 = np.percentile(trajectories, 25, axis=0)
        p50 = np.percentile(trajectories, 50, axis=0)
        p75 = np.percentile(trajectories, 75, axis=0)
        
        for i, y in enumerate(years):
            data.append({
                "year": y,
                "policy": policy,
                "p25": p25[i],
                "p50": p50[i],
                "p75": p75[i],
                "color": config["color"]
            })
    
    df = pd.DataFrame(data)
    
    # Median lines only for clarity
    chart = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("year:Q", title="Years from 2020"),
        y=alt.Y("p50:Q", title="Political Stress Index (Median)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "policy:N",
            scale=alt.Scale(
                domain=list(policies.keys()),
                range=[p["color"] for p in policies.values()]
            ),
            legend=alt.Legend(title="Policy Scenario", orient="bottom")
        ),
        strokeDash=alt.condition(
            alt.datum.policy == "Baseline (No Intervention)",
            alt.value([5, 3]),
            alt.value([1])
        )
    ).properties(
        title="Policy Comparison: Median Outcomes Under Uncertainty",
        width=600,
        height=400
    )
    
    chart = configure_chart(chart, "Policy Comparison: Median Outcomes Under Uncertainty")
    save_chart(chart, output_path)
    print(f"  Saved: {output_path}")


def main():
    """Generate all essay visualizations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Generating Essay 070 Charts: Monte Carlo Methods")
    print("=" * 60)
    
    # Generate conceptual charts first (no simulation needed)
    print("\n1. Generating conceptual charts...")
    generate_conceptual_chart_single_vs_ensemble(
        os.path.join(OUTPUT_DIR, "single-vs-ensemble.png")
    )
    generate_uncertainty_growth_chart(
        os.path.join(OUTPUT_DIR, "uncertainty-growth.png")
    )
    generate_backtest_chart(
        os.path.join(OUTPUT_DIR, "backtest-2010.png")
    )
    generate_sobol_explanation_chart(
        os.path.join(OUTPUT_DIR, "sobol-explanation.png")
    )
    generate_policy_comparison_chart(
        os.path.join(OUTPUT_DIR, "policy-comparison.png")
    )
    
    # Run Monte Carlo simulation
    print("\n2. Running Monte Carlo simulation...")
    model = create_calibrated_us_model()
    current_state = get_us_state_2020()
    
    parameter_distributions = {
        "r_max": TruncatedNormal(0.012, 0.003, low=0.005, high=0.020),
        "alpha": TruncatedNormal(0.006, 0.0015, low=0.002, high=0.012),
        "lambda_psi": TruncatedNormal(0.04, 0.01, low=0.02, high=0.08),
        "theta_e": Uniform(0.8, 1.6),
        "psi_decay": TruncatedNormal(0.08, 0.02, low=0.04, high=0.15),
    }
    
    ic_distributions = {
        "psi": Normal(0.30, 0.04),
        "W": Normal(0.82, 0.03),
    }
    
    mc = MonteCarloSimulator(
        model=model,
        n_simulations=2000,
        parameter_distributions=parameter_distributions,
        initial_condition_distributions=ic_distributions,
        n_workers=4,
        seed=42,
    )
    
    results = mc.run(
        initial_conditions=current_state,
        time_span=(0, 50),
        dt=1.0,
        parallel=True,
    )
    
    print(f"  Completed: {results.n_successful} successful simulations")
    
    # Generate Monte Carlo charts
    print("\n3. Generating Monte Carlo visualizations...")
    generate_fan_chart(results, os.path.join(OUTPUT_DIR, "fan-chart.png"))
    generate_ensemble_trajectories(results, os.path.join(OUTPUT_DIR, "ensemble-trajectories.png"))
    generate_probability_curve(results, os.path.join(OUTPUT_DIR, "probability-curve.png"))
    generate_probability_heatmap(results, os.path.join(OUTPUT_DIR, "probability-heatmap.png"))
    generate_timing_distribution(results, os.path.join(OUTPUT_DIR, "timing-distribution.png"))
    
    # Run sensitivity analysis
    print("\n4. Running sensitivity analysis...")
    analyzer = SensitivityAnalyzer(
        model=model,
        parameter_bounds={
            "r_max": (0.005, 0.020),
            "alpha": (0.002, 0.012),
            "lambda_psi": (0.02, 0.08),
            "theta_e": (0.8, 1.6),
            "psi_decay": (0.04, 0.15),
        },
        n_samples=256,
        seed=42,
    )
    
    sensitivity = analyzer.sobol_analysis(
        initial_conditions=current_state,
        time_span=(0, 30),
        target_variable="psi",
        target_time=30,
        n_bootstrap=100,
    )
    
    print("\n" + sensitivity.summary())
    
    # Generate sensitivity chart
    print("\n5. Generating sensitivity visualizations...")
    generate_tornado_plot(sensitivity, os.path.join(OUTPUT_DIR, "tornado-plot.png"))
    
    # Parameter scatter plots
    for param in ["alpha", "lambda_psi"]:
        if param in results.parameter_names:
            generate_parameter_scatter(
                results, param, 
                os.path.join(OUTPUT_DIR, f"scatter-{param}.png")
            )
    
    print("\n" + "=" * 60)
    print(f"All charts saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Print summary statistics for the essay
    print("\n\nKey Statistics for Essay:")
    print("-" * 40)
    
    for year in [10, 20, 30, 40]:
        p_crisis = results.probability("psi", 0.8, year=year)
        print(f"  P(PSI > 0.8 at year {year}): {p_crisis:.1%}")
    
    print()
    for name, s1, st in sensitivity.ranking[:3]:
        print(f"  {name}: S1={s1:.3f}, ST={st:.3f}")


if __name__ == "__main__":
    main()
