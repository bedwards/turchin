#!/usr/bin/env python3
"""Example: Monte Carlo forecast for U.S. Political Stress Index.

This script demonstrates the Monte Carlo simulation framework by generating
probabilistic forecasts for the United States based on Turchin's Ages of
Discord model.

Key questions answered:
1. What's the probability of major instability by 2030? 2040?
2. Which model parameters matter most for predictions?
3. How uncertain are our forecasts?

Output:
- Fan chart showing PSI forecast with confidence bands
- Probability over time of exceeding crisis thresholds
- Tornado plot of parameter sensitivity
- Summary statistics

Usage:
    python scripts/examples/monte_carlo_us_forecast.py

References:
    Turchin, P. (2016). Ages of Discord. Beresta Books.
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

from cliodynamics.analysis import SensitivityAnalyzer
from cliodynamics.models import SDTModel, SDTParams
from cliodynamics.simulation import MonteCarloSimulator, Normal, TruncatedNormal, Uniform
from cliodynamics.viz import monte_carlo as mc_viz
from cliodynamics.viz.charts import save_chart


def create_us_model() -> SDTModel:
    """Create SDT model calibrated for U.S. dynamics.

    Parameters are based on Ages of Discord Chapter 8, calibrated to
    reproduce the observed pattern of secular cycles in U.S. history.

    Returns:
        SDTModel with U.S.-calibrated parameters.
    """
    params = SDTParams(
        # Population dynamics - slower modern growth
        r_max=0.015,
        K_0=1.0,
        beta=0.8,
        # Elite dynamics - high modern mobility
        mu=0.25,
        alpha=0.008,
        delta_e=0.015,
        # Wage dynamics - moderate sensitivity
        gamma=1.5,
        eta=0.8,
        # State dynamics
        rho=0.18,
        sigma=0.08,
        epsilon=0.04,
        # Instability dynamics - key for PSI
        lambda_psi=0.06,
        theta_w=1.2,  # Wage stress weight
        theta_e=1.5,  # Elite stress weight (high for U.S.)
        theta_s=0.8,  # State stress weight
        psi_decay=0.025,
        # Reference values (normalized)
        W_0=1.0,
        E_0=0.08,
        S_0=1.0,
    )
    return SDTModel(params)


def get_current_us_state() -> dict[str, float]:
    """Estimate current U.S. state variables (circa 2020).

    Based on data from Ages of Discord and subsequent updates.
    Values are normalized relative to historical baselines.

    Returns:
        Dict of current state variable estimates.
    """
    return {
        "N": 0.85,  # Population near carrying capacity
        "E": 0.12,  # Elevated elite fraction (lawyer/PhD overproduction)
        "W": 0.78,  # Real wages below 1960s peak
        "S": 0.85,  # State fiscal stress (debt, polarization)
        "psi": 0.35,  # Elevated PSI (post-2020 unrest)
    }


def run_monte_carlo_forecast(
    n_simulations: int = 1000,
    forecast_years: int = 50,
    output_dir: str = "output",
) -> None:
    """Run Monte Carlo forecast and generate visualizations.

    Args:
        n_simulations: Number of Monte Carlo samples.
        forecast_years: Years to forecast.
        output_dir: Directory for output files.
    """
    print("=" * 60)
    print("Monte Carlo Forecast for U.S. Political Stress Index")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup model
    model = create_us_model()
    current_state = get_current_us_state()

    print(f"\nCurrent State (circa 2020):")
    for var, value in current_state.items():
        print(f"  {var}: {value:.3f}")

    # Define parameter uncertainty distributions
    # Based on calibration uncertainty from Ages of Discord
    parameter_distributions = {
        "r_max": TruncatedNormal(0.015, 0.003, low=0.005, high=0.025),
        "alpha": TruncatedNormal(0.008, 0.002, low=0.003, high=0.015),
        "lambda_psi": TruncatedNormal(0.06, 0.015, low=0.02, high=0.12),
        "theta_e": Uniform(1.0, 2.0),
        "psi_decay": TruncatedNormal(0.025, 0.008, low=0.01, high=0.05),
    }

    # Initial condition uncertainty
    ic_distributions = {
        "psi": Normal(0.35, 0.05),  # Uncertainty in current PSI
        "W": Normal(0.78, 0.03),  # Uncertainty in wage measurement
    }

    print(f"\nRunning {n_simulations} Monte Carlo simulations...")

    mc = MonteCarloSimulator(
        model=model,
        n_simulations=n_simulations,
        parameter_distributions=parameter_distributions,
        initial_condition_distributions=ic_distributions,
        n_workers=4,
        seed=42,
    )

    results = mc.run(
        initial_conditions=current_state,
        time_span=(0, forecast_years),
        dt=1.0,
        parallel=True,
    )

    print(f"  Completed: {results.n_successful} successful simulations")
    print(f"  Failed: {results.failed_simulations} simulations")

    # Print summary statistics
    print("\n" + results.summary())

    # Calculate key probabilities
    print("\n" + "=" * 60)
    print("Crisis Probability Analysis")
    print("=" * 60)

    thresholds = [0.5, 1.0, 1.5]
    years = [10, 20, 30, 40, 50]

    print("\nP(PSI > threshold) by year:")
    print("-" * 50)
    print(f"{'Threshold':<12}", end="")
    for y in years:
        print(f"Year {y:<5}", end="")
    print()
    print("-" * 50)

    for thresh in thresholds:
        print(f"PSI > {thresh:<5.1f}", end="")
        for y in years:
            prob = results.probability("psi", thresh, year=y)
            print(f"{prob:>8.1%}", end="")
        print()

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Fan chart
    print("  - Fan chart (PSI forecast with uncertainty)...")
    fan_chart = mc_viz.plot_fan_chart(
        results,
        variable="psi",
        title="U.S. Political Stress Index Forecast (Monte Carlo)",
        time_label="Years from 2020",
        show_median=True,
        show_mean=True,
    )
    save_chart(fan_chart, os.path.join(output_dir, "us_psi_fan_chart.png"))

    # 2. Probability over time
    print("  - Probability curves...")
    prob_chart = mc_viz.plot_probability_over_time(
        results,
        variable="psi",
        threshold=1.0,
        title="Probability of Major Instability (PSI > 1.0)",
    )
    save_chart(prob_chart, os.path.join(output_dir, "us_crisis_probability.png"))

    # 3. Probability heatmap
    print("  - Probability heatmap...")
    heatmap = mc_viz.plot_probability_heatmap(
        results,
        variable="psi",
        thresholds=[0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0],
        title="P(PSI > threshold) Over Time",
    )
    save_chart(heatmap, os.path.join(output_dir, "us_probability_heatmap.png"))

    # 4. Ensemble trajectories
    print("  - Sample trajectories...")
    ensemble_chart = mc_viz.plot_ensemble_trajectories(
        results,
        variable="psi",
        n_trajectories=50,
        highlight_percentiles=True,
        title="Sample PSI Trajectories (50 of 1000)",
    )
    save_chart(ensemble_chart, os.path.join(output_dir, "us_ensemble_trajectories.png"))

    # 5. First crossing time distribution
    print("  - Crisis timing distribution...")
    timing_chart = mc_viz.plot_timing_distribution(
        results,
        variable="psi",
        threshold=1.0,
        title="When Does PSI First Exceed 1.0?",
    )
    save_chart(timing_chart, os.path.join(output_dir, "us_crisis_timing.png"))

    # 6. Sensitivity analysis
    print("\nRunning sensitivity analysis...")
    analyzer = SensitivityAnalyzer(
        model=model,
        parameter_bounds={
            "r_max": (0.005, 0.025),
            "alpha": (0.003, 0.015),
            "lambda_psi": (0.02, 0.12),
            "theta_e": (1.0, 2.0),
            "psi_decay": (0.01, 0.05),
        },
        n_samples=128,
        seed=42,
    )

    sensitivity = analyzer.sobol_analysis(
        initial_conditions=current_state,
        time_span=(0, 30),
        target_variable="psi",
        target_time=30,
        n_bootstrap=50,
    )

    print("\n" + sensitivity.summary())

    # 7. Tornado plot
    print("\n  - Tornado plot (parameter sensitivity)...")
    tornado = mc_viz.plot_tornado(
        sensitivity,
        title="Which Parameters Drive PSI Uncertainty?",
        show_interactions=True,
    )
    save_chart(tornado, os.path.join(output_dir, "us_sensitivity_tornado.png"))

    # 8. Parameter scatter plots
    print("  - Parameter scatter plots...")
    for param in ["lambda_psi", "theta_e"]:
        if param in results.parameter_names:
            scatter = mc_viz.plot_parameter_scatter(
                results,
                parameter=param,
                variable="psi",
                target_time=30,
                title=f"{param} vs PSI at Year 30",
            )
            save_chart(
                scatter, os.path.join(output_dir, f"us_scatter_{param}.png")
            )

    print("\n" + "=" * 60)
    print("Output files saved to:", output_dir)
    print("=" * 60)

    # Return key results for programmatic use
    return {
        "results": results,
        "sensitivity": sensitivity,
        "p_crisis_2030": results.probability("psi", 1.0, year=10),
        "p_crisis_2050": results.probability("psi", 1.0, year=30),
    }


if __name__ == "__main__":
    # Run with fewer simulations for quick testing
    # Increase to 10000 for production-quality results
    run_monte_carlo_forecast(
        n_simulations=1000,
        forecast_years=50,
        output_dir="output/monte_carlo_us",
    )
