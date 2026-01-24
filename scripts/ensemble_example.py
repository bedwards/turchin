#!/usr/bin/env python3
"""Example script demonstrating the ensemble simulation system.

This script shows how to:
1. Set up parameter grids for exploration
2. Run ensemble simulations
3. Analyze stability regions
4. Detect bifurcations
5. Generate visualizations

Run with:
    python scripts/ensemble_example.py
"""

import numpy as np

from cliodynamics.models import SDTModel, SDTParams
from cliodynamics.simulation import EnsembleSimulator
from cliodynamics.viz import ensemble as ens_viz


def main() -> None:
    """Run ensemble example."""
    print("=" * 60)
    print("Ensemble Simulation Example: Parameter Space Exploration")
    print("=" * 60)

    # Create model with base parameters
    base_params = SDTParams(
        r_max=0.02,
        K_0=1.0,
        beta=1.0,
        mu=0.2,
        alpha=0.005,  # Will vary this
        delta_e=0.02,
        gamma=0.01,  # Will vary this
        eta=0.01,
        rho=0.2,
        sigma=0.1,
        epsilon=0.05,
        lambda_psi=0.05,
        theta_w=1.0,
        theta_e=1.0,
        theta_s=1.0,
        psi_decay=0.02,
        W_0=1.0,
        E_0=0.1,
        S_0=1.0,
    )

    model = SDTModel(base_params)

    # Define parameter grid
    # We explore:
    # - alpha (elite recruitment): 0.003 to 0.012
    # - lambda_psi (instability growth): 0.03 to 0.08
    print("\n1. Setting up parameter grid...")
    print("   - alpha (elite recruitment): 0.003 to 0.012 (10 points)")
    print("   - lambda_psi (instability growth): 0.03 to 0.08 (10 points)")
    print(f"   - Total simulations: {10 * 10} = 100")

    ensemble = EnsembleSimulator(
        model=model,
        parameter_grid={
            "alpha": np.linspace(0.003, 0.012, 10),
            "lambda_psi": np.linspace(0.03, 0.08, 10),
        },
        n_workers=4,  # Use 4 parallel workers
    )

    # Define initial conditions
    initial_conditions = {
        "N": 0.5,  # Population at half carrying capacity
        "E": 0.05,  # Small elite fraction
        "W": 1.0,  # Wages at baseline
        "S": 1.0,  # State health at baseline
        "psi": 0.0,  # No initial instability
    }

    # Run ensemble
    print("\n2. Running ensemble simulations...")
    results = ensemble.run(
        initial_conditions=initial_conditions,
        time_span=(0, 200),  # 200 years
        dt=1.0,
        parallel=True,
        psi_threshold=1.0,
        collapse_threshold=5.0,
        show_progress=True,
    )

    # Print summary
    print("\n3. Results Summary")
    print("-" * 40)
    print(results.summary())

    # Analyze stability
    print("\n4. Stability Analysis")
    print("-" * 40)
    stable_fraction = results.stability_region_area("stable", psi_threshold=1.0)
    unstable_fraction = results.stability_region_area("unstable", psi_threshold=1.0)
    collapse_fraction = results.stability_region_area("collapse", psi_threshold=1.0)

    print(f"   Stable region: {stable_fraction:.1%} of parameter space")
    print(f"   Unstable region: {unstable_fraction:.1%} of parameter space")
    print(f"   Collapse region: {collapse_fraction:.1%} of parameter space")

    # Find bifurcations
    print("\n5. Bifurcation Analysis")
    print("-" * 40)

    for param in ["alpha", "lambda_psi"]:
        bifurcations = results.find_bifurcation(
            parameter=param,
            psi_threshold=1.0,
        )
        if bifurcations:
            print(f"   {param}:")
            for bif in bifurcations:
                print(f"      - Bifurcation at {param} = {bif.value:.4f} ({bif.direction})")
        else:
            print(f"   {param}: No bifurcations detected")

    # Generate visualizations
    print("\n6. Generating Visualizations...")
    print("-" * 40)
    output_dir = "output/ensemble_example"

    import os

    os.makedirs(output_dir, exist_ok=True)

    # Phase diagram
    print("   - Phase diagram...")
    chart = ens_viz.plot_phase_diagram(
        results,
        x_param="alpha",
        y_param="lambda_psi",
        metric="max_psi",
        title="Stability Phase Diagram",
    )
    path = f"{output_dir}/phase_diagram.png"
    ens_viz.save_chart(chart, path)
    print(f"     Saved: {path}")

    # Stability map
    print("   - Stability map...")
    chart = ens_viz.plot_stability_map(
        results,
        x_param="alpha",
        y_param="lambda_psi",
        psi_threshold=1.0,
        title="Stability Classification Map",
    )
    path = f"{output_dir}/stability_map.png"
    ens_viz.save_chart(chart, path)
    print(f"     Saved: {path}")

    # Bifurcation diagrams
    print("   - Bifurcation diagrams...")
    for param in ["alpha", "lambda_psi"]:
        chart = ens_viz.plot_bifurcation_diagram(
            results,
            parameter=param,
            metric="max_psi",
            psi_threshold=1.0,
            title=f"Bifurcation Diagram: {param}",
        )
        path = f"{output_dir}/bifurcation_{param}.png"
        ens_viz.save_chart(chart, path)
        print(f"     Saved: {path}")

    # Stability fraction
    print("   - Stability fraction plots...")
    for param in ["alpha", "lambda_psi"]:
        chart = ens_viz.plot_stability_fraction(
            results,
            parameter=param,
            psi_threshold=1.0,
        )
        path = f"{output_dir}/stability_fraction_{param}.png"
        ens_viz.save_chart(chart, path)
        print(f"     Saved: {path}")

    # Outcome distribution
    print("   - Outcome distribution...")
    chart = ens_viz.plot_outcome_distribution(
        results,
        metric="max_psi",
        title="Distribution of Maximum PSI",
    )
    path = f"{output_dir}/outcome_distribution.png"
    ens_viz.save_chart(chart, path)
    print(f"     Saved: {path}")

    print("\n" + "=" * 60)
    print("Example complete! Check output/ensemble_example/ for visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    main()
