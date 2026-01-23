#!/usr/bin/env python3
"""
Demonstrate calibration functionality for Essay #16.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cliodynamics.calibration import Calibrator, generate_synthetic_data
from cliodynamics.models import SDTModel, SDTParams

# Generate synthetic data with known parameters
true_params = SDTParams(
    r_max=0.025,
    K_0=1.0,
    alpha=0.006,
    gamma=2.5,
)

initial_conditions = {"N": 0.5, "E": 0.05, "W": 1.0, "S": 1.0, "psi": 0.0}

print("Generating synthetic data...")
# Generate data
data = generate_synthetic_data(
    true_params,
    initial_conditions,
    time_span=(0, 200),
    dt=10.0,  # Sparse observations every 10 years
    noise_std=0.02,
    variables=["N", "psi"],
    seed=42,
)

print("Synthetic data generated:")
print(data.head(10))

# Create calibrator
calibrator = Calibrator(
    model=SDTModel, observed_data=data, fit_variables=["N", "psi"], time_column="year"
)

print("\nRunning calibration...")
# Run calibration
result = calibrator.fit(
    param_bounds={
        "r_max": (0.01, 0.05),
        "alpha": (0.001, 0.01),
    },
    method="differential_evolution",
    seed=42,
    maxiter=100,
)

print()
print(result.summary())
print()
print("True r_max:", true_params.r_max)
print("True alpha:", true_params.alpha)
print("Recovery error r_max:", abs(result.best_params["r_max"] - true_params.r_max))
print("Recovery error alpha:", abs(result.best_params["alpha"] - true_params.alpha))
