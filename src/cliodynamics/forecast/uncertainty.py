"""Uncertainty quantification for cliodynamics forecasts.

This module provides tools for quantifying and propagating uncertainty
in forecasts, including:
- Parameter uncertainty from calibration
- Initial condition uncertainty
- Model structural uncertainty
- Ensemble generation and aggregation

The uncertainty framework supports both frequentist (bootstrap, ensemble)
and Bayesian (MCMC posterior) approaches.

Example:
    >>> from cliodynamics.forecast.uncertainty import (
    ...     UncertaintyQuantifier,
    ...     combine_uncertainties,
    ... )
    >>>
    >>> uq = UncertaintyQuantifier(
    ...     parameter_uncertainty=0.1,
    ...     initial_condition_uncertainty=0.05,
    ...     model_uncertainty=0.05
    ... )
    >>> total_uncertainty = uq.total_uncertainty(horizon_years=20)

References:
    Turchin, P. (2016). Ages of Discord. Beresta Books.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from cliodynamics.calibration import CalibrationResult


@dataclass
class UncertaintyEstimate:
    """Container for uncertainty estimates at a point in time.

    Attributes:
        parameter_cv: Coefficient of variation from parameter uncertainty.
        initial_condition_cv: CV from initial condition uncertainty.
        model_cv: CV from model structural uncertainty.
        total_cv: Combined total coefficient of variation.
        confidence_interval: (lower, upper) bounds at specified confidence.
        confidence_level: Confidence level for interval.
    """

    parameter_cv: float
    initial_condition_cv: float
    model_cv: float
    total_cv: float
    confidence_interval: tuple[float, float]
    confidence_level: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary.

        Returns:
            Dictionary with all uncertainty components.
        """
        return {
            "parameter_cv": self.parameter_cv,
            "initial_condition_cv": self.initial_condition_cv,
            "model_cv": self.model_cv,
            "total_cv": self.total_cv,
            "ci_lower": self.confidence_interval[0],
            "ci_upper": self.confidence_interval[1],
            "confidence_level": self.confidence_level,
        }


class UncertaintyQuantifier:
    """Quantify and propagate uncertainty in forecasts.

    The UncertaintyQuantifier tracks different sources of uncertainty
    and computes how they grow over the forecast horizon.

    Sources of uncertainty:
    1. **Parameter uncertainty**: Uncertainty in model parameters from
       calibration (can come from bootstrap or MCMC samples).
    2. **Initial condition uncertainty**: Uncertainty in the current
       state measurements.
    3. **Model uncertainty**: Structural uncertainty from model
       mis-specification or simplifying assumptions.

    Attributes:
        parameter_cv: Coefficient of variation for parameters.
        initial_condition_cv: CV for initial conditions.
        model_cv: CV for model uncertainty.
        growth_rate: Rate at which uncertainty grows with horizon.

    Example:
        >>> uq = UncertaintyQuantifier(
        ...     parameter_cv=0.1,
        ...     initial_condition_cv=0.05,
        ...     model_cv=0.03
        ... )
        >>> # Get uncertainty at 20 years
        >>> estimate = uq.estimate_uncertainty(
        ...     horizon_years=20,
        ...     mean_value=0.5
        ... )
        >>> print(f"Total CV: {estimate.total_cv:.2%}")
    """

    def __init__(
        self,
        parameter_cv: float = 0.1,
        initial_condition_cv: float = 0.05,
        model_cv: float = 0.03,
        growth_rate: float = 0.02,
    ) -> None:
        """Initialize the UncertaintyQuantifier.

        Args:
            parameter_cv: Coefficient of variation for parameter uncertainty.
            initial_condition_cv: CV for initial condition uncertainty.
            model_cv: CV for model structural uncertainty.
            growth_rate: Rate at which uncertainty grows per year.
                Captures chaotic divergence of trajectories.
        """
        self.parameter_cv = parameter_cv
        self.initial_condition_cv = initial_condition_cv
        self.model_cv = model_cv
        self.growth_rate = growth_rate

    @classmethod
    def from_calibration_result(
        cls,
        result: "CalibrationResult",
        initial_condition_cv: float = 0.05,
        model_cv: float = 0.03,
    ) -> "UncertaintyQuantifier":
        """Create UncertaintyQuantifier from calibration results.

        Extracts parameter uncertainty from bootstrap confidence intervals
        in the calibration result.

        Args:
            result: CalibrationResult with confidence intervals.
            initial_condition_cv: CV for initial conditions.
            model_cv: CV for model uncertainty.

        Returns:
            Configured UncertaintyQuantifier.
        """
        # Estimate parameter CV from confidence intervals
        if result.confidence_intervals:
            cvs = []
            for param_name, (lower, upper) in result.confidence_intervals.items():
                best = result.best_params.get(param_name)
                if best and best != 0:
                    # Estimate CV from CI width
                    width = upper - lower
                    # Assuming ~95% CI, width is ~4 standard deviations
                    std = width / 4
                    cv = std / abs(best)
                    cvs.append(cv)
            parameter_cv = float(np.mean(cvs)) if cvs else 0.1
        else:
            parameter_cv = 0.1

        return cls(
            parameter_cv=parameter_cv,
            initial_condition_cv=initial_condition_cv,
            model_cv=model_cv,
        )

    def total_cv(self, horizon_years: float = 0.0) -> float:
        """Compute total coefficient of variation.

        Combines all uncertainty sources, with uncertainty growing
        over the forecast horizon.

        Args:
            horizon_years: Years into the forecast.

        Returns:
            Total coefficient of variation.
        """
        # Base uncertainties (independent, so add in quadrature)
        base_cv_squared = (
            self.parameter_cv**2 + self.initial_condition_cv**2 + self.model_cv**2
        )

        # Uncertainty grows with forecast horizon
        # Using exponential growth model
        growth_factor = np.exp(self.growth_rate * horizon_years)

        return float(np.sqrt(base_cv_squared) * growth_factor)

    def estimate_uncertainty(
        self,
        horizon_years: float,
        mean_value: float,
        confidence_level: float = 0.90,
    ) -> UncertaintyEstimate:
        """Estimate uncertainty at a forecast horizon.

        Args:
            horizon_years: Years into the forecast.
            mean_value: Mean forecast value (for scaling).
            confidence_level: Confidence level for interval.

        Returns:
            UncertaintyEstimate with all components.
        """
        # Compute time-dependent CVs
        growth_factor = np.exp(self.growth_rate * horizon_years)

        param_cv = self.parameter_cv * growth_factor
        ic_cv = self.initial_condition_cv * growth_factor
        model_cv_t = self.model_cv * growth_factor

        total = self.total_cv(horizon_years)

        # Compute confidence interval assuming log-normal
        # For log-normal, CV is approximately sigma for small sigma
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence_level, 1.645)

        # Standard error
        se = total * abs(mean_value)

        lower = mean_value - z * se
        upper = mean_value + z * se

        return UncertaintyEstimate(
            parameter_cv=param_cv,
            initial_condition_cv=ic_cv,
            model_cv=model_cv_t,
            total_cv=total,
            confidence_interval=(lower, upper),
            confidence_level=confidence_level,
        )

    def uncertainty_profile(
        self,
        horizon_years: float,
        dt: float = 1.0,
        mean_trajectory: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Compute uncertainty profile over forecast horizon.

        Args:
            horizon_years: Total forecast horizon.
            dt: Time step.
            mean_trajectory: Optional mean forecast trajectory for scaling.
                If None, returns unitless CVs.

        Returns:
            Dictionary with time array and uncertainty components.
        """
        time = np.arange(0, horizon_years + dt, dt)
        n_times = len(time)

        # Compute time-dependent CVs
        growth_factors = np.exp(self.growth_rate * time)

        param_cv = self.parameter_cv * growth_factors
        ic_cv = self.initial_condition_cv * growth_factors
        model_cv_t = self.model_cv * growth_factors

        total_cv = np.sqrt(param_cv**2 + ic_cv**2 + model_cv_t**2)

        result = {
            "time": time,
            "parameter_cv": param_cv,
            "initial_condition_cv": ic_cv,
            "model_cv": model_cv_t,
            "total_cv": total_cv,
        }

        # Add scaled uncertainty bounds if trajectory provided
        if mean_trajectory is not None and len(mean_trajectory) == n_times:
            se = total_cv * np.abs(mean_trajectory)
            result["ci_lower"] = mean_trajectory - 1.645 * se
            result["ci_upper"] = mean_trajectory + 1.645 * se

        return result


def combine_uncertainties(*cvs: float) -> float:
    """Combine independent uncertainties in quadrature.

    For independent error sources, total variance is sum of variances.

    Args:
        *cvs: Coefficients of variation to combine.

    Returns:
        Combined coefficient of variation.

    Example:
        >>> total = combine_uncertainties(0.1, 0.05, 0.03)
        >>> print(f"Combined CV: {total:.3f}")
    """
    return float(np.sqrt(sum(cv**2 for cv in cvs)))


def compute_ensemble_statistics(
    ensemble: NDArray[np.float64],
    confidence_level: float = 0.90,
) -> dict[str, NDArray[np.float64]]:
    """Compute statistics from ensemble of trajectories.

    Args:
        ensemble: Ensemble trajectories (n_samples, n_times, n_vars)
            or (n_samples, n_times) for single variable.
        confidence_level: Confidence level for intervals.

    Returns:
        Dictionary with 'mean', 'std', 'ci_lower', 'ci_upper', 'cv'.
    """
    alpha = 1 - confidence_level
    lower_pct = alpha / 2 * 100
    upper_pct = (1 - alpha / 2) * 100

    mean = np.mean(ensemble, axis=0)
    std = np.std(ensemble, axis=0)
    ci_lower = np.percentile(ensemble, lower_pct, axis=0)
    ci_upper = np.percentile(ensemble, upper_pct, axis=0)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(mean != 0, std / np.abs(mean), 0.0)

    return {
        "mean": mean,
        "std": std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cv": cv,
    }


def generate_parameter_samples(
    base_params: dict[str, float],
    n_samples: int,
    cv: float = 0.1,
    seed: int | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> list[dict[str, float]]:
    """Generate parameter samples for ensemble forecasting.

    Samples parameters from log-normal distributions centered on
    base values with specified coefficient of variation.

    Args:
        base_params: Dictionary of base parameter values.
        n_samples: Number of samples to generate.
        cv: Coefficient of variation for sampling.
        seed: Random seed for reproducibility.
        bounds: Optional bounds {param_name: (min, max)} to clip samples.

    Returns:
        List of parameter dictionaries.

    Example:
        >>> base = {'r_max': 0.02, 'K_0': 1.0, 'mu': 0.2}
        >>> samples = generate_parameter_samples(base, n_samples=100, cv=0.1)
        >>> print(f"Generated {len(samples)} parameter sets")
    """
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(n_samples):
        sample = {}
        for name, value in base_params.items():
            if value > 0:
                # Log-normal sampling for positive parameters
                log_std = np.sqrt(np.log(1 + cv**2))
                log_mean = np.log(value) - log_std**2 / 2
                sampled = rng.lognormal(log_mean, log_std)
            else:
                # Normal sampling for zero or negative values
                sampled = rng.normal(value, cv * 0.1)

            # Apply bounds if provided
            if bounds and name in bounds:
                min_val, max_val = bounds[name]
                sampled = np.clip(sampled, min_val, max_val)

            sample[name] = float(sampled)
        samples.append(sample)

    return samples


def generate_initial_condition_samples(
    base_state: dict[str, float],
    n_samples: int,
    cv: float = 0.05,
    seed: int | None = None,
) -> list[dict[str, float]]:
    """Generate initial condition samples for ensemble forecasting.

    Args:
        base_state: Dictionary of base state values.
        n_samples: Number of samples to generate.
        cv: Coefficient of variation for sampling.
        seed: Random seed for reproducibility.

    Returns:
        List of state dictionaries.
    """
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(n_samples):
        sample = {}
        for name, value in base_state.items():
            if value > 0:
                # Log-normal for positive values
                log_std = np.sqrt(np.log(1 + cv**2))
                log_mean = np.log(value) - log_std**2 / 2
                sample[name] = float(rng.lognormal(log_mean, log_std))
            elif value == 0:
                # Half-normal for zero values (can only go up)
                sample[name] = float(abs(rng.normal(0, cv * 0.1)))
            else:
                # Normal for negative values
                sample[name] = float(rng.normal(value, abs(value) * cv))
        samples.append(sample)

    return samples


__all__ = [
    "UncertaintyQuantifier",
    "UncertaintyEstimate",
    "combine_uncertainties",
    "compute_ensemble_statistics",
    "generate_parameter_samples",
    "generate_initial_condition_samples",
]
