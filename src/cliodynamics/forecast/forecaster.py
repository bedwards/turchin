"""Instability forecasting pipeline for cliodynamics models.

This module provides the core Forecaster class for generating probabilistic
forecasts of societal instability using calibrated SDT models.

The forecaster supports:
- Ensemble-based uncertainty quantification
- Multiple scenario analysis
- Confidence intervals that grow with forecast horizon
- Peak instability probability estimation

Example:
    >>> from cliodynamics.forecast import Forecaster
    >>> from cliodynamics.models import SDTModel, SDTParams
    >>> from cliodynamics.calibration import CalibrationResult
    >>>
    >>> # Create model with calibrated parameters
    >>> params = SDTParams(r_max=0.02, K_0=1.0)
    >>> model = SDTModel(params)
    >>>
    >>> # Initialize forecaster
    >>> forecaster = Forecaster(
    ...     model=model,
    ...     uncertainty_method='ensemble'
    ... )
    >>>
    >>> # Generate forecast
    >>> forecast = forecaster.predict(
    ...     current_state={'N': 0.8, 'E': 0.15, 'W': 0.85, 'S': 0.9, 'psi': 0.3},
    ...     horizon_years=20
    ... )

References:
    Turchin, P. (2016). Ages of Discord. Beresta Books.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from cliodynamics.calibration import CalibrationResult
    from cliodynamics.models import SDTModel, SDTParams


@dataclass
class ForecastResult:
    """Container for forecast results with uncertainty quantification.

    Attributes:
        time: Array of forecast time points.
        mean: DataFrame with mean forecast trajectory for each variable.
        ci_lower: DataFrame with lower confidence interval bound.
        ci_upper: DataFrame with upper confidence interval bound.
        confidence_level: Confidence level for intervals (e.g., 0.90).
        ensemble: Array of ensemble member trajectories (n_samples x n_times x n_vars).
        peak_probability: Probability of instability peak in forecast horizon.
        peak_time_distribution: Distribution of predicted peak times.
        scenarios: Dictionary of scenario-specific forecasts (if multiple).
        metadata: Additional information about the forecast.
    """

    time: NDArray[np.float64]
    mean: pd.DataFrame
    ci_lower: pd.DataFrame
    ci_upper: pd.DataFrame
    confidence_level: float
    ensemble: NDArray[np.float64] | None = None
    peak_probability: float = 0.0
    peak_time_distribution: NDArray[np.float64] | None = None
    scenarios: dict[str, "ForecastResult"] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ci_90(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return 90% confidence interval (lower, upper).

        Note: If forecast was generated with different confidence level,
        this returns the stored interval regardless of level.
        """
        return (self.ci_lower, self.ci_upper)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecast to a single DataFrame with all statistics.

        Returns:
            DataFrame with columns: time, {var}_mean, {var}_lower, {var}_upper
            for each state variable.
        """
        result = pd.DataFrame({"time": self.time})

        for col in self.mean.columns:
            result[f"{col}_mean"] = self.mean[col].values
            result[f"{col}_lower"] = self.ci_lower[col].values
            result[f"{col}_upper"] = self.ci_upper[col].values

        return result

    def summary(self) -> str:
        """Generate a human-readable summary of the forecast.

        Returns:
            Formatted string with forecast summary.
        """
        lines = [
            "Forecast Summary",
            "=" * 40,
            f"Horizon: {self.time[-1] - self.time[0]:.1f} years",
            f"Confidence level: {self.confidence_level * 100:.0f}%",
            f"Peak instability probability: {self.peak_probability * 100:.1f}%",
            "",
            "Final state predictions (mean [CI]):",
        ]

        final_idx = len(self.time) - 1
        for var in self.mean.columns:
            mean_val = self.mean[var].iloc[final_idx]
            lower_val = self.ci_lower[var].iloc[final_idx]
            upper_val = self.ci_upper[var].iloc[final_idx]
            lines.append(f"  {var}: {mean_val:.3f} [{lower_val:.3f}, {upper_val:.3f}]")

        if self.scenarios:
            lines.append("")
            lines.append(f"Scenarios analyzed: {list(self.scenarios.keys())}")

        return "\n".join(lines)


class Forecaster:
    """Generate probabilistic forecasts from calibrated SDT models.

    The Forecaster takes a calibrated model and produces forecasts with
    uncertainty quantification using ensemble methods. It supports
    scenario analysis for exploring different policy interventions
    or external shocks.

    Attributes:
        model: The calibrated SDT model to use for forecasting.
        uncertainty_method: Method for uncertainty quantification
            ('ensemble' or 'mcmc').
        n_ensemble: Number of ensemble members for uncertainty estimation.
        calibration_result: Optional CalibrationResult with parameter
            uncertainty from calibration.

    Example:
        >>> forecaster = Forecaster(model, uncertainty_method='ensemble')
        >>> forecast = forecaster.predict(current_state, horizon_years=20)
        >>> print(forecast.peak_probability)
    """

    # State variable names in standard order
    STATE_NAMES = ("N", "E", "W", "S", "psi")

    def __init__(
        self,
        model: "SDTModel",
        uncertainty_method: Literal["ensemble", "mcmc"] = "ensemble",
        n_ensemble: int = 100,
        calibration_result: "CalibrationResult | None" = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the Forecaster.

        Args:
            model: Calibrated SDT model to use for forecasting.
            uncertainty_method: Method for uncertainty quantification:
                - 'ensemble': Perturb parameters and initial conditions
                - 'mcmc': Use MCMC samples if available from calibration
            n_ensemble: Number of ensemble members for uncertainty estimation.
            calibration_result: Optional CalibrationResult containing
                parameter uncertainty (e.g., bootstrap samples).
            seed: Random seed for reproducibility.
        """
        self.model = model
        self.uncertainty_method = uncertainty_method
        self.n_ensemble = n_ensemble
        self.calibration_result = calibration_result
        self._rng = np.random.default_rng(seed)

        # Cache model parameters
        self._base_params = model.params

    def predict(
        self,
        current_state: dict[str, float],
        horizon_years: float,
        dt: float = 1.0,
        confidence_level: float = 0.90,
        scenarios: list[str] | None = None,
        scenario_modifiers: dict[str, dict[str, float]] | None = None,
        initial_condition_uncertainty: float = 0.05,
        parameter_uncertainty: float = 0.1,
    ) -> ForecastResult:
        """Generate a probabilistic forecast from the current state.

        Args:
            current_state: Dictionary with current values for state variables
                {'N': float, 'E': float, 'W': float, 'S': float, 'psi': float}.
            horizon_years: Number of years to forecast.
            dt: Time step for forecast output.
            confidence_level: Confidence level for intervals (e.g., 0.90 for 90%).
            scenarios: List of scenario names to analyze. If None, only
                generates baseline forecast.
            scenario_modifiers: Dictionary mapping scenario names to
                parameter modifications. Each modifier is a dict of
                {param_name: new_value} to override during simulation.
            initial_condition_uncertainty: Relative uncertainty in initial
                conditions (coefficient of variation).
            parameter_uncertainty: Relative uncertainty in parameters
                (coefficient of variation) if no calibration result.

        Returns:
            ForecastResult with mean trajectory, confidence intervals,
            and scenario comparisons.

        Raises:
            ValueError: If current_state is missing required variables.

        Example:
            >>> forecast = forecaster.predict(
            ...     current_state={
            ...         'N': 0.8, 'E': 0.15, 'W': 0.85, 'S': 0.9, 'psi': 0.3
            ...     },
            ...     horizon_years=20,
            ...     scenarios=['baseline', 'wealth_pump_off'],
            ...     scenario_modifiers={'wealth_pump_off': {'mu': 0.05}}
            ... )
        """
        # Validate current state
        missing = set(self.STATE_NAMES) - set(current_state.keys())
        if missing:
            raise ValueError(f"current_state missing variables: {missing}")

        # Generate time array
        time = np.arange(0, horizon_years + dt, dt)

        # Run baseline forecast with ensemble
        baseline_result = self._run_ensemble_forecast(
            current_state=current_state,
            time=time,
            confidence_level=confidence_level,
            initial_condition_uncertainty=initial_condition_uncertainty,
            parameter_uncertainty=parameter_uncertainty,
        )

        # Compute peak probability
        peak_prob, peak_times = self._compute_peak_probability(
            baseline_result["ensemble"]
        )

        # Build baseline ForecastResult
        result = ForecastResult(
            time=time,
            mean=baseline_result["mean"],
            ci_lower=baseline_result["ci_lower"],
            ci_upper=baseline_result["ci_upper"],
            confidence_level=confidence_level,
            ensemble=baseline_result["ensemble"],
            peak_probability=peak_prob,
            peak_time_distribution=peak_times,
            metadata={
                "current_state": current_state,
                "horizon_years": horizon_years,
                "n_ensemble": self.n_ensemble,
                "uncertainty_method": self.uncertainty_method,
            },
        )

        # Run scenario forecasts if requested
        if scenarios and scenario_modifiers:
            for scenario_name in scenarios:
                if scenario_name == "baseline":
                    # Baseline is already computed
                    result.scenarios[scenario_name] = result
                elif scenario_name in scenario_modifiers:
                    scenario_result = self._run_scenario_forecast(
                        current_state=current_state,
                        time=time,
                        confidence_level=confidence_level,
                        param_modifiers=scenario_modifiers[scenario_name],
                        initial_condition_uncertainty=initial_condition_uncertainty,
                        parameter_uncertainty=parameter_uncertainty,
                    )
                    result.scenarios[scenario_name] = scenario_result

        return result

    def _run_ensemble_forecast(
        self,
        current_state: dict[str, float],
        time: NDArray[np.float64],
        confidence_level: float,
        initial_condition_uncertainty: float,
        parameter_uncertainty: float,
        param_modifiers: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Run ensemble of forecasts for uncertainty quantification.

        Args:
            current_state: Current state values.
            time: Time array for forecast.
            confidence_level: Confidence level for intervals.
            initial_condition_uncertainty: CV for initial conditions.
            parameter_uncertainty: CV for parameters.
            param_modifiers: Optional parameter overrides for scenarios.

        Returns:
            Dictionary with 'mean', 'ci_lower', 'ci_upper', 'ensemble'.
        """
        from cliodynamics.simulation import Simulator

        # Storage for ensemble trajectories
        # Shape: (n_ensemble, n_times, n_vars)
        n_times = len(time)
        n_vars = len(self.STATE_NAMES)
        ensemble = np.zeros((self.n_ensemble, n_times, n_vars))

        # Time span for simulator
        time_span = (float(time[0]), float(time[-1]))
        dt = float(time[1] - time[0]) if len(time) > 1 else 1.0

        for i in range(self.n_ensemble):
            # Perturb initial conditions
            perturbed_ic = self._perturb_initial_conditions(
                current_state, initial_condition_uncertainty
            )

            # Perturb parameters
            perturbed_params = self._perturb_parameters(
                parameter_uncertainty, param_modifiers
            )

            # Create model with perturbed parameters
            from cliodynamics.models import SDTModel

            perturbed_model = SDTModel(perturbed_params)
            sim = Simulator(perturbed_model)

            # Run simulation
            try:
                result = sim.run(
                    initial_conditions=perturbed_ic,
                    time_span=time_span,
                    dt=dt,
                    method="RK45",
                )

                # Interpolate to output times if needed
                for j, var in enumerate(self.STATE_NAMES):
                    ensemble[i, :, j] = np.interp(
                        time, result.df["t"].values, result.df[var].values
                    )
            except Exception:
                # If simulation fails, use previous ensemble member or baseline
                if i > 0:
                    ensemble[i, :, :] = ensemble[i - 1, :, :]
                else:
                    # Use deterministic forecast
                    determ = self._run_deterministic_forecast(current_state, time)
                    for j, var in enumerate(self.STATE_NAMES):
                        ensemble[i, :, j] = determ[var].values

        # Compute statistics
        alpha = 1 - confidence_level
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100

        mean_data = {}
        lower_data = {}
        upper_data = {}

        for j, var in enumerate(self.STATE_NAMES):
            var_ensemble = ensemble[:, :, j]
            mean_data[var] = np.mean(var_ensemble, axis=0)
            lower_data[var] = np.percentile(var_ensemble, lower_pct, axis=0)
            upper_data[var] = np.percentile(var_ensemble, upper_pct, axis=0)

        return {
            "mean": pd.DataFrame(mean_data),
            "ci_lower": pd.DataFrame(lower_data),
            "ci_upper": pd.DataFrame(upper_data),
            "ensemble": ensemble,
        }

    def _run_deterministic_forecast(
        self,
        current_state: dict[str, float],
        time: NDArray[np.float64],
    ) -> pd.DataFrame:
        """Run a single deterministic forecast.

        Args:
            current_state: Initial state.
            time: Time array.

        Returns:
            DataFrame with forecast trajectory.
        """
        from cliodynamics.simulation import Simulator

        time_span = (float(time[0]), float(time[-1]))
        dt = float(time[1] - time[0]) if len(time) > 1 else 1.0

        sim = Simulator(self.model)
        result = sim.run(
            initial_conditions=current_state,
            time_span=time_span,
            dt=dt,
            method="RK45",
        )

        # Interpolate to exact output times
        interp_data = {}
        for var in self.STATE_NAMES:
            interp_data[var] = np.interp(
                time, result.df["t"].values, result.df[var].values
            )

        return pd.DataFrame(interp_data)

    def _run_scenario_forecast(
        self,
        current_state: dict[str, float],
        time: NDArray[np.float64],
        confidence_level: float,
        param_modifiers: dict[str, float],
        initial_condition_uncertainty: float,
        parameter_uncertainty: float,
    ) -> ForecastResult:
        """Run a forecast for a specific scenario.

        Args:
            current_state: Initial state.
            time: Time array.
            confidence_level: Confidence level.
            param_modifiers: Parameter overrides for this scenario.
            initial_condition_uncertainty: CV for initial conditions.
            parameter_uncertainty: CV for parameters.

        Returns:
            ForecastResult for the scenario.
        """
        ensemble_result = self._run_ensemble_forecast(
            current_state=current_state,
            time=time,
            confidence_level=confidence_level,
            initial_condition_uncertainty=initial_condition_uncertainty,
            parameter_uncertainty=parameter_uncertainty,
            param_modifiers=param_modifiers,
        )

        peak_prob, peak_times = self._compute_peak_probability(
            ensemble_result["ensemble"]
        )

        return ForecastResult(
            time=time,
            mean=ensemble_result["mean"],
            ci_lower=ensemble_result["ci_lower"],
            ci_upper=ensemble_result["ci_upper"],
            confidence_level=confidence_level,
            ensemble=ensemble_result["ensemble"],
            peak_probability=peak_prob,
            peak_time_distribution=peak_times,
            metadata={"scenario_modifiers": param_modifiers},
        )

    def _perturb_initial_conditions(
        self,
        current_state: dict[str, float],
        cv: float,
    ) -> dict[str, float]:
        """Perturb initial conditions with random noise.

        Args:
            current_state: Base state values.
            cv: Coefficient of variation for perturbation.

        Returns:
            Perturbed state dictionary.
        """
        perturbed = {}
        for var, value in current_state.items():
            # Add log-normal noise to ensure positive values
            if value > 0:
                log_std = np.sqrt(np.log(1 + cv**2))
                log_mean = np.log(value) - log_std**2 / 2
                perturbed[var] = self._rng.lognormal(log_mean, log_std)
            else:
                # For zero or negative values, use normal with clipping
                perturbed[var] = max(0.0, value + self._rng.normal(0, cv * 0.1))
        return perturbed

    def _perturb_parameters(
        self,
        cv: float,
        param_modifiers: dict[str, float] | None = None,
    ) -> "SDTParams":
        """Perturb model parameters with random noise.

        If calibration_result has bootstrap_params, samples from those.
        Otherwise, adds log-normal noise to base parameters.

        Args:
            cv: Coefficient of variation for perturbation.
            param_modifiers: Optional overrides for specific parameters.

        Returns:
            Perturbed SDTParams instance.
        """
        from dataclasses import fields

        from cliodynamics.models import SDTParams

        # Start with base parameters
        param_dict = {
            f.name: getattr(self._base_params, f.name)
            for f in fields(self._base_params)
        }

        # If we have bootstrap samples from calibration, use those
        if (
            self.calibration_result is not None
            and self.calibration_result.bootstrap_params is not None
        ):
            # Sample a random bootstrap parameter set
            n_bootstrap = len(self.calibration_result.bootstrap_params)
            idx = self._rng.integers(0, n_bootstrap)
            bootstrap_sample = self.calibration_result.bootstrap_params[idx]

            # Get parameter names from calibration result
            cal_param_names = list(self.calibration_result.best_params.keys())
            for i, name in enumerate(cal_param_names):
                param_dict[name] = bootstrap_sample[i]
        else:
            # Perturb parameters with log-normal noise
            for name, value in param_dict.items():
                if isinstance(value, (int, float)) and value > 0:
                    log_std = np.sqrt(np.log(1 + cv**2))
                    log_mean = np.log(value) - log_std**2 / 2
                    param_dict[name] = self._rng.lognormal(log_mean, log_std)

        # Apply scenario modifiers (deterministic overrides)
        if param_modifiers:
            for name, value in param_modifiers.items():
                if name in param_dict:
                    param_dict[name] = value

        return SDTParams(**param_dict)

    def _compute_peak_probability(
        self,
        ensemble: NDArray[np.float64],
        psi_threshold: float = 0.5,
    ) -> tuple[float, NDArray[np.float64]]:
        """Compute probability and timing of instability peak.

        A peak is defined as psi reaching a local maximum above threshold
        within the forecast horizon.

        Args:
            ensemble: Ensemble trajectories (n_samples, n_times, n_vars).
            psi_threshold: Threshold for considering a peak significant.

        Returns:
            Tuple of (peak_probability, peak_time_distribution).
            peak_probability is fraction of ensemble members with a peak.
            peak_time_distribution is array of peak times for each member
            (NaN if no peak detected).
        """
        n_samples, n_times, _ = ensemble.shape
        psi_idx = list(self.STATE_NAMES).index("psi")

        peak_times = np.full(n_samples, np.nan)

        for i in range(n_samples):
            psi_trajectory = ensemble[i, :, psi_idx]

            # Find local maxima
            # A point is a local max if it's greater than neighbors
            local_maxima = []
            for t in range(1, n_times - 1):
                if (
                    psi_trajectory[t] > psi_trajectory[t - 1]
                    and psi_trajectory[t] > psi_trajectory[t + 1]
                    and psi_trajectory[t] > psi_threshold
                ):
                    local_maxima.append(t)

            # Also check if trajectory is still rising at end (potential peak)
            if (
                n_times > 1
                and psi_trajectory[-1] > psi_trajectory[-2]
                and psi_trajectory[-1] > psi_threshold
            ):
                local_maxima.append(n_times - 1)

            # Record first significant peak
            if local_maxima:
                peak_times[i] = local_maxima[0]

        # Compute probability as fraction with peaks
        n_peaks = np.sum(~np.isnan(peak_times))
        peak_probability = n_peaks / n_samples

        return peak_probability, peak_times

    def forecast_peak_timing(
        self,
        current_state: dict[str, float],
        max_horizon: float = 50.0,
        dt: float = 1.0,
        psi_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Forecast the timing of the next instability peak.

        Runs forecast until peak is reached or max_horizon, estimating
        the probability distribution of peak timing.

        Args:
            current_state: Current state values.
            max_horizon: Maximum years to forecast.
            dt: Time step.
            psi_threshold: Threshold for peak detection.

        Returns:
            Dictionary with:
                - 'peak_probability': P(peak within horizon)
                - 'expected_time': Expected time to peak (if any)
                - 'time_distribution': Distribution of peak times
                - 'confidence_interval': CI for peak timing
        """
        forecast = self.predict(
            current_state=current_state,
            horizon_years=max_horizon,
            dt=dt,
            confidence_level=0.90,
        )

        peak_times = forecast.peak_time_distribution
        valid_peaks = peak_times[~np.isnan(peak_times)] * dt  # Convert to years

        if len(valid_peaks) > 0:
            expected_time = float(np.mean(valid_peaks))
            ci_lower = float(np.percentile(valid_peaks, 5))
            ci_upper = float(np.percentile(valid_peaks, 95))
        else:
            expected_time = np.nan
            ci_lower = np.nan
            ci_upper = np.nan

        return {
            "peak_probability": forecast.peak_probability,
            "expected_time": expected_time,
            "time_distribution": valid_peaks,
            "confidence_interval": (ci_lower, ci_upper),
        }


__all__ = ["Forecaster", "ForecastResult"]
