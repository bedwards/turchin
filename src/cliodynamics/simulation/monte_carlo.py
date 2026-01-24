"""Monte Carlo simulation framework for probabilistic forecasting.

This module provides a comprehensive Monte Carlo simulator that runs thousands
of simulations with parameter uncertainty to generate probability distributions
instead of point predictions. This enables honest forecasting by quantifying
uncertainty in model outputs.

Key features:
- Parameter distributions (Normal, Uniform, LogNormal, etc.)
- Parallel execution with multiprocessing for speed
- Results aggregation (percentiles, confidence intervals)
- Probability threshold queries
- Historical backtesting support

Example:
    >>> from cliodynamics.simulation import MonteCarloSimulator
    >>> from cliodynamics.simulation.monte_carlo import Normal, Uniform
    >>> from cliodynamics.models import SDTModel
    >>>
    >>> # Run 10,000 simulations with parameter uncertainty
    >>> mc = MonteCarloSimulator(
    ...     model=SDTModel(),
    ...     n_simulations=10000,
    ...     parameter_distributions={
    ...         'r_max': Normal(0.02, 0.005),
    ...         'alpha': Normal(0.1, 0.02),
    ...         'beta': Uniform(0.5, 1.5),
    ...     }
    ... )
    >>> results = mc.run(
    ...     initial_conditions=initial_state,
    ...     time_span=(0, 100)
    ... )
    >>>
    >>> # Get probability distribution of outcomes
    >>> p_crisis = results.probability(variable='psi', threshold=1.5, year=30)
    >>> peak_timing = results.peak_timing_distribution('psi')
    >>> sensitivity = results.sensitivity_indices()  # Which params matter?

References:
    Turchin, P. (2016). Ages of Discord. Beresta Books.
    Saltelli, A. et al. (2008). Global Sensitivity Analysis. Wiley.
"""

from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from cliodynamics.models import SDTModel


# =============================================================================
# Parameter Distribution Classes
# =============================================================================


class Distribution(ABC):
    """Abstract base class for parameter distributions."""

    @abstractmethod
    def sample(self, rng: np.random.Generator, n: int = 1) -> NDArray[np.float64]:
        """Draw random samples from the distribution.

        Args:
            rng: NumPy random generator.
            n: Number of samples to draw.

        Returns:
            Array of n samples.
        """
        ...

    @abstractmethod
    def mean(self) -> float:
        """Return the distribution mean."""
        ...

    @abstractmethod
    def std(self) -> float:
        """Return the distribution standard deviation."""
        ...


@dataclass
class Normal(Distribution):
    """Normal (Gaussian) distribution.

    Attributes:
        mu: Mean of the distribution.
        sigma: Standard deviation (must be positive).
    """

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")

    def sample(self, rng: np.random.Generator, n: int = 1) -> NDArray[np.float64]:
        return rng.normal(self.mu, self.sigma, n)

    def mean(self) -> float:
        return self.mu

    def std(self) -> float:
        return self.sigma


@dataclass
class Uniform(Distribution):
    """Uniform distribution over [low, high].

    Attributes:
        low: Lower bound.
        high: Upper bound.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(f"low must be < high, got {self.low} >= {self.high}")

    def sample(self, rng: np.random.Generator, n: int = 1) -> NDArray[np.float64]:
        return rng.uniform(self.low, self.high, n)

    def mean(self) -> float:
        return (self.low + self.high) / 2

    def std(self) -> float:
        return (self.high - self.low) / np.sqrt(12)


@dataclass
class LogNormal(Distribution):
    """Log-normal distribution.

    Parameterized so that the underlying normal has mean mu and std sigma.
    The resulting distribution is always positive.

    Attributes:
        mu: Mean of the underlying normal distribution (log scale).
        sigma: Standard deviation of the underlying normal (log scale).
    """

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")

    def sample(self, rng: np.random.Generator, n: int = 1) -> NDArray[np.float64]:
        return rng.lognormal(self.mu, self.sigma, n)

    def mean(self) -> float:
        return np.exp(self.mu + self.sigma**2 / 2)

    def std(self) -> float:
        var = (np.exp(self.sigma**2) - 1) * np.exp(2 * self.mu + self.sigma**2)
        return np.sqrt(var)


@dataclass
class Triangular(Distribution):
    """Triangular distribution with mode (most likely value).

    Useful when you have minimum, maximum, and best estimate.

    Attributes:
        low: Lower bound.
        mode: Most likely value.
        high: Upper bound.
    """

    low: float
    mode: float
    high: float

    def __post_init__(self) -> None:
        if not (self.low <= self.mode <= self.high):
            msg = (
                f"Must have low <= mode <= high, "
                f"got {self.low}, {self.mode}, {self.high}"
            )
            raise ValueError(msg)

    def sample(self, rng: np.random.Generator, n: int = 1) -> NDArray[np.float64]:
        return rng.triangular(self.low, self.mode, self.high, n)

    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3

    def std(self) -> float:
        a, c, b = self.low, self.mode, self.high
        var = (a**2 + b**2 + c**2 - a * b - a * c - b * c) / 18
        return np.sqrt(var)


@dataclass
class Constant(Distribution):
    """Constant (degenerate) distribution - always returns the same value.

    Useful for fixing a parameter while varying others.

    Attributes:
        value: The constant value.
    """

    value: float

    def sample(self, rng: np.random.Generator, n: int = 1) -> NDArray[np.float64]:
        return np.full(n, self.value)

    def mean(self) -> float:
        return self.value

    def std(self) -> float:
        return 0.0


@dataclass
class TruncatedNormal(Distribution):
    """Truncated normal distribution bounded to [low, high].

    Uses rejection sampling to ensure values stay in bounds.
    Useful for parameters that must be positive.

    Attributes:
        mu: Mean of the underlying normal.
        sigma: Standard deviation of the underlying normal.
        low: Lower bound (default 0).
        high: Upper bound (default inf).
    """

    mu: float
    sigma: float
    low: float = 0.0
    high: float = float("inf")

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.low >= self.high:
            raise ValueError(f"low must be < high, got {self.low} >= {self.high}")

    def sample(self, rng: np.random.Generator, n: int = 1) -> NDArray[np.float64]:
        samples = []
        while len(samples) < n:
            # Oversample and reject out-of-bounds values
            batch_size = max(n * 2, 100)
            candidates = rng.normal(self.mu, self.sigma, batch_size)
            valid = candidates[(candidates >= self.low) & (candidates <= self.high)]
            samples.extend(valid.tolist())
        return np.array(samples[:n])

    def mean(self) -> float:
        # Approximate mean for truncated normal
        return np.clip(self.mu, self.low, self.high)

    def std(self) -> float:
        # Approximate std - will be less than original sigma due to truncation
        return self.sigma


# =============================================================================
# Monte Carlo Results
# =============================================================================


@dataclass
class MonteCarloResults:
    """Container for Monte Carlo simulation results.

    Attributes:
        time: Array of time points.
        ensemble: Array of ensemble trajectories (n_sims x n_times x n_vars).
        parameter_samples: Array of sampled parameters (n_sims x n_params).
        parameter_names: List of parameter names.
        state_names: Names of state variables.
        n_simulations: Number of simulations run.
        failed_simulations: Number of simulations that failed.
        seed: Random seed used.
        metadata: Additional information about the run.
    """

    time: NDArray[np.float64]
    ensemble: NDArray[np.float64]
    parameter_samples: NDArray[np.float64]
    parameter_names: list[str]
    state_names: tuple[str, ...]
    n_simulations: int
    failed_simulations: int = 0
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate dimensions."""
        n_sims, n_times, n_vars = self.ensemble.shape
        if n_sims != self.n_simulations - self.failed_simulations:
            # Allow for failed simulations - successful ones stored
            pass
        if n_times != len(self.time):
            raise ValueError("Ensemble time dimension doesn't match time array")
        if n_vars != len(self.state_names):
            raise ValueError("Ensemble var dimension doesn't match state_names")

    @property
    def n_successful(self) -> int:
        """Number of successful simulations."""
        return self.ensemble.shape[0]

    def get_variable(self, variable: str) -> NDArray[np.float64]:
        """Get ensemble data for a single variable.

        Args:
            variable: Variable name (e.g., 'psi', 'N').

        Returns:
            Array of shape (n_sims, n_times) with the variable values.
        """
        if variable not in self.state_names:
            raise ValueError(f"Variable '{variable}' not in {self.state_names}")
        idx = self.state_names.index(variable)
        return self.ensemble[:, :, idx]

    def mean(self) -> pd.DataFrame:
        """Get mean trajectory across all simulations.

        Returns:
            DataFrame with time and mean values for each variable.
        """
        means = np.mean(self.ensemble, axis=0)
        data = {"time": self.time}
        for i, name in enumerate(self.state_names):
            data[name] = means[:, i]
        return pd.DataFrame(data)

    def std(self) -> pd.DataFrame:
        """Get standard deviation at each time point.

        Returns:
            DataFrame with time and std for each variable.
        """
        stds = np.std(self.ensemble, axis=0)
        data = {"time": self.time}
        for i, name in enumerate(self.state_names):
            data[name] = stds[:, i]
        return pd.DataFrame(data)

    def percentile(
        self, q: float | list[float]
    ) -> pd.DataFrame | dict[float, pd.DataFrame]:
        """Get percentiles across simulations at each time point.

        Args:
            q: Percentile(s) to compute (0-100).

        Returns:
            DataFrame or dict of DataFrames with percentile values.
        """
        if isinstance(q, (int, float)):
            pcts = np.percentile(self.ensemble, q, axis=0)
            data = {"time": self.time}
            for i, name in enumerate(self.state_names):
                data[name] = pcts[:, i]
            return pd.DataFrame(data)
        else:
            result = {}
            for qi in q:
                result[qi] = self.percentile(qi)
            return result

    def confidence_interval(
        self, confidence: float = 0.90
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Get confidence interval bounds.

        Args:
            confidence: Confidence level (e.g., 0.90 for 90% CI).

        Returns:
            Tuple of (lower, upper) DataFrames.
        """
        alpha = 1 - confidence
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100
        return self.percentile(lower_pct), self.percentile(upper_pct)

    def probability(
        self,
        variable: str,
        threshold: float,
        year: float | None = None,
        comparison: str = "greater",
    ) -> float:
        """Compute probability of exceeding/falling below a threshold.

        Args:
            variable: Variable to check (e.g., 'psi').
            threshold: Threshold value.
            year: Specific year to check. If None, checks if threshold is
                ever exceeded during the simulation.
            comparison: 'greater' or 'less'.

        Returns:
            Probability (0-1) of crossing the threshold.

        Example:
            >>> p = results.probability('psi', threshold=1.5, year=30)
            >>> print(f"P(PSI > 1.5 at year 30) = {p:.2%}")
        """
        var_data = self.get_variable(variable)

        if year is not None:
            # Find closest time index
            idx = np.argmin(np.abs(self.time - year))
            values = var_data[:, idx]
        else:
            # Check if threshold is ever exceeded across all time
            if comparison == "greater":
                values = np.max(var_data, axis=1)
            else:
                values = np.min(var_data, axis=1)

        if comparison == "greater":
            count = np.sum(values > threshold)
        else:
            count = np.sum(values < threshold)

        return count / len(values)

    def probability_by_year(
        self,
        variable: str,
        threshold: float,
        comparison: str = "greater",
    ) -> pd.DataFrame:
        """Compute probability of threshold crossing at each year.

        Args:
            variable: Variable to check.
            threshold: Threshold value.
            comparison: 'greater' or 'less'.

        Returns:
            DataFrame with 'year' and 'probability' columns.
        """
        var_data = self.get_variable(variable)
        probs = []

        for t_idx, t in enumerate(self.time):
            values = var_data[:, t_idx]
            if comparison == "greater":
                p = np.sum(values > threshold) / len(values)
            else:
                p = np.sum(values < threshold) / len(values)
            probs.append(p)

        return pd.DataFrame({"year": self.time, "probability": probs})

    def peak_timing_distribution(
        self, variable: str, threshold: float = 0.0
    ) -> NDArray[np.float64]:
        """Get distribution of when variable reaches its peak.

        Args:
            variable: Variable to analyze.
            threshold: Minimum peak value to consider (filters noise).

        Returns:
            Array of peak times (NaN for simulations without a peak).
        """
        var_data = self.get_variable(variable)
        peak_times = np.full(self.n_successful, np.nan)

        for i in range(self.n_successful):
            trajectory = var_data[i, :]
            max_val = np.max(trajectory)
            if max_val > threshold:
                peak_times[i] = self.time[np.argmax(trajectory)]

        return peak_times

    def first_crossing_distribution(
        self,
        variable: str,
        threshold: float,
        comparison: str = "greater",
    ) -> NDArray[np.float64]:
        """Get distribution of when threshold is first crossed.

        Args:
            variable: Variable to check.
            threshold: Threshold value.
            comparison: 'greater' or 'less'.

        Returns:
            Array of first crossing times (NaN if never crossed).
        """
        var_data = self.get_variable(variable)
        crossing_times = np.full(self.n_successful, np.nan)

        for i in range(self.n_successful):
            trajectory = var_data[i, :]
            if comparison == "greater":
                crosses = np.where(trajectory > threshold)[0]
            else:
                crosses = np.where(trajectory < threshold)[0]

            if len(crosses) > 0:
                crossing_times[i] = self.time[crosses[0]]

        return crossing_times

    def to_fan_chart_data(
        self,
        variable: str,
        percentiles: list[float] | None = None,
    ) -> pd.DataFrame:
        """Prepare data for fan chart visualization.

        Args:
            variable: Variable to prepare data for.
            percentiles: List of percentiles to include.
                Defaults to [5, 10, 25, 50, 75, 90, 95].

        Returns:
            DataFrame with 'time' and columns for each percentile.
        """
        if percentiles is None:
            percentiles = [5, 10, 25, 50, 75, 90, 95]

        var_data = self.get_variable(variable)
        data = {"time": self.time}

        for p in percentiles:
            data[f"p{int(p)}"] = np.percentile(var_data, p, axis=0)

        return pd.DataFrame(data)

    def summary(self) -> str:
        """Generate a human-readable summary of results.

        Returns:
            Formatted summary string.
        """
        lines = [
            "Monte Carlo Simulation Summary",
            "=" * 50,
            f"Simulations: {self.n_simulations} "
            f"({self.n_successful} successful, {self.failed_simulations} failed)",
            f"Time range: {self.time[0]:.1f} to {self.time[-1]:.1f}",
            f"Variables: {', '.join(self.state_names)}",
            f"Parameters varied: {len(self.parameter_names)}",
            "",
            "Final State Statistics (mean [90% CI]):",
        ]

        lower, upper = self.confidence_interval(0.90)
        mean_df = self.mean()

        for var in self.state_names:
            m = mean_df[var].iloc[-1]
            lo = lower[var].iloc[-1]
            hi = upper[var].iloc[-1]
            lines.append(f"  {var}: {m:.3f} [{lo:.3f}, {hi:.3f}]")

        # Add probability info for psi if present
        if "psi" in self.state_names:
            p50 = self.probability("psi", 0.5)
            p100 = self.probability("psi", 1.0)
            lines.extend(
                [
                    "",
                    "Crisis Probability (psi threshold):",
                    f"  P(psi > 0.5 ever): {p50:.1%}",
                    f"  P(psi > 1.0 ever): {p100:.1%}",
                ]
            )

        return "\n".join(lines)


# =============================================================================
# Monte Carlo Simulator
# =============================================================================


def _run_single_simulation(args: tuple[Any, ...]) -> tuple[bool, Any, Any]:
    """Worker function for parallel simulation.

    Args:
        args: Tuple of (model_class, params_dict, initial_conditions,
                       time_span, dt, state_names)

    Returns:
        Tuple of (success, trajectory, params_values) or (False, None, None).
    """
    from cliodynamics.models import SDTModel, SDTParams
    from cliodynamics.simulation import Simulator

    (
        model_class_name,
        params_dict,
        initial_conditions,
        time_span,
        dt,
        state_names,
    ) = args

    try:
        # Recreate model with sampled parameters
        params = SDTParams(**params_dict)
        model = SDTModel(params)
        sim = Simulator(model, state_names=state_names)

        result = sim.run(
            initial_conditions=initial_conditions,
            time_span=time_span,
            dt=dt,
            method="RK45",
            rtol=1e-5,
            atol=1e-8,
        )

        # Extract trajectory as numpy array
        trajectory = np.column_stack([result.df[name].values for name in state_names])
        return (True, trajectory, list(params_dict.values()))

    except Exception:
        return (False, None, None)


class MonteCarloSimulator:
    """Monte Carlo simulator for probabilistic forecasting.

    Runs thousands of simulations with parameter uncertainty to generate
    probability distributions of model outputs.

    Attributes:
        model: Base SDT model instance.
        n_simulations: Number of Monte Carlo samples.
        parameter_distributions: Dict mapping param names to distributions.
        initial_condition_distributions: Dict mapping state vars to distributions.
        n_workers: Number of parallel workers.
        seed: Random seed for reproducibility.

    Example:
        >>> mc = MonteCarloSimulator(
        ...     model=SDTModel(),
        ...     n_simulations=10000,
        ...     parameter_distributions={
        ...         'r_max': Normal(0.02, 0.005),
        ...         'alpha': Uniform(0.003, 0.007),
        ...     },
        ...     n_workers=4
        ... )
        >>> results = mc.run(
        ...     initial_conditions=initial_state,
        ...     time_span=(0, 100)
        ... )
    """

    # Default state variable names
    STATE_NAMES = ("N", "E", "W", "S", "psi")

    def __init__(
        self,
        model: "SDTModel",
        n_simulations: int = 1000,
        parameter_distributions: dict[str, Distribution] | None = None,
        initial_condition_distributions: dict[str, Distribution] | None = None,
        n_workers: int | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the Monte Carlo simulator.

        Args:
            model: Base SDT model instance (provides default parameters).
            n_simulations: Number of Monte Carlo samples (default 1000).
            parameter_distributions: Dict mapping parameter names to
                Distribution objects. Parameters not specified will use
                the base model values.
            initial_condition_distributions: Dict mapping state variable
                names to Distribution objects for uncertain initial conditions.
            n_workers: Number of parallel workers. Default is CPU count - 1.
            seed: Random seed for reproducibility.
        """
        self.model = model
        self.n_simulations = n_simulations
        self.parameter_distributions = parameter_distributions or {}
        self.initial_condition_distributions = initial_condition_distributions or {}
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Get base parameter values
        self._base_params = self._get_param_dict()

    def _get_param_dict(self) -> dict[str, float]:
        """Extract parameter dict from model."""
        from dataclasses import fields

        return {
            f.name: getattr(self.model.params, f.name)
            for f in fields(self.model.params)
        }

    def _sample_parameters(self, n: int) -> tuple[NDArray[np.float64], list[str]]:
        """Sample parameters from distributions.

        Args:
            n: Number of parameter sets to sample.

        Returns:
            Tuple of (samples array, parameter names list).
        """
        param_names = list(self.parameter_distributions.keys())
        samples = np.zeros((n, len(param_names)))

        for i, name in enumerate(param_names):
            dist = self.parameter_distributions[name]
            samples[:, i] = dist.sample(self._rng, n)

        return samples, param_names

    def _sample_initial_conditions(
        self, base_ic: dict[str, float], n: int
    ) -> list[dict[str, float]]:
        """Sample initial conditions from distributions.

        Args:
            base_ic: Base initial conditions dict.
            n: Number of IC sets to sample.

        Returns:
            List of n initial condition dicts.
        """
        ic_list = []

        for _ in range(n):
            ic = base_ic.copy()
            for name, dist in self.initial_condition_distributions.items():
                if name in ic:
                    ic[name] = float(dist.sample(self._rng, 1)[0])
            ic_list.append(ic)

        return ic_list

    def run(
        self,
        initial_conditions: dict[str, float],
        time_span: tuple[float, float],
        dt: float = 1.0,
        parallel: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> MonteCarloResults:
        """Run Monte Carlo simulation ensemble.

        Args:
            initial_conditions: Base initial conditions dict.
            time_span: Tuple of (start_time, end_time).
            dt: Output time step.
            parallel: Whether to use parallel execution.
            progress_callback: Optional callback(completed, total) for progress.

        Returns:
            MonteCarloResults with ensemble of trajectories.

        Example:
            >>> results = mc.run(
            ...     initial_conditions=initial_state,
            ...     time_span=(0, 100),
            ...     dt=1.0
            ... )
        """
        from cliodynamics.simulation import Simulator

        # Sample parameters
        if self.parameter_distributions:
            param_samples, param_names = self._sample_parameters(self.n_simulations)
        else:
            param_samples = np.array([])
            param_names = []

        # Sample initial conditions
        ic_samples = self._sample_initial_conditions(
            initial_conditions, self.n_simulations
        )

        # Generate time array
        t_start, t_end = time_span
        time = np.arange(t_start, t_end + dt, dt)
        time = time[time <= t_end]
        n_times = len(time)
        n_vars = len(self.STATE_NAMES)

        # Storage for results
        successful_trajectories = []
        successful_params = []
        failed_count = 0

        if parallel and self.n_simulations > 10:
            # Parallel execution
            # Prepare arguments for worker function
            args_list = []
            for i in range(self.n_simulations):
                params_dict = self._base_params.copy()
                for j, name in enumerate(param_names):
                    params_dict[name] = float(param_samples[i, j])

                args_list.append(
                    (
                        "SDTModel",
                        params_dict,
                        ic_samples[i],
                        time_span,
                        dt,
                        self.STATE_NAMES,
                    )
                )

            # Use ProcessPoolExecutor for parallel execution
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(_run_single_simulation, args): idx
                    for idx, args in enumerate(args_list)
                }

                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, self.n_simulations)

                    _ = futures[future]  # noqa: F841
                    try:
                        success, trajectory, params = future.result()
                        if success:
                            # Interpolate to exact time points if needed
                            if trajectory.shape[0] == n_times:
                                successful_trajectories.append(trajectory)
                            else:
                                # Interpolate to match expected time points
                                interp_traj = np.zeros((n_times, n_vars))
                                sim_time = np.linspace(
                                    t_start, t_end, trajectory.shape[0]
                                )
                                for v in range(n_vars):
                                    interp_traj[:, v] = np.interp(
                                        time, sim_time, trajectory[:, v]
                                    )
                                successful_trajectories.append(interp_traj)

                            if len(params) > 0:
                                successful_params.append(params)
                        else:
                            failed_count += 1
                    except Exception:
                        failed_count += 1

        else:
            # Sequential execution
            sim = Simulator(self.model, state_names=self.STATE_NAMES)

            for i in range(self.n_simulations):
                if progress_callback and i % 100 == 0:
                    progress_callback(i, self.n_simulations)

                try:
                    # Update model parameters
                    params_dict = self._base_params.copy()
                    for j, name in enumerate(param_names):
                        params_dict[name] = float(param_samples[i, j])

                    from cliodynamics.models import SDTModel, SDTParams

                    params = SDTParams(**params_dict)
                    model = SDTModel(params)
                    sim = Simulator(model, state_names=self.STATE_NAMES)

                    result = sim.run(
                        initial_conditions=ic_samples[i],
                        time_span=time_span,
                        dt=dt,
                        method="RK45",
                        rtol=1e-5,
                        atol=1e-8,
                    )

                    # Extract trajectory
                    trajectory = np.zeros((n_times, n_vars))
                    for v, name in enumerate(self.STATE_NAMES):
                        trajectory[:, v] = np.interp(
                            time, result.df["t"].values, result.df[name].values
                        )

                    successful_trajectories.append(trajectory)
                    if len(param_names) > 0:
                        successful_params.append(param_samples[i, :].tolist())

                except Exception:
                    failed_count += 1

        # Combine results
        ensemble = np.stack(successful_trajectories, axis=0)
        if successful_params:
            param_array = np.array(successful_params)
        else:
            param_array = np.array([]).reshape(0, 0)

        return MonteCarloResults(
            time=time,
            ensemble=ensemble,
            parameter_samples=param_array,
            parameter_names=param_names,
            state_names=self.STATE_NAMES,
            n_simulations=self.n_simulations,
            failed_simulations=failed_count,
            seed=self.seed,
            metadata={
                "time_span": time_span,
                "dt": dt,
                "base_initial_conditions": initial_conditions,
                "parameter_distributions": {
                    k: type(v).__name__ for k, v in self.parameter_distributions.items()
                },
            },
        )

    def sensitivity_analysis(
        self,
        initial_conditions: dict[str, float],
        time_span: tuple[float, float],
        target_variable: str = "psi",
        target_time: float | None = None,
        dt: float = 1.0,
    ) -> dict[str, float]:
        """Compute simple sensitivity indices for parameters.

        Uses correlation-based sensitivity: how strongly does each parameter
        correlate with the output variable?

        For full Sobol sensitivity analysis, use the sensitivity module.

        Args:
            initial_conditions: Base initial conditions.
            time_span: Simulation time span.
            target_variable: Variable to analyze (default 'psi').
            target_time: Time at which to evaluate (default: end time).
            dt: Time step.

        Returns:
            Dict mapping parameter names to correlation coefficients.
        """
        results = self.run(initial_conditions, time_span, dt, parallel=True)

        if target_time is None:
            target_time = time_span[1]

        # Get target variable values at target time
        time_idx = np.argmin(np.abs(results.time - target_time))
        var_idx = results.state_names.index(target_variable)
        target_values = results.ensemble[:, time_idx, var_idx]

        # Compute correlations with each parameter
        sensitivities = {}
        for i, name in enumerate(results.parameter_names):
            param_values = results.parameter_samples[:, i]
            # Pearson correlation
            corr = np.corrcoef(param_values, target_values)[0, 1]
            sensitivities[name] = corr

        # Sort by absolute correlation
        sorted_items = sorted(
            sensitivities.items(), key=lambda x: abs(x[1]), reverse=True
        )
        return dict(sorted_items)


__all__ = [
    "MonteCarloSimulator",
    "MonteCarloResults",
    "Distribution",
    "Normal",
    "Uniform",
    "LogNormal",
    "Triangular",
    "Constant",
    "TruncatedNormal",
]
