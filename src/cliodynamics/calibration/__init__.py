"""Parameter calibration framework for cliodynamics models.

This module provides tools for calibrating SDT model parameters against
historical data using global optimization and uncertainty quantification.

Example:
    >>> from cliodynamics.models import SDTModel, SDTParams
    >>> from cliodynamics.calibration import Calibrator, CalibrationResult
    >>> import pandas as pd
    >>>
    >>> # Prepare observed data
    >>> observed = pd.DataFrame({
    ...     'year': [0, 50, 100, 150, 200],
    ...     'N': [0.5, 0.6, 0.7, 0.65, 0.55],
    ...     'psi': [0.0, 0.1, 0.3, 0.5, 0.4]
    ... })
    >>>
    >>> # Create calibrator
    >>> calibrator = Calibrator(
    ...     model=SDTModel,
    ...     observed_data=observed,
    ...     fit_variables=['N', 'psi'],
    ...     time_column='year'
    ... )
    >>>
    >>> # Run calibration
    >>> result = calibrator.fit(
    ...     param_bounds={
    ...         'r_max': (0.01, 0.05),
    ...         'K_0': (0.8, 1.2),
    ...     },
    ...     method='differential_evolution'
    ... )
    >>>
    >>> print(result.best_params)
    >>> print(result.loss)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Type

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize

if TYPE_CHECKING:
    from cliodynamics.models import SDTModel, SDTParams
    from cliodynamics.simulation import Simulator


@dataclass
class CalibrationResult:
    """Results from parameter calibration.

    Attributes:
        best_params: Dictionary of optimized parameter values.
        loss: Final loss value at best parameters.
        n_iterations: Number of optimization iterations.
        converged: Whether optimization converged successfully.
        message: Status message from optimizer.
        initial_conditions: Initial state used for simulations.
        param_bounds: Parameter bounds used in optimization.
        confidence_intervals: Bootstrap confidence intervals (if computed).
            Maps parameter names to (lower, upper) tuples.
        bootstrap_params: Parameter samples from bootstrap (if computed).
            Each row is a bootstrap sample.
        loss_history: History of loss values during optimization (if available).
    """

    best_params: dict[str, float]
    loss: float
    n_iterations: int
    converged: bool
    message: str
    initial_conditions: dict[str, float]
    param_bounds: dict[str, tuple[float, float]]
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    bootstrap_params: NDArray[np.float64] | None = None
    loss_history: list[float] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a human-readable summary of calibration results.

        Returns:
            Formatted string with calibration summary.
        """
        lines = [
            "Calibration Results",
            "=" * 40,
            f"Converged: {self.converged}",
            f"Loss: {self.loss:.6f}",
            f"Iterations: {self.n_iterations}",
            f"Message: {self.message}",
            "",
            "Best Parameters:",
        ]

        for name, value in sorted(self.best_params.items()):
            if name in self.confidence_intervals:
                ci_low, ci_high = self.confidence_intervals[name]
                lines.append(f"  {name}: {value:.6f} [{ci_low:.6f}, {ci_high:.6f}]")
            else:
                lines.append(f"  {name}: {value:.6f}")

        return "\n".join(lines)

    def to_params(self, base_params: "SDTParams") -> "SDTParams":
        """Create SDTParams with calibrated values.

        Args:
            base_params: Base parameters to update with calibrated values.

        Returns:
            New SDTParams with calibrated values.
        """
        from dataclasses import replace

        return replace(base_params, **self.best_params)


class Calibrator:
    """Calibrate SDT model parameters against historical data.

    The Calibrator fits model parameters by minimizing a loss function
    that compares simulated trajectories to observed data. It supports
    multiple optimization methods and uncertainty quantification via
    bootstrap resampling.

    Attributes:
        model_class: The model class to instantiate (e.g., SDTModel).
        observed_data: DataFrame with observed time series.
        fit_variables: List of state variables to fit (e.g., ['N', 'psi']).
        time_column: Name of the time column in observed_data.
        weights: Optional weights for each fit variable.
        loss_type: Type of loss function ('mse' or 'likelihood').

    Example:
        >>> calibrator = Calibrator(
        ...     model=SDTModel,
        ...     observed_data=data,
        ...     fit_variables=['N', 'psi'],
        ...     time_column='year'
        ... )
        >>> result = calibrator.fit(
        ...     param_bounds={'r_max': (0.01, 0.05)},
        ...     method='differential_evolution'
        ... )
    """

    def __init__(
        self,
        model: Type["SDTModel"],
        observed_data: pd.DataFrame,
        fit_variables: list[str],
        time_column: str = "year",
        weights: dict[str, float] | None = None,
        loss_type: Literal["mse", "likelihood"] = "mse",
    ) -> None:
        """Initialize the Calibrator.

        Args:
            model: Model class to use for simulations (not an instance).
            observed_data: DataFrame with observed time series data.
                Must contain the time_column and all fit_variables.
            fit_variables: List of state variable names to fit.
            time_column: Name of the time column in observed_data.
            weights: Optional dictionary of weights for each fit variable.
                If None, equal weights are used.
            loss_type: Type of loss function:
                - 'mse': Mean squared error (default)
                - 'likelihood': Negative log-likelihood (assumes Gaussian noise)

        Raises:
            ValueError: If observed_data is missing required columns.
        """
        self.model_class = model
        self.observed_data = observed_data.copy()
        self.fit_variables = fit_variables
        self.time_column = time_column
        self.loss_type = loss_type

        # Validate observed data
        missing_cols = []
        if time_column not in observed_data.columns:
            missing_cols.append(time_column)
        for var in fit_variables:
            if var not in observed_data.columns:
                missing_cols.append(var)

        if missing_cols:
            raise ValueError(f"observed_data missing columns: {missing_cols}")

        # Sort by time
        self.observed_data = self.observed_data.sort_values(time_column).reset_index(
            drop=True
        )

        # Set weights
        if weights is None:
            self.weights = {var: 1.0 for var in fit_variables}
        else:
            self.weights = weights.copy()
            for var in fit_variables:
                if var not in self.weights:
                    self.weights[var] = 1.0

        # Cache for optimization
        self._param_names: list[str] = []
        self._param_bounds: list[tuple[float, float]] = []
        self._base_params: "SDTParams | None" = None
        self._initial_conditions: dict[str, float] = {}
        self._time_span: tuple[float, float] = (0.0, 0.0)
        self._dt: float = 1.0
        self._observed_times: NDArray[np.float64] = np.array([])
        self._observed_values: dict[str, NDArray[np.float64]] = {}
        self._loss_history: list[float] = []

    def _compute_loss(
        self, param_values: NDArray[np.float64]
    ) -> float:
        """Compute the loss for a set of parameter values.

        Args:
            param_values: Array of parameter values in order of _param_names.

        Returns:
            Loss value (lower is better).
        """
        from cliodynamics.models import SDTParams
        from cliodynamics.simulation import Simulator

        # Build parameters
        param_dict = dict(zip(self._param_names, param_values))

        # Create params with updated values
        if self._base_params is not None:
            params = SDTParams(
                **{
                    **{
                        k: getattr(self._base_params, k)
                        for k in SDTParams.__dataclass_fields__
                    },
                    **param_dict,
                }
            )
        else:
            params = SDTParams(**param_dict)

        # Create model and simulator
        model = self.model_class(params)
        sim = Simulator(model)

        # Run simulation
        try:
            result = sim.run(
                initial_conditions=self._initial_conditions,
                time_span=self._time_span,
                dt=self._dt,
            )
        except Exception:
            # Return high loss for failed simulations
            return 1e10

        # Interpolate simulation to observed time points
        sim_df = result.df

        total_loss = 0.0
        for var in self.fit_variables:
            # Get observed values
            observed = self._observed_values[var]

            # Interpolate simulated values to observed times
            try:
                simulated = np.interp(
                    self._observed_times, sim_df["t"].values, sim_df[var].values
                )
            except Exception:
                return 1e10

            # Check for NaN or Inf
            if np.any(~np.isfinite(simulated)):
                return 1e10

            # Compute loss based on type
            if self.loss_type == "mse":
                # Mean squared error
                mse = np.mean((observed - simulated) ** 2)
                total_loss += self.weights[var] * mse
            elif self.loss_type == "likelihood":
                # Negative log-likelihood (Gaussian)
                # Estimate variance from residuals
                residuals = observed - simulated
                variance = max(np.var(residuals), 1e-10)
                n = len(observed)
                nll = 0.5 * n * np.log(2 * np.pi * variance) + np.sum(
                    residuals**2
                ) / (2 * variance)
                total_loss += self.weights[var] * nll

        return total_loss

    def fit(
        self,
        param_bounds: dict[str, tuple[float, float]],
        initial_conditions: dict[str, float] | None = None,
        base_params: "SDTParams | None" = None,
        method: Literal[
            "differential_evolution", "basinhopping", "minimize"
        ] = "differential_evolution",
        dt: float = 1.0,
        seed: int | None = None,
        maxiter: int = 1000,
        tol: float = 1e-6,
        workers: int = 1,
        **optimizer_kwargs: Any,
    ) -> CalibrationResult:
        """Fit model parameters to observed data.

        Args:
            param_bounds: Dictionary mapping parameter names to (min, max) bounds.
            initial_conditions: Initial state for simulations. If None, uses
                the first row of observed_data for fit_variables and defaults
                for others.
            base_params: Base SDTParams to use for non-fitted parameters.
                If None, uses SDTParams defaults.
            method: Optimization method:
                - 'differential_evolution': Global optimizer (default)
                - 'basinhopping': Global optimizer with local refinement
                - 'minimize': Local optimizer (L-BFGS-B)
            dt: Time step for simulations.
            seed: Random seed for reproducibility.
            maxiter: Maximum iterations for optimizer.
            tol: Tolerance for convergence.
            workers: Number of parallel workers (for differential_evolution).
            **optimizer_kwargs: Additional arguments passed to scipy optimizer.

        Returns:
            CalibrationResult with optimized parameters and diagnostics.

        Raises:
            ValueError: If param_bounds is empty or initial_conditions invalid.
        """
        from cliodynamics.models import SDTParams

        if not param_bounds:
            raise ValueError("param_bounds cannot be empty")

        # Set up parameter arrays
        self._param_names = list(param_bounds.keys())
        self._param_bounds = [param_bounds[name] for name in self._param_names]
        self._base_params = base_params
        self._dt = dt
        self._loss_history = []

        # Get time span from observed data
        times = self.observed_data[self.time_column].values
        self._observed_times = times.astype(float)
        self._time_span = (float(times.min()), float(times.max()))

        # Cache observed values
        self._observed_values = {
            var: self.observed_data[var].values.astype(float)
            for var in self.fit_variables
        }

        # Set up initial conditions
        if initial_conditions is None:
            # Use first observation for fit variables, defaults for others
            initial_conditions = {
                "N": 0.5,
                "E": 0.05,
                "W": 1.0,
                "S": 1.0,
                "psi": 0.0,
            }
            for var in self.fit_variables:
                if var in self.observed_data.columns:
                    initial_conditions[var] = float(
                        self.observed_data[var].iloc[0]
                    )
        self._initial_conditions = initial_conditions.copy()

        # Set random state
        rng = np.random.default_rng(seed)

        # Run optimization
        if method == "differential_evolution":
            result = optimize.differential_evolution(
                self._compute_loss,
                bounds=self._param_bounds,
                seed=rng.integers(0, 2**31),
                maxiter=maxiter,
                tol=tol,
                workers=workers,
                updating="deferred" if workers > 1 else "immediate",
                **optimizer_kwargs,
            )
            best_params = dict(zip(self._param_names, result.x))
            n_iterations = result.nit
            converged = result.success
            message = result.message

        elif method == "basinhopping":
            # Start from middle of bounds
            x0 = np.array(
                [(b[0] + b[1]) / 2 for b in self._param_bounds]
            )

            # Define bounds for local optimizer
            bounds_list = list(self._param_bounds)

            result = optimize.basinhopping(
                self._compute_loss,
                x0,
                minimizer_kwargs={
                    "method": "L-BFGS-B",
                    "bounds": bounds_list,
                },
                niter=maxiter,
                seed=rng.integers(0, 2**31),
                **optimizer_kwargs,
            )
            best_params = dict(zip(self._param_names, result.x))
            n_iterations = result.nit
            converged = result.lowest_optimization_result.success
            message = str(result.message)

        elif method == "minimize":
            # Start from middle of bounds
            x0 = np.array(
                [(b[0] + b[1]) / 2 for b in self._param_bounds]
            )

            result = optimize.minimize(
                self._compute_loss,
                x0,
                method="L-BFGS-B",
                bounds=self._param_bounds,
                options={"maxiter": maxiter, "ftol": tol},
                **optimizer_kwargs,
            )
            best_params = dict(zip(self._param_names, result.x))
            n_iterations = result.nit
            converged = result.success
            message = result.message

        else:
            raise ValueError(f"Unknown method: {method}")

        return CalibrationResult(
            best_params=best_params,
            loss=float(result.fun),
            n_iterations=n_iterations,
            converged=converged,
            message=message,
            initial_conditions=self._initial_conditions.copy(),
            param_bounds=param_bounds.copy(),
            loss_history=self._loss_history.copy(),
        )

    def compute_confidence_intervals(
        self,
        result: CalibrationResult,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
        seed: int | None = None,
        method: Literal[
            "differential_evolution", "basinhopping", "minimize"
        ] = "minimize",
        maxiter: int = 100,
    ) -> CalibrationResult:
        """Compute bootstrap confidence intervals for parameters.

        Uses bootstrap resampling of the observed data to estimate
        parameter uncertainty. For each bootstrap sample, re-runs
        the optimization and records the best parameters.

        Args:
            result: CalibrationResult from fit() to augment.
            n_bootstrap: Number of bootstrap samples.
            confidence_level: Confidence level (e.g., 0.95 for 95% CI).
            seed: Random seed for reproducibility.
            method: Optimization method for bootstrap fits.
            maxiter: Maximum iterations per bootstrap fit.

        Returns:
            Updated CalibrationResult with confidence_intervals and
            bootstrap_params populated.
        """
        rng = np.random.default_rng(seed)
        n_obs = len(self.observed_data)

        bootstrap_params_list: list[dict[str, float]] = []

        for i in range(n_bootstrap):
            # Resample observed data with replacement
            indices = rng.integers(0, n_obs, size=n_obs)
            resampled_data = self.observed_data.iloc[indices].copy()
            resampled_data = resampled_data.sort_values(
                self.time_column
            ).reset_index(drop=True)

            # Create new calibrator with resampled data
            bootstrap_calibrator = Calibrator(
                model=self.model_class,
                observed_data=resampled_data,
                fit_variables=self.fit_variables,
                time_column=self.time_column,
                weights=self.weights,
                loss_type=self.loss_type,
            )

            # Run optimization
            try:
                bootstrap_result = bootstrap_calibrator.fit(
                    param_bounds=result.param_bounds,
                    initial_conditions=result.initial_conditions,
                    base_params=self._base_params,
                    method=method,
                    dt=self._dt,
                    seed=rng.integers(0, 2**31),
                    maxiter=maxiter,
                )
                bootstrap_params_list.append(bootstrap_result.best_params)
            except Exception:
                # Skip failed bootstrap samples
                continue

        if len(bootstrap_params_list) < 10:
            # Not enough samples for reliable CI
            return result

        # Convert to array
        param_names = list(result.best_params.keys())
        bootstrap_array = np.array(
            [[params[name] for name in param_names] for params in bootstrap_params_list]
        )

        # Compute confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        confidence_intervals: dict[str, tuple[float, float]] = {}
        for i, name in enumerate(param_names):
            lower = float(np.percentile(bootstrap_array[:, i], lower_percentile))
            upper = float(np.percentile(bootstrap_array[:, i], upper_percentile))
            confidence_intervals[name] = (lower, upper)

        # Update result
        return CalibrationResult(
            best_params=result.best_params,
            loss=result.loss,
            n_iterations=result.n_iterations,
            converged=result.converged,
            message=result.message,
            initial_conditions=result.initial_conditions,
            param_bounds=result.param_bounds,
            confidence_intervals=confidence_intervals,
            bootstrap_params=bootstrap_array,
            loss_history=result.loss_history,
        )

    def cross_validate(
        self,
        param_bounds: dict[str, tuple[float, float]],
        n_folds: int = 5,
        initial_conditions: dict[str, float] | None = None,
        base_params: "SDTParams | None" = None,
        method: Literal[
            "differential_evolution", "basinhopping", "minimize"
        ] = "differential_evolution",
        dt: float = 1.0,
        seed: int | None = None,
        maxiter: int = 1000,
    ) -> dict[str, Any]:
        """Perform time-series cross-validation.

        Splits the data into training and validation sets using
        expanding window cross-validation (train on early data,
        validate on later data).

        Args:
            param_bounds: Dictionary mapping parameter names to bounds.
            n_folds: Number of cross-validation folds.
            initial_conditions: Initial state for simulations.
            base_params: Base SDTParams for non-fitted parameters.
            method: Optimization method.
            dt: Time step for simulations.
            seed: Random seed for reproducibility.
            maxiter: Maximum iterations per fold.

        Returns:
            Dictionary with:
                - 'fold_results': List of CalibrationResult per fold
                - 'train_losses': List of training losses
                - 'val_losses': List of validation losses
                - 'mean_val_loss': Mean validation loss
                - 'std_val_loss': Std of validation losses
        """
        rng = np.random.default_rng(seed)
        n_obs = len(self.observed_data)

        if n_folds < 2:
            raise ValueError("n_folds must be at least 2")

        # Compute fold boundaries (expanding window)
        fold_size = n_obs // n_folds
        if fold_size < 2:
            raise ValueError("Not enough data for requested number of folds")

        fold_results: list[CalibrationResult] = []
        train_losses: list[float] = []
        val_losses: list[float] = []

        for fold in range(1, n_folds):
            # Training data: first fold*fold_size points
            train_end = fold * fold_size
            train_data = self.observed_data.iloc[:train_end].copy()

            # Validation data: next fold_size points
            val_start = train_end
            val_end = min(val_start + fold_size, n_obs)
            val_data = self.observed_data.iloc[val_start:val_end].copy()

            if len(train_data) < 2 or len(val_data) < 1:
                continue

            # Train on training data
            train_calibrator = Calibrator(
                model=self.model_class,
                observed_data=train_data,
                fit_variables=self.fit_variables,
                time_column=self.time_column,
                weights=self.weights,
                loss_type=self.loss_type,
            )

            try:
                result = train_calibrator.fit(
                    param_bounds=param_bounds,
                    initial_conditions=initial_conditions,
                    base_params=base_params,
                    method=method,
                    dt=dt,
                    seed=rng.integers(0, 2**31),
                    maxiter=maxiter,
                )
                fold_results.append(result)
                train_losses.append(result.loss)

                # Evaluate on validation data
                val_calibrator = Calibrator(
                    model=self.model_class,
                    observed_data=val_data,
                    fit_variables=self.fit_variables,
                    time_column=self.time_column,
                    weights=self.weights,
                    loss_type=self.loss_type,
                )

                # Compute validation loss with trained parameters
                val_calibrator._param_names = list(result.best_params.keys())
                val_calibrator._param_bounds = [
                    param_bounds[name] for name in val_calibrator._param_names
                ]
                val_calibrator._base_params = base_params
                val_calibrator._dt = dt
                val_calibrator._observed_times = val_data[
                    self.time_column
                ].values.astype(float)
                val_calibrator._time_span = (
                    float(val_data[self.time_column].min()),
                    float(val_data[self.time_column].max()),
                )
                val_calibrator._observed_values = {
                    var: val_data[var].values.astype(float)
                    for var in self.fit_variables
                }

                # Use initial conditions from end of training period
                val_initial = initial_conditions.copy() if initial_conditions else {
                    "N": 0.5,
                    "E": 0.05,
                    "W": 1.0,
                    "S": 1.0,
                    "psi": 0.0,
                }
                for var in self.fit_variables:
                    if var in train_data.columns:
                        val_initial[var] = float(train_data[var].iloc[-1])
                val_calibrator._initial_conditions = val_initial

                # Compute loss
                param_values = np.array(
                    [result.best_params[name] for name in val_calibrator._param_names]
                )
                val_loss = val_calibrator._compute_loss(param_values)
                val_losses.append(val_loss)

            except Exception:
                continue

        if not val_losses:
            return {
                "fold_results": [],
                "train_losses": [],
                "val_losses": [],
                "mean_val_loss": float("inf"),
                "std_val_loss": float("inf"),
            }

        return {
            "fold_results": fold_results,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "mean_val_loss": float(np.mean(val_losses)),
            "std_val_loss": float(np.std(val_losses)),
        }


def generate_synthetic_data(
    params: "SDTParams",
    initial_conditions: dict[str, float],
    time_span: tuple[float, float],
    dt: float = 1.0,
    noise_std: float = 0.0,
    variables: list[str] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic data from SDT model.

    Useful for testing calibration on known parameters.

    Args:
        params: SDTParams to use for simulation.
        initial_conditions: Initial state.
        time_span: (start, end) time range.
        dt: Time step for output.
        noise_std: Standard deviation of Gaussian noise to add.
        variables: List of variables to include. If None, includes all.
        seed: Random seed for noise generation.

    Returns:
        DataFrame with time series data.
    """
    from cliodynamics.models import SDTModel
    from cliodynamics.simulation import Simulator

    model = SDTModel(params)
    sim = Simulator(model)

    result = sim.run(
        initial_conditions=initial_conditions,
        time_span=time_span,
        dt=dt,
    )

    df = result.df.copy()

    # Rename time column
    df = df.rename(columns={"t": "year"})

    # Add noise
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        if variables is None:
            variables = ["N", "E", "W", "S", "psi"]
        for var in variables:
            if var in df.columns:
                noise = rng.normal(0, noise_std, len(df))
                df[var] = df[var] + noise

    # Select variables
    if variables is not None:
        cols = ["year"] + [v for v in variables if v in df.columns]
        df = df[cols]

    return df


__all__ = [
    "Calibrator",
    "CalibrationResult",
    "generate_synthetic_data",
]
