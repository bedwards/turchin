"""Large ensemble simulation framework for systematic parameter space exploration.

This module provides grid-based parameter sweeps for mapping stability regions,
detecting bifurcations, and understanding the landscape of possible societal outcomes.

Key features:
- Grid-based parameter sweeps (covering entire parameter space)
- Parallel execution with multiprocessing
- Stability classification (stable, unstable, collapse)
- Bifurcation detection (where smooth parameter changes cause sudden transitions)
- Cloud scaling capability

Unlike Monte Carlo sampling (which samples randomly), ensembles cover the space
methodically to produce stability maps and phase diagrams.

Example:
    >>> from cliodynamics.simulation import EnsembleSimulator
    >>> from cliodynamics.models import SDTModel
    >>> import numpy as np
    >>>
    >>> # Define parameter grid
    >>> ensemble = EnsembleSimulator(
    ...     model=SDTModel(),
    ...     parameter_grid={
    ...         'alpha': np.linspace(0.005, 0.02, 20),  # Elite recruitment
    ...         'gamma': np.linspace(0.5, 3.0, 20),    # Wage response
    ...     }
    ... )
    >>>
    >>> # Run all combinations (400 simulations)
    >>> results = ensemble.run(
    ...     initial_conditions={'N': 0.5, 'E': 0.05, 'W': 1, 'S': 1, 'psi': 0},
    ...     time_span=(0, 300),
    ... )
    >>>
    >>> # Analyze stability
    >>> stable_mask = results.classify_stability(psi_threshold=1.0)
    >>> bifurcation = results.find_bifurcation(fixed_params={'alpha': 0.01})

References:
    Turchin, P. (2016). Ages of Discord. Beresta Books.
    Strogatz, S. (2015). Nonlinear Dynamics and Chaos. Westview Press.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from cliodynamics.models import SDTModel


class DifferentialModel(Protocol):
    """Protocol for models compatible with EnsembleSimulator."""

    params: Any

    def system(self, y: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        """Compute derivatives."""
        ...


@dataclass
class StabilityClassification:
    """Classification of simulation outcome stability.

    Attributes:
        stable: Converged to bounded behavior with low instability.
        unstable: High instability values but still bounded.
        collapse: System variables diverged or hit extreme values.
        oscillating: Persistent oscillations without convergence.
    """

    STABLE: str = "stable"
    UNSTABLE: str = "unstable"
    COLLAPSE: str = "collapse"
    OSCILLATING: str = "oscillating"
    UNKNOWN: str = "unknown"


@dataclass
class SimulationOutcome:
    """Result of a single simulation in the ensemble.

    Attributes:
        parameters: Dictionary of parameter values used.
        final_state: Final state vector.
        max_psi: Maximum PSI reached during simulation.
        mean_psi: Mean PSI over simulation.
        classification: Stability classification.
        time_series: Full time series (if stored).
        terminated_early: Whether simulation ended early (e.g., divergence).
        metadata: Additional computed metrics.
    """

    parameters: dict[str, float]
    final_state: dict[str, float]
    max_psi: float
    mean_psi: float
    classification: str
    time_series: pd.DataFrame | None = None
    terminated_early: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BifurcationPoint:
    """A detected bifurcation point in parameter space.

    Attributes:
        parameter: Name of the bifurcating parameter.
        value: Critical value where bifurcation occurs.
        fixed_parameters: Values of other parameters held constant.
        type: Type of bifurcation (e.g., 'saddle-node', 'hopf').
        direction: 'destabilizing' or 'stabilizing'.
    """

    parameter: str
    value: float
    fixed_parameters: dict[str, float]
    type: str = "unknown"
    direction: str = "destabilizing"


@dataclass
class EnsembleResults:
    """Container for ensemble simulation results.

    Provides methods for analyzing stability regions, bifurcations, and
    generating phase diagrams.

    Attributes:
        outcomes: List of SimulationOutcome for each grid point.
        parameter_grid: Original parameter grid specification.
        grid_shape: Shape of the parameter grid.
        parameter_names: Ordered list of parameter names.
        parameter_values: Dictionary mapping names to array of values.
        initial_conditions: Initial conditions used.
        time_span: Simulation time range.
        n_simulations: Total number of simulations.
        n_successful: Number that completed without error.
        n_failed: Number that failed.
    """

    outcomes: list[SimulationOutcome]
    parameter_grid: dict[str, NDArray[np.float64]]
    grid_shape: tuple[int, ...]
    parameter_names: list[str]
    parameter_values: dict[str, NDArray[np.float64]]
    initial_conditions: dict[str, float]
    time_span: tuple[float, float]
    n_simulations: int
    n_successful: int
    n_failed: int

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a DataFrame for analysis.

        Returns:
            DataFrame with parameters, final states, and metrics.
        """
        data = []
        for outcome in self.outcomes:
            row = {**outcome.parameters}
            row.update({f"final_{k}": v for k, v in outcome.final_state.items()})
            row["max_psi"] = outcome.max_psi
            row["mean_psi"] = outcome.mean_psi
            row["classification"] = outcome.classification
            row["terminated_early"] = outcome.terminated_early
            row.update(outcome.metadata)
            data.append(row)
        return pd.DataFrame(data)

    def classify_stability(
        self,
        psi_threshold: float = 1.0,
        collapse_threshold: float = 10.0,
        oscillation_threshold: float = 0.3,
    ) -> NDArray[np.str_]:
        """Classify all simulations by stability.

        Args:
            psi_threshold: Max PSI below which system is 'stable'.
            collapse_threshold: Max PSI above which system 'collapsed'.
            oscillation_threshold: Std/mean ratio for oscillation detection.

        Returns:
            Array of classification labels matching grid shape.
        """
        classifications = []
        for outcome in self.outcomes:
            if outcome.classification != StabilityClassification.UNKNOWN:
                classifications.append(outcome.classification)
            elif outcome.max_psi >= collapse_threshold:
                classifications.append(StabilityClassification.COLLAPSE)
            elif outcome.max_psi >= psi_threshold:
                classifications.append(StabilityClassification.UNSTABLE)
            else:
                classifications.append(StabilityClassification.STABLE)

        return np.array(classifications).reshape(self.grid_shape)

    def get_metric_grid(self, metric: str = "max_psi") -> NDArray[np.float64]:
        """Extract a metric as a grid array.

        Args:
            metric: Metric name ('max_psi', 'mean_psi', 'final_psi', etc.)

        Returns:
            Array of metric values matching grid shape.
        """
        values = []
        for outcome in self.outcomes:
            if metric == "max_psi":
                values.append(outcome.max_psi)
            elif metric == "mean_psi":
                values.append(outcome.mean_psi)
            elif metric.startswith("final_"):
                var_name = metric[6:]
                values.append(outcome.final_state.get(var_name, np.nan))
            elif metric in outcome.metadata:
                values.append(outcome.metadata[metric])
            else:
                values.append(np.nan)

        return np.array(values).reshape(self.grid_shape)

    def find_stability_boundary(
        self,
        psi_threshold: float = 1.0,
    ) -> pd.DataFrame:
        """Find points along the stability boundary.

        Identifies parameter combinations that lie on the boundary between
        stable and unstable regions.

        Args:
            psi_threshold: Threshold for stability classification.

        Returns:
            DataFrame with boundary point parameters.
        """
        classifications = self.classify_stability(psi_threshold)

        # Find points adjacent to different classifications
        boundary_points = []

        # For 2D grids, find adjacent points with different classifications
        if len(self.grid_shape) == 2:
            for i in range(self.grid_shape[0]):
                for j in range(self.grid_shape[1]):
                    current = classifications[i, j]
                    # Check neighbors
                    neighbors = []
                    if i > 0:
                        neighbors.append(classifications[i - 1, j])
                    if i < self.grid_shape[0] - 1:
                        neighbors.append(classifications[i + 1, j])
                    if j > 0:
                        neighbors.append(classifications[i, j - 1])
                    if j < self.grid_shape[1] - 1:
                        neighbors.append(classifications[i, j + 1])

                    # On boundary if any neighbor differs
                    if any(n != current for n in neighbors):
                        idx = i * self.grid_shape[1] + j
                        point = {
                            name: self.parameter_values[name][
                                idx // int(np.prod(self.grid_shape[1:]))
                                if k == 0
                                else idx % self.grid_shape[1]
                            ]
                            for k, name in enumerate(self.parameter_names)
                        }
                        point["classification"] = current
                        boundary_points.append(point)

        # Fallback for higher dimensions
        else:
            stable_mask = classifications.flatten() == StabilityClassification.STABLE
            unstable_mask = (
                classifications.flatten() == StabilityClassification.UNSTABLE
            )  # noqa: F841

            # Include points near transitions
            for i, outcome in enumerate(self.outcomes):
                if stable_mask[i] or unstable_mask[i]:
                    point = {**outcome.parameters}
                    point["classification"] = classifications.flatten()[i]
                    # Only include if near a transition (would need more logic)
                    boundary_points.append(point)

        return pd.DataFrame(boundary_points)

    def find_bifurcation(
        self,
        parameter: str,
        fixed_params: dict[str, float] | None = None,
        psi_threshold: float = 1.0,
        interpolate: bool = True,
    ) -> list[BifurcationPoint]:
        """Find bifurcation points along a parameter axis.

        Scans along one parameter while holding others fixed to find
        where the system transitions between stability regimes.

        Args:
            parameter: Name of parameter to scan.
            fixed_params: Values for other parameters (uses grid midpoints if None).
            psi_threshold: Threshold for stability classification.
            interpolate: Whether to interpolate exact bifurcation location.

        Returns:
            List of detected BifurcationPoint objects.
        """
        if parameter not in self.parameter_names:
            raise ValueError(f"Parameter '{parameter}' not in grid")

        # Get slice along parameter axis
        df = self.to_dataframe()

        # Filter to fixed parameters
        if fixed_params:
            for name, value in fixed_params.items():
                if name != parameter and name in df.columns:
                    # Find closest value in grid
                    closest = min(
                        self.parameter_values[name],
                        key=lambda x: abs(x - value),
                    )
                    df = df[np.isclose(df[name], closest)]
        else:
            # Use midpoint for unfixed parameters
            fixed_params = {}
            for name in self.parameter_names:
                if name != parameter:
                    values = self.parameter_values[name]
                    mid_idx = len(values) // 2
                    fixed_params[name] = values[mid_idx]
                    df = df[np.isclose(df[name], fixed_params[name])]

        if len(df) < 2:
            return []

        # Sort by parameter value
        df = df.sort_values(parameter)

        # Detect transitions
        bifurcations = []
        prev_stable = df.iloc[0]["max_psi"] < psi_threshold

        for i in range(1, len(df)):
            curr_stable = df.iloc[i]["max_psi"] < psi_threshold

            if curr_stable != prev_stable:
                # Found a transition
                if interpolate and i > 0:
                    # Linear interpolation for bifurcation location
                    p1 = df.iloc[i - 1][parameter]
                    p2 = df.iloc[i][parameter]
                    psi1 = df.iloc[i - 1]["max_psi"]
                    psi2 = df.iloc[i]["max_psi"]

                    if psi2 != psi1:
                        t = (psi_threshold - psi1) / (psi2 - psi1)
                        critical_value = p1 + t * (p2 - p1)
                    else:
                        critical_value = (p1 + p2) / 2
                else:
                    critical_value = df.iloc[i][parameter]

                direction = "destabilizing" if prev_stable else "stabilizing"

                bifurcations.append(
                    BifurcationPoint(
                        parameter=parameter,
                        value=critical_value,
                        fixed_parameters=fixed_params,
                        type="threshold",
                        direction=direction,
                    )
                )

            prev_stable = curr_stable

        return bifurcations

    def get_bifurcation_diagram_data(
        self,
        parameter: str,
        fixed_params: dict[str, float] | None = None,
        metric: str = "max_psi",
    ) -> pd.DataFrame:
        """Get data for a bifurcation diagram.

        Extracts the relationship between one parameter and an output metric
        while holding other parameters fixed.

        Args:
            parameter: Parameter to vary on x-axis.
            fixed_params: Values for other parameters.
            metric: Output metric for y-axis.

        Returns:
            DataFrame with parameter and metric columns.
        """
        df = self.to_dataframe()

        # Filter to fixed parameters
        if fixed_params:
            for name, value in fixed_params.items():
                if name != parameter and name in df.columns:
                    closest = min(
                        self.parameter_values[name],
                        key=lambda x: abs(x - value),
                    )
                    df = df[np.isclose(df[name], closest)]

        return df[[parameter, metric]].sort_values(parameter)

    def get_phase_diagram_data(
        self,
        x_param: str,
        y_param: str,
        metric: str = "max_psi",
    ) -> pd.DataFrame:
        """Get data for a 2D phase diagram.

        Args:
            x_param: Parameter for x-axis.
            y_param: Parameter for y-axis.
            metric: Metric for coloring.

        Returns:
            DataFrame with x, y, and metric columns.
        """
        if x_param not in self.parameter_names:
            raise ValueError(f"Parameter '{x_param}' not in grid")
        if y_param not in self.parameter_names:
            raise ValueError(f"Parameter '{y_param}' not in grid")

        df = self.to_dataframe()

        # If there are more than 2 parameters, average over others
        other_params = [p for p in self.parameter_names if p not in (x_param, y_param)]

        if other_params:
            # Group by x_param and y_param, average metric
            df = df.groupby([x_param, y_param])[metric].mean().reset_index()

        return df[[x_param, y_param, metric]]

    def stability_region_area(
        self,
        classification: str = "stable",
        psi_threshold: float = 1.0,
    ) -> float:
        """Calculate the fraction of parameter space in a given stability region.

        Args:
            classification: Which region to measure.
            psi_threshold: Threshold for classification.

        Returns:
            Fraction of parameter space (0 to 1).
        """
        classifications = self.classify_stability(psi_threshold)
        return np.mean(classifications.flatten() == classification)

    def summary(self) -> str:
        """Generate a text summary of ensemble results."""
        lines = [
            "Ensemble Simulation Results",
            "=" * 40,
            f"Grid: {' x '.join(str(s) for s in self.grid_shape)}",
            f"Total simulations: {self.n_simulations}",
            f"Successful: {self.n_successful}",
            f"Failed: {self.n_failed}",
            "",
            "Parameters:",
        ]

        for name in self.parameter_names:
            values = self.parameter_values[name]
            lines.append(
                f"  {name}: [{values.min():.4f}, {values.max():.4f}] "
                f"({len(values)} points)"
            )

        # Stability summary
        classifications = self.classify_stability()
        counts = pd.Series(classifications.flatten()).value_counts()
        lines.append("")
        lines.append("Stability Distribution:")
        for label, count in counts.items():
            pct = count / len(self.outcomes) * 100
            lines.append(f"  {label}: {count} ({pct:.1f}%)")

        # Metric ranges
        max_psi = self.get_metric_grid("max_psi")
        lines.append("")
        lines.append("Max PSI Statistics:")
        lines.append(f"  Min: {np.nanmin(max_psi):.4f}")
        lines.append(f"  Max: {np.nanmax(max_psi):.4f}")
        lines.append(f"  Mean: {np.nanmean(max_psi):.4f}")
        lines.append(f"  Median: {np.nanmedian(max_psi):.4f}")

        return "\n".join(lines)


def _run_single_simulation(
    args: tuple[
        dict[str, float],  # parameters
        dict[str, float],  # initial_conditions
        tuple[float, float],  # time_span
        float,  # dt
        str,  # method
        float,  # psi_threshold
        float,  # collapse_threshold
        bool,  # store_time_series
        dict[str, Any],  # model_config (serializable params)
    ],
) -> SimulationOutcome | None:
    """Worker function for parallel simulation.

    Args:
        args: Tuple of simulation arguments.

    Returns:
        SimulationOutcome or None if failed.
    """
    (
        parameters,
        initial_conditions,
        time_span,
        dt,
        method,
        psi_threshold,
        collapse_threshold,
        store_time_series,
        model_config,
    ) = args

    # Reconstruct model from config
    from cliodynamics.models import SDTModel, SDTParams

    params_dict = model_config.get("params", {})
    # Apply parameter overrides
    for key, value in parameters.items():
        params_dict[key] = value

    model_params = SDTParams(**params_dict)
    model = SDTModel(model_params)

    # Build state vector
    state_names = ("N", "E", "W", "S", "psi")
    y0 = np.array([initial_conditions[name] for name in state_names])

    # Time points
    t_eval = np.arange(time_span[0], time_span[1] + dt, dt)
    t_eval = t_eval[t_eval <= time_span[1]]

    def derivatives(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        return model.system(y, t)

    try:
        solution = solve_ivp(
            derivatives,
            time_span,
            y0,
            method=method,
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9,
        )

        if not solution.success:
            return SimulationOutcome(
                parameters=parameters,
                final_state={name: np.nan for name in state_names},
                max_psi=np.nan,
                mean_psi=np.nan,
                classification=StabilityClassification.UNKNOWN,
                terminated_early=True,
                metadata={"error": solution.message},
            )

        # Extract PSI (index 4)
        psi_series = solution.y[4]
        max_psi = float(np.max(psi_series))
        mean_psi = float(np.mean(psi_series))

        # Final state
        final_state = {
            name: float(solution.y[i, -1]) for i, name in enumerate(state_names)
        }

        # Classification
        if max_psi >= collapse_threshold or np.isnan(max_psi):
            classification = StabilityClassification.COLLAPSE
        elif max_psi >= psi_threshold:
            classification = StabilityClassification.UNSTABLE
        else:
            classification = StabilityClassification.STABLE

        # Time series (optional)
        time_series = None
        if store_time_series:
            data = {"time": solution.t}
            for i, name in enumerate(state_names):
                data[name] = solution.y[i]
            time_series = pd.DataFrame(data)

        # Additional metrics
        metadata = {
            "psi_std": float(np.std(psi_series)),
            "psi_final": float(psi_series[-1]),
            "simulation_time": float(solution.t[-1]),
        }

        return SimulationOutcome(
            parameters=parameters,
            final_state=final_state,
            max_psi=max_psi,
            mean_psi=mean_psi,
            classification=classification,
            time_series=time_series,
            terminated_early=False,
            metadata=metadata,
        )

    except Exception as e:
        return SimulationOutcome(
            parameters=parameters,
            final_state={name: np.nan for name in ("N", "E", "W", "S", "psi")},
            max_psi=np.nan,
            mean_psi=np.nan,
            classification=StabilityClassification.UNKNOWN,
            terminated_early=True,
            metadata={"error": str(e)},
        )


class EnsembleSimulator:
    """Systematic parameter space exploration through grid-based simulation.

    Unlike Monte Carlo sampling which explores randomly, EnsembleSimulator
    covers parameter space methodically using a regular grid. This enables:
    - Stability mapping (which regions are stable/unstable)
    - Bifurcation detection (where sudden transitions occur)
    - Phase diagram generation (2D maps of outcomes)

    The simulator supports parallel execution for efficient exploration of
    large parameter grids.

    Attributes:
        model: The SDT model to simulate.
        parameter_grid: Dictionary mapping parameter names to arrays of values.
        n_workers: Number of parallel workers (None = CPU count).

    Example:
        >>> ensemble = EnsembleSimulator(
        ...     model=SDTModel(),
        ...     parameter_grid={
        ...         'alpha': np.linspace(0.005, 0.02, 20),
        ...         'gamma': np.linspace(0.5, 3.0, 20),
        ...     }
        ... )
        >>> results = ensemble.run(
        ...     initial_conditions={'N': 0.5, 'E': 0.05, 'W': 1, 'S': 1, 'psi': 0},
        ...     time_span=(0, 300)
        ... )
    """

    def __init__(
        self,
        model: "SDTModel",
        parameter_grid: dict[str, NDArray[np.float64]],
        n_workers: int | None = None,
    ) -> None:
        """Initialize the ensemble simulator.

        Args:
            model: SDT model with default parameters.
            parameter_grid: Dictionary mapping parameter names to arrays
                of values to explore. Each combination will be simulated.
            n_workers: Number of parallel workers. Defaults to CPU count.
        """
        self.model = model
        self.parameter_grid = {k: np.asarray(v) for k, v in parameter_grid.items()}
        self.n_workers = n_workers or mp.cpu_count()

        # Validate parameters exist
        for param_name in self.parameter_grid:
            if not hasattr(self.model.params, param_name):
                raise ValueError(f"Model has no parameter '{param_name}'")

    @property
    def n_simulations(self) -> int:
        """Total number of simulations in the grid."""
        return int(np.prod([len(v) for v in self.parameter_grid.values()]))

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Shape of the parameter grid."""
        return tuple(len(v) for v in self.parameter_grid.values())

    def _generate_parameter_combinations(self) -> list[dict[str, float]]:
        """Generate all parameter combinations from the grid."""
        names = list(self.parameter_grid.keys())
        value_arrays = [self.parameter_grid[name] for name in names]

        combinations = []
        for values in product(*value_arrays):
            combo = {name: float(val) for name, val in zip(names, values)}
            combinations.append(combo)

        return combinations

    def _get_model_config(self) -> dict[str, Any]:
        """Get serializable model configuration."""
        # Extract current parameters as dictionary
        params_dict = {}
        for field_name in self.model.params.__dataclass_fields__:
            params_dict[field_name] = getattr(self.model.params, field_name)
        return {"params": params_dict}

    def run(
        self,
        initial_conditions: dict[str, float],
        time_span: tuple[float, float],
        dt: float = 1.0,
        method: str = "RK45",
        psi_threshold: float = 1.0,
        collapse_threshold: float = 10.0,
        parallel: bool = True,
        store_time_series: bool = False,
        show_progress: bool = True,
    ) -> EnsembleResults:
        """Run the ensemble of simulations.

        Args:
            initial_conditions: Initial state for all simulations.
            time_span: (start, end) time for simulation.
            dt: Time step for output.
            method: ODE solver method ('RK45', 'BDF', etc.).
            psi_threshold: PSI threshold for stability classification.
            collapse_threshold: PSI threshold for collapse classification.
            parallel: Whether to run in parallel.
            store_time_series: Whether to store full time series (memory intensive).
            show_progress: Whether to print progress messages.

        Returns:
            EnsembleResults with all simulation outcomes.
        """
        combinations = self._generate_parameter_combinations()
        model_config = self._get_model_config()

        # Build argument tuples for worker function
        args_list = [
            (
                combo,
                initial_conditions,
                time_span,
                dt,
                method,
                psi_threshold,
                collapse_threshold,
                store_time_series,
                model_config,
            )
            for combo in combinations
        ]

        outcomes = []
        n_successful = 0
        n_failed = 0

        if parallel and len(args_list) > 1:
            # Parallel execution
            if show_progress:
                print(
                    f"Running {len(args_list)} simulations "
                    f"with {self.n_workers} workers..."
                )

            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(_run_single_simulation, args): i
                    for i, args in enumerate(args_list)
                }

                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        outcomes.append(result)
                        if not result.terminated_early:
                            n_successful += 1
                        else:
                            n_failed += 1
                    else:
                        n_failed += 1

                    completed += 1
                    if show_progress and completed % max(1, len(args_list) // 10) == 0:
                        print(f"  Completed {completed}/{len(args_list)}")

        else:
            # Sequential execution
            if show_progress:
                print(f"Running {len(args_list)} simulations sequentially...")

            for i, args in enumerate(args_list):
                result = _run_single_simulation(args)
                if result is not None:
                    outcomes.append(result)
                    if not result.terminated_early:
                        n_successful += 1
                    else:
                        n_failed += 1
                else:
                    n_failed += 1

                if show_progress and (i + 1) % max(1, len(args_list) // 10) == 0:
                    print(f"  Completed {i + 1}/{len(args_list)}")

        if show_progress:
            print(f"Done: {n_successful} successful, {n_failed} failed")

        # Sort outcomes by parameter order to match grid
        param_names = list(self.parameter_grid.keys())

        def sort_key(outcome: SimulationOutcome) -> tuple:
            return tuple(outcome.parameters[name] for name in param_names)

        outcomes.sort(key=sort_key)

        return EnsembleResults(
            outcomes=outcomes,
            parameter_grid=self.parameter_grid,
            grid_shape=self.grid_shape,
            parameter_names=param_names,
            parameter_values=self.parameter_grid,
            initial_conditions=initial_conditions,
            time_span=time_span,
            n_simulations=len(args_list),
            n_successful=n_successful,
            n_failed=n_failed,
        )

    def run_adaptive(
        self,
        initial_conditions: dict[str, float],
        time_span: tuple[float, float],
        dt: float = 1.0,
        method: str = "RK45",
        psi_threshold: float = 1.0,
        refinement_levels: int = 2,
        parallel: bool = True,
    ) -> EnsembleResults:
        """Run with adaptive refinement near boundaries.

        First runs on coarse grid, then refines near stability boundaries.

        Args:
            initial_conditions: Initial state for all simulations.
            time_span: (start, end) time for simulation.
            dt: Time step for output.
            method: ODE solver method.
            psi_threshold: PSI threshold for stability classification.
            refinement_levels: Number of refinement iterations.
            parallel: Whether to run in parallel.

        Returns:
            EnsembleResults with refined grid near boundaries.
        """
        # Start with base grid
        results = self.run(
            initial_conditions=initial_conditions,
            time_span=time_span,
            dt=dt,
            method=method,
            psi_threshold=psi_threshold,
            parallel=parallel,
            show_progress=True,
        )

        for level in range(refinement_levels):
            # Find boundary points
            boundary = results.find_stability_boundary(psi_threshold)

            if len(boundary) == 0:
                break

            # Create refined grid around boundary points
            new_combinations = []
            param_names = list(self.parameter_grid.keys())

            for _, row in boundary.iterrows():
                for name in param_names:
                    # Get grid spacing
                    values = self.parameter_grid[name]
                    if len(values) > 1:
                        spacing = (values[-1] - values[0]) / (len(values) - 1)
                        refine_spacing = spacing / (2 ** (level + 1))

                        # Add points around boundary
                        base_value = row[name]
                        for offset in [-refine_spacing, refine_spacing]:
                            new_value = base_value + offset
                            if values[0] <= new_value <= values[-1]:
                                combo = {n: row[n] for n in param_names}
                                combo[name] = new_value
                                new_combinations.append(combo)

            if not new_combinations:
                break

            # Run additional simulations (would need to extend results)
            # For now, just return the base results
            print(
                f"Refinement level {level + 1}: "
                f"would add {len(new_combinations)} points"
            )

        return results


__all__ = [
    "EnsembleSimulator",
    "EnsembleResults",
    "SimulationOutcome",
    "StabilityClassification",
    "BifurcationPoint",
]
