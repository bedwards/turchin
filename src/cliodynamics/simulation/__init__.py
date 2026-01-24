"""ODE solver integration and simulation runner for cliodynamics models.

This module provides a flexible simulation harness for running SDT and other
models using scipy's ODE solvers with support for:
- Multiple solver methods (RK45, BDF for stiff systems)
- Event detection (e.g., state collapse thresholds)
- Results as pandas DataFrame

Example:
    >>> from cliodynamics.models import SDTModel, SDTParams
    >>> from cliodynamics.simulation import Simulator
    >>>
    >>> params = SDTParams(r_max=0.02, K_0=100_000_000)
    >>> model = SDTModel(params)
    >>> sim = Simulator(model)
    >>> initial = {'N': 50_000_000, 'E': 100_000, 'W': 1.0, 'S': 1.0, 'psi': 0.1}
    >>> results = sim.run(initial, time_span=(0, 300), dt=1.0)
    >>> results.to_csv("simulation_output.csv")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp


class DifferentialModel(Protocol):
    """Protocol for models compatible with the Simulator.

    Any model implementing this protocol can be used with Simulator.
    The model must provide a system() method that computes derivatives.
    """

    def system(self, y: ArrayLike, t: float) -> NDArray[np.float64]:
        """Compute the derivatives of all state variables.

        Args:
            y: Current state vector.
            t: Current time.

        Returns:
            Array of derivatives dy/dt.
        """
        ...


@dataclass
class Event:
    """Event specification for detecting threshold crossings during simulation.

    Events are used to detect when state variables cross specified thresholds,
    which can optionally terminate the simulation.

    Attributes:
        name: Human-readable name for the event.
        variable: Name of the state variable to monitor (e.g., 'psi', 'S').
        threshold: Value at which the event triggers.
        direction: Direction of crossing to detect:
            - 'rising': trigger when variable crosses threshold from below
            - 'falling': trigger when variable crosses threshold from above
            - 'both': trigger on any crossing
        terminal: If True, stop simulation when event occurs.

    Example:
        >>> # Detect state collapse (S drops below 0.1)
        >>> collapse_event = Event(
        ...     name="state_collapse",
        ...     variable="S",
        ...     threshold=0.1,
        ...     direction="falling",
        ...     terminal=True
        ... )
    """

    name: str
    variable: str
    threshold: float
    direction: str = "both"  # 'rising', 'falling', or 'both'
    terminal: bool = False

    def __post_init__(self) -> None:
        """Validate event configuration."""
        if self.direction not in ("rising", "falling", "both"):
            msg = "direction must be 'rising', 'falling', or 'both', "
            msg += f"got '{self.direction}'"
            raise ValueError(msg)


@dataclass
class EventRecord:
    """Record of an event occurrence during simulation.

    Attributes:
        event: The Event specification that triggered.
        time: Time at which the event occurred.
        state: State vector at the time of the event.
    """

    event: Event
    time: float
    state: dict[str, float]


@dataclass
class SimulationResult:
    """Container for simulation results with convenient accessors.

    Attributes:
        df: DataFrame with time series of all state variables.
        events: List of events that occurred during simulation.
        terminated_by_event: True if simulation ended due to a terminal event.
        solver_message: Status message from the ODE solver.
    """

    df: pd.DataFrame
    events: list[EventRecord] = field(default_factory=list)
    terminated_by_event: bool = False
    solver_message: str = ""

    def to_csv(self, path: str, **kwargs: Any) -> None:
        """Save results to CSV file.

        Args:
            path: Output file path.
            **kwargs: Additional arguments passed to DataFrame.to_csv().
        """
        self.df.to_csv(path, index=False, **kwargs)

    def __getitem__(self, key: str) -> pd.Series:
        """Access a column by name.

        Args:
            key: Column name (e.g., 't', 'N', 'psi').

        Returns:
            Series with the requested data.
        """
        return self.df[key]

    @property
    def t(self) -> NDArray[np.float64]:
        """Time values as numpy array."""
        return self.df["t"].values

    @property
    def final_state(self) -> dict[str, float]:
        """State variables at the final time point."""
        return {col: self.df[col].iloc[-1] for col in self.df.columns if col != "t"}


class Simulator:
    """Simulation harness for ODE-based models.

    Wraps scipy.integrate.solve_ivp to provide a convenient interface for
    running simulations with event detection and DataFrame output.

    The Simulator is designed to work with any model that implements the
    DifferentialModel protocol (i.e., has a system(y, t) method).

    Attributes:
        model: The differential equation model to simulate.
        state_names: Names of state variables in order.

    Example:
        >>> from cliodynamics.models import SDTModel, SDTParams
        >>> params = SDTParams(r_max=0.02, K_0=1.0)
        >>> model = SDTModel(params)
        >>> sim = Simulator(model)
        >>> initial = {'N': 0.5, 'E': 0.05, 'W': 1.0, 'S': 1.0, 'psi': 0.0}
        >>> results = sim.run(initial, time_span=(0, 100), dt=0.5)
    """

    # Default state variable names for SDT model
    DEFAULT_STATE_NAMES = ("N", "E", "W", "S", "psi")

    def __init__(
        self,
        model: DifferentialModel,
        state_names: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize the simulator.

        Args:
            model: A model with a system(y, t) method for computing derivatives.
            state_names: Names of state variables. Defaults to SDT names
                ('N', 'E', 'W', 'S', 'psi').
        """
        self.model = model
        self.state_names = state_names or self.DEFAULT_STATE_NAMES

    def _derivatives_for_ivp(
        self, t: float, y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Wrapper to match solve_ivp's expected signature (t, y).

        scipy.integrate.solve_ivp expects f(t, y), but our model.system
        uses (y, t) to match odeint convention.

        Args:
            t: Current time.
            y: Current state vector.

        Returns:
            Derivatives dy/dt.
        """
        return self.model.system(y, t)

    def _create_event_function(
        self, event: Event, var_index: int
    ) -> Callable[[float, NDArray[np.float64]], float]:
        """Create an event function for solve_ivp.

        Args:
            event: Event specification.
            var_index: Index of the variable in the state vector.

        Returns:
            Event function that returns 0 at threshold crossing.
        """

        def event_func(t: float, y: NDArray[np.float64]) -> float:
            return y[var_index] - event.threshold

        # Set direction attribute for solve_ivp
        if event.direction == "rising":
            event_func.direction = 1  # type: ignore[attr-defined]
        elif event.direction == "falling":
            event_func.direction = -1  # type: ignore[attr-defined]
        else:
            event_func.direction = 0  # type: ignore[attr-defined]

        event_func.terminal = event.terminal  # type: ignore[attr-defined]

        return event_func

    def run(
        self,
        initial_conditions: dict[str, float],
        time_span: tuple[float, float],
        dt: float = 1.0,
        method: str = "RK45",
        events: list[Event] | None = None,
        dense_output: bool = False,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        max_step: float = np.inf,
    ) -> SimulationResult:
        """Run a simulation with the configured model.

        Args:
            initial_conditions: Dictionary mapping state variable names to
                initial values. Must include all state variables.
            time_span: Tuple of (start_time, end_time) for simulation.
            dt: Time step for output (solver may use smaller internal steps).
            method: ODE solver method. Options:
                - 'RK45': Explicit Runge-Kutta (default, good for non-stiff)
                - 'RK23': Lower-order Runge-Kutta
                - 'DOP853': High-order Runge-Kutta
                - 'Radau': Implicit Runge-Kutta (good for stiff systems)
                - 'BDF': Backward differentiation (good for stiff systems)
                - 'LSODA': Auto-switching stiff/non-stiff
            events: List of Event objects for threshold detection.
            dense_output: If True, use continuous solution for output times.
            rtol: Relative tolerance for solver.
            atol: Absolute tolerance for solver.
            max_step: Maximum step size for solver.

        Returns:
            SimulationResult with DataFrame of time series and event records.

        Raises:
            ValueError: If initial_conditions is missing required variables.

        Example:
            >>> initial = {'N': 0.5, 'E': 0.05, 'W': 1.0, 'S': 1.0, 'psi': 0.0}
            >>> collapse = Event(
            ...     name="collapse", variable="S", threshold=0.1,
            ...     direction="falling", terminal=True
            ... )
            >>> results = sim.run(
            ...     initial, time_span=(0, 300), dt=1.0,
            ...     method='BDF', events=[collapse]
            ... )
        """
        # Validate initial conditions
        missing = set(self.state_names) - set(initial_conditions.keys())
        if missing:
            raise ValueError(f"Missing initial conditions for: {missing}")

        # Build initial state vector in correct order
        y0 = np.array([initial_conditions[name] for name in self.state_names])

        # Build evaluation time points
        t_start, t_end = time_span
        t_eval = np.arange(t_start, t_end + dt, dt)
        # Ensure we don't go past t_end
        t_eval = t_eval[t_eval <= t_end]

        # Build event functions
        event_funcs: list[Callable[[float, NDArray[np.float64]], float]] = []
        event_specs: list[Event] = events or []
        for event in event_specs:
            if event.variable not in self.state_names:
                msg = f"Event variable '{event.variable}' not in "
                msg += f"state_names: {self.state_names}"
                raise ValueError(msg)
            var_index = self.state_names.index(event.variable)
            event_funcs.append(self._create_event_function(event, var_index))

        # Run solver
        solution = solve_ivp(
            self._derivatives_for_ivp,
            time_span,
            y0,
            method=method,
            t_eval=t_eval,
            events=event_funcs if event_funcs else None,
            dense_output=dense_output,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )

        # Build DataFrame from solution
        data = {"t": solution.t}
        for i, name in enumerate(self.state_names):
            data[name] = solution.y[i]
        df = pd.DataFrame(data)

        # Process events
        event_records: list[EventRecord] = []
        terminated_by_event = False

        if solution.t_events is not None:
            for i, (t_events, y_events) in enumerate(
                zip(solution.t_events, solution.y_events)
            ):
                for t_event, y_event in zip(t_events, y_events):
                    state = {
                        name: y_event[j] for j, name in enumerate(self.state_names)
                    }
                    event_records.append(
                        EventRecord(event=event_specs[i], time=t_event, state=state)
                    )
                    if event_specs[i].terminal:
                        terminated_by_event = True

        return SimulationResult(
            df=df,
            events=event_records,
            terminated_by_event=terminated_by_event,
            solver_message=solution.message,
        )

    def run_with_parameter_sweep(
        self,
        initial_conditions: dict[str, float],
        time_span: tuple[float, float],
        parameter_name: str,
        parameter_values: ArrayLike,
        dt: float = 1.0,
        method: str = "RK45",
        **kwargs: Any,
    ) -> dict[float, SimulationResult]:
        """Run multiple simulations varying a single parameter.

        Useful for sensitivity analysis and exploring parameter space.

        Args:
            initial_conditions: Initial state for all runs.
            time_span: Simulation time range.
            parameter_name: Name of parameter to vary
                (must be attribute of model.params).
            parameter_values: Array of parameter values to test.
            dt: Output time step.
            method: ODE solver method.
            **kwargs: Additional arguments passed to run().

        Returns:
            Dictionary mapping parameter values to SimulationResult objects.

        Example:
            >>> initial = {'N': 0.5, 'E': 0.05, 'W': 1.0, 'S': 1.0, 'psi': 0.0}
            >>> results = sim.run_with_parameter_sweep(
            ...     initial, time_span=(0, 200),
            ...     parameter_name='r_max',
            ...     parameter_values=[0.01, 0.02, 0.03, 0.04],
            ... )
        """
        results: dict[float, SimulationResult] = {}

        # Store original parameter value
        original_value = getattr(self.model.params, parameter_name)

        try:
            for value in parameter_values:
                setattr(self.model.params, parameter_name, value)
                results[float(value)] = self.run(
                    initial_conditions=initial_conditions,
                    time_span=time_span,
                    dt=dt,
                    method=method,
                    **kwargs,
                )
        finally:
            # Restore original value
            setattr(self.model.params, parameter_name, original_value)

        return results



# Import Monte Carlo classes
from cliodynamics.simulation.monte_carlo import (
    Constant,
    Distribution,
    LogNormal,
    MonteCarloResults,
    MonteCarloSimulator,
    Normal,
    Triangular,
    TruncatedNormal,
    Uniform,
)

__all__ = [
    "Simulator",
    "Event",
    "EventRecord",
    "SimulationResult",
    "DifferentialModel",
    # Monte Carlo
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
