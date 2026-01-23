"""Counterfactual simulation engine for policy analysis.

This module provides the core simulation infrastructure for running
counterfactual scenarios with policy interventions applied.

The CounterfactualEngine wraps the standard Simulator to apply
interventions during integration, allowing comparison of trajectories
with and without policy changes.

Example:
    >>> from cliodynamics.policy import CounterfactualEngine, EliteCap
    >>> from cliodynamics.models import SDTModel, SDTParams
    >>>
    >>> model = SDTModel(SDTParams())
    >>> engine = CounterfactualEngine(model)
    >>>
    >>> # Get baseline result
    >>> baseline = engine.run_baseline(
    ...     {'N': 0.5, 'E': 0.05, 'W': 1.0, 'S': 1.0, 'psi': 0.0},
    ...     time_span=(0, 200)
    ... )
    >>>
    >>> # Run counterfactual with elite cap
    >>> intervention = EliteCap(name="Elite cap", start_time=50, max_elite_ratio=1.5)
    >>> result = engine.run_intervention(baseline, intervention)
    >>>
    >>> # Compare PSI peaks
    >>> print(f"Baseline PSI peak: {baseline.result.df['psi'].max():.2f}")
    >>> print(f"Intervention PSI peak: {result.result.df['psi'].max():.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from cliodynamics.models import SDTModel
    from cliodynamics.models.params import SDTParams
    from cliodynamics.simulation import SimulationResult

from cliodynamics.policy.interventions import Intervention


@dataclass
class CounterfactualResult:
    """Container for counterfactual simulation results.

    Attributes:
        result: The SimulationResult from the counterfactual run.
        intervention: The intervention that was applied (or None for baseline).
        baseline_reference: Reference to the baseline CounterfactualResult
            if this is an intervention run.
        intervention_start_state: State at intervention start time.
        metadata: Additional metadata about the run.
    """

    result: "SimulationResult"
    intervention: Intervention | None = None
    baseline_reference: "CounterfactualResult | None" = None
    intervention_start_state: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def df(self) -> pd.DataFrame:
        """Access the results DataFrame."""
        return self.result.df

    @property
    def is_baseline(self) -> bool:
        """Check if this is a baseline (no intervention) result."""
        return self.intervention is None

    @property
    def psi_peak(self) -> float:
        """Maximum PSI value in the trajectory."""
        return float(self.result.df["psi"].max())

    @property
    def psi_peak_time(self) -> float:
        """Time at which PSI reaches maximum."""
        df = self.result.df
        return float(df.loc[df["psi"].idxmax(), "t"])

    @property
    def final_state(self) -> dict[str, float]:
        """State at the end of simulation."""
        return self.result.final_state

    def state_at_time(self, t: float) -> dict[str, float]:
        """Get interpolated state at a specific time.

        Args:
            t: Time to query.

        Returns:
            Dictionary of state variables at time t.
        """
        df = self.result.df
        state = {}
        for col in ["N", "E", "W", "S", "psi"]:
            if col in df.columns:
                state[col] = float(np.interp(t, df["t"], df[col]))
        return state


class InterventionModel:
    """SDT model wrapper that applies interventions during simulation.

    This class wraps an SDTModel and modifies its dynamics according
    to active interventions. It implements the same interface as SDTModel
    so it can be used with the Simulator.

    Attributes:
        base_model: The underlying SDTModel.
        interventions: List of active interventions.
    """

    def __init__(
        self,
        base_model: "SDTModel",
        interventions: list[Intervention] | None = None,
    ) -> None:
        """Initialize the intervention model.

        Args:
            base_model: The SDTModel to wrap.
            interventions: List of interventions to apply.
        """
        self.base_model = base_model
        self.interventions = interventions or []

    @property
    def params(self) -> "SDTParams":
        """Access the underlying model parameters."""
        return self.base_model.params

    def system(self, y: ArrayLike, t: float) -> NDArray[np.float64]:
        """Compute system derivatives with interventions applied.

        First applies state modifications, then computes derivatives,
        then applies derivative modifications.

        Args:
            y: Current state vector [N, E, W, S, psi].
            t: Current time.

        Returns:
            Modified derivatives.
        """
        state = np.array(y, dtype=np.float64)

        # Apply state modifications from interventions
        for intervention in self.interventions:
            state = intervention.modify_state(state, t, self.params)

        # Compute base derivatives
        derivatives = self.base_model.system(state, t)

        # Apply derivative modifications from interventions
        for intervention in self.interventions:
            derivatives = intervention.modify_derivatives(
                derivatives, state, t, self.params
            )

        return derivatives


class CounterfactualEngine:
    """Engine for running counterfactual policy simulations.

    The engine provides methods for:
    - Running baseline simulations
    - Applying interventions at specified times
    - Comparing trajectories between scenarios
    - Batch simulation of multiple interventions

    Attributes:
        model: The SDTModel to use for simulations.
        state_names: Names of state variables.
    """

    DEFAULT_STATE_NAMES = ("N", "E", "W", "S", "psi")

    def __init__(
        self,
        model: "SDTModel",
        state_names: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize the counterfactual engine.

        Args:
            model: SDTModel to use for simulations.
            state_names: Names of state variables.
        """
        self.model = model
        self.state_names = state_names or self.DEFAULT_STATE_NAMES

    def run_baseline(
        self,
        initial_conditions: dict[str, float],
        time_span: tuple[float, float],
        dt: float = 1.0,
        method: str = "RK45",
        **kwargs: Any,
    ) -> CounterfactualResult:
        """Run a baseline simulation without interventions.

        Args:
            initial_conditions: Initial state dictionary.
            time_span: (start_time, end_time) tuple.
            dt: Output time step.
            method: ODE solver method.
            **kwargs: Additional arguments for solve_ivp.

        Returns:
            CounterfactualResult for the baseline trajectory.
        """
        from cliodynamics.simulation import Simulator

        sim = Simulator(self.model, state_names=self.state_names)
        result = sim.run(
            initial_conditions=initial_conditions,
            time_span=time_span,
            dt=dt,
            method=method,
            **kwargs,
        )

        return CounterfactualResult(
            result=result,
            intervention=None,
            baseline_reference=None,
            metadata={
                "initial_conditions": initial_conditions.copy(),
                "time_span": time_span,
                "dt": dt,
                "method": method,
            },
        )

    def run_intervention(
        self,
        baseline: CounterfactualResult,
        intervention: Intervention,
        dt: float | None = None,
        method: str | None = None,
        **kwargs: Any,
    ) -> CounterfactualResult:
        """Run a counterfactual with an intervention applied.

        The simulation uses the same initial conditions and time span
        as the baseline, but with the intervention modifying dynamics.

        Args:
            baseline: The baseline CounterfactualResult to compare against.
            intervention: The intervention to apply.
            dt: Output time step (defaults to baseline's dt).
            method: ODE solver method (defaults to baseline's method).
            **kwargs: Additional arguments for solve_ivp.

        Returns:
            CounterfactualResult for the intervention scenario.
        """
        # Get baseline parameters
        meta = baseline.metadata
        initial_conditions = meta.get("initial_conditions", baseline.final_state)
        time_span = meta.get("time_span", (0.0, 200.0))
        dt = dt or meta.get("dt", 1.0)
        method = method or meta.get("method", "RK45")

        # Create intervention model
        intervention_model = InterventionModel(self.model, [intervention])

        # Run simulation using solve_ivp directly
        y0 = np.array([initial_conditions[name] for name in self.state_names])
        t_start, t_end = time_span
        t_eval = np.arange(t_start, t_end + dt, dt)
        t_eval = t_eval[t_eval <= t_end]

        def system(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            return intervention_model.system(y, t)

        solution = solve_ivp(
            system,
            time_span,
            y0,
            method=method,
            t_eval=t_eval,
            **kwargs,
        )

        # Build DataFrame
        data = {"t": solution.t}
        for i, name in enumerate(self.state_names):
            data[name] = solution.y[i]
        df = pd.DataFrame(data)

        # Create SimulationResult-like object
        from cliodynamics.simulation import SimulationResult

        result = SimulationResult(
            df=df,
            events=[],
            terminated_by_event=False,
            solver_message=solution.message,
        )

        # Get state at intervention start
        intervention_start_state = None
        if intervention.start_time >= time_span[0]:
            intervention_start_state = baseline.state_at_time(intervention.start_time)

        return CounterfactualResult(
            result=result,
            intervention=intervention,
            baseline_reference=baseline,
            intervention_start_state=intervention_start_state,
            metadata={
                "initial_conditions": initial_conditions.copy(),
                "time_span": time_span,
                "dt": dt,
                "method": method,
            },
        )

    def run_interventions(
        self,
        baseline: CounterfactualResult,
        interventions: list[Intervention],
        dt: float | None = None,
        method: str | None = None,
        **kwargs: Any,
    ) -> list[CounterfactualResult]:
        """Run multiple interventions against the same baseline.

        Args:
            baseline: The baseline CounterfactualResult.
            interventions: List of interventions to test.
            dt: Output time step.
            method: ODE solver method.
            **kwargs: Additional arguments for solve_ivp.

        Returns:
            List of CounterfactualResults, one per intervention.
        """
        results = []
        for intervention in interventions:
            result = self.run_intervention(
                baseline, intervention, dt=dt, method=method, **kwargs
            )
            results.append(result)
        return results

    def run_from_state(
        self,
        state: dict[str, float],
        time_span: tuple[float, float],
        intervention: Intervention | None = None,
        dt: float = 1.0,
        method: str = "RK45",
        **kwargs: Any,
    ) -> CounterfactualResult:
        """Run simulation from a specified state.

        Useful for branching simulations: run baseline until time T,
        then branch with an intervention from that point.

        Args:
            state: State dictionary to start from.
            time_span: (start_time, end_time) tuple.
            intervention: Optional intervention to apply.
            dt: Output time step.
            method: ODE solver method.
            **kwargs: Additional arguments for solve_ivp.

        Returns:
            CounterfactualResult for the trajectory.
        """
        if intervention is None:
            return self.run_baseline(
                initial_conditions=state,
                time_span=time_span,
                dt=dt,
                method=method,
                **kwargs,
            )

        # Create intervention model
        intervention_model = InterventionModel(self.model, [intervention])

        # Run simulation
        y0 = np.array([state[name] for name in self.state_names])
        t_start, t_end = time_span
        t_eval = np.arange(t_start, t_end + dt, dt)
        t_eval = t_eval[t_eval <= t_end]

        def system(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            return intervention_model.system(y, t)

        solution = solve_ivp(
            system,
            time_span,
            y0,
            method=method,
            t_eval=t_eval,
            **kwargs,
        )

        # Build DataFrame
        data = {"t": solution.t}
        for i, name in enumerate(self.state_names):
            data[name] = solution.y[i]
        df = pd.DataFrame(data)

        from cliodynamics.simulation import SimulationResult

        result = SimulationResult(
            df=df,
            events=[],
            terminated_by_event=False,
            solver_message=solution.message,
        )

        return CounterfactualResult(
            result=result,
            intervention=intervention,
            baseline_reference=None,
            intervention_start_state=state.copy(),
            metadata={
                "initial_conditions": state.copy(),
                "time_span": time_span,
                "dt": dt,
                "method": method,
            },
        )

    def find_intervention_timing(
        self,
        baseline: CounterfactualResult,
        intervention_factory: Any,  # Callable[[float], Intervention]
        start_times: list[float],
        metric: str = "psi_peak",
    ) -> dict[float, tuple[float, CounterfactualResult]]:
        """Find optimal intervention timing.

        Tests an intervention applied at different times to find
        when it is most effective.

        Args:
            baseline: The baseline CounterfactualResult.
            intervention_factory: Function(start_time) -> Intervention.
            start_times: List of start times to test.
            metric: Metric to optimize ("psi_peak" or "final_psi").

        Returns:
            Dictionary mapping start_time to (metric_value, result).
        """
        results: dict[float, tuple[float, CounterfactualResult]] = {}

        for start_time in start_times:
            intervention = intervention_factory(start_time)
            result = self.run_intervention(baseline, intervention)

            if metric == "psi_peak":
                value = result.psi_peak
            elif metric == "final_psi":
                value = result.df["psi"].iloc[-1]
            else:
                raise ValueError(f"Unknown metric: {metric}")

            results[start_time] = (value, result)

        return results


__all__ = [
    "CounterfactualEngine",
    "CounterfactualResult",
    "InterventionModel",
]
