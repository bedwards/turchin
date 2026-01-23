"""Scenario generation for cliodynamics forecasting.

This module provides tools for defining and managing forecast scenarios,
including baseline projections, policy interventions, and external shocks.

Scenarios allow exploration of "what if" questions:
- What if the wealth pump reverses?
- What happens with elite reduction policies?
- How do external shocks (war, pandemic) affect trajectories?

Example:
    >>> from cliodynamics.forecast.scenarios import (
    ...     Scenario, ScenarioManager, create_standard_scenarios
    ... )
    >>>
    >>> # Create custom scenario
    >>> wealth_pump_off = Scenario(
    ...     name="wealth_pump_off",
    ...     description="Elite extraction rate reduced by policy",
    ...     param_changes={"mu": 0.05},
    ...     start_year=5,
    ...     ramp_years=3
    ... )
    >>>
    >>> # Use scenario manager
    >>> manager = ScenarioManager()
    >>> manager.add_scenario(wealth_pump_off)
    >>> modifiers = manager.get_param_modifiers("wealth_pump_off", t=10)

References:
    Turchin, P. (2016). Ages of Discord. Beresta Books.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scenario:
    """Definition of a forecast scenario.

    A scenario represents a hypothetical modification to the baseline
    forecast, such as policy changes or external shocks.

    Attributes:
        name: Unique identifier for the scenario.
        description: Human-readable description.
        param_changes: Dictionary of parameter changes from baseline.
            Keys are parameter names, values are new parameter values.
        state_changes: Dictionary of state variable shocks.
            Keys are variable names, values are additive changes.
        start_year: Year when scenario begins (relative to forecast start).
        end_year: Year when scenario ends (None for permanent change).
        ramp_years: Years over which change is gradually applied.
            If 0, change is instantaneous.
        probability: Prior probability of scenario occurring (for
            probabilistic scenario analysis).

    Example:
        >>> policy_scenario = Scenario(
        ...     name="elite_tax",
        ...     description="Progressive wealth tax reduces elite extraction",
        ...     param_changes={"mu": 0.1, "alpha": 0.003},
        ...     start_year=5,
        ...     ramp_years=5
        ... )
    """

    name: str
    description: str = ""
    param_changes: dict[str, float] = field(default_factory=dict)
    state_changes: dict[str, float] = field(default_factory=dict)
    start_year: float = 0.0
    end_year: float | None = None
    ramp_years: float = 0.0
    probability: float = 1.0

    def get_param_value(
        self,
        param_name: str,
        baseline_value: float,
        t: float,
    ) -> float:
        """Get parameter value at time t with ramping.

        Args:
            param_name: Name of the parameter.
            baseline_value: Baseline (pre-scenario) parameter value.
            t: Current time (years from forecast start).

        Returns:
            Parameter value accounting for scenario timing and ramping.
        """
        if param_name not in self.param_changes:
            return baseline_value

        target_value = self.param_changes[param_name]

        # Before scenario starts
        if t < self.start_year:
            return baseline_value

        # After scenario ends (if applicable)
        if self.end_year is not None and t > self.end_year:
            return baseline_value

        # During ramp period
        if self.ramp_years > 0 and t < self.start_year + self.ramp_years:
            ramp_fraction = (t - self.start_year) / self.ramp_years
            return baseline_value + ramp_fraction * (target_value - baseline_value)

        # Fully active
        return target_value

    def get_state_shock(
        self,
        var_name: str,
        t: float,
        dt: float = 1.0,
    ) -> float:
        """Get state variable shock at time t.

        Shocks are applied as additive impulses at the start_year.

        Args:
            var_name: Name of the state variable.
            t: Current time.
            dt: Time step for detecting shock application.

        Returns:
            Additive shock to apply (0 if not at shock time).
        """
        if var_name not in self.state_changes:
            return 0.0

        # Apply shock at start_year
        if abs(t - self.start_year) < dt / 2:
            return self.state_changes[var_name]

        return 0.0

    def is_active(self, t: float) -> bool:
        """Check if scenario is active at time t.

        Args:
            t: Current time.

        Returns:
            True if scenario affects dynamics at time t.
        """
        if t < self.start_year:
            return False
        if self.end_year is not None and t > self.end_year:
            return False
        return True


class ScenarioManager:
    """Manage multiple scenarios for forecast analysis.

    The ScenarioManager provides a convenient interface for:
    - Registering and retrieving scenarios
    - Computing combined parameter modifications
    - Scenario comparison and reporting

    Attributes:
        scenarios: Dictionary of registered scenarios.

    Example:
        >>> manager = ScenarioManager()
        >>> manager.add_scenario(Scenario("baseline", "No intervention"))
        >>> manager.add_scenario(Scenario("reform", "Policy reform", {"mu": 0.05}))
        >>> for name, scenario in manager.items():
        ...     print(f"{name}: {scenario.description}")
    """

    def __init__(self) -> None:
        """Initialize the ScenarioManager."""
        self._scenarios: dict[str, Scenario] = {}

    def add_scenario(self, scenario: Scenario) -> None:
        """Register a scenario.

        Args:
            scenario: Scenario to add.

        Raises:
            ValueError: If scenario with same name already exists.
        """
        if scenario.name in self._scenarios:
            raise ValueError(f"Scenario '{scenario.name}' already exists")
        self._scenarios[scenario.name] = scenario

    def get_scenario(self, name: str) -> Scenario:
        """Get a scenario by name.

        Args:
            name: Scenario name.

        Returns:
            The requested Scenario.

        Raises:
            KeyError: If scenario not found.
        """
        if name not in self._scenarios:
            raise KeyError(f"Scenario '{name}' not found")
        return self._scenarios[name]

    def remove_scenario(self, name: str) -> None:
        """Remove a scenario.

        Args:
            name: Scenario name to remove.

        Raises:
            KeyError: If scenario not found.
        """
        if name not in self._scenarios:
            raise KeyError(f"Scenario '{name}' not found")
        del self._scenarios[name]

    def list_scenarios(self) -> list[str]:
        """List all registered scenario names.

        Returns:
            List of scenario names.
        """
        return list(self._scenarios.keys())

    def items(self) -> list[tuple[str, Scenario]]:
        """Iterate over scenarios.

        Returns:
            List of (name, scenario) tuples.
        """
        return list(self._scenarios.items())

    def get_param_modifiers(
        self,
        scenario_name: str,
        t: float = 0.0,
    ) -> dict[str, float]:
        """Get parameter modifiers for a scenario at time t.

        Args:
            scenario_name: Name of the scenario.
            t: Time for evaluating time-dependent modifications.

        Returns:
            Dictionary of {param_name: value} for modified parameters.
        """
        scenario = self.get_scenario(scenario_name)
        return {
            name: scenario.get_param_value(name, 0.0, t)
            for name in scenario.param_changes
        }

    @property
    def scenarios(self) -> dict[str, Scenario]:
        """Access the scenarios dictionary."""
        return self._scenarios.copy()


def create_standard_scenarios() -> ScenarioManager:
    """Create a manager with standard policy scenarios.

    Returns a ScenarioManager populated with common scenarios
    discussed in cliodynamics literature:

    - baseline: No intervention (status quo)
    - wealth_pump_off: Elite extraction reduced
    - elite_reduction: Policies reducing elite overproduction
    - economic_shock: Recession/crisis event
    - reform_package: Combined reforms

    Returns:
        ScenarioManager with standard scenarios.

    Example:
        >>> manager = create_standard_scenarios()
        >>> wealth_off = manager.get_scenario("wealth_pump_off")
        >>> print(wealth_off.description)
    """
    manager = ScenarioManager()

    # Baseline - no changes
    manager.add_scenario(
        Scenario(
            name="baseline",
            description="Continuation of current trends",
            param_changes={},
        )
    )

    # Wealth pump off - reduce elite extraction
    manager.add_scenario(
        Scenario(
            name="wealth_pump_off",
            description="Reduced elite extraction through policy intervention "
            "(progressive taxation, labor protections)",
            param_changes={"mu": 0.05, "eta": 0.5},
            ramp_years=5,
        )
    )

    # Elite reduction - limit elite overproduction
    manager.add_scenario(
        Scenario(
            name="elite_reduction",
            description="Policies reducing elite overproduction "
            "(credential reform, reduced elite positions)",
            param_changes={"alpha": 0.002, "delta_e": 0.03},
            ramp_years=10,
        )
    )

    # Economic shock - sudden crisis
    manager.add_scenario(
        Scenario(
            name="economic_shock",
            description="Major economic crisis (recession, market crash)",
            param_changes={},
            state_changes={"W": -0.1, "S": -0.2},
            start_year=5,
        )
    )

    # Reform package - combined interventions
    manager.add_scenario(
        Scenario(
            name="reform_package",
            description="Comprehensive reform: reduced extraction + elite limits",
            param_changes={"mu": 0.08, "alpha": 0.003, "eta": 0.7, "epsilon": 0.03},
            ramp_years=5,
        )
    )

    # State capacity building
    manager.add_scenario(
        Scenario(
            name="state_strengthening",
            description="Improved state capacity and fiscal reform",
            param_changes={"rho": 0.25, "sigma": 0.08},
            ramp_years=10,
        )
    )

    return manager


def create_shock_scenario(
    name: str,
    description: str,
    shock_type: str = "economic",
    magnitude: float = 0.2,
    start_year: float = 5.0,
) -> Scenario:
    """Create a shock scenario of a given type and magnitude.

    Factory function for creating external shock scenarios.

    Args:
        name: Scenario name.
        description: Description of the shock.
        shock_type: Type of shock:
            - 'economic': Affects wages and state finances
            - 'demographic': Affects population
            - 'political': Increases instability
        magnitude: Size of shock (0-1 scale, fraction of baseline).
        start_year: When shock occurs.

    Returns:
        Scenario configured for the shock.

    Example:
        >>> pandemic = create_shock_scenario(
        ...     "pandemic",
        ...     "Major pandemic event",
        ...     shock_type="demographic",
        ...     magnitude=0.05,
        ...     start_year=2
        ... )
    """
    state_changes: dict[str, float] = {}

    if shock_type == "economic":
        state_changes = {"W": -magnitude, "S": -magnitude * 1.5}
    elif shock_type == "demographic":
        state_changes = {"N": -magnitude}
    elif shock_type == "political":
        state_changes = {"psi": magnitude * 2, "S": -magnitude * 0.5}
    else:
        raise ValueError(f"Unknown shock_type: {shock_type}")

    return Scenario(
        name=name,
        description=description,
        state_changes=state_changes,
        start_year=start_year,
    )


__all__ = [
    "Scenario",
    "ScenarioManager",
    "create_standard_scenarios",
    "create_shock_scenario",
]
