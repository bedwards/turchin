"""Policy intervention types for counterfactual analysis.

This module defines various policy interventions that can be applied
to SDT simulations to explore counterfactual scenarios.

Interventions modify the dynamics of the system at specified times,
allowing exploration of questions like:
- "What if Rome had capped elite growth in 100 BCE?"
- "What if US wages had tracked productivity since 1970?"

Intervention Categories:
1. Elite Management: caps, purges, co-optation, credentialing
2. Popular Welfare: wage floors, land reform, labor rights
3. Fiscal Policy: taxation, spending, debt
4. Demographics: migration, frontier expansion
5. Institutions: democratic reform, representation

Each intervention specifies:
- When it activates (start_time, end_time)
- What it modifies (target variable or parameter)
- How it modifies (absolute value, multiplier, cap, floor)

Example:
    >>> intervention = EliteCap(
    ...     start_time=100,
    ...     max_elite_ratio=1.5,
    ...     name="Cap elite growth to 1.5x baseline"
    ... )
    >>> # Intervention activates at t=100 and prevents E from exceeding 1.5*E_0

References:
    Turchin (2016), Ages of Discord, Chapter 10
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from cliodynamics.models.params import SDTParams


@dataclass
class Intervention(ABC):
    """Base class for all policy interventions.

    An intervention modifies the SDT system dynamics at specified times.
    Subclasses implement specific intervention types (elite caps, wage
    floors, etc.).

    Attributes:
        name: Human-readable name for the intervention.
        start_time: Time at which intervention becomes active.
        end_time: Time at which intervention ends. If None, runs forever.
        description: Detailed description of the intervention.

    Note:
        Interventions are immutable after creation. To modify, create
        a new intervention with updated parameters.
    """

    name: str
    start_time: float
    end_time: float | None = None
    description: str = ""

    def is_active(self, t: float) -> bool:
        """Check if intervention is active at time t.

        Args:
            t: Current simulation time.

        Returns:
            True if start_time <= t < end_time (or forever if end_time is None).
        """
        if t < self.start_time:
            return False
        if self.end_time is not None and t >= self.end_time:
            return False
        return True

    @abstractmethod
    def modify_state(
        self,
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """Modify the state vector during simulation.

        This method is called at each time step to potentially modify
        the state variables. The default implementation returns the
        state unchanged.

        Args:
            state: Current state vector [N, E, W, S, psi].
            t: Current time.
            params: Model parameters.

        Returns:
            Modified state vector.
        """
        ...

    @abstractmethod
    def modify_derivatives(
        self,
        derivatives: NDArray[np.float64],
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """Modify the derivatives during simulation.

        This method is called to modify dY/dt before integration.
        Use this for interventions that change dynamics rather than
        directly setting state values.

        Args:
            derivatives: Current derivatives [dN/dt, dE/dt, dW/dt, dS/dt, dpsi/dt].
            state: Current state vector [N, E, W, S, psi].
            t: Current time.
            params: Model parameters.

        Returns:
            Modified derivatives.
        """
        ...


# State variable indices for SDT model
_N_IDX = 0
_E_IDX = 1
_W_IDX = 2
_S_IDX = 3
_PSI_IDX = 4


@dataclass
class EliteCap(Intervention):
    """Cap elite population growth.

    Limits the elite population to a maximum ratio of the baseline E_0.
    This simulates policies that restrict entry into the elite class:
    - Credential restrictions (limiting law degrees, PhDs)
    - Wealth taxes that prevent accumulation
    - Primogeniture or inheritance limits

    Historical examples:
    - Roman Lex Claudia restricting senatorial commerce
    - Medieval guild restrictions
    - Modern professional licensing

    Attributes:
        max_elite_ratio: Maximum E/E_0 ratio allowed.
    """

    max_elite_ratio: float = 1.5

    def modify_state(
        self,
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """Cap elite population at max_elite_ratio * E_0."""
        if not self.is_active(t):
            return state

        modified = state.copy()
        max_E = self.max_elite_ratio * params.E_0
        if modified[_E_IDX] > max_E:
            modified[_E_IDX] = max_E
        return modified

    def modify_derivatives(
        self,
        derivatives: NDArray[np.float64],
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """Prevent positive dE/dt when at cap."""
        if not self.is_active(t):
            return derivatives

        modified = derivatives.copy()
        max_E = self.max_elite_ratio * params.E_0

        # If at or above cap and derivatives positive, zero out growth
        if state[_E_IDX] >= max_E and modified[_E_IDX] > 0:
            modified[_E_IDX] = 0.0

        return modified


@dataclass
class ElitePurge(Intervention):
    """Reduce elite population by a fraction.

    Simulates events that reduce elite numbers:
    - Civil wars eliminating rival factions
    - Revolutions (French, Russian)
    - Land reforms redistributing wealth

    This is a discrete intervention: it reduces E once at start_time.

    Attributes:
        reduction_fraction: Fraction of elites removed (0.0 to 1.0).
    """

    reduction_fraction: float = 0.2

    def __post_init__(self) -> None:
        """Validate reduction fraction."""
        if not 0 <= self.reduction_fraction <= 1:
            raise ValueError("reduction_fraction must be between 0 and 1")
        # Track if already applied
        self._applied = False

    def modify_state(
        self,
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """Apply one-time reduction at start_time."""
        if not self.is_active(t):
            return state

        # Only apply once
        if self._applied:
            return state

        modified = state.copy()
        modified[_E_IDX] *= 1 - self.reduction_fraction
        self._applied = True
        return modified

    def modify_derivatives(
        self,
        derivatives: NDArray[np.float64],
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """No derivative modification for discrete intervention."""
        return derivatives


@dataclass
class WageFloor(Intervention):
    """Set a minimum wage level.

    Prevents wages from falling below a specified fraction of baseline.
    Simulates minimum wage laws, labor protections, or unionization.

    Historical examples:
    - Medieval guild wage standards
    - New Deal labor reforms
    - Modern minimum wage legislation

    Attributes:
        min_wage_ratio: Minimum W/W_0 ratio enforced.
    """

    min_wage_ratio: float = 0.8

    def modify_state(
        self,
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """Enforce wage floor."""
        if not self.is_active(t):
            return state

        modified = state.copy()
        min_W = self.min_wage_ratio * params.W_0
        if modified[_W_IDX] < min_W:
            modified[_W_IDX] = min_W
        return modified

    def modify_derivatives(
        self,
        derivatives: NDArray[np.float64],
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """Prevent negative dW/dt when at floor."""
        if not self.is_active(t):
            return derivatives

        modified = derivatives.copy()
        min_W = self.min_wage_ratio * params.W_0

        # If at or below floor and derivatives negative, zero out decline
        if state[_W_IDX] <= min_W and modified[_W_IDX] < 0:
            modified[_W_IDX] = 0.0

        return modified


@dataclass
class WageBoost(Intervention):
    """Boost wage growth rate.

    Increases wages by a specified percentage per year. Simulates
    policies that raise worker bargaining power or productivity gains.

    Historical examples:
    - Post-WWII labor accords
    - Productivity sharing agreements
    - Immigration restrictions tightening labor markets

    Attributes:
        boost_rate: Additional annual growth rate (e.g., 0.02 for 2%/year).
    """

    boost_rate: float = 0.02

    def modify_state(
        self,
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """No direct state modification."""
        return state

    def modify_derivatives(
        self,
        derivatives: NDArray[np.float64],
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """Add boost to wage growth."""
        if not self.is_active(t):
            return derivatives

        modified = derivatives.copy()
        # Add boost_rate * W to dW/dt
        modified[_W_IDX] += self.boost_rate * state[_W_IDX]
        return modified


@dataclass
class TaxProgressivity(Intervention):
    """Increase tax progressivity to redistribute from elites.

    Increases state revenue while reducing elite growth.
    Models progressive taxation that captures elite surplus
    and funds public goods.

    Historical examples:
    - Roman tribute distributions
    - Progressive income taxation (20th century)
    - Estate/inheritance taxes

    Attributes:
        revenue_boost: Additional tax revenue rate (added to rho).
        elite_drain: Additional reduction in elite growth (added to delta_e).
    """

    revenue_boost: float = 0.05
    elite_drain: float = 0.01

    def modify_state(
        self,
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """No direct state modification."""
        return state

    def modify_derivatives(
        self,
        derivatives: NDArray[np.float64],
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """Boost state revenue, drain elite growth."""
        if not self.is_active(t):
            return derivatives

        modified = derivatives.copy()
        N, E, W, S, psi = state

        # Additional revenue from progressive taxes
        additional_revenue = self.revenue_boost * W * N
        modified[_S_IDX] += additional_revenue

        # Additional drain on elite (taxing their accumulation)
        modified[_E_IDX] -= self.elite_drain * E

        return modified


@dataclass
class FiscalAusterity(Intervention):
    """Reduce state spending to improve fiscal health.

    Reduces state expenditure rate, allowing faster recovery
    of state fiscal health. May have negative effects on
    legitimacy and popular welfare.

    Historical examples:
    - Diocletian's reforms
    - Post-crisis austerity measures
    - Debt reduction programs

    Attributes:
        spending_reduction: Fractional reduction in sigma (0.0 to 0.9).
    """

    spending_reduction: float = 0.2

    def __post_init__(self) -> None:
        """Validate spending reduction."""
        if not 0 <= self.spending_reduction < 1:
            raise ValueError("spending_reduction must be between 0 and 0.9")

    def modify_state(
        self,
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """No direct state modification."""
        return state

    def modify_derivatives(
        self,
        derivatives: NDArray[np.float64],
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """Reduce state expenditure."""
        if not self.is_active(t):
            return derivatives

        modified = derivatives.copy()
        S = state[_S_IDX]

        # Reduce the negative expenditure term
        expenditure_saved = self.spending_reduction * params.sigma * S
        modified[_S_IDX] += expenditure_saved

        return modified


@dataclass
class FiscalStimulus(Intervention):
    """Increase state spending to boost wages and reduce instability.

    Uses state resources to fund public works, welfare programs,
    or redistribution that raises popular welfare.

    Historical examples:
    - Roman grain doles
    - New Deal public works
    - Modern stimulus programs

    Attributes:
        wage_boost: Boost to wage growth funded by state.
        psi_reduction: Reduction in instability from social spending.
        spending_rate: State resources consumed per year.
    """

    wage_boost: float = 0.02
    psi_reduction: float = 0.01
    spending_rate: float = 0.1

    def modify_state(
        self,
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """No direct state modification."""
        return state

    def modify_derivatives(
        self,
        derivatives: NDArray[np.float64],
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """Boost wages, reduce PSI, consume state resources."""
        if not self.is_active(t):
            return derivatives

        modified = derivatives.copy()
        W, S, psi = state[_W_IDX], state[_S_IDX], state[_PSI_IDX]

        # Only spend if state has resources
        if S > 0.1:
            # Boost wages
            modified[_W_IDX] += self.wage_boost * W
            # Reduce instability
            modified[_PSI_IDX] -= self.psi_reduction * psi
            # Consume state resources
            modified[_S_IDX] -= self.spending_rate * S

        return modified


@dataclass
class MigrationControl(Intervention):
    """Control population through migration policy.

    Can either restrict population growth (emigration, lower immigration)
    or boost it (open borders, pro-natalist policies).

    Historical examples:
    - Roman colonization (population export)
    - Chinese one-child policy
    - Immigration restrictions (1920s USA)

    Attributes:
        population_effect: Modifier to population growth rate.
            Positive = encourage growth, negative = restrict growth.
    """

    population_effect: float = -0.01

    def modify_state(
        self,
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """No direct state modification."""
        return state

    def modify_derivatives(
        self,
        derivatives: NDArray[np.float64],
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """Modify population growth rate."""
        if not self.is_active(t):
            return derivatives

        modified = derivatives.copy()
        N = state[_N_IDX]

        # Add/subtract from population growth
        modified[_N_IDX] += self.population_effect * N

        return modified


@dataclass
class FrontierExpansion(Intervention):
    """Expand carrying capacity through territorial expansion.

    Models acquiring new resources/territory that relieves
    population pressure and provides elite opportunities.

    Historical examples:
    - Roman conquest (Republic period)
    - American westward expansion
    - Colonial expansion (European powers)

    Attributes:
        carrying_capacity_boost: Fractional increase in effective K.
        elite_opportunity_boost: Additional elite outlets reducing competition.
    """

    carrying_capacity_boost: float = 0.1
    elite_opportunity_boost: float = 0.01

    def modify_state(
        self,
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """No direct state modification."""
        return state

    def modify_derivatives(
        self,
        derivatives: NDArray[np.float64],
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """Boost population capacity and provide elite outlets."""
        if not self.is_active(t):
            return derivatives

        modified = derivatives.copy()
        N, E, W = state[_N_IDX], state[_E_IDX], state[_W_IDX]

        # Effective carrying capacity boost reduces N/K ratio
        # This allows more population growth when near capacity
        # Approximated by adding to dN/dt when near K
        capacity_factor = N / params.K_0
        if capacity_factor > 0.7:
            # More room to grow
            capacity_relief = self.carrying_capacity_boost * params.r_max * N
            modified[_N_IDX] += capacity_relief

        # Elite opportunities in new territories reduce competition
        # This reduces dE/dt slightly (elites emigrate to frontier)
        modified[_E_IDX] -= self.elite_opportunity_boost * E

        # Wages may rise due to labor scarcity in expansion
        modified[_W_IDX] += 0.01 * W

        return modified


@dataclass
class InstitutionalReform(Intervention):
    """Reform institutions to improve governance and reduce instability.

    Models political reforms that improve representation,
    reduce corruption, or enhance state legitimacy.

    Historical examples:
    - Augustan reforms
    - Constitutional reforms
    - Democratization

    Attributes:
        legitimacy_boost: Reduction in instability accumulation.
        efficiency_boost: Improvement in state revenue efficiency.
        elite_restraint: Reduction in elite burden on state.
    """

    legitimacy_boost: float = 0.02
    efficiency_boost: float = 0.02
    elite_restraint: float = 0.01

    def modify_state(
        self,
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """No direct state modification."""
        return state

    def modify_derivatives(
        self,
        derivatives: NDArray[np.float64],
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """Improve governance outcomes."""
        if not self.is_active(t):
            return derivatives

        modified = derivatives.copy()
        N, E, W, S, psi = state

        # Reduce instability accumulation
        if modified[_PSI_IDX] > 0:
            modified[_PSI_IDX] *= 1 - self.legitimacy_boost

        # Improve state revenue
        additional_revenue = self.efficiency_boost * params.rho * W * N
        modified[_S_IDX] += additional_revenue

        # Reduce elite burden on state
        elite_burden_reduction = self.elite_restraint * params.epsilon * E
        modified[_S_IDX] += elite_burden_reduction

        return modified


@dataclass
class CompositeIntervention(Intervention):
    """Combine multiple interventions into a policy package.

    Allows testing comprehensive policy reforms that address
    multiple SDT variables simultaneously.

    Example:
        >>> composite = CompositeIntervention(
        ...     name="New Deal Package",
        ...     start_time=1933,
        ...     interventions=[
        ...         WageFloor(name="Minimum wage", start_time=1933, min_wage_ratio=0.7),
        ...         FiscalStimulus(
        ...             name="Public works", start_time=1933, wage_boost=0.03
        ...         ),
        ...         TaxProgressivity(
        ...             name="Progressive tax", start_time=1933,
        ...             revenue_boost=0.1
        ...         ),
        ...     ]
        ... )

    Attributes:
        interventions: List of interventions to apply.
    """

    interventions: list[Intervention] = field(default_factory=list)

    def modify_state(
        self,
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """Apply all constituent interventions' state modifications."""
        if not self.is_active(t):
            return state

        modified = state.copy()
        for intervention in self.interventions:
            modified = intervention.modify_state(modified, t, params)
        return modified

    def modify_derivatives(
        self,
        derivatives: NDArray[np.float64],
        state: NDArray[np.float64],
        t: float,
        params: "SDTParams",
    ) -> NDArray[np.float64]:
        """Apply all constituent interventions' derivative modifications."""
        if not self.is_active(t):
            return derivatives

        modified = derivatives.copy()
        for intervention in self.interventions:
            modified = intervention.modify_derivatives(modified, state, t, params)
        return modified


# Factory functions for common intervention patterns


def create_elite_management_policy(
    start_time: float,
    cap_ratio: float = 1.5,
    purge_fraction: float = 0.0,
    end_time: float | None = None,
) -> Intervention:
    """Create an elite management policy.

    Args:
        start_time: When policy begins.
        cap_ratio: Maximum elite ratio (if > 1).
        purge_fraction: Initial elite reduction (if > 0).
        end_time: When policy ends.

    Returns:
        Intervention (single or composite).
    """
    interventions: list[Intervention] = []

    if purge_fraction > 0:
        interventions.append(
            ElitePurge(
                name="Elite purge",
                start_time=start_time,
                end_time=start_time + 1,  # One-time event
                reduction_fraction=purge_fraction,
            )
        )

    if cap_ratio > 0:
        interventions.append(
            EliteCap(
                name="Elite cap",
                start_time=start_time,
                end_time=end_time,
                max_elite_ratio=cap_ratio,
            )
        )

    if len(interventions) == 1:
        return interventions[0]
    else:
        return CompositeIntervention(
            name="Elite management",
            start_time=start_time,
            end_time=end_time,
            interventions=interventions,
        )


def create_welfare_policy(
    start_time: float,
    wage_floor: float = 0.8,
    wage_boost: float = 0.01,
    end_time: float | None = None,
) -> Intervention:
    """Create a popular welfare policy.

    Args:
        start_time: When policy begins.
        wage_floor: Minimum wage ratio to baseline.
        wage_boost: Additional wage growth rate.
        end_time: When policy ends.

    Returns:
        Composite intervention.
    """
    return CompositeIntervention(
        name="Welfare policy",
        start_time=start_time,
        end_time=end_time,
        interventions=[
            WageFloor(
                name="Wage floor",
                start_time=start_time,
                end_time=end_time,
                min_wage_ratio=wage_floor,
            ),
            WageBoost(
                name="Wage boost",
                start_time=start_time,
                end_time=end_time,
                boost_rate=wage_boost,
            ),
        ],
    )


__all__ = [
    "Intervention",
    "EliteCap",
    "ElitePurge",
    "WageFloor",
    "WageBoost",
    "TaxProgressivity",
    "FiscalAusterity",
    "FiscalStimulus",
    "MigrationControl",
    "FrontierExpansion",
    "InstitutionalReform",
    "CompositeIntervention",
    "create_elite_management_policy",
    "create_welfare_policy",
]
