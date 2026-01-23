"""Policy simulation framework for counterfactual analysis.

This module provides tools for simulating policy interventions and
extracting actionable recommendations from SDT models. It enables
"what-if" analysis: given a historical trajectory, what policies
might have changed outcomes?

Key Components:
- Interventions: Define policy actions (elite caps, wage floors, etc.)
- Counterfactual engine: Run simulations with interventions applied
- Analysis: Compare outcomes and identify effective policies

Example:
    >>> from cliodynamics.policy import (
    ...     Intervention, EliteCap, WageFloor,
    ...     CounterfactualEngine, OutcomeComparison
    ... )
    >>> from cliodynamics.models import SDTModel, SDTParams
    >>> from cliodynamics.simulation import Simulator, SimulationResult
    >>>
    >>> # Run baseline simulation
    >>> model = SDTModel(SDTParams())
    >>> sim = Simulator(model)
    >>> baseline = sim.run(
    ...     {'N': 0.5, 'E': 0.05, 'W': 1.0, 'S': 1.0, 'psi': 0.0},
    ...     time_span=(0, 200)
    ... )
    >>>
    >>> # Define intervention: cap elite growth at year 50
    >>> intervention = EliteCap(
    ...     start_time=50,
    ...     max_elite_ratio=1.5,  # max 1.5x baseline
    ...     name="Elite cap at year 50"
    ... )
    >>>
    >>> # Run counterfactual
    >>> engine = CounterfactualEngine(model)
    >>> result = engine.run_intervention(baseline, intervention)
    >>>
    >>> # Compare outcomes
    >>> comparison = OutcomeComparison(baseline, result)
    >>> print(comparison.psi_peak_reduction())

References:
    Turchin, P. (2016). Ages of Discord, Chapter 10: "What Can We Do?"
    Turchin, P. & Nefedov, S. (2009). Secular Cycles, Conclusion.
"""

from cliodynamics.policy.analysis import (
    OutcomeComparison,
    PolicyRecommendation,
    SensitivityAnalysis,
    recommend_interventions,
)
from cliodynamics.policy.counterfactual import (
    CounterfactualEngine,
    CounterfactualResult,
    InterventionModel,
)
from cliodynamics.policy.interventions import (
    CompositeIntervention,
    EliteCap,
    ElitePurge,
    FiscalAusterity,
    FiscalStimulus,
    FrontierExpansion,
    InstitutionalReform,
    Intervention,
    MigrationControl,
    TaxProgressivity,
    WageBoost,
    WageFloor,
)

__all__ = [
    # Interventions
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
    # Counterfactual engine
    "CounterfactualEngine",
    "CounterfactualResult",
    "InterventionModel",
    # Analysis
    "OutcomeComparison",
    "SensitivityAnalysis",
    "PolicyRecommendation",
    "recommend_interventions",
]
