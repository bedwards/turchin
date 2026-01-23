"""Analysis tools for comparing counterfactual outcomes.

This module provides tools for analyzing and comparing results from
counterfactual simulations, including:

- OutcomeComparison: Compare metrics between baseline and intervention
- SensitivityAnalysis: Measure which interventions have largest effects
- PolicyRecommendation: Rank interventions by effectiveness

Example:
    >>> from cliodynamics.policy import OutcomeComparison, SensitivityAnalysis
    >>>
    >>> # Compare baseline and intervention
    >>> comparison = OutcomeComparison(baseline, intervention_result)
    >>> print(f"PSI reduction: {comparison.psi_peak_reduction():.1%}")
    >>>
    >>> # Analyze sensitivity to different interventions
    >>> analysis = SensitivityAnalysis(baseline, [result1, result2, result3])
    >>> ranking = analysis.rank_by_psi_reduction()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from cliodynamics.policy.counterfactual import CounterfactualResult
    from cliodynamics.policy.interventions import Intervention


@dataclass
class OutcomeMetrics:
    """Container for key outcome metrics from a simulation.

    Attributes:
        psi_peak: Maximum PSI value.
        psi_peak_time: Time at PSI peak.
        psi_final: Final PSI value.
        psi_integral: Integral of PSI over time (cumulative instability).
        collapse_time: Time when S < 0.1 (if occurred).
        recovery_time: Time from peak PSI to PSI < 0.1 (if occurred).
        wage_minimum: Minimum wage value.
        elite_maximum: Maximum elite population.
        state_minimum: Minimum state fiscal health.
    """

    psi_peak: float
    psi_peak_time: float
    psi_final: float
    psi_integral: float
    collapse_time: float | None
    recovery_time: float | None
    wage_minimum: float
    elite_maximum: float
    state_minimum: float

    @classmethod
    def from_result(cls, result: "CounterfactualResult") -> "OutcomeMetrics":
        """Compute metrics from a CounterfactualResult.

        Args:
            result: The simulation result to analyze.

        Returns:
            OutcomeMetrics instance.
        """
        df = result.df

        # PSI metrics
        psi_peak = float(df["psi"].max())
        psi_peak_idx = df["psi"].idxmax()
        psi_peak_time = float(df.loc[psi_peak_idx, "t"])
        psi_final = float(df["psi"].iloc[-1])

        # Integral using trapezoidal rule
        psi_integral = float(np.trapezoid(df["psi"], df["t"]))

        # Collapse detection (S < 0.1)
        collapse_mask = df["S"] < 0.1
        if collapse_mask.any():
            collapse_time = float(df.loc[collapse_mask.idxmax(), "t"])
        else:
            collapse_time = None

        # Recovery time (from peak to PSI < 0.1)
        post_peak = df[df["t"] >= psi_peak_time]
        recovery_mask = post_peak["psi"] < 0.1
        if recovery_mask.any():
            recovery_end = float(post_peak.loc[recovery_mask.idxmax(), "t"])
            recovery_time = recovery_end - psi_peak_time
        else:
            recovery_time = None

        return cls(
            psi_peak=psi_peak,
            psi_peak_time=psi_peak_time,
            psi_final=psi_final,
            psi_integral=psi_integral,
            collapse_time=collapse_time,
            recovery_time=recovery_time,
            wage_minimum=float(df["W"].min()),
            elite_maximum=float(df["E"].max()),
            state_minimum=float(df["S"].min()),
        )


@dataclass
class OutcomeComparison:
    """Compare outcomes between baseline and intervention scenarios.

    Provides methods for computing the effect of an intervention
    on various outcome metrics.

    Attributes:
        baseline: The baseline CounterfactualResult.
        intervention: The intervention CounterfactualResult.
        baseline_metrics: Computed metrics for baseline.
        intervention_metrics: Computed metrics for intervention.
    """

    baseline: "CounterfactualResult"
    intervention: "CounterfactualResult"
    baseline_metrics: OutcomeMetrics = field(init=False)
    intervention_metrics: OutcomeMetrics = field(init=False)

    def __post_init__(self) -> None:
        """Compute metrics for both scenarios."""
        self.baseline_metrics = OutcomeMetrics.from_result(self.baseline)
        self.intervention_metrics = OutcomeMetrics.from_result(self.intervention)

    def psi_peak_reduction(self) -> float:
        """Compute fractional reduction in PSI peak.

        Returns:
            Fraction by which PSI peak was reduced (positive = improvement).
        """
        base = self.baseline_metrics.psi_peak
        interv = self.intervention_metrics.psi_peak
        if base == 0:
            return 0.0
        return (base - interv) / base

    def psi_peak_delay(self) -> float:
        """Compute delay in PSI peak time.

        Returns:
            Time delay (positive = intervention delays crisis).
        """
        return (
            self.intervention_metrics.psi_peak_time
            - self.baseline_metrics.psi_peak_time
        )

    def psi_integral_reduction(self) -> float:
        """Compute fractional reduction in cumulative instability.

        Returns:
            Fraction by which PSI integral was reduced.
        """
        base = self.baseline_metrics.psi_integral
        interv = self.intervention_metrics.psi_integral
        if base == 0:
            return 0.0
        return (base - interv) / base

    def collapse_prevented(self) -> bool:
        """Check if intervention prevented collapse.

        Returns:
            True if baseline collapsed but intervention did not.
        """
        return (
            self.baseline_metrics.collapse_time is not None
            and self.intervention_metrics.collapse_time is None
        )

    def collapse_delayed(self) -> float | None:
        """Compute delay in collapse time.

        Returns:
            Time delay if both collapsed, None otherwise.
        """
        base = self.baseline_metrics.collapse_time
        interv = self.intervention_metrics.collapse_time
        if base is not None and interv is not None:
            return interv - base
        return None

    def recovery_speedup(self) -> float | None:
        """Compute speedup in recovery time.

        Returns:
            Time saved in recovery (positive = faster recovery).
        """
        base = self.baseline_metrics.recovery_time
        interv = self.intervention_metrics.recovery_time
        if base is not None and interv is not None:
            return base - interv
        return None

    def wage_improvement(self) -> float:
        """Compute improvement in minimum wage.

        Returns:
            Fractional improvement (positive = higher wage floor).
        """
        base = self.baseline_metrics.wage_minimum
        interv = self.intervention_metrics.wage_minimum
        if base == 0:
            return 0.0
        return (interv - base) / base

    def state_improvement(self) -> float:
        """Compute improvement in minimum state health.

        Returns:
            Fractional improvement (positive = stronger state).
        """
        base = self.baseline_metrics.state_minimum
        interv = self.intervention_metrics.state_minimum
        if base == 0:
            return 0.0
        return (interv - base) / base

    def trajectory_difference(self, variable: str = "psi") -> pd.DataFrame:
        """Compute difference in trajectories.

        Args:
            variable: Variable to compare (N, E, W, S, or psi).

        Returns:
            DataFrame with time, baseline, intervention, and difference.
        """
        df_base = self.baseline.df
        df_int = self.intervention.df

        # Interpolate to common time grid
        t_min = max(df_base["t"].min(), df_int["t"].min())
        t_max = min(df_base["t"].max(), df_int["t"].max())
        t_common = np.linspace(t_min, t_max, 200)

        base_vals = np.interp(t_common, df_base["t"], df_base[variable])
        int_vals = np.interp(t_common, df_int["t"], df_int[variable])

        return pd.DataFrame(
            {
                "t": t_common,
                "baseline": base_vals,
                "intervention": int_vals,
                "difference": int_vals - base_vals,
            }
        )

    def summary(self) -> str:
        """Generate a human-readable comparison summary.

        Returns:
            Formatted string with key comparisons.
        """
        intervention_name = (
            self.intervention.intervention.name
            if self.intervention.intervention
            else "N/A"
        )
        base_peak = self.baseline_metrics.psi_peak
        base_peak_time = self.baseline_metrics.psi_peak_time
        int_peak = self.intervention_metrics.psi_peak
        int_peak_time = self.intervention_metrics.psi_peak_time

        lines = [
            "Outcome Comparison",
            "=" * 50,
            f"Intervention: {intervention_name}",
            "",
            "PSI Metrics:",
            f"  Peak reduction: {self.psi_peak_reduction():.1%}",
            f"  Peak delay: {self.psi_peak_delay():.1f} time units",
            f"  Cumulative reduction: {self.psi_integral_reduction():.1%}",
            "",
            f"  Baseline peak: {base_peak:.3f} at t={base_peak_time:.1f}",
            f"  Intervention peak: {int_peak:.3f} at t={int_peak_time:.1f}",
            "",
            "Collapse:",
            f"  Prevented: {self.collapse_prevented()}",
        ]

        delay = self.collapse_delayed()
        if delay is not None:
            lines.append(f"  Delayed by: {delay:.1f} time units")

        lines.extend(
            [
                "",
                "Other Improvements:",
                f"  Wage floor: {self.wage_improvement():.1%}",
                f"  State minimum: {self.state_improvement():.1%}",
            ]
        )

        return "\n".join(lines)


@dataclass
class SensitivityAnalysis:
    """Analyze sensitivity of outcomes to different interventions.

    Compare multiple interventions against the same baseline to
    identify which policies have the largest effects.

    Attributes:
        baseline: The baseline CounterfactualResult.
        results: List of intervention CounterfactualResults.
        comparisons: Computed comparisons for each intervention.
    """

    baseline: "CounterfactualResult"
    results: list["CounterfactualResult"]
    comparisons: list[OutcomeComparison] = field(init=False)

    def __post_init__(self) -> None:
        """Compute comparisons for all interventions."""
        self.comparisons = [
            OutcomeComparison(self.baseline, result) for result in self.results
        ]

    def rank_by_psi_reduction(self) -> list[tuple[str, float, "CounterfactualResult"]]:
        """Rank interventions by PSI peak reduction.

        Returns:
            List of (intervention_name, reduction, result) tuples,
            sorted by reduction (highest first).
        """
        ranked = []
        for comparison, result in zip(self.comparisons, self.results):
            name = result.intervention.name if result.intervention else "Unknown"
            reduction = comparison.psi_peak_reduction()
            ranked.append((name, reduction, result))

        return sorted(ranked, key=lambda x: x[1], reverse=True)

    def rank_by_integral_reduction(
        self,
    ) -> list[tuple[str, float, "CounterfactualResult"]]:
        """Rank interventions by cumulative instability reduction.

        Returns:
            List of (intervention_name, reduction, result) tuples.
        """
        ranked = []
        for comparison, result in zip(self.comparisons, self.results):
            name = result.intervention.name if result.intervention else "Unknown"
            reduction = comparison.psi_integral_reduction()
            ranked.append((name, reduction, result))

        return sorted(ranked, key=lambda x: x[1], reverse=True)

    def find_point_of_no_return(
        self,
        intervention_factory: Callable[[float], "Intervention"],
        time_points: list[float],
        threshold: float = 0.1,
    ) -> float | None:
        """Find the point after which interventions become ineffective.

        Tests an intervention at different times to find when it
        no longer reduces PSI peak by at least the threshold.

        Args:
            intervention_factory: Function(start_time) -> Intervention.
            time_points: Times to test.
            threshold: Minimum PSI reduction to consider effective.

        Returns:
            Time of point of no return, or None if always effective.
        """
        for t in sorted(time_points):
            # Find result with intervention starting near this time
            matching = None
            for result in self.results:
                if result.intervention and abs(result.intervention.start_time - t) < 1:
                    matching = result
                    break

            if matching is None:
                continue

            comparison = OutcomeComparison(self.baseline, matching)
            if comparison.psi_peak_reduction() < threshold:
                return t

        return None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert analysis to a DataFrame.

        Returns:
            DataFrame with one row per intervention and metric columns.
        """
        rows = []
        for comparison, result in zip(self.comparisons, self.results):
            name = result.intervention.name if result.intervention else "Baseline"
            start = result.intervention.start_time if result.intervention else None

            rows.append(
                {
                    "intervention": name,
                    "start_time": start,
                    "psi_peak_reduction": comparison.psi_peak_reduction(),
                    "psi_integral_reduction": comparison.psi_integral_reduction(),
                    "psi_peak_delay": comparison.psi_peak_delay(),
                    "collapse_prevented": comparison.collapse_prevented(),
                    "wage_improvement": comparison.wage_improvement(),
                    "state_improvement": comparison.state_improvement(),
                    "baseline_psi_peak": comparison.baseline_metrics.psi_peak,
                    "intervention_psi_peak": comparison.intervention_metrics.psi_peak,
                }
            )

        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Generate summary of sensitivity analysis.

        Returns:
            Formatted string with rankings and key findings.
        """
        baseline_psi = OutcomeMetrics.from_result(self.baseline).psi_peak
        lines = [
            "Sensitivity Analysis",
            "=" * 60,
            f"Baseline PSI peak: {baseline_psi:.3f}",
            f"Number of interventions tested: {len(self.results)}",
            "",
            "Ranked by PSI Peak Reduction:",
            "-" * 40,
        ]

        for i, (name, reduction, _) in enumerate(self.rank_by_psi_reduction()[:10]):
            lines.append(f"  {i + 1}. {name}: {reduction:.1%}")

        lines.extend(
            [
                "",
                "Ranked by Cumulative Instability Reduction:",
                "-" * 40,
            ]
        )

        for i, (name, reduction, _) in enumerate(
            self.rank_by_integral_reduction()[:10]
        ):
            lines.append(f"  {i + 1}. {name}: {reduction:.1%}")

        return "\n".join(lines)


@dataclass
class PolicyRecommendation:
    """A policy recommendation with justification.

    Attributes:
        intervention: The recommended intervention.
        effectiveness: PSI peak reduction achieved.
        timing: Recommended start time.
        confidence: Confidence level (0-1).
        rationale: Explanation of why this is recommended.
        trade_offs: List of potential downsides.
        historical_precedents: Historical examples of similar policies.
    """

    intervention: "Intervention"
    effectiveness: float
    timing: float
    confidence: float
    rationale: str
    trade_offs: list[str] = field(default_factory=list)
    historical_precedents: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate recommendation summary.

        Returns:
            Formatted string with recommendation details.
        """
        lines = [
            f"Policy Recommendation: {self.intervention.name}",
            "=" * 60,
            f"Effectiveness: {self.effectiveness:.1%} PSI reduction",
            f"Optimal timing: t = {self.timing:.1f}",
            f"Confidence: {self.confidence:.0%}",
            "",
            "Rationale:",
            f"  {self.rationale}",
        ]

        if self.trade_offs:
            lines.extend(["", "Trade-offs:"])
            for trade_off in self.trade_offs:
                lines.append(f"  - {trade_off}")

        if self.historical_precedents:
            lines.extend(["", "Historical precedents:"])
            for precedent in self.historical_precedents:
                lines.append(f"  - {precedent}")

        return "\n".join(lines)


def recommend_interventions(
    baseline: "CounterfactualResult",
    intervention_results: list["CounterfactualResult"],
    objective: str = "minimize_psi_peak",
    constraints: list[str] | None = None,
    max_recommendations: int = 5,
) -> list[PolicyRecommendation]:
    """Generate ranked policy recommendations.

    Analyzes intervention results and generates recommendations
    based on effectiveness and constraints.

    Args:
        baseline: The baseline CounterfactualResult.
        intervention_results: Results from tested interventions.
        objective: What to optimize:
            - "minimize_psi_peak": Reduce peak instability
            - "minimize_psi_integral": Reduce cumulative instability
            - "prevent_collapse": Avoid state fiscal crisis
        constraints: List of constraints to apply:
            - "economically_viable": Avoid severe economic disruption
            - "politically_feasible": Limit radical changes
        max_recommendations: Maximum number of recommendations.

    Returns:
        List of PolicyRecommendation objects, ranked by effectiveness.
    """
    constraints = constraints or []
    analysis = SensitivityAnalysis(baseline, intervention_results)

    # Rank by objective
    if objective == "minimize_psi_peak":
        ranked = analysis.rank_by_psi_reduction()
    elif objective == "minimize_psi_integral":
        ranked = analysis.rank_by_integral_reduction()
    elif objective == "prevent_collapse":
        # Filter to those that prevent collapse, then rank by PSI reduction
        ranked = [
            (name, reduction, result)
            for name, reduction, result in analysis.rank_by_psi_reduction()
            if OutcomeComparison(baseline, result).collapse_prevented()
            or OutcomeMetrics.from_result(baseline).collapse_time is None
        ]
    else:
        raise ValueError(f"Unknown objective: {objective}")

    # Apply constraints (simplified filtering)
    if "economically_viable" in constraints:
        # Filter out interventions with wage boost > 5% or extreme changes
        ranked = [
            (name, reduction, result)
            for name, reduction, result in ranked
            if not _is_economically_extreme(result.intervention)
        ]

    if "politically_feasible" in constraints:
        # Filter out purges and other radical interventions
        ranked = [
            (name, reduction, result)
            for name, reduction, result in ranked
            if not _is_politically_radical(result.intervention)
        ]

    # Generate recommendations
    recommendations = []
    for name, reduction, result in ranked[:max_recommendations]:
        intervention = result.intervention
        if intervention is None:
            continue

        # Generate rationale based on intervention type
        rationale = _generate_rationale(intervention, reduction)
        trade_offs = _identify_trade_offs(intervention, result, baseline)
        precedents = _find_historical_precedents(intervention)

        recommendations.append(
            PolicyRecommendation(
                intervention=intervention,
                effectiveness=reduction,
                timing=intervention.start_time,
                confidence=_estimate_confidence(reduction, len(intervention_results)),
                rationale=rationale,
                trade_offs=trade_offs,
                historical_precedents=precedents,
            )
        )

    return recommendations


def _is_economically_extreme(intervention: "Intervention | None") -> bool:
    """Check if intervention is economically extreme."""
    if intervention is None:
        return False

    from cliodynamics.policy.interventions import ElitePurge, WageBoost

    if isinstance(intervention, WageBoost) and intervention.boost_rate > 0.05:
        return True
    if isinstance(intervention, ElitePurge) and intervention.reduction_fraction > 0.3:
        return True
    return False


def _is_politically_radical(intervention: "Intervention | None") -> bool:
    """Check if intervention is politically radical."""
    if intervention is None:
        return False

    from cliodynamics.policy.interventions import ElitePurge

    if isinstance(intervention, ElitePurge):
        return True
    return False


def _generate_rationale(intervention: "Intervention", effectiveness: float) -> str:
    """Generate explanation for why intervention is effective."""
    from cliodynamics.policy.interventions import (
        EliteCap,
        FiscalStimulus,
        InstitutionalReform,
        TaxProgressivity,
        WageFloor,
    )

    if isinstance(intervention, EliteCap):
        return (
            f"Capping elite growth prevents overproduction that drives "
            f"intra-elite competition, reducing instability by {effectiveness:.0%}."
        )
    elif isinstance(intervention, WageFloor):
        return (
            f"Maintaining minimum wages prevents popular immiseration, "
            f"a key driver of instability, achieving {effectiveness:.0%} reduction."
        )
    elif isinstance(intervention, TaxProgressivity):
        return (
            f"Progressive taxation redistributes resources from elites to state, "
            f"strengthening fiscal capacity and reducing PSI by {effectiveness:.0%}."
        )
    elif isinstance(intervention, FiscalStimulus):
        return (
            f"State-funded welfare programs directly address popular grievances, "
            f"achieving {effectiveness:.0%} PSI reduction."
        )
    elif isinstance(intervention, InstitutionalReform):
        return (
            f"Institutional reforms improve legitimacy and governance efficiency, "
            f"yielding {effectiveness:.0%} reduction in instability."
        )
    else:
        return f"This intervention achieves {effectiveness:.0%} PSI reduction."


def _identify_trade_offs(
    intervention: "Intervention",
    result: "CounterfactualResult",
    baseline: "CounterfactualResult",
) -> list[str]:
    """Identify potential downsides of the intervention."""
    trade_offs = []

    comparison = OutcomeComparison(baseline, result)

    # Check if any metric got worse
    if comparison.wage_improvement() < -0.05:
        trade_offs.append("May reduce wage levels in short term")

    if comparison.state_improvement() < -0.05:
        trade_offs.append("May strain state fiscal capacity")

    from cliodynamics.policy.interventions import (
        EliteCap,
        FiscalAusterity,
        TaxProgressivity,
    )

    if isinstance(intervention, EliteCap):
        trade_offs.append("May reduce economic dynamism and innovation")
        trade_offs.append("Requires sustained enforcement against elite resistance")

    if isinstance(intervention, TaxProgressivity):
        trade_offs.append("May face political opposition from wealthy elites")
        trade_offs.append("Could trigger capital flight")

    if isinstance(intervention, FiscalAusterity):
        trade_offs.append("May worsen popular welfare in short term")
        trade_offs.append("Could trigger social unrest if perceived as unfair")

    return trade_offs


def _find_historical_precedents(intervention: "Intervention") -> list[str]:
    """Find historical examples of similar policies."""
    from cliodynamics.policy.interventions import (
        EliteCap,
        ElitePurge,
        FiscalStimulus,
        FrontierExpansion,
        InstitutionalReform,
        TaxProgressivity,
        WageFloor,
    )

    if isinstance(intervention, EliteCap):
        return [
            "Roman Lex Claudia (218 BCE) - restricted senatorial commerce",
            "Medieval guild restrictions on master craftsmen",
            "Modern professional licensing requirements",
        ]
    elif isinstance(intervention, ElitePurge):
        return [
            "Roman proscriptions under Sulla and Second Triumvirate",
            "French Revolution's Reign of Terror",
            "Russian Revolution - dispossession of nobility",
        ]
    elif isinstance(intervention, WageFloor):
        return [
            "Medieval just price doctrines",
            "New Deal minimum wage (1938)",
            "Post-WWII labor accords in Western Europe",
        ]
    elif isinstance(intervention, TaxProgressivity):
        return [
            "Roman tribute distributions",
            "British income tax (1799, reintroduced 1842)",
            "Progressive income tax (US 1913, peak rates 1950s)",
        ]
    elif isinstance(intervention, FiscalStimulus):
        return [
            "Roman grain doles (annona)",
            "New Deal public works programs",
            "Post-2008 stimulus packages",
        ]
    elif isinstance(intervention, FrontierExpansion):
        return [
            "Roman colonization during Republic",
            "American westward expansion (19th century)",
            "European colonial expansion (16th-19th century)",
        ]
    elif isinstance(intervention, InstitutionalReform):
        return [
            "Augustan constitutional settlement (27 BCE)",
            "British Reform Acts (1832, 1867, 1884)",
            "New Deal institutional reforms (1930s)",
        ]
    else:
        return []


def _estimate_confidence(effectiveness: float, n_samples: int) -> float:
    """Estimate confidence in the recommendation."""
    # Simple heuristic: higher effectiveness and more samples = more confidence
    base_confidence = min(0.9, effectiveness * 2)  # Cap at 90%
    sample_factor = min(1.0, n_samples / 10)  # Approaches 1 with 10+ samples
    return base_confidence * sample_factor


__all__ = [
    "OutcomeMetrics",
    "OutcomeComparison",
    "SensitivityAnalysis",
    "PolicyRecommendation",
    "recommend_interventions",
]
