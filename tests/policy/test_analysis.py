"""Tests for policy analysis tools.

Tests cover:
- OutcomeMetrics computation
- OutcomeComparison between scenarios
- SensitivityAnalysis across multiple interventions
- PolicyRecommendation generation
"""

import pandas as pd
import pytest

from cliodynamics.policy.analysis import (
    OutcomeComparison,
    OutcomeMetrics,
    PolicyRecommendation,
    SensitivityAnalysis,
    recommend_interventions,
)
from cliodynamics.policy.counterfactual import CounterfactualResult
from cliodynamics.policy.interventions import EliteCap, TaxProgressivity, WageFloor
from cliodynamics.simulation import SimulationResult


def create_mock_result(
    psi_values: list[float],
    S_values: list[float] | None = None,
    W_values: list[float] | None = None,
    intervention=None,
) -> CounterfactualResult:
    """Create a mock CounterfactualResult for testing."""
    n = len(psi_values)
    t = list(range(n))

    if S_values is None:
        S_values = [1.0 - 0.3 * (i / (n - 1)) for i in range(n)]
    if W_values is None:
        W_values = [1.0 - 0.2 * (i / (n - 1)) for i in range(n)]

    df = pd.DataFrame(
        {
            "t": t,
            "N": [0.5 + 0.1 * (i / (n - 1)) for i in range(n)],
            "E": [0.05 + 0.05 * (i / (n - 1)) for i in range(n)],
            "W": W_values,
            "S": S_values,
            "psi": psi_values,
        }
    )

    sim_result = SimulationResult(df=df, events=[], terminated_by_event=False)

    return CounterfactualResult(
        result=sim_result,
        intervention=intervention,
    )


class TestOutcomeMetrics:
    """Tests for OutcomeMetrics computation."""

    def test_psi_metrics(self):
        """PSI metrics computed correctly."""
        # PSI peaks at index 6 (value 0.6), then declines
        psi_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3]
        result = create_mock_result(psi_values)

        metrics = OutcomeMetrics.from_result(result)

        assert metrics.psi_peak == pytest.approx(0.6)
        assert metrics.psi_peak_time == pytest.approx(6)
        assert metrics.psi_final == pytest.approx(0.3)

    def test_psi_integral(self):
        """PSI integral computed using trapezoidal rule."""
        psi_values = [0.0, 0.2, 0.4, 0.2, 0.0]  # Triangle-like
        result = create_mock_result(psi_values)

        metrics = OutcomeMetrics.from_result(result)

        # Trapezoidal integral for [0,0.2,0.4,0.2,0] over t=[0,1,2,3,4]
        # = 0.5*(0+0.2) + 0.5*(0.2+0.4) + 0.5*(0.4+0.2) + 0.5*(0.2+0)
        # = 0.1 + 0.3 + 0.3 + 0.1 = 0.8
        assert metrics.psi_integral == pytest.approx(0.8)

    def test_collapse_detection(self):
        """Collapse detected when S < 0.1."""
        # S drops below 0.1 at t=7
        S_values = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.08, 0.05, 0.05]
        result = create_mock_result([0.0] * 10, S_values=S_values)

        metrics = OutcomeMetrics.from_result(result)

        assert metrics.collapse_time == pytest.approx(7)

    def test_no_collapse(self):
        """No collapse when S stays above 0.1."""
        S_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15]
        result = create_mock_result([0.0] * 10, S_values=S_values)

        metrics = OutcomeMetrics.from_result(result)

        assert metrics.collapse_time is None

    def test_recovery_time(self):
        """Recovery time computed from peak to PSI < 0.1."""
        # PSI peaks at t=5, drops below 0.1 at t=8
        psi_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.3, 0.15, 0.05, 0.02]
        result = create_mock_result(psi_values)

        metrics = OutcomeMetrics.from_result(result)

        assert metrics.psi_peak_time == pytest.approx(5)
        assert metrics.recovery_time == pytest.approx(3)  # 8 - 5

    def test_wage_and_state_minima(self):
        """Minimum wage and state values tracked."""
        W_values = [1.0, 0.9, 0.8, 0.6, 0.5, 0.55, 0.6, 0.7, 0.8, 0.85]
        S_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.5, 0.6, 0.7]
        result = create_mock_result([0.0] * 10, S_values=S_values, W_values=W_values)

        metrics = OutcomeMetrics.from_result(result)

        assert metrics.wage_minimum == pytest.approx(0.5)
        assert metrics.state_minimum == pytest.approx(0.45)


class TestOutcomeComparison:
    """Tests for OutcomeComparison."""

    @pytest.fixture
    def baseline(self):
        """Baseline with moderate crisis."""
        psi_values = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.7, 0.5, 0.3, 0.2]
        return create_mock_result(psi_values)

    @pytest.fixture
    def intervention(self):
        """Intervention that reduces crisis."""
        psi_values = [0.0, 0.1, 0.15, 0.25, 0.35, 0.4, 0.35, 0.25, 0.15, 0.1]
        cap = EliteCap(name="Elite cap", start_time=20)
        return create_mock_result(psi_values, intervention=cap)

    def test_psi_peak_reduction(self, baseline, intervention):
        """PSI peak reduction computed correctly."""
        comparison = OutcomeComparison(baseline, intervention)

        # Baseline peak = 0.8, intervention peak = 0.4
        # Reduction = (0.8 - 0.4) / 0.8 = 0.5
        assert comparison.psi_peak_reduction() == pytest.approx(0.5)

    def test_psi_peak_delay(self, baseline, intervention):
        """PSI peak delay computed correctly."""
        comparison = OutcomeComparison(baseline, intervention)

        # Baseline peak at t=5, intervention peak at t=5
        # (They happen to peak at same time in this example)
        assert comparison.psi_peak_delay() == pytest.approx(0)

    def test_collapse_prevented(self):
        """Collapse prevention detected."""
        # Baseline collapses (S < 0.1)
        baseline = create_mock_result(
            [0.0] * 10, S_values=[1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.05, 0.05, 0.05]
        )

        # Intervention prevents collapse
        intervention_result = create_mock_result(
            [0.0] * 10,
            S_values=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.5, 0.55, 0.6],
            intervention=EliteCap(name="Cap", start_time=0),
        )

        comparison = OutcomeComparison(baseline, intervention_result)
        assert comparison.collapse_prevented()

    def test_wage_improvement(self, baseline, intervention):
        """Wage improvement computed correctly."""
        comparison = OutcomeComparison(baseline, intervention)

        # Both have same W trajectory in this mock, so improvement = 0
        assert comparison.wage_improvement() == pytest.approx(0)

    def test_trajectory_difference(self, baseline, intervention):
        """Trajectory difference DataFrame created correctly."""
        comparison = OutcomeComparison(baseline, intervention)

        diff_df = comparison.trajectory_difference("psi")

        assert "t" in diff_df.columns
        assert "baseline" in diff_df.columns
        assert "intervention" in diff_df.columns
        assert "difference" in diff_df.columns

    def test_summary(self, baseline, intervention):
        """Summary string generated."""
        comparison = OutcomeComparison(baseline, intervention)
        summary = comparison.summary()

        assert "Outcome Comparison" in summary
        assert "Peak reduction" in summary


class TestSensitivityAnalysis:
    """Tests for SensitivityAnalysis."""

    @pytest.fixture
    def baseline(self):
        psi_values = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.7, 0.5, 0.3, 0.2]
        return create_mock_result(psi_values)

    @pytest.fixture
    def results(self):
        """Multiple intervention results with different effectiveness."""
        return [
            create_mock_result(
                [0.0, 0.1, 0.15, 0.25, 0.35, 0.4, 0.35, 0.25, 0.15, 0.1],
                intervention=EliteCap(
                    name="Strong cap", start_time=0, max_elite_ratio=1.0
                ),
            ),
            create_mock_result(
                [0.0, 0.1, 0.18, 0.32, 0.48, 0.6, 0.55, 0.42, 0.28, 0.18],
                intervention=EliteCap(
                    name="Weak cap", start_time=0, max_elite_ratio=1.5
                ),
            ),
            create_mock_result(
                [0.0, 0.1, 0.17, 0.3, 0.45, 0.55, 0.5, 0.38, 0.25, 0.15],
                intervention=WageFloor(
                    name="Wage floor", start_time=0, min_wage_ratio=0.8
                ),
            ),
        ]

    def test_rank_by_psi_reduction(self, baseline, results):
        """Interventions ranked by PSI peak reduction."""
        analysis = SensitivityAnalysis(baseline, results)
        ranked = analysis.rank_by_psi_reduction()

        assert len(ranked) == 3
        # Strong cap (0.4 peak) should be first (best reduction)
        assert ranked[0][0] == "Strong cap"
        assert ranked[0][1] == pytest.approx(0.5)  # (0.8 - 0.4) / 0.8

    def test_rank_by_integral_reduction(self, baseline, results):
        """Interventions ranked by cumulative instability reduction."""
        analysis = SensitivityAnalysis(baseline, results)
        ranked = analysis.rank_by_integral_reduction()

        assert len(ranked) == 3

    def test_to_dataframe(self, baseline, results):
        """Analysis converts to DataFrame."""
        analysis = SensitivityAnalysis(baseline, results)
        df = analysis.to_dataframe()

        assert len(df) == 3
        assert "intervention" in df.columns
        assert "psi_peak_reduction" in df.columns
        assert "collapse_prevented" in df.columns

    def test_summary(self, baseline, results):
        """Summary string generated."""
        analysis = SensitivityAnalysis(baseline, results)
        summary = analysis.summary()

        assert "Sensitivity Analysis" in summary
        assert "Ranked by PSI Peak Reduction" in summary


class TestPolicyRecommendation:
    """Tests for PolicyRecommendation."""

    def test_summary(self):
        """Recommendation summary generated."""
        intervention = EliteCap(name="Elite cap", start_time=50)
        rec = PolicyRecommendation(
            intervention=intervention,
            effectiveness=0.3,
            timing=50.0,
            confidence=0.8,
            rationale="Limits elite overproduction.",
            trade_offs=["May reduce innovation"],
            historical_precedents=["Roman Lex Claudia"],
        )

        summary = rec.summary()

        assert "Elite cap" in summary
        assert "30" in summary  # Matches 30%, 30.0%, or 0.30
        assert "Rationale" in summary
        assert "Trade-offs" in summary
        assert "Historical precedents" in summary


class TestRecommendInterventions:
    """Tests for recommend_interventions function."""

    @pytest.fixture
    def baseline(self):
        psi_values = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.7, 0.5, 0.3, 0.2]
        return create_mock_result(psi_values)

    @pytest.fixture
    def results(self):
        return [
            create_mock_result(
                [0.0, 0.1, 0.15, 0.25, 0.35, 0.4, 0.35, 0.25, 0.15, 0.1],
                intervention=EliteCap(name="Elite cap", start_time=0),
            ),
            create_mock_result(
                [0.0, 0.1, 0.18, 0.32, 0.48, 0.55, 0.5, 0.38, 0.25, 0.15],
                intervention=WageFloor(name="Wage floor", start_time=0),
            ),
            create_mock_result(
                [0.0, 0.1, 0.16, 0.28, 0.42, 0.5, 0.45, 0.35, 0.22, 0.12],
                intervention=TaxProgressivity(name="Tax reform", start_time=0),
            ),
        ]

    def test_minimize_psi_peak(self, baseline, results):
        """Recommendations for minimizing PSI peak."""
        recs = recommend_interventions(
            baseline=baseline,
            intervention_results=results,
            objective="minimize_psi_peak",
            max_recommendations=3,
        )

        assert len(recs) <= 3
        # Best should be elite cap (peak 0.4 vs baseline 0.8)
        assert recs[0].intervention.name == "Elite cap"

    def test_minimize_psi_integral(self, baseline, results):
        """Recommendations for minimizing cumulative instability."""
        recs = recommend_interventions(
            baseline=baseline,
            intervention_results=results,
            objective="minimize_psi_integral",
            max_recommendations=3,
        )

        assert len(recs) <= 3

    def test_with_constraints(self, baseline, results):
        """Constraints filter out ineligible interventions."""
        # Add a politically radical intervention
        from cliodynamics.policy.interventions import ElitePurge

        radical_result = create_mock_result(
            [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.2, 0.15, 0.1, 0.05],
            intervention=ElitePurge(
                name="Elite purge", start_time=0, reduction_fraction=0.3
            ),
        )
        results.append(radical_result)

        recs = recommend_interventions(
            baseline=baseline,
            intervention_results=results,
            objective="minimize_psi_peak",
            constraints=["politically_feasible"],
            max_recommendations=5,
        )

        # Purge should be filtered out
        intervention_names = [r.intervention.name for r in recs]
        assert "Elite purge" not in intervention_names
