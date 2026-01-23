"""Tests for counterfactual simulation engine.

Tests cover:
- Baseline simulation
- Intervention application
- InterventionModel wrapper
- Batch intervention testing
- Intervention timing analysis

Note: The SDT model with default parameters can exhibit exponentially growing
dynamics (particularly for wages W). Tests use parameters tuned for more
stable dynamics.
"""

import numpy as np
import pandas as pd
import pytest

from cliodynamics.models import SDTModel
from cliodynamics.models.params import SDTParams
from cliodynamics.policy.counterfactual import (
    CounterfactualEngine,
    CounterfactualResult,
    InterventionModel,
)
from cliodynamics.policy.interventions import (
    EliteCap,
    WageBoost,
    WageFloor,
)


def stable_params() -> SDTParams:
    """Return parameters tuned for stable, bounded dynamics.

    These parameters reduce the wage sensitivity coefficients to prevent
    exponential wage growth that can make simulations computationally expensive.
    """
    return SDTParams(
        r_max=0.02,
        K_0=1.0,
        beta=1.0,
        mu=0.2,
        alpha=0.005,
        delta_e=0.02,
        gamma=0.01,  # Very small for stable tests (default is 2.0)
        eta=0.01,  # Very small for stable tests (default is 1.0)
        rho=0.2,
        sigma=0.1,
        epsilon=0.05,
        lambda_psi=0.05,
        theta_w=1.0,
        theta_e=1.0,
        theta_s=1.0,
        psi_decay=0.02,
        W_0=1.0,
        E_0=0.1,
        S_0=1.0,
    )


class TestInterventionModel:
    """Tests for InterventionModel wrapper."""

    @pytest.fixture
    def base_model(self):
        return SDTModel(stable_params())

    @pytest.fixture
    def intervention(self):
        return EliteCap(name="Cap", start_time=0, max_elite_ratio=1.5)

    def test_wraps_base_model(self, base_model, intervention):
        """InterventionModel exposes base model params."""
        model = InterventionModel(base_model, [intervention])
        assert model.params == base_model.params

    def test_applies_intervention(self, base_model, intervention):
        """InterventionModel applies intervention during system()."""
        model = InterventionModel(base_model, [intervention])

        # State with elite above cap
        state = np.array([0.5, 0.2, 1.0, 1.0, 0.1])

        # Base model would have positive dE/dt
        base_derivs = base_model.system(state, 10)

        # Intervention model should cap dE/dt if at limit
        # First apply state modification, which caps E at 0.15
        # Then derivatives should be computed from capped state
        derivs = model.system(state, 10)

        # The derivatives should differ due to intervention
        assert not np.allclose(derivs, base_derivs)


class TestCounterfactualEngine:
    """Tests for CounterfactualEngine."""

    @pytest.fixture
    def model(self):
        return SDTModel(stable_params())

    @pytest.fixture
    def engine(self, model):
        return CounterfactualEngine(model)

    @pytest.fixture
    def initial_conditions(self):
        return {"N": 0.5, "E": 0.05, "W": 1.0, "S": 1.0, "psi": 0.0}

    def test_run_baseline(self, engine, initial_conditions):
        """Baseline simulation runs without intervention."""
        result = engine.run_baseline(
            initial_conditions=initial_conditions,
            time_span=(0, 100),
            dt=1.0,
        )

        assert isinstance(result, CounterfactualResult)
        assert result.is_baseline
        assert result.intervention is None
        assert len(result.df) > 0
        assert "t" in result.df.columns
        assert "psi" in result.df.columns

    def test_baseline_metadata(self, engine, initial_conditions):
        """Baseline stores correct metadata."""
        result = engine.run_baseline(
            initial_conditions=initial_conditions,
            time_span=(0, 100),
            dt=1.0,
        )

        assert result.metadata["time_span"] == (0, 100)
        assert result.metadata["dt"] == 1.0
        assert result.metadata["initial_conditions"] == initial_conditions

    def test_run_intervention(self, engine, initial_conditions):
        """Intervention simulation applies the intervention."""
        baseline = engine.run_baseline(
            initial_conditions=initial_conditions,
            time_span=(0, 100),
            dt=1.0,
        )

        intervention = EliteCap(
            name="Elite cap",
            start_time=50,
            max_elite_ratio=1.2,
        )

        result = engine.run_intervention(baseline, intervention)

        assert not result.is_baseline
        assert result.intervention == intervention
        assert result.baseline_reference == baseline

    def test_intervention_changes_trajectory(self, engine, initial_conditions):
        """Intervention modifies the trajectory.

        Tests that a WageBoost intervention produces different wage outcomes
        than the baseline. This is a more direct test since WageBoost always
        adds to dW/dt when active.
        """
        baseline = engine.run_baseline(
            initial_conditions=initial_conditions,
            time_span=(0, 100),
            dt=1.0,
        )

        # Use WageBoost which always increases wages when active
        intervention = WageBoost(
            name="Wage boost",
            start_time=10,
            boost_rate=0.05,  # 5% annual boost
        )

        result = engine.run_intervention(baseline, intervention)

        # Compare wage trajectories at end
        baseline_W_final = baseline.df["W"].iloc[-1]
        intervention_W_final = result.df["W"].iloc[-1]

        # Intervention should boost wages above baseline
        assert intervention_W_final > baseline_W_final

    def test_run_interventions_batch(self, engine, initial_conditions):
        """Batch intervention testing runs multiple scenarios."""
        baseline = engine.run_baseline(
            initial_conditions=initial_conditions,
            time_span=(0, 100),
            dt=1.0,
        )

        interventions = [
            EliteCap(name="Cap 1.2", start_time=50, max_elite_ratio=1.2),
            EliteCap(name="Cap 1.5", start_time=50, max_elite_ratio=1.5),
            WageFloor(name="Wage 0.8", start_time=50, min_wage_ratio=0.8),
        ]

        results = engine.run_interventions(baseline, interventions)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, CounterfactualResult)

    def test_run_from_state(self, engine):
        """run_from_state starts from arbitrary state."""
        state = {"N": 0.7, "E": 0.15, "W": 0.8, "S": 0.9, "psi": 0.3}

        result = engine.run_from_state(
            state=state,
            time_span=(50, 150),
            dt=1.0,
        )

        # Should start near the specified state
        first_row = result.df.iloc[0]
        assert first_row["t"] == pytest.approx(50)
        assert first_row["N"] == pytest.approx(0.7)
        assert first_row["psi"] == pytest.approx(0.3)

    def test_run_from_state_with_intervention(self, engine):
        """run_from_state applies intervention."""
        state = {"N": 0.7, "E": 0.15, "W": 0.8, "S": 0.9, "psi": 0.3}

        intervention = EliteCap(name="Cap", start_time=50, max_elite_ratio=1.0)

        result = engine.run_from_state(
            state=state,
            time_span=(50, 150),
            intervention=intervention,
            dt=1.0,
        )

        assert result.intervention == intervention


class TestCounterfactualResult:
    """Tests for CounterfactualResult properties."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample result for testing."""
        df = pd.DataFrame(
            {
                "t": [0, 10, 20, 30, 40, 50],
                "N": [0.5, 0.55, 0.6, 0.65, 0.7, 0.72],
                "E": [0.05, 0.06, 0.08, 0.1, 0.12, 0.14],
                "W": [1.0, 0.95, 0.9, 0.85, 0.8, 0.78],
                "S": [1.0, 0.95, 0.9, 0.85, 0.7, 0.6],
                "psi": [0.0, 0.05, 0.15, 0.3, 0.4, 0.35],
            }
        )

        from cliodynamics.simulation import SimulationResult

        sim_result = SimulationResult(df=df, events=[], terminated_by_event=False)

        return CounterfactualResult(
            result=sim_result,
            intervention=None,
        )

    def test_psi_peak(self, sample_result):
        """psi_peak returns maximum PSI."""
        assert sample_result.psi_peak == pytest.approx(0.4)

    def test_psi_peak_time(self, sample_result):
        """psi_peak_time returns time of maximum PSI."""
        assert sample_result.psi_peak_time == pytest.approx(40)

    def test_final_state(self, sample_result):
        """final_state returns last row values."""
        final = sample_result.final_state
        assert final["psi"] == pytest.approx(0.35)
        assert final["N"] == pytest.approx(0.72)

    def test_state_at_time(self, sample_result):
        """state_at_time interpolates correctly."""
        # Exact time point
        state = sample_result.state_at_time(20)
        assert state["psi"] == pytest.approx(0.15)

        # Interpolated time point
        state = sample_result.state_at_time(15)
        assert state["psi"] == pytest.approx(0.1)  # Midpoint

    def test_is_baseline(self, sample_result):
        """is_baseline correctly identifies baseline."""
        assert sample_result.is_baseline

        # Add an intervention
        sample_result.intervention = EliteCap(name="test", start_time=0)
        assert not sample_result.is_baseline


class TestInterventionTiming:
    """Tests for finding optimal intervention timing."""

    @pytest.fixture
    def model(self):
        return SDTModel(stable_params())

    @pytest.fixture
    def engine(self, model):
        return CounterfactualEngine(model)

    def test_find_intervention_timing(self, engine):
        """find_intervention_timing tests multiple start times."""
        initial = {"N": 0.5, "E": 0.05, "W": 1.0, "S": 1.0, "psi": 0.0}
        baseline = engine.run_baseline(
            initial_conditions=initial,
            time_span=(0, 100),
            dt=1.0,
        )

        def intervention_factory(t):
            return EliteCap(name=f"Cap at {t}", start_time=t, max_elite_ratio=1.2)

        results = engine.find_intervention_timing(
            baseline=baseline,
            intervention_factory=intervention_factory,
            start_times=[10, 30, 50, 70],
            metric="psi_peak",
        )

        assert len(results) == 4
        for t, (value, result) in results.items():
            assert isinstance(value, float)
            assert isinstance(result, CounterfactualResult)
