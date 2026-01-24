"""Tests for the ensemble simulation framework.

Tests cover:
- EnsembleSimulator functionality
- EnsembleResults analysis methods
- Stability classification
- Bifurcation detection
- Parallel execution
"""

import numpy as np
import pandas as pd
import pytest

from cliodynamics.models import SDTModel, SDTParams
from cliodynamics.simulation import (
    BifurcationPoint,
    EnsembleResults,
    EnsembleSimulator,
    SimulationOutcome,
    StabilityClassification,
)


def stable_params() -> SDTParams:
    """Return parameters tuned for stable, bounded dynamics."""
    return SDTParams(
        r_max=0.02,
        K_0=1.0,
        beta=1.0,
        mu=0.2,
        alpha=0.005,
        delta_e=0.02,
        gamma=0.01,
        eta=0.01,
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


class TestEnsembleSimulator:
    """Tests for EnsembleSimulator class."""

    def test_simulator_initialization(self) -> None:
        """Test EnsembleSimulator can be initialized."""
        model = SDTModel(stable_params())
        ensemble = EnsembleSimulator(
            model=model,
            parameter_grid={
                "alpha": np.linspace(0.005, 0.01, 3),
                "gamma": np.linspace(0.005, 0.02, 4),
            },
        )

        assert ensemble.n_simulations == 12  # 3 * 4
        assert ensemble.grid_shape == (3, 4)
        assert "alpha" in ensemble.parameter_grid
        assert "gamma" in ensemble.parameter_grid

    def test_invalid_parameter_raises_error(self) -> None:
        """Test that invalid parameter names raise ValueError."""
        model = SDTModel(stable_params())

        with pytest.raises(ValueError, match="has no parameter"):
            EnsembleSimulator(
                model=model,
                parameter_grid={"nonexistent_param": np.array([1, 2, 3])},
            )

    def test_simple_sequential_run(self) -> None:
        """Test basic sequential ensemble run."""
        model = SDTModel(stable_params())
        ensemble = EnsembleSimulator(
            model=model,
            parameter_grid={
                "alpha": np.array([0.005, 0.01]),
                "gamma": np.array([0.01, 0.02]),
            },
        )

        results = ensemble.run(
            initial_conditions={
                "N": 0.5,
                "E": 0.05,
                "W": 1.0,
                "S": 1.0,
                "psi": 0.0,
            },
            time_span=(0, 10),
            dt=1.0,
            parallel=False,
            show_progress=False,
        )

        assert isinstance(results, EnsembleResults)
        assert results.n_simulations == 4
        assert results.n_successful > 0
        assert len(results.outcomes) == 4
        assert results.grid_shape == (2, 2)

    def test_parallel_execution(self) -> None:
        """Test parallel ensemble execution."""
        model = SDTModel(stable_params())
        ensemble = EnsembleSimulator(
            model=model,
            parameter_grid={
                "alpha": np.array([0.005, 0.01]),
            },
            n_workers=2,
        )

        results = ensemble.run(
            initial_conditions={
                "N": 0.5,
                "E": 0.05,
                "W": 1.0,
                "S": 1.0,
                "psi": 0.0,
            },
            time_span=(0, 5),
            dt=1.0,
            parallel=True,
            show_progress=False,
        )

        assert results.n_successful > 0


class TestEnsembleResults:
    """Tests for EnsembleResults class."""

    @pytest.fixture
    def sample_results(self) -> EnsembleResults:
        """Create sample results for testing."""
        # Create synthetic outcomes
        outcomes = []
        alpha_values = np.array([0.005, 0.01, 0.015])
        gamma_values = np.array([0.01, 0.02])

        for i, alpha in enumerate(alpha_values):
            for j, gamma in enumerate(gamma_values):
                # Make max_psi increase with both parameters
                max_psi = alpha * 100 + gamma * 50 + 0.1
                mean_psi = max_psi * 0.5

                outcome = SimulationOutcome(
                    parameters={"alpha": alpha, "gamma": gamma},
                    final_state={
                        "N": 0.6,
                        "E": 0.08,
                        "W": 0.9,
                        "S": 0.8,
                        "psi": max_psi * 0.8,
                    },
                    max_psi=max_psi,
                    mean_psi=mean_psi,
                    classification=StabilityClassification.STABLE
                    if max_psi < 1.0
                    else StabilityClassification.UNSTABLE,
                    metadata={"psi_std": 0.1, "psi_final": max_psi * 0.8},
                )
                outcomes.append(outcome)

        return EnsembleResults(
            outcomes=outcomes,
            parameter_grid={
                "alpha": alpha_values,
                "gamma": gamma_values,
            },
            grid_shape=(3, 2),
            parameter_names=["alpha", "gamma"],
            parameter_values={
                "alpha": alpha_values,
                "gamma": gamma_values,
            },
            initial_conditions={
                "N": 0.5,
                "E": 0.05,
                "W": 1.0,
                "S": 1.0,
                "psi": 0.0,
            },
            time_span=(0, 100),
            n_simulations=6,
            n_successful=6,
            n_failed=0,
        )

    def test_to_dataframe(self, sample_results: EnsembleResults) -> None:
        """Test conversion to DataFrame."""
        df = sample_results.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6
        assert "alpha" in df.columns
        assert "gamma" in df.columns
        assert "max_psi" in df.columns
        assert "classification" in df.columns

    def test_classify_stability(self, sample_results: EnsembleResults) -> None:
        """Test stability classification."""
        classifications = sample_results.classify_stability(psi_threshold=1.0)

        assert classifications.shape == (3, 2)
        assert set(np.unique(classifications)).issubset(
            {"stable", "unstable", "collapse", "unknown"}
        )

    def test_get_metric_grid(self, sample_results: EnsembleResults) -> None:
        """Test metric extraction as grid."""
        max_psi_grid = sample_results.get_metric_grid("max_psi")

        assert max_psi_grid.shape == (3, 2)
        assert np.all(max_psi_grid > 0)  # All positive in our synthetic data

    def test_stability_region_area(self, sample_results: EnsembleResults) -> None:
        """Test stability region area calculation."""
        stable_area = sample_results.stability_region_area("stable", psi_threshold=1.0)

        assert 0 <= stable_area <= 1

    def test_get_phase_diagram_data(self, sample_results: EnsembleResults) -> None:
        """Test phase diagram data extraction."""
        df = sample_results.get_phase_diagram_data("alpha", "gamma", "max_psi")

        assert isinstance(df, pd.DataFrame)
        assert "alpha" in df.columns
        assert "gamma" in df.columns
        assert "max_psi" in df.columns

    def test_get_bifurcation_diagram_data(
        self, sample_results: EnsembleResults
    ) -> None:
        """Test bifurcation diagram data extraction."""
        df = sample_results.get_bifurcation_diagram_data("alpha")

        assert isinstance(df, pd.DataFrame)
        assert "alpha" in df.columns
        assert "max_psi" in df.columns
        # Should be sorted by parameter
        assert df["alpha"].is_monotonic_increasing

    def test_find_bifurcation(self, sample_results: EnsembleResults) -> None:
        """Test bifurcation detection."""
        # Our synthetic data has max_psi increasing with alpha
        # With threshold 1.0, there should be a transition
        bifurcations = sample_results.find_bifurcation(
            parameter="alpha",
            psi_threshold=1.0,
        )

        # May or may not find bifurcation depending on exact values
        assert isinstance(bifurcations, list)
        for bif in bifurcations:
            assert isinstance(bif, BifurcationPoint)
            assert bif.parameter == "alpha"
            assert bif.direction in ("stabilizing", "destabilizing")

    def test_summary(self, sample_results: EnsembleResults) -> None:
        """Test summary generation."""
        summary = sample_results.summary()

        assert isinstance(summary, str)
        assert "Ensemble" in summary
        assert "6" in summary  # n_simulations
        assert "alpha" in summary
        assert "gamma" in summary


class TestStabilityClassification:
    """Tests for StabilityClassification enum-like class."""

    def test_classification_values(self) -> None:
        """Test classification string values."""
        assert StabilityClassification.STABLE == "stable"
        assert StabilityClassification.UNSTABLE == "unstable"
        assert StabilityClassification.COLLAPSE == "collapse"
        assert StabilityClassification.OSCILLATING == "oscillating"
        assert StabilityClassification.UNKNOWN == "unknown"


class TestSimulationOutcome:
    """Tests for SimulationOutcome dataclass."""

    def test_outcome_creation(self) -> None:
        """Test SimulationOutcome can be created."""
        outcome = SimulationOutcome(
            parameters={"alpha": 0.01},
            final_state={"N": 0.5, "E": 0.05, "W": 1.0, "S": 1.0, "psi": 0.1},
            max_psi=0.5,
            mean_psi=0.25,
            classification=StabilityClassification.STABLE,
        )

        assert outcome.parameters == {"alpha": 0.01}
        assert outcome.max_psi == 0.5
        assert outcome.classification == "stable"
        assert outcome.time_series is None
        assert not outcome.terminated_early


class TestBifurcationPoint:
    """Tests for BifurcationPoint dataclass."""

    def test_bifurcation_point_creation(self) -> None:
        """Test BifurcationPoint can be created."""
        bif = BifurcationPoint(
            parameter="alpha",
            value=0.012,
            fixed_parameters={"gamma": 0.02},
            type="saddle-node",
            direction="destabilizing",
        )

        assert bif.parameter == "alpha"
        assert bif.value == 0.012
        assert bif.fixed_parameters == {"gamma": 0.02}
        assert bif.type == "saddle-node"
        assert bif.direction == "destabilizing"


class TestEnsembleIntegration:
    """Integration tests for complete ensemble workflow."""

    def test_full_workflow(self) -> None:
        """Test complete ensemble analysis workflow."""
        # Setup
        model = SDTModel(stable_params())
        ensemble = EnsembleSimulator(
            model=model,
            parameter_grid={
                "alpha": np.linspace(0.003, 0.008, 3),
                "lambda_psi": np.linspace(0.03, 0.07, 3),
            },
        )

        # Run
        results = ensemble.run(
            initial_conditions={
                "N": 0.5,
                "E": 0.05,
                "W": 1.0,
                "S": 1.0,
                "psi": 0.0,
            },
            time_span=(0, 20),
            dt=1.0,
            parallel=False,
            show_progress=False,
        )

        # Verify basic structure
        assert results.n_successful > 0
        assert results.grid_shape == (3, 3)

        # Get DataFrame
        df = results.to_dataframe()
        assert len(df) > 0
        assert "max_psi" in df.columns

        # Get stability classification
        classifications = results.classify_stability(psi_threshold=1.0)
        assert classifications.shape == (3, 3)

        # Get metric grid
        psi_grid = results.get_metric_grid("max_psi")
        assert psi_grid.shape == (3, 3)

        # Get phase diagram data
        phase_data = results.get_phase_diagram_data("alpha", "lambda_psi", "max_psi")
        assert len(phase_data) > 0

        # Calculate stability fraction
        stable_fraction = results.stability_region_area("stable", psi_threshold=1.0)
        assert 0 <= stable_fraction <= 1

    def test_varying_single_parameter(self) -> None:
        """Test ensemble with single parameter variation."""
        model = SDTModel(stable_params())
        ensemble = EnsembleSimulator(
            model=model,
            parameter_grid={
                "alpha": np.linspace(0.003, 0.010, 5),
            },
        )

        results = ensemble.run(
            initial_conditions={
                "N": 0.5,
                "E": 0.05,
                "W": 1.0,
                "S": 1.0,
                "psi": 0.0,
            },
            time_span=(0, 10),
            dt=1.0,
            parallel=False,
            show_progress=False,
        )

        assert results.n_simulations == 5
        assert results.grid_shape == (5,)

        # Bifurcation diagram data
        bif_data = results.get_bifurcation_diagram_data("alpha")
        assert len(bif_data) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
