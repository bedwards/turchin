"""Tests for Monte Carlo visualization module.

Tests cover:
- Fan chart generation
- Tornado plots
- Probability heatmaps
- Ensemble trajectory plots
"""

import numpy as np
import pytest

from cliodynamics.analysis import SensitivityResults
from cliodynamics.simulation.monte_carlo import MonteCarloResults
from cliodynamics.viz import monte_carlo as mc_viz


@pytest.fixture
def sample_mc_results() -> MonteCarloResults:
    """Create sample Monte Carlo results for testing."""
    time = np.arange(0, 51, 1.0)
    n_sims = 100
    n_times = len(time)
    n_vars = 5

    rng = np.random.default_rng(42)
    ensemble = np.zeros((n_sims, n_times, n_vars))

    for i in range(n_sims):
        for j in range(n_vars):
            # Create realistic-looking trajectories
            trend = np.linspace(0, 1, n_times)
            noise = 0.1 * rng.standard_normal(n_times)
            offset = 0.3 * rng.random()
            ensemble[i, :, j] = np.clip(trend + noise + offset, 0, 2)

    return MonteCarloResults(
        time=time,
        ensemble=ensemble,
        parameter_samples=rng.random((n_sims, 3)),
        parameter_names=["r_max", "alpha", "gamma"],
        state_names=("N", "E", "W", "S", "psi"),
        n_simulations=100,
        failed_simulations=0,
        seed=42,
    )


@pytest.fixture
def sample_sensitivity_results() -> SensitivityResults:
    """Create sample sensitivity results for testing."""
    return SensitivityResults(
        first_order={
            "r_max": 0.30,
            "alpha": 0.25,
            "gamma": 0.15,
            "lambda_psi": 0.10,
            "beta": 0.05,
        },
        total_order={
            "r_max": 0.40,
            "alpha": 0.35,
            "gamma": 0.25,
            "lambda_psi": 0.15,
            "beta": 0.08,
        },
        confidence_intervals={},
        parameter_names=["r_max", "alpha", "gamma", "lambda_psi", "beta"],
        target_variable="psi",
        target_time=100.0,
        n_samples=1000,
        method="sobol",
    )


class TestFanChart:
    """Tests for fan chart visualization."""

    def test_fan_chart_creation(self, sample_mc_results: MonteCarloResults) -> None:
        """Test that fan chart can be created."""
        chart = mc_viz.plot_fan_chart(sample_mc_results, variable="psi")

        assert chart is not None
        # Chart should be a layer chart with multiple layers
        assert hasattr(chart, "layer") or hasattr(chart, "data")

    def test_fan_chart_with_options(
        self, sample_mc_results: MonteCarloResults
    ) -> None:
        """Test fan chart with various options."""
        chart = mc_viz.plot_fan_chart(
            sample_mc_results,
            variable="psi",
            title="Custom Title",
            show_median=True,
            show_mean=True,
            percentiles=[5, 25, 50, 75, 95],
        )

        assert chart is not None


class TestTornadoPlot:
    """Tests for tornado plot visualization."""

    def test_tornado_creation(
        self, sample_sensitivity_results: SensitivityResults
    ) -> None:
        """Test that tornado plot can be created."""
        chart = mc_viz.plot_tornado(sample_sensitivity_results)

        assert chart is not None

    def test_tornado_with_interactions(
        self, sample_sensitivity_results: SensitivityResults
    ) -> None:
        """Test tornado plot showing interactions."""
        chart = mc_viz.plot_tornado(
            sample_sensitivity_results,
            show_interactions=True,
            title="Parameter Sensitivity",
        )

        assert chart is not None

    def test_tornado_max_params(
        self, sample_sensitivity_results: SensitivityResults
    ) -> None:
        """Test limiting number of parameters shown."""
        chart = mc_viz.plot_tornado(
            sample_sensitivity_results,
            max_params=3,
        )

        assert chart is not None


class TestProbabilityHeatmap:
    """Tests for probability heatmap."""

    def test_heatmap_creation(self, sample_mc_results: MonteCarloResults) -> None:
        """Test that probability heatmap can be created."""
        chart = mc_viz.plot_probability_heatmap(
            sample_mc_results,
            variable="psi",
        )

        assert chart is not None

    def test_heatmap_custom_thresholds(
        self, sample_mc_results: MonteCarloResults
    ) -> None:
        """Test heatmap with custom thresholds."""
        chart = mc_viz.plot_probability_heatmap(
            sample_mc_results,
            variable="psi",
            thresholds=[0.25, 0.5, 0.75, 1.0, 1.25],
        )

        assert chart is not None


class TestProbabilityOverTime:
    """Tests for probability over time plot."""

    def test_probability_line(self, sample_mc_results: MonteCarloResults) -> None:
        """Test probability over time line chart."""
        chart = mc_viz.plot_probability_over_time(
            sample_mc_results,
            variable="psi",
            threshold=0.5,
        )

        assert chart is not None


class TestTimingDistribution:
    """Tests for timing distribution histogram."""

    def test_timing_histogram(self, sample_mc_results: MonteCarloResults) -> None:
        """Test timing distribution histogram."""
        chart = mc_viz.plot_timing_distribution(
            sample_mc_results,
            variable="psi",
            threshold=0.5,
        )

        assert chart is not None


class TestEnsembleTrajectories:
    """Tests for ensemble trajectory plot."""

    def test_ensemble_plot(self, sample_mc_results: MonteCarloResults) -> None:
        """Test ensemble trajectory visualization."""
        chart = mc_viz.plot_ensemble_trajectories(
            sample_mc_results,
            variable="psi",
            n_trajectories=20,
        )

        assert chart is not None

    def test_ensemble_with_percentiles(
        self, sample_mc_results: MonteCarloResults
    ) -> None:
        """Test ensemble plot with percentile highlights."""
        chart = mc_viz.plot_ensemble_trajectories(
            sample_mc_results,
            variable="psi",
            n_trajectories=30,
            highlight_percentiles=True,
        )

        assert chart is not None


class TestParameterScatter:
    """Tests for parameter-output scatter plot."""

    def test_scatter_plot(self, sample_mc_results: MonteCarloResults) -> None:
        """Test parameter vs output scatter plot."""
        chart = mc_viz.plot_parameter_scatter(
            sample_mc_results,
            parameter="r_max",
            variable="psi",
        )

        assert chart is not None

    def test_scatter_invalid_parameter(
        self, sample_mc_results: MonteCarloResults
    ) -> None:
        """Test error for invalid parameter."""
        with pytest.raises(ValueError, match="not in results"):
            mc_viz.plot_parameter_scatter(
                sample_mc_results,
                parameter="nonexistent",
                variable="psi",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
