"""Tests for the Monte Carlo simulation framework.

Tests cover:
- Distribution classes
- MonteCarloSimulator functionality
- MonteCarloResults aggregation
- Probability calculations
- Parallel execution
"""

import numpy as np
import pandas as pd
import pytest

from cliodynamics.models import SDTModel, SDTParams
from cliodynamics.simulation import (
    Constant,
    LogNormal,
    MonteCarloResults,
    MonteCarloSimulator,
    Normal,
    Triangular,
    TruncatedNormal,
    Uniform,
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


class TestDistributions:
    """Tests for distribution classes."""

    def test_normal_distribution(self) -> None:
        """Test Normal distribution sampling."""
        dist = Normal(mu=5.0, sigma=1.0)
        rng = np.random.default_rng(42)

        samples = dist.sample(rng, 10000)

        assert len(samples) == 10000
        assert abs(np.mean(samples) - 5.0) < 0.1
        assert abs(np.std(samples) - 1.0) < 0.1
        assert dist.mean() == 5.0
        assert dist.std() == 1.0

    def test_normal_requires_positive_sigma(self) -> None:
        """Test that Normal raises error for non-positive sigma."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            Normal(mu=0.0, sigma=0.0)

        with pytest.raises(ValueError, match="sigma must be positive"):
            Normal(mu=0.0, sigma=-1.0)

    def test_uniform_distribution(self) -> None:
        """Test Uniform distribution sampling."""
        dist = Uniform(low=2.0, high=8.0)
        rng = np.random.default_rng(42)

        samples = dist.sample(rng, 10000)

        assert len(samples) == 10000
        assert np.all(samples >= 2.0)
        assert np.all(samples <= 8.0)
        assert abs(np.mean(samples) - 5.0) < 0.1
        assert dist.mean() == 5.0

    def test_uniform_requires_low_less_than_high(self) -> None:
        """Test that Uniform raises error for invalid bounds."""
        with pytest.raises(ValueError, match="low must be < high"):
            Uniform(low=5.0, high=5.0)

        with pytest.raises(ValueError, match="low must be < high"):
            Uniform(low=6.0, high=5.0)

    def test_lognormal_distribution(self) -> None:
        """Test LogNormal distribution sampling."""
        dist = LogNormal(mu=0.0, sigma=0.5)
        rng = np.random.default_rng(42)

        samples = dist.sample(rng, 10000)

        assert len(samples) == 10000
        assert np.all(samples > 0)  # Log-normal is always positive

    def test_triangular_distribution(self) -> None:
        """Test Triangular distribution sampling."""
        dist = Triangular(low=0.0, mode=3.0, high=5.0)
        rng = np.random.default_rng(42)

        samples = dist.sample(rng, 10000)

        assert len(samples) == 10000
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 5.0)
        # Mode should be common
        assert np.median(samples) > 2.0 and np.median(samples) < 4.0

    def test_triangular_requires_valid_bounds(self) -> None:
        """Test that Triangular raises error for invalid bounds."""
        with pytest.raises(ValueError, match="Must have low <= mode <= high"):
            Triangular(low=3.0, mode=2.0, high=5.0)

    def test_constant_distribution(self) -> None:
        """Test Constant distribution always returns same value."""
        dist = Constant(value=42.0)
        rng = np.random.default_rng(42)

        samples = dist.sample(rng, 100)

        assert np.all(samples == 42.0)
        assert dist.mean() == 42.0
        assert dist.std() == 0.0

    def test_truncated_normal(self) -> None:
        """Test TruncatedNormal stays within bounds."""
        dist = TruncatedNormal(mu=0.5, sigma=0.5, low=0.0, high=1.0)
        rng = np.random.default_rng(42)

        samples = dist.sample(rng, 10000)

        assert len(samples) == 10000
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)


class TestMonteCarloSimulator:
    """Tests for MonteCarloSimulator class."""

    def test_simulator_initialization(self) -> None:
        """Test MonteCarloSimulator can be initialized."""
        model = SDTModel(stable_params())
        mc = MonteCarloSimulator(
            model=model,
            n_simulations=100,
            parameter_distributions={"r_max": Normal(0.02, 0.005)},
            seed=42,
        )

        assert mc.n_simulations == 100
        assert "r_max" in mc.parameter_distributions
        assert mc.seed == 42

    def test_simple_run(self) -> None:
        """Test basic Monte Carlo run."""
        model = SDTModel(stable_params())
        mc = MonteCarloSimulator(
            model=model,
            n_simulations=50,
            parameter_distributions={"r_max": Uniform(0.015, 0.025)},
            seed=42,
        )

        results = mc.run(
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
        )

        assert isinstance(results, MonteCarloResults)
        assert results.n_simulations == 50
        assert len(results.time) == 11  # 0, 1, ..., 10
        assert results.ensemble.shape[0] <= 50  # May have some failures
        assert results.ensemble.shape[1] == 11  # time points
        assert results.ensemble.shape[2] == 5  # state variables

    def test_run_without_parameter_distributions(self) -> None:
        """Test Monte Carlo with only IC uncertainty."""
        model = SDTModel(stable_params())
        mc = MonteCarloSimulator(
            model=model,
            n_simulations=20,
            initial_condition_distributions={"N": Uniform(0.4, 0.6)},
            seed=42,
        )

        results = mc.run(
            initial_conditions={
                "N": 0.5,
                "E": 0.05,
                "W": 1.0,
                "S": 1.0,
                "psi": 0.0,
            },
            time_span=(0, 5),
            dt=1.0,
            parallel=False,
        )

        assert results.n_successful > 0
        assert len(results.parameter_names) == 0

    def test_parallel_execution(self) -> None:
        """Test parallel Monte Carlo execution."""
        model = SDTModel(stable_params())
        mc = MonteCarloSimulator(
            model=model,
            n_simulations=20,
            parameter_distributions={"r_max": Uniform(0.015, 0.025)},
            n_workers=2,
            seed=42,
        )

        results = mc.run(
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
        )

        assert results.n_successful > 0


class TestMonteCarloResults:
    """Tests for MonteCarloResults class."""

    @pytest.fixture
    def sample_results(self) -> MonteCarloResults:
        """Create sample results for testing."""
        time = np.arange(0, 11, 1.0)
        n_sims = 100
        n_times = len(time)
        n_vars = 5

        # Create synthetic ensemble data
        rng = np.random.default_rng(42)
        ensemble = np.zeros((n_sims, n_times, n_vars))

        for i in range(n_sims):
            for j in range(n_vars):
                # Simple linear trend with noise
                ensemble[i, :, j] = (
                    np.linspace(0, 1, n_times)
                    + 0.1 * rng.standard_normal(n_times)
                    + 0.2 * rng.random()  # Simulation-specific offset
                )

        return MonteCarloResults(
            time=time,
            ensemble=ensemble,
            parameter_samples=rng.random((n_sims, 2)),
            parameter_names=["param1", "param2"],
            state_names=("N", "E", "W", "S", "psi"),
            n_simulations=100,
            failed_simulations=0,
            seed=42,
        )

    def test_get_variable(self, sample_results: MonteCarloResults) -> None:
        """Test extracting single variable from ensemble."""
        psi_data = sample_results.get_variable("psi")

        assert psi_data.shape == (100, 11)

    def test_mean_and_std(self, sample_results: MonteCarloResults) -> None:
        """Test mean and std computations."""
        mean_df = sample_results.mean()
        std_df = sample_results.std()

        assert isinstance(mean_df, pd.DataFrame)
        assert isinstance(std_df, pd.DataFrame)
        assert "time" in mean_df.columns
        assert "psi" in mean_df.columns
        assert len(mean_df) == 11

    def test_percentiles(self, sample_results: MonteCarloResults) -> None:
        """Test percentile computation."""
        p50 = sample_results.percentile(50)
        assert isinstance(p50, pd.DataFrame)
        assert "psi" in p50.columns

        multi_p = sample_results.percentile([10, 50, 90])
        assert isinstance(multi_p, dict)
        assert 50 in multi_p

    def test_confidence_interval(self, sample_results: MonteCarloResults) -> None:
        """Test confidence interval computation."""
        lower, upper = sample_results.confidence_interval(0.90)

        assert isinstance(lower, pd.DataFrame)
        assert isinstance(upper, pd.DataFrame)
        # Lower should be less than upper
        assert np.all(lower["psi"].values <= upper["psi"].values)

    def test_probability_at_year(self, sample_results: MonteCarloResults) -> None:
        """Test probability calculation at specific year."""
        # Values are roughly 0-1, so threshold of 0.5 at year 5 should be ~50%
        prob = sample_results.probability("psi", threshold=0.5, year=5)

        assert 0 <= prob <= 1

    def test_probability_ever(self, sample_results: MonteCarloResults) -> None:
        """Test probability of ever exceeding threshold."""
        # All trajectories end around 1, so prob(>0.5 ever) should be high
        prob = sample_results.probability("psi", threshold=0.5)

        assert prob > 0.5  # Should be high for this data

    def test_probability_by_year(self, sample_results: MonteCarloResults) -> None:
        """Test probability over time."""
        prob_df = sample_results.probability_by_year("psi", threshold=0.5)

        assert isinstance(prob_df, pd.DataFrame)
        assert "year" in prob_df.columns
        assert "probability" in prob_df.columns
        assert len(prob_df) == 11

    def test_peak_timing_distribution(self, sample_results: MonteCarloResults) -> None:
        """Test peak timing distribution."""
        peaks = sample_results.peak_timing_distribution("psi")

        assert len(peaks) == 100  # One per simulation

    def test_first_crossing_distribution(
        self, sample_results: MonteCarloResults
    ) -> None:
        """Test first crossing time distribution."""
        crossings = sample_results.first_crossing_distribution("psi", 0.5)

        assert len(crossings) == 100

    def test_fan_chart_data(self, sample_results: MonteCarloResults) -> None:
        """Test fan chart data preparation."""
        df = sample_results.to_fan_chart_data("psi")

        assert "time" in df.columns
        assert "p50" in df.columns
        assert len(df) == 11

    def test_summary(self, sample_results: MonteCarloResults) -> None:
        """Test summary generation."""
        summary = sample_results.summary()

        assert isinstance(summary, str)
        assert "Monte Carlo" in summary
        assert "100" in summary  # n_simulations


class TestMonteCarloIntegration:
    """Integration tests for complete Monte Carlo workflow."""

    def test_full_workflow(self) -> None:
        """Test complete Monte Carlo workflow."""
        # Setup
        model = SDTModel(stable_params())
        mc = MonteCarloSimulator(
            model=model,
            n_simulations=30,
            parameter_distributions={
                "r_max": Uniform(0.015, 0.025),
                "alpha": Normal(0.005, 0.001),
            },
            seed=42,
        )

        # Run
        results = mc.run(
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
        )

        # Verify
        assert results.n_successful > 0

        # Statistics
        mean = results.mean()
        assert "psi" in mean.columns

        # Probability queries
        prob = results.probability("psi", 0.1, year=10)
        assert 0 <= prob <= 1

        # Fan chart data
        fan_data = results.to_fan_chart_data("psi")
        assert len(fan_data) > 0

    def test_sensitivity_analysis(self) -> None:
        """Test sensitivity analysis from Monte Carlo."""
        model = SDTModel(stable_params())
        mc = MonteCarloSimulator(
            model=model,
            n_simulations=50,
            parameter_distributions={
                "r_max": Uniform(0.015, 0.025),
                "lambda_psi": Uniform(0.03, 0.07),
            },
            seed=42,
        )

        sensitivities = mc.sensitivity_analysis(
            initial_conditions={
                "N": 0.5,
                "E": 0.05,
                "W": 1.0,
                "S": 1.0,
                "psi": 0.0,
            },
            time_span=(0, 10),
            target_variable="psi",
        )

        assert isinstance(sensitivities, dict)
        assert "r_max" in sensitivities or "lambda_psi" in sensitivities


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
