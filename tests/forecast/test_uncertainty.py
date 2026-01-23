"""Tests for the uncertainty module."""

from __future__ import annotations

import numpy as np

from cliodynamics.forecast.uncertainty import (
    UncertaintyEstimate,
    UncertaintyQuantifier,
    combine_uncertainties,
    compute_ensemble_statistics,
    generate_initial_condition_samples,
    generate_parameter_samples,
)


class TestUncertaintyQuantifier:
    """Tests for the UncertaintyQuantifier class."""

    def test_basic_creation(self) -> None:
        """Test UncertaintyQuantifier can be created."""
        uq = UncertaintyQuantifier()

        assert uq.parameter_cv == 0.1
        assert uq.initial_condition_cv == 0.05
        assert uq.model_cv == 0.03

    def test_custom_params(self) -> None:
        """Test UncertaintyQuantifier with custom parameters."""
        uq = UncertaintyQuantifier(
            parameter_cv=0.15,
            initial_condition_cv=0.08,
            model_cv=0.05,
            growth_rate=0.03,
        )

        assert uq.parameter_cv == 0.15
        assert uq.growth_rate == 0.03

    def test_total_cv_at_zero_horizon(self) -> None:
        """Test total CV at horizon=0."""
        uq = UncertaintyQuantifier(
            parameter_cv=0.1,
            initial_condition_cv=0.05,
            model_cv=0.03,
        )

        # At horizon 0, CV should be sqrt(0.1^2 + 0.05^2 + 0.03^2)
        expected = np.sqrt(0.01 + 0.0025 + 0.0009)
        result = uq.total_cv(horizon_years=0)

        assert abs(result - expected) < 0.001

    def test_total_cv_grows_with_horizon(self) -> None:
        """Test that total CV grows with forecast horizon."""
        uq = UncertaintyQuantifier(growth_rate=0.02)

        cv_0 = uq.total_cv(horizon_years=0)
        cv_10 = uq.total_cv(horizon_years=10)
        cv_20 = uq.total_cv(horizon_years=20)

        assert cv_10 > cv_0
        assert cv_20 > cv_10

    def test_estimate_uncertainty(self) -> None:
        """Test estimate_uncertainty method."""
        uq = UncertaintyQuantifier(
            parameter_cv=0.1,
            initial_condition_cv=0.05,
            model_cv=0.03,
        )

        estimate = uq.estimate_uncertainty(
            horizon_years=10,
            mean_value=0.5,
            confidence_level=0.90,
        )

        assert isinstance(estimate, UncertaintyEstimate)
        assert estimate.confidence_level == 0.90
        assert estimate.total_cv > 0
        assert estimate.confidence_interval[0] < 0.5
        assert estimate.confidence_interval[1] > 0.5

    def test_uncertainty_profile(self) -> None:
        """Test uncertainty profile over time."""
        uq = UncertaintyQuantifier()

        profile = uq.uncertainty_profile(horizon_years=20, dt=1.0)

        assert "time" in profile
        assert "total_cv" in profile
        assert len(profile["time"]) == 21  # 0 to 20 inclusive

        # CV should increase monotonically
        cv = profile["total_cv"]
        assert all(cv[i + 1] >= cv[i] for i in range(len(cv) - 1))

    def test_uncertainty_profile_with_trajectory(self) -> None:
        """Test uncertainty profile with mean trajectory."""
        uq = UncertaintyQuantifier()

        mean_trajectory = np.linspace(0.5, 0.8, 21)
        profile = uq.uncertainty_profile(
            horizon_years=20,
            dt=1.0,
            mean_trajectory=mean_trajectory,
        )

        assert "ci_lower" in profile
        assert "ci_upper" in profile
        assert all(profile["ci_lower"] <= mean_trajectory)
        assert all(profile["ci_upper"] >= mean_trajectory)


class TestUncertaintyEstimate:
    """Tests for the UncertaintyEstimate class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        estimate = UncertaintyEstimate(
            parameter_cv=0.1,
            initial_condition_cv=0.05,
            model_cv=0.03,
            total_cv=0.12,
            confidence_interval=(0.4, 0.6),
            confidence_level=0.90,
        )

        d = estimate.to_dict()

        assert d["parameter_cv"] == 0.1
        assert d["ci_lower"] == 0.4
        assert d["ci_upper"] == 0.6
        assert d["confidence_level"] == 0.90


class TestCombineUncertainties:
    """Tests for combine_uncertainties function."""

    def test_single_cv(self) -> None:
        """Test combining single CV."""
        result = combine_uncertainties(0.1)
        assert result == 0.1

    def test_two_cvs(self) -> None:
        """Test combining two CVs."""
        result = combine_uncertainties(0.1, 0.05)
        expected = np.sqrt(0.01 + 0.0025)
        assert abs(result - expected) < 0.001

    def test_multiple_cvs(self) -> None:
        """Test combining multiple CVs."""
        result = combine_uncertainties(0.1, 0.1, 0.1)
        expected = np.sqrt(0.03)  # 3 * 0.01
        assert abs(result - expected) < 0.001


class TestComputeEnsembleStatistics:
    """Tests for compute_ensemble_statistics function."""

    def test_basic_statistics(self) -> None:
        """Test basic ensemble statistics computation."""
        # Create simple ensemble: 100 samples, 10 time points
        rng = np.random.default_rng(42)
        ensemble = rng.normal(0.5, 0.1, size=(100, 10))

        stats = compute_ensemble_statistics(ensemble)

        assert "mean" in stats
        assert "std" in stats
        assert "ci_lower" in stats
        assert "ci_upper" in stats
        assert "cv" in stats

        assert stats["mean"].shape == (10,)
        assert np.allclose(stats["mean"], 0.5, atol=0.05)

    def test_ci_ordering(self) -> None:
        """Test that CI bounds are properly ordered."""
        rng = np.random.default_rng(42)
        ensemble = rng.normal(1.0, 0.2, size=(100, 5))

        stats = compute_ensemble_statistics(ensemble, confidence_level=0.90)

        assert np.all(stats["ci_lower"] <= stats["mean"])
        assert np.all(stats["mean"] <= stats["ci_upper"])

    def test_3d_ensemble(self) -> None:
        """Test with 3D ensemble (samples, times, variables)."""
        rng = np.random.default_rng(42)
        ensemble = rng.normal(0.5, 0.1, size=(50, 10, 3))

        stats = compute_ensemble_statistics(ensemble)

        # Should compute statistics over first axis
        assert stats["mean"].shape == (10, 3)


class TestGenerateParameterSamples:
    """Tests for generate_parameter_samples function."""

    def test_basic_sampling(self) -> None:
        """Test basic parameter sampling."""
        base_params = {"r_max": 0.02, "K_0": 1.0, "mu": 0.2}

        samples = generate_parameter_samples(
            base_params, n_samples=100, cv=0.1, seed=42
        )

        assert len(samples) == 100
        assert all(isinstance(s, dict) for s in samples)
        assert all(set(s.keys()) == set(base_params.keys()) for s in samples)

    def test_samples_near_base(self) -> None:
        """Test that samples are centered near base values."""
        base_params = {"r_max": 0.02}

        samples = generate_parameter_samples(
            base_params, n_samples=1000, cv=0.1, seed=42
        )

        values = [s["r_max"] for s in samples]
        mean = np.mean(values)

        # Mean should be close to base value (allowing for log-normal bias)
        assert abs(mean - 0.02) < 0.005

    def test_samples_with_bounds(self) -> None:
        """Test parameter sampling with bounds."""
        base_params = {"mu": 0.2}
        bounds = {"mu": (0.05, 0.3)}

        samples = generate_parameter_samples(
            base_params, n_samples=100, cv=0.3, seed=42, bounds=bounds
        )

        values = [s["mu"] for s in samples]
        assert all(0.05 <= v <= 0.3 for v in values)

    def test_reproducibility(self) -> None:
        """Test that same seed produces same samples."""
        base_params = {"r_max": 0.02}

        samples1 = generate_parameter_samples(base_params, n_samples=10, seed=42)
        samples2 = generate_parameter_samples(base_params, n_samples=10, seed=42)

        for s1, s2 in zip(samples1, samples2):
            assert s1["r_max"] == s2["r_max"]


class TestGenerateInitialConditionSamples:
    """Tests for generate_initial_condition_samples function."""

    def test_basic_sampling(self) -> None:
        """Test basic IC sampling."""
        base_state = {"N": 0.8, "E": 0.1, "W": 0.9, "S": 0.95, "psi": 0.2}

        samples = generate_initial_condition_samples(
            base_state, n_samples=50, cv=0.05, seed=42
        )

        assert len(samples) == 50
        assert all(set(s.keys()) == set(base_state.keys()) for s in samples)

    def test_positive_values(self) -> None:
        """Test that samples remain positive."""
        base_state = {"N": 0.5, "psi": 0.1}

        samples = generate_initial_condition_samples(
            base_state, n_samples=100, cv=0.1, seed=42
        )

        for s in samples:
            assert s["N"] > 0
            assert s["psi"] >= 0

    def test_zero_value_handling(self) -> None:
        """Test handling of zero base values."""
        base_state = {"psi": 0.0}  # PSI can start at zero

        samples = generate_initial_condition_samples(
            base_state, n_samples=50, cv=0.1, seed=42
        )

        # Should produce small positive values
        values = [s["psi"] for s in samples]
        assert all(v >= 0 for v in values)
