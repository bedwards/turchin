"""Tests for the sensitivity analysis module.

Tests cover:
- SensitivityAnalyzer initialization
- Sobol sensitivity analysis
- Correlation-based analysis
- PRCC analysis
- Variance decomposition
"""

import numpy as np
import pandas as pd
import pytest

from cliodynamics.analysis import SensitivityAnalyzer, SensitivityResults
from cliodynamics.models import SDTModel, SDTParams


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


class TestSensitivityAnalyzerInit:
    """Tests for SensitivityAnalyzer initialization."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        model = SDTModel(stable_params())
        analyzer = SensitivityAnalyzer(
            model=model,
            parameter_bounds={
                "r_max": (0.01, 0.03),
                "alpha": (0.003, 0.007),
            },
            n_samples=100,
            seed=42,
        )

        assert analyzer.n_params == 2
        assert "r_max" in analyzer.parameter_names
        assert "alpha" in analyzer.parameter_names
        assert analyzer.n_samples == 100


class TestSensitivityResults:
    """Tests for SensitivityResults class."""

    @pytest.fixture
    def sample_results(self) -> SensitivityResults:
        """Create sample sensitivity results for testing."""
        return SensitivityResults(
            first_order={
                "r_max": 0.3,
                "alpha": 0.2,
                "gamma": 0.1,
            },
            total_order={
                "r_max": 0.4,
                "alpha": 0.35,
                "gamma": 0.15,
            },
            confidence_intervals={
                "r_max": ((0.25, 0.35), (0.35, 0.45)),
                "alpha": ((0.15, 0.25), (0.3, 0.4)),
                "gamma": ((0.05, 0.15), (0.1, 0.2)),
            },
            parameter_names=["r_max", "alpha", "gamma"],
            target_variable="psi",
            target_time=100.0,
            n_samples=1000,
            method="sobol",
        )

    def test_ranking(self, sample_results: SensitivityResults) -> None:
        """Test parameter ranking by total-order index."""
        ranking = sample_results.ranking

        assert len(ranking) == 3
        # Should be sorted by ST descending
        assert ranking[0][0] == "r_max"  # Highest ST
        assert ranking[2][0] == "gamma"  # Lowest ST

    def test_interaction_strength(self, sample_results: SensitivityResults) -> None:
        """Test interaction strength calculation."""
        interactions = sample_results.interaction_strength

        # Interaction = ST - S1
        assert abs(interactions["r_max"] - 0.1) < 1e-6
        assert abs(interactions["alpha"] - 0.15) < 1e-6
        assert abs(interactions["gamma"] - 0.05) < 1e-6

    def test_to_dataframe(self, sample_results: SensitivityResults) -> None:
        """Test conversion to DataFrame."""
        df = sample_results.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "parameter" in df.columns
        assert "S1" in df.columns
        assert "ST" in df.columns
        assert "interaction" in df.columns
        assert len(df) == 3

    def test_summary(self, sample_results: SensitivityResults) -> None:
        """Test summary generation."""
        summary = sample_results.summary()

        assert isinstance(summary, str)
        assert "Sensitivity" in summary
        assert "r_max" in summary


class TestSobolAnalysis:
    """Tests for Sobol sensitivity analysis."""

    def test_sobol_analysis_runs(self) -> None:
        """Test that Sobol analysis completes without error."""
        model = SDTModel(stable_params())
        analyzer = SensitivityAnalyzer(
            model=model,
            parameter_bounds={
                "lambda_psi": (0.03, 0.07),
            },
            n_samples=32,  # Small for fast tests
            seed=42,
        )

        results = analyzer.sobol_analysis(
            initial_conditions={
                "N": 0.5,
                "E": 0.05,
                "W": 1.0,
                "S": 1.0,
                "psi": 0.0,
            },
            time_span=(0, 10),
            target_variable="psi",
            target_time=10,
            dt=1.0,
            n_bootstrap=10,
        )

        assert isinstance(results, SensitivityResults)
        assert results.method == "sobol"
        assert "lambda_psi" in results.first_order
        assert "lambda_psi" in results.total_order

    def test_sobol_indices_bounded(self) -> None:
        """Test that Sobol indices are in valid range [0, 1]."""
        model = SDTModel(stable_params())
        analyzer = SensitivityAnalyzer(
            model=model,
            parameter_bounds={
                "lambda_psi": (0.03, 0.07),
                "psi_decay": (0.01, 0.03),
            },
            n_samples=32,
            seed=42,
        )

        results = analyzer.sobol_analysis(
            initial_conditions={
                "N": 0.5,
                "E": 0.05,
                "W": 1.0,
                "S": 1.0,
                "psi": 0.0,
            },
            time_span=(0, 10),
            target_variable="psi",
            n_bootstrap=0,
        )

        for name in results.parameter_names:
            s1 = results.first_order[name]
            st = results.total_order[name]
            assert 0 <= s1 <= 1, f"S1 for {name} = {s1}"
            assert 0 <= st <= 1, f"ST for {name} = {st}"


class TestCorrelationAnalysis:
    """Tests for correlation-based sensitivity analysis."""

    def test_pearson_correlation(self) -> None:
        """Test Pearson correlation analysis."""
        model = SDTModel(stable_params())
        analyzer = SensitivityAnalyzer(
            model=model,
            parameter_bounds={
                "lambda_psi": (0.03, 0.07),
            },
            n_samples=50,
            seed=42,
        )

        results = analyzer.correlation_analysis(
            initial_conditions={
                "N": 0.5,
                "E": 0.05,
                "W": 1.0,
                "S": 1.0,
                "psi": 0.0,
            },
            time_span=(0, 10),
            target_variable="psi",
            method="pearson",
        )

        assert results.method == "correlation_pearson"
        assert "lambda_psi" in results.first_order

    def test_spearman_correlation(self) -> None:
        """Test Spearman correlation analysis."""
        model = SDTModel(stable_params())
        analyzer = SensitivityAnalyzer(
            model=model,
            parameter_bounds={
                "lambda_psi": (0.03, 0.07),
            },
            n_samples=50,
            seed=42,
        )

        results = analyzer.correlation_analysis(
            initial_conditions={
                "N": 0.5,
                "E": 0.05,
                "W": 1.0,
                "S": 1.0,
                "psi": 0.0,
            },
            time_span=(0, 10),
            target_variable="psi",
            method="spearman",
        )

        assert results.method == "correlation_spearman"


class TestPRCCAnalysis:
    """Tests for PRCC analysis."""

    def test_prcc_analysis(self) -> None:
        """Test PRCC analysis completes."""
        model = SDTModel(stable_params())
        analyzer = SensitivityAnalyzer(
            model=model,
            parameter_bounds={
                "lambda_psi": (0.03, 0.07),
                "psi_decay": (0.01, 0.03),
            },
            n_samples=50,
            seed=42,
        )

        results = analyzer.prcc_analysis(
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

        assert results.method == "prcc"
        assert len(results.parameter_names) == 2


class TestVarianceDecomposition:
    """Tests for variance decomposition."""

    def test_variance_decomposition(self) -> None:
        """Test variance decomposition returns expected structure."""
        model = SDTModel(stable_params())
        analyzer = SensitivityAnalyzer(
            model=model,
            parameter_bounds={
                "lambda_psi": (0.03, 0.07),
            },
            n_samples=32,
            seed=42,
        )

        df = analyzer.variance_decomposition(
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

        assert isinstance(df, pd.DataFrame)
        assert "parameter" in df.columns
        assert "variance_contribution" in df.columns
        assert "percentage" in df.columns
        assert "cumulative_percentage" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
