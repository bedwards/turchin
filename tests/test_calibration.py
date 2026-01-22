"""Tests for the calibration module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cliodynamics.calibration import (
    Calibrator,
    CalibrationResult,
    generate_synthetic_data,
)
from cliodynamics.models import SDTModel, SDTParams


class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""

    def test_creation(self) -> None:
        """Test basic creation of CalibrationResult."""
        result = CalibrationResult(
            best_params={"r_max": 0.02, "K_0": 1.0},
            loss=0.01,
            n_iterations=100,
            converged=True,
            message="Optimization converged",
            initial_conditions={"N": 0.5, "E": 0.01, "W": 1.0, "S": 1.0, "psi": 0.0},
            param_bounds={"r_max": (0.01, 0.05), "K_0": (0.5, 2.0)},
        )
        assert result.best_params["r_max"] == 0.02
        assert result.loss == 0.01
        assert result.converged is True

    def test_summary(self) -> None:
        """Test summary string generation."""
        result = CalibrationResult(
            best_params={"r_max": 0.02},
            loss=0.01,
            n_iterations=100,
            converged=True,
            message="Success",
            initial_conditions={"N": 0.5},
            param_bounds={"r_max": (0.01, 0.05)},
        )
        summary = result.summary()
        assert "Calibration Results" in summary
        assert "r_max" in summary
        assert "0.02" in summary

    def test_confidence_intervals(self) -> None:
        """Test with confidence intervals."""
        result = CalibrationResult(
            best_params={"r_max": 0.02},
            loss=0.01,
            n_iterations=100,
            converged=True,
            message="Success",
            initial_conditions={"N": 0.5},
            param_bounds={"r_max": (0.01, 0.05)},
            confidence_intervals={"r_max": (0.015, 0.025)},
        )
        assert result.confidence_intervals["r_max"] == (0.015, 0.025)


class TestGenerateSyntheticData:
    """Tests for synthetic data generation."""

    def test_basic_generation(self) -> None:
        """Test basic synthetic data generation."""
        params = SDTParams()
        initial = {"N": 0.5, "E": 0.01, "W": 1.0, "S": 1.0, "psi": 0.0}

        df = generate_synthetic_data(
            params=params,
            initial_conditions=initial,
            time_span=(0, 100),
            dt=1.0,
        )

        assert "year" in df.columns
        assert "N" in df.columns
        assert len(df) > 0

    def test_with_noise(self) -> None:
        """Test synthetic data with noise."""
        params = SDTParams()
        initial = {"N": 0.5, "E": 0.01, "W": 1.0, "S": 1.0, "psi": 0.0}

        df1 = generate_synthetic_data(
            params=params,
            initial_conditions=initial,
            time_span=(0, 50),
            noise_std=0.0,
            seed=42,
        )

        df2 = generate_synthetic_data(
            params=params,
            initial_conditions=initial,
            time_span=(0, 50),
            noise_std=0.1,
            seed=42,
        )

        # With noise, values should differ
        assert not np.allclose(df1["N"].values, df2["N"].values)

    def test_variable_selection(self) -> None:
        """Test selecting specific variables."""
        params = SDTParams()
        initial = {"N": 0.5, "E": 0.01, "W": 1.0, "S": 1.0, "psi": 0.0}

        df = generate_synthetic_data(
            params=params,
            initial_conditions=initial,
            time_span=(0, 50),
            variables=["N", "psi"],
        )

        assert "year" in df.columns
        assert "N" in df.columns
        assert "psi" in df.columns
        assert "E" not in df.columns
        assert "W" not in df.columns

    def test_reproducibility_with_seed(self) -> None:
        """Test that seed provides reproducibility."""
        params = SDTParams()
        initial = {"N": 0.5, "E": 0.01, "W": 1.0, "S": 1.0, "psi": 0.0}

        df1 = generate_synthetic_data(
            params=params,
            initial_conditions=initial,
            time_span=(0, 50),
            noise_std=0.1,
            seed=42,
        )

        df2 = generate_synthetic_data(
            params=params,
            initial_conditions=initial,
            time_span=(0, 50),
            noise_std=0.1,
            seed=42,
        )

        np.testing.assert_array_almost_equal(df1["N"].values, df2["N"].values)


class TestCalibrator:
    """Tests for the Calibrator class."""

    @pytest.fixture
    def simple_data(self) -> pd.DataFrame:
        """Create simple test data."""
        params = SDTParams(r_max=0.02, K_0=1.0)
        initial = {"N": 0.5, "E": 0.01, "W": 1.0, "S": 1.0, "psi": 0.0}
        return generate_synthetic_data(
            params=params,
            initial_conditions=initial,
            time_span=(0, 100),
            dt=10.0,
            variables=["N"],
        )

    def test_calibrator_creation(self, simple_data: pd.DataFrame) -> None:
        """Test creating a Calibrator instance."""
        calibrator = Calibrator(
            model=SDTModel,
            observed_data=simple_data,
            fit_variables=["N"],
            time_column="year",
        )
        assert calibrator.model == SDTModel
        assert calibrator.fit_variables == ["N"]

    def test_basic_fit(self, simple_data: pd.DataFrame) -> None:
        """Test basic parameter fitting."""
        calibrator = Calibrator(
            model=SDTModel,
            observed_data=simple_data,
            fit_variables=["N"],
            time_column="year",
        )

        result = calibrator.fit(
            param_bounds={"r_max": (0.01, 0.05)},
            method="L-BFGS-B",  # Fast local optimizer
            maxiter=50,
        )

        assert isinstance(result, CalibrationResult)
        assert "r_max" in result.best_params
        assert result.loss >= 0

    def test_fit_recovers_known_params(self) -> None:
        """Test that calibration can recover known parameters."""
        # Generate data with known parameters
        true_r_max = 0.025
        params = SDTParams(r_max=true_r_max, K_0=1.0)
        initial = {"N": 0.5, "E": 0.01, "W": 1.0, "S": 1.0, "psi": 0.0}

        data = generate_synthetic_data(
            params=params,
            initial_conditions=initial,
            time_span=(0, 200),
            dt=20.0,
            variables=["N"],
            noise_std=0.0,  # No noise for exact recovery
        )

        calibrator = Calibrator(
            model=SDTModel,
            observed_data=data,
            fit_variables=["N"],
            time_column="year",
        )

        result = calibrator.fit(
            param_bounds={"r_max": (0.01, 0.05)},
            method="differential_evolution",
            maxiter=100,
            seed=42,
        )

        # Should recover parameter within reasonable tolerance
        assert abs(result.best_params["r_max"] - true_r_max) < 0.01

    def test_multiple_parameters(self) -> None:
        """Test fitting multiple parameters simultaneously."""
        params = SDTParams(r_max=0.02, K_0=1.0)
        initial = {"N": 0.5, "E": 0.01, "W": 1.0, "S": 1.0, "psi": 0.0}

        data = generate_synthetic_data(
            params=params,
            initial_conditions=initial,
            time_span=(0, 100),
            dt=10.0,
            variables=["N", "psi"],
        )

        calibrator = Calibrator(
            model=SDTModel,
            observed_data=data,
            fit_variables=["N", "psi"],
            time_column="year",
        )

        result = calibrator.fit(
            param_bounds={
                "r_max": (0.01, 0.05),
                "K_0": (0.5, 2.0),
            },
            method="L-BFGS-B",
            maxiter=50,
        )

        assert "r_max" in result.best_params
        assert "K_0" in result.best_params

    def test_custom_loss_function(self, simple_data: pd.DataFrame) -> None:
        """Test using a custom loss function."""
        def mae_loss(pred: np.ndarray, obs: np.ndarray) -> float:
            return float(np.mean(np.abs(pred - obs)))

        calibrator = Calibrator(
            model=SDTModel,
            observed_data=simple_data,
            fit_variables=["N"],
            time_column="year",
            loss_function=mae_loss,
        )

        result = calibrator.fit(
            param_bounds={"r_max": (0.01, 0.05)},
            method="L-BFGS-B",
            maxiter=50,
        )

        assert result.loss >= 0

    def test_different_optimizers(self, simple_data: pd.DataFrame) -> None:
        """Test different optimization methods."""
        calibrator = Calibrator(
            model=SDTModel,
            observed_data=simple_data,
            fit_variables=["N"],
            time_column="year",
        )

        # Test L-BFGS-B
        result1 = calibrator.fit(
            param_bounds={"r_max": (0.01, 0.05)},
            method="L-BFGS-B",
            maxiter=30,
        )
        assert result1.loss >= 0

        # Test differential evolution
        result2 = calibrator.fit(
            param_bounds={"r_max": (0.01, 0.05)},
            method="differential_evolution",
            maxiter=30,
            seed=42,
        )
        assert result2.loss >= 0


class TestConfidenceIntervals:
    """Tests for confidence interval computation."""

    def test_bootstrap_ci(self) -> None:
        """Test bootstrap confidence interval computation."""
        params = SDTParams(r_max=0.02)
        initial = {"N": 0.5, "E": 0.01, "W": 1.0, "S": 1.0, "psi": 0.0}

        data = generate_synthetic_data(
            params=params,
            initial_conditions=initial,
            time_span=(0, 100),
            dt=10.0,
            variables=["N"],
            noise_std=0.05,  # Add some noise
            seed=42,
        )

        calibrator = Calibrator(
            model=SDTModel,
            observed_data=data,
            fit_variables=["N"],
            time_column="year",
        )

        result = calibrator.fit(
            param_bounds={"r_max": (0.01, 0.05)},
            method="L-BFGS-B",
            maxiter=30,
        )

        result_with_ci = calibrator.compute_confidence_intervals(
            result=result,
            n_bootstrap=10,  # Small for testing
            confidence_level=0.9,
            seed=42,
        )

        assert "r_max" in result_with_ci.confidence_intervals
        ci = result_with_ci.confidence_intervals["r_max"]
        assert ci[0] <= ci[1]  # Lower <= upper


class TestCrossValidation:
    """Tests for cross-validation."""

    def test_time_series_cv(self) -> None:
        """Test time series cross-validation."""
        params = SDTParams(r_max=0.02)
        initial = {"N": 0.5, "E": 0.01, "W": 1.0, "S": 1.0, "psi": 0.0}

        data = generate_synthetic_data(
            params=params,
            initial_conditions=initial,
            time_span=(0, 200),
            dt=10.0,
            variables=["N"],
        )

        calibrator = Calibrator(
            model=SDTModel,
            observed_data=data,
            fit_variables=["N"],
            time_column="year",
        )

        cv_results = calibrator.cross_validate(
            param_bounds={"r_max": (0.01, 0.05)},
            n_folds=3,
            method="L-BFGS-B",
            maxiter=20,
        )

        assert "mean_val_loss" in cv_results
        assert "std_val_loss" in cv_results
        assert len(cv_results["val_losses"]) <= 3


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_calibration_pipeline(self) -> None:
        """Test complete calibration pipeline."""
        # 1. Generate synthetic data with known parameters
        true_params = SDTParams(r_max=0.025, K_0=1.0)
        initial = {"N": 0.5, "E": 0.01, "W": 1.0, "S": 1.0, "psi": 0.0}

        data = generate_synthetic_data(
            params=true_params,
            initial_conditions=initial,
            time_span=(0, 150),
            dt=15.0,
            variables=["N"],
            noise_std=0.02,
            seed=42,
        )

        # 2. Create calibrator
        calibrator = Calibrator(
            model=SDTModel,
            observed_data=data,
            fit_variables=["N"],
            time_column="year",
        )

        # 3. Fit parameters
        result = calibrator.fit(
            param_bounds={"r_max": (0.01, 0.05)},
            method="differential_evolution",
            maxiter=50,
            seed=42,
        )

        # 4. Verify convergence
        assert result.converged or result.loss < 0.1

        # 5. Get summary
        summary = result.summary()
        assert "r_max" in summary

    def test_module_exports(self) -> None:
        """Test that all expected items are exported."""
        from cliodynamics import calibration

        assert hasattr(calibration, "Calibrator")
        assert hasattr(calibration, "CalibrationResult")
        assert hasattr(calibration, "generate_synthetic_data")
