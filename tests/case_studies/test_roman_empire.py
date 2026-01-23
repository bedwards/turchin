"""Tests for Roman Empire case study module.

Tests cover:
- Data loading and synthetic data generation
- Model calibration
- Simulation execution
- Secular cycle detection
- Report generation
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cliodynamics.case_studies import roman_empire as rome
from cliodynamics.case_studies.roman_empire import (
    RomanEmpireStudy,
    RomanHistoricalData,
    SecularCycle,
    _detect_secular_cycles,
    _generate_roman_estimates,
)
from cliodynamics.models import SDTParams
from cliodynamics.simulation import SimulationResult


class TestRomanHistoricalData:
    """Tests for RomanHistoricalData dataclass."""

    def test_create_with_defaults(self):
        """Should create with default values."""
        df = pd.DataFrame({"year": [0], "N": [0.5]})
        data = RomanHistoricalData(df=df)
        assert data.source == "unknown"
        assert data.time_range == (-500, 500)
        assert len(data.polities) == 0

    def test_summary(self):
        """Should generate readable summary."""
        df = pd.DataFrame({"year": [0, 100], "N": [0.5, 0.6], "psi": [0.0, 0.1]})
        data = RomanHistoricalData(
            df=df,
            source="polaris2025",
            time_range=(-100, 200),
            variables=["N", "psi"],
        )
        summary = data.summary()
        assert "polaris2025" in summary
        assert "-100" in summary
        assert "200" in summary


class TestGenerateRomanEstimates:
    """Tests for synthetic data generation."""

    def test_returns_dataframe(self):
        """Should return DataFrame with correct columns."""
        df = _generate_roman_estimates((-500, 500))
        assert isinstance(df, pd.DataFrame)
        assert "year" in df.columns
        assert "N" in df.columns
        assert "E" in df.columns
        assert "W" in df.columns
        assert "S" in df.columns
        assert "psi" in df.columns

    def test_time_range_respected(self):
        """Should generate data within time range."""
        df = _generate_roman_estimates((-200, 300), step=50)
        assert df["year"].min() >= -200
        assert df["year"].max() <= 300

    def test_values_normalized(self):
        """Values should be in reasonable normalized range."""
        df = _generate_roman_estimates((-500, 500))
        for var in ["N", "E", "W", "S"]:
            assert df[var].min() >= 0.0
            assert df[var].max() <= 1.5  # Allow some headroom
        assert df["psi"].min() >= 0.0
        assert df["psi"].max() <= 1.0

    def test_crisis_third_century_elevated(self):
        """PSI should be elevated during Crisis of Third Century."""
        df = _generate_roman_estimates((-500, 500), step=10)

        # Get PSI values for different periods
        pre_crisis = df[(df["year"] >= 100) & (df["year"] < 235)]["psi"].mean()
        crisis = df[(df["year"] >= 235) & (df["year"] <= 284)]["psi"].mean()

        # Crisis period should have higher PSI
        assert crisis > pre_crisis, (
            f"Crisis PSI ({crisis}) should exceed pre-crisis ({pre_crisis})"
        )

    def test_population_peak_around_150ce(self):
        """Population should peak around Principate era."""
        df = _generate_roman_estimates((-500, 500), step=10)

        # Find peak
        peak_idx = df["N"].idxmax()
        peak_year = df.loc[peak_idx, "year"]

        # Peak should be roughly 100-200 CE (Antonine period)
        assert 50 <= peak_year <= 250, (
            f"Population peak at {peak_year}, expected 50-250 CE"
        )


class TestLoadData:
    """Tests for load_data function."""

    def test_load_without_db(self):
        """Should load synthetic data when no database provided."""
        data = rome.load_data(db=None, dataset="polaris2025")
        assert isinstance(data, RomanHistoricalData)
        assert not data.df.empty
        assert "N" in data.df.columns

    def test_load_with_time_range(self):
        """Should respect time range parameter."""
        data = rome.load_data(db=None, time_range=(-100, 200))
        assert data.time_range == (-100, 200)

    def test_load_sets_source(self):
        """Should set correct data source."""
        data = rome.load_data(db=None, dataset="equinox2020")
        assert data.source == "equinox2020"


class TestCalibrate:
    """Tests for calibrate function."""

    def test_calibrate_returns_params_and_result(self):
        """Calibration should return SDTParams and CalibrationResult."""
        data = rome.load_data(db=None, time_range=(-100, 100))

        # Use small bounds and few iterations for fast test
        params, result = rome.calibrate(
            data,
            param_bounds={"r_max": (0.01, 0.02)},
            method="minimize",
            maxiter=10,
        )

        assert isinstance(params, SDTParams)
        assert hasattr(result, "loss")
        assert hasattr(result, "best_params")

    def test_calibrated_params_within_bounds(self):
        """Calibrated parameters should be within specified bounds."""
        data = rome.load_data(db=None, time_range=(-100, 100))

        bounds = {
            "r_max": (0.01, 0.02),
            "alpha": (0.002, 0.008),
        }

        params, result = rome.calibrate(
            data,
            param_bounds=bounds,
            method="minimize",
            maxiter=10,
        )

        for name, (low, high) in bounds.items():
            value = result.best_params[name]
            assert low <= value <= high, f"{name}={value} not in [{low}, {high}]"

    def test_calibrate_with_different_fit_variables(self):
        """Should calibrate to specified fit variables."""
        data = rome.load_data(db=None, time_range=(-100, 100))

        # Calibrate to different variables
        params, result = rome.calibrate(
            data,
            param_bounds={"r_max": (0.01, 0.02)},
            fit_variables=["N"],
            method="minimize",
            maxiter=10,
        )

        assert result.loss >= 0


class TestSimulate:
    """Tests for simulate function."""

    def test_simulate_returns_result(self):
        """Simulation should return SimulationResult."""
        params = SDTParams()
        result = rome.simulate(params, time_span=(-100, 100))

        assert isinstance(result, SimulationResult)
        assert not result.df.empty

    def test_simulate_respects_time_span(self):
        """Simulation should cover specified time span."""
        params = SDTParams()
        result = rome.simulate(params, time_span=(-50, 150))

        assert result.df["t"].min() == -50
        assert result.df["t"].max() == 150

    def test_simulate_with_initial_conditions(self):
        """Simulation should use specified initial conditions."""
        params = SDTParams()
        initial = {"N": 0.8, "E": 0.1, "W": 0.9, "S": 0.8, "psi": 0.1}

        result = rome.simulate(
            params,
            time_span=(0, 10),
            initial_conditions=initial,
        )

        # Check first row matches initial conditions
        first_row = result.df.iloc[0]
        for var, value in initial.items():
            assert abs(first_row[var] - value) < 0.01, f"{var} initial mismatch"

    def test_simulate_detects_events(self):
        """Simulation should detect defined events."""
        # Use params that might trigger events
        params = SDTParams(lambda_psi=0.2, theta_w=2.0, theta_e=2.0)

        result = rome.simulate(
            params,
            time_span=(0, 200),
            initial_conditions={"N": 0.9, "E": 0.2, "W": 0.6, "S": 0.5, "psi": 0.3},
        )

        # Events list should exist (may be empty if no thresholds crossed)
        assert hasattr(result, "events")


class TestDetectSecularCycles:
    """Tests for _detect_secular_cycles function."""

    def test_detect_with_peaks(self):
        """Should detect cycles when PSI has peaks."""
        # Create mock result with clear peaks
        t = np.arange(-100, 300, 1)
        psi = np.zeros_like(t, dtype=float)

        # Add peaks at 0 CE and 200 CE
        for peak_year, width in [(0, 30), (200, 25)]:
            idx = np.where(t == peak_year)[0][0]
            for i in range(-width, width + 1):
                if 0 <= idx + i < len(psi):
                    psi[idx + i] += 0.5 * np.exp(-abs(i) / 10)

        df = pd.DataFrame(
            {
                "t": t,
                "N": np.ones_like(t),
                "E": np.ones_like(t) * 0.1,
                "W": np.ones_like(t),
                "S": np.ones_like(t),
                "psi": psi,
            }
        )
        result = SimulationResult(df=df)

        cycles = _detect_secular_cycles(result, psi_threshold=0.2)

        assert len(cycles) >= 1, "Should detect at least one cycle"

    def test_detect_returns_secular_cycles(self):
        """Should return list of SecularCycle objects."""
        params = SDTParams()
        result = rome.simulate(params, time_span=(-100, 300))

        cycles = _detect_secular_cycles(result, psi_threshold=0.1)

        assert isinstance(cycles, list)
        for cycle in cycles:
            assert isinstance(cycle, SecularCycle)
            assert cycle.start_year <= cycle.end_year


class TestGenerateReport:
    """Tests for generate_report function."""

    def test_generate_creates_file(self):
        """Should create HTML report file."""
        data = rome.load_data(db=None, time_range=(-100, 100))
        params = SDTParams()
        result = rome.simulate(params, time_span=(-100, 100))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"

            path = rome.generate_report(result, data, str(output_path))

            assert path.exists()
            assert path.suffix == ".html"

    def test_report_contains_key_sections(self):
        """Report should contain key analysis sections."""
        data = rome.load_data(db=None, time_range=(-100, 100))
        params = SDTParams()
        result = rome.simulate(params, time_span=(-100, 100))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"

            rome.generate_report(result, data, str(output_path), params=params)

            content = output_path.read_text()

            assert "Roman Empire" in content
            assert "Parameters" in content
            assert "Historical Events" in content

    def test_report_with_cycles(self):
        """Report should include detected cycles."""
        data = rome.load_data(db=None, time_range=(-100, 100))
        params = SDTParams()
        result = rome.simulate(params, time_span=(-100, 100))

        cycles = [
            SecularCycle(
                name="Test Cycle",
                start_year=-100,
                end_year=100,
                crisis_peak=50,
                crisis_psi=0.5,
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"

            rome.generate_report(result, data, str(output_path), cycles=cycles)

            content = output_path.read_text()
            assert "Test Cycle" in content


class TestRomanEmpireStudy:
    """Tests for RomanEmpireStudy class."""

    def test_create_study(self):
        """Should create study with defaults."""
        study = RomanEmpireStudy()
        assert study.time_range == (-500, 500)
        assert study.data_source == "polaris2025"

    def test_load_data(self):
        """Should load data into study."""
        study = RomanEmpireStudy(time_range=(-100, 100))
        data = study.load_data()

        assert study.data is not None
        assert isinstance(data, RomanHistoricalData)

    def test_calibrate_requires_data(self):
        """Calibration should fail without data."""
        study = RomanEmpireStudy()

        with pytest.raises(ValueError, match="Must load data"):
            study.calibrate()

    def test_simulate_requires_params(self):
        """Simulation should fail without parameters."""
        study = RomanEmpireStudy()

        with pytest.raises(ValueError, match="Must calibrate or provide"):
            study.simulate()

    def test_full_workflow(self):
        """Should complete full analysis workflow."""
        study = RomanEmpireStudy(time_range=(-100, 100))

        # Load data
        study.load_data()
        assert study.data is not None

        # Calibrate with minimal iterations
        study.calibrate(
            param_bounds={"r_max": (0.01, 0.02)},
            method="minimize",
            maxiter=5,
        )
        assert study.params is not None

        # Simulate
        study.simulate()
        assert study.result is not None

        # Detect cycles
        cycles = study.detect_cycles()
        assert isinstance(cycles, list)

    def test_generate_report_requires_simulation(self):
        """Report generation should fail without simulation."""
        study = RomanEmpireStudy()
        study.load_data()

        with pytest.raises(ValueError, match="Must simulate"):
            study.generate_report("output.html")


class TestHistoricalEvents:
    """Tests for historical event constants."""

    def test_events_defined(self):
        """Should have key historical events defined."""
        assert "crisis_begins" in rome.HISTORICAL_EVENTS
        assert "crisis_ends" in rome.HISTORICAL_EVENTS
        assert "fall_west" in rome.HISTORICAL_EVENTS

    def test_events_chronological(self):
        """Events should be in roughly chronological order."""
        crisis_begins = rome.HISTORICAL_EVENTS["crisis_begins"]
        crisis_ends = rome.HISTORICAL_EVENTS["crisis_ends"]
        fall_west = rome.HISTORICAL_EVENTS["fall_west"]
        assert crisis_begins < crisis_ends
        assert crisis_ends < fall_west

    def test_crisis_third_century_dates(self):
        """Crisis of Third Century dates should be accurate."""
        assert rome.HISTORICAL_EVENTS["crisis_begins"] == 235
        assert rome.HISTORICAL_EVENTS["crisis_ends"] == 284


class TestConstants:
    """Tests for module constants."""

    def test_polity_ids_defined(self):
        """Should have Roman polity IDs defined."""
        assert len(rome.ROMAN_POLITY_IDS) > 0
        assert any("Rom" in pid for pid in rome.ROMAN_POLITY_IDS)

    def test_default_param_bounds(self):
        """Should have default parameter bounds defined."""
        assert "r_max" in rome.DEFAULT_PARAM_BOUNDS
        assert "alpha" in rome.DEFAULT_PARAM_BOUNDS

        # Bounds should be valid (min < max)
        for name, (low, high) in rome.DEFAULT_PARAM_BOUNDS.items():
            assert low < high, f"{name} bounds invalid: {low} >= {high}"
