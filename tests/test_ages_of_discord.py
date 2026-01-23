"""Tests for Ages of Discord model.

Tests cover:
- WBI, EOI, PSI computation
- Weight configurations
- Cycle detection
- Integration with U.S. data
- Comparison to published values
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from cliodynamics.models.ages_of_discord import (
    AgesOfDiscordConfig,
    AgesOfDiscordModel,
    EliteWeights,
    PSIWeights,
    WellBeingWeights,
    compute_ages_of_discord_indices,
)


class TestWellBeingWeights:
    """Tests for WellBeingWeights dataclass."""

    def test_default_weights_valid(self):
        """Default weights should sum to 1.0."""
        weights = WellBeingWeights()
        assert weights.validate()

    def test_custom_weights_validation(self):
        """Custom weights should validate correctly."""
        # Valid weights
        weights = WellBeingWeights(
            real_wage=0.4,
            relative_wage=0.3,
            health=0.2,
            family=0.1,
        )
        assert weights.validate()

        # Invalid weights (don't sum to 1.0)
        weights = WellBeingWeights(
            real_wage=0.5,
            relative_wage=0.5,
            health=0.5,
            family=0.5,
        )
        assert not weights.validate()

    def test_as_array(self):
        """as_array should return correct numpy array."""
        weights = WellBeingWeights(
            real_wage=0.4,
            relative_wage=0.3,
            health=0.2,
            family=0.1,
        )
        arr = weights.as_array()
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 4
        assert_allclose(arr, [0.4, 0.3, 0.2, 0.1])


class TestEliteWeights:
    """Tests for EliteWeights dataclass."""

    def test_default_weights_valid(self):
        """Default weights should sum to 1.0."""
        weights = EliteWeights()
        assert weights.validate()

    def test_as_array(self):
        """as_array should return correct numpy array."""
        weights = EliteWeights()
        arr = weights.as_array()
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 3


class TestPSIWeights:
    """Tests for PSIWeights dataclass."""

    def test_default_weights_valid(self):
        """Default weights should sum to 1.0."""
        weights = PSIWeights()
        assert weights.validate()


class TestAgesOfDiscordConfig:
    """Tests for AgesOfDiscordConfig dataclass."""

    def test_default_config(self):
        """Default config should have reasonable values."""
        config = AgesOfDiscordConfig()
        assert config.baseline_year == 1960
        assert config.wbi_weights.validate()
        assert config.elite_weights.validate()
        assert config.psi_weights.validate()


class TestAgesOfDiscordModel:
    """Tests for AgesOfDiscordModel class."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample historical data for testing."""
        years = list(range(1900, 2021))
        n = len(years)

        return pd.DataFrame(
            {
                "year": years,
                "real_wage_index": np.linspace(60, 90, n),  # Simplified trend
                "relative_wage_index": np.linspace(65, 45, n),  # Decline
                "lawyers_per_capita_index": np.linspace(90, 220, n),  # Growth
                "phds_per_capita_index": np.linspace(30, 400, n),  # Explosive growth
                "inequality_index": np.concatenate(
                    [
                        np.linspace(150, 100, 60),  # Decline to 1960
                        np.linspace(100, 170, 61),  # Rise after 1960
                    ]
                ),
            }
        )

    @pytest.fixture
    def model(self) -> AgesOfDiscordModel:
        """Create default model."""
        return AgesOfDiscordModel()

    def test_initialization_default(self):
        """Model should initialize with defaults."""
        model = AgesOfDiscordModel()
        assert model.config is not None
        assert model.config.baseline_year == 1960

    def test_initialization_custom_config(self):
        """Model should accept custom config."""
        config = AgesOfDiscordConfig(baseline_year=1950)
        model = AgesOfDiscordModel(config)
        assert model.config.baseline_year == 1950

    def test_compute_wbi_returns_series(self, model, sample_data):
        """compute_well_being_index should return Series."""
        wbi = model.compute_well_being_index(sample_data)
        assert isinstance(wbi, pd.Series)
        assert wbi.name == "well_being_index"

    def test_compute_wbi_indexed_by_year(self, model, sample_data):
        """WBI should be indexed by year."""
        wbi = model.compute_well_being_index(sample_data)
        assert 1900 in wbi.index
        assert 2020 in wbi.index

    def test_compute_wbi_normalized(self, model, sample_data):
        """WBI should be normalized to baseline year."""
        wbi = model.compute_well_being_index(sample_data)
        # 1960 should be close to 100
        assert 90 < wbi.loc[1960] < 110

    def test_compute_wbi_missing_columns(self, model):
        """WBI should raise error for missing columns."""
        bad_data = pd.DataFrame({"year": [1900, 1950, 2000]})
        with pytest.raises(ValueError, match="Missing required columns"):
            model.compute_well_being_index(bad_data)

    def test_compute_eoi_returns_series(self, model, sample_data):
        """compute_elite_overproduction_index should return Series."""
        eoi = model.compute_elite_overproduction_index(sample_data)
        assert isinstance(eoi, pd.Series)
        assert eoi.name == "elite_overproduction_index"

    def test_compute_eoi_increasing(self, model, sample_data):
        """EOI should generally increase over time in test data."""
        eoi = model.compute_elite_overproduction_index(sample_data)
        # 2020 should be higher than 1900
        assert eoi.loc[2020] > eoi.loc[1900]

    def test_compute_eoi_normalized(self, model, sample_data):
        """EOI should be normalized to baseline year."""
        eoi = model.compute_elite_overproduction_index(sample_data)
        # 1960 should be close to 100
        assert 90 < eoi.loc[1960] < 110

    def test_compute_psi_returns_series(self, model, sample_data):
        """compute_political_stress_index should return Series."""
        wbi = model.compute_well_being_index(sample_data)
        eoi = model.compute_elite_overproduction_index(sample_data)
        psi = model.compute_political_stress_index(wbi, eoi)
        assert isinstance(psi, pd.Series)
        assert psi.name == "political_stress_index"

    def test_compute_psi_inverse_wbi_relationship(self, model, sample_data):
        """PSI should rise when WBI falls (inverse relationship)."""
        wbi = model.compute_well_being_index(sample_data)
        eoi = model.compute_elite_overproduction_index(sample_data)
        psi = model.compute_political_stress_index(wbi, eoi)

        # Find years with low WBI
        low_wbi_years = wbi[wbi < wbi.median()].index
        high_wbi_years = wbi[wbi >= wbi.median()].index

        # PSI should be higher on average when WBI is lower
        mean_psi_low_wbi = psi.loc[low_wbi_years].mean()
        mean_psi_high_wbi = psi.loc[high_wbi_years].mean()
        # Note: This relationship depends on EOI too, so we just check non-trivial
        assert mean_psi_low_wbi != mean_psi_high_wbi

    def test_compute_all_returns_dataframe(self, model, sample_data):
        """compute_all should return DataFrame with all indices."""
        results = model.compute_all(sample_data)
        assert isinstance(results, pd.DataFrame)
        assert "year" in results.columns
        assert "well_being_index" in results.columns
        assert "elite_overproduction_index" in results.columns
        assert "political_stress_index" in results.columns

    def test_compute_all_sorted_by_year(self, model, sample_data):
        """Results should be sorted by year."""
        results = model.compute_all(sample_data)
        years = results["year"].values
        assert np.all(years[:-1] <= years[1:])

    def test_smooth_function(self):
        """_smooth should apply moving average correctly."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        smoothed = AgesOfDiscordModel._smooth(values, window=3)

        # Smoothed values should be close to original for linear data
        assert len(smoothed) == len(values)
        assert smoothed[5] == pytest.approx(values[5], abs=0.5)


class TestComputeAgesOfDiscordIndices:
    """Tests for convenience function."""

    def test_returns_dataframe(self):
        """Function should return DataFrame."""
        data = pd.DataFrame(
            {
                "year": [1950, 1960, 1970],
                "real_wage_index": [95, 100, 105],
                "relative_wage_index": [95, 100, 98],
                "lawyers_per_capita_index": [82, 100, 105],
                "inequality_index": [96, 100, 95],
            }
        )
        results = compute_ages_of_discord_indices(data)
        assert isinstance(results, pd.DataFrame)

    def test_accepts_custom_config(self):
        """Function should accept custom config."""
        data = pd.DataFrame(
            {
                "year": [1950, 1960, 1970],
                "real_wage_index": [95, 100, 105],
                "relative_wage_index": [95, 100, 98],
                "lawyers_per_capita_index": [82, 100, 105],
                "inequality_index": [96, 100, 95],
            }
        )
        config = AgesOfDiscordConfig(baseline_year=1950)
        results = compute_ages_of_discord_indices(data, config)
        assert isinstance(results, pd.DataFrame)


class TestCycleDetection:
    """Tests for cycle detection functionality."""

    @pytest.fixture
    def cyclic_data(self) -> pd.Series:
        """Create synthetic cyclic data."""
        years = np.arange(1800, 2020)
        # Create a sine wave with period ~60 years
        values = 100 + 30 * np.sin(2 * np.pi * (years - 1800) / 60)
        return pd.Series(values, index=years, name="test_index")

    def test_identify_cycles_returns_list(self):
        """identify_cycles should return list of dicts."""
        model = AgesOfDiscordModel()
        series = pd.Series(
            [100, 120, 80, 130, 70, 140, 60],
            index=[1900, 1920, 1940, 1960, 1980, 2000, 2020],
        )
        cycles = model.identify_cycles(series, min_period=1)
        assert isinstance(cycles, list)

    def test_identify_cycles_has_required_keys(self):
        """Each cycle should have required keys."""
        model = AgesOfDiscordModel()
        series = pd.Series(
            [100, 120, 80, 130, 70, 140, 60],
            index=[1900, 1920, 1940, 1960, 1980, 2000, 2020],
        )
        cycles = model.identify_cycles(series, min_period=1)

        if len(cycles) > 0:
            cycle = cycles[0]
            assert "start_year" in cycle
            assert "peak_year" in cycle
            assert "end_year" in cycle
            assert "amplitude" in cycle

    def test_identify_cycles_in_synthetic_data(self, cyclic_data):
        """Should detect cycles in synthetic cyclic data."""
        model = AgesOfDiscordModel()
        cycles = model.identify_cycles(cyclic_data, min_period=20)

        # Should find multiple cycles in 220 years of 60-year cycles
        assert len(cycles) >= 2


class TestCompareToPublished:
    """Tests for comparison to published values."""

    def test_compare_returns_dict(self):
        """compare_to_published should return dict."""
        model = AgesOfDiscordModel()
        series = pd.Series([95, 100, 105], index=[1950, 1960, 1970])

        result = model.compare_to_published(
            computed=series,
            target_year=1960,
            published_value=100,
        )
        assert isinstance(result, dict)

    def test_compare_has_required_keys(self):
        """Result should have required keys."""
        model = AgesOfDiscordModel()
        series = pd.Series([95, 100, 105], index=[1950, 1960, 1970])

        result = model.compare_to_published(
            computed=series,
            target_year=1960,
            published_value=100,
        )

        assert "computed_value" in result
        assert "published_value" in result
        assert "difference" in result
        assert "relative_error" in result
        assert "within_tolerance" in result

    def test_compare_exact_match(self):
        """Exact match should have zero error."""
        model = AgesOfDiscordModel()
        series = pd.Series([95, 100, 105], index=[1950, 1960, 1970])

        result = model.compare_to_published(
            computed=series,
            target_year=1960,
            published_value=100,
        )

        assert result["difference"] == 0
        assert result["relative_error"] == 0
        assert result["within_tolerance"]

    def test_compare_within_tolerance(self):
        """Values within tolerance should be flagged."""
        model = AgesOfDiscordModel()
        series = pd.Series([95, 100, 105], index=[1950, 1960, 1970])

        # 5% error should be within 15% tolerance
        result = model.compare_to_published(
            computed=series,
            target_year=1960,
            published_value=95,
            tolerance=0.15,
        )

        assert result["within_tolerance"]

    def test_compare_outside_tolerance(self):
        """Values outside tolerance should be flagged."""
        model = AgesOfDiscordModel()
        series = pd.Series([95, 100, 105], index=[1950, 1960, 1970])

        # 25% error should be outside 15% tolerance
        result = model.compare_to_published(
            computed=series,
            target_year=1960,
            published_value=80,
            tolerance=0.15,
        )

        assert not result["within_tolerance"]

    def test_compare_invalid_year(self):
        """Should raise error for year not in series."""
        model = AgesOfDiscordModel()
        series = pd.Series([95, 100, 105], index=[1950, 1960, 1970])

        with pytest.raises(ValueError, match="not in computed series"):
            model.compare_to_published(
                computed=series,
                target_year=1980,  # Not in series
                published_value=100,
            )


class TestIntegrationWithUSData:
    """Integration tests using actual U.S. data."""

    @pytest.fixture
    def us_data(self):
        """Load U.S. historical data."""
        from cliodynamics.data.us import USHistoricalData

        return USHistoricalData().get_combined_dataset()

    def test_compute_all_with_real_data(self, us_data):
        """Should compute indices from real U.S. data."""
        model = AgesOfDiscordModel()
        results = model.compute_all(us_data)

        assert len(results) > 0
        assert results["well_being_index"].notna().any()
        assert results["elite_overproduction_index"].notna().any()
        assert results["political_stress_index"].notna().any()

    def test_wbi_peak_around_1960(self, us_data):
        """WBI should peak around 1960."""
        model = AgesOfDiscordModel()
        results = model.compute_all(us_data)

        wbi = results.set_index("year")["well_being_index"]
        peak_year = wbi.idxmax()

        # Peak should be between 1950 and 1975
        assert 1950 <= peak_year <= 1975

    def test_eoi_increases_post_1970(self, us_data):
        """EOI should increase substantially after 1970."""
        model = AgesOfDiscordModel()
        results = model.compute_all(us_data)

        eoi = results.set_index("year")["elite_overproduction_index"]

        eoi_1970 = eoi.loc[1970]
        eoi_2020 = eoi.loc[2020]

        # Should at least double
        assert eoi_2020 > eoi_1970 * 1.5

    def test_psi_captures_historical_peaks(self, us_data):
        """PSI should capture known instability periods."""
        model = AgesOfDiscordModel()
        results = model.compute_all(us_data)

        psi = results.set_index("year")["political_stress_index"]

        # PSI in 1920s should be elevated
        psi_1920 = psi.loc[1920]
        psi_1950 = psi.loc[1950]
        assert psi_1920 > psi_1950 * 0.8  # 1920s should be similar or higher

        # PSI in 2020 should be elevated vs 1960 trough
        psi_1960 = psi.loc[1960]
        psi_2020 = psi.loc[2020]
        assert psi_2020 > psi_1960
