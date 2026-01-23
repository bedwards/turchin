"""Tests for U.S. historical data loader.

Tests cover:
- Data loading functions
- USHistoricalData class
- Data range and interpolation
- Data quality and consistency
"""

import numpy as np
import pandas as pd

from cliodynamics.data.us import (
    USHistoricalData,
    get_elite_indicators,
    get_instability_indicators,
    get_real_wages,
    get_relative_wages,
    get_wealth_inequality,
)


class TestGetRealWages:
    """Tests for get_real_wages function."""

    def test_returns_dataframe(self):
        """Should return a DataFrame."""
        df = get_real_wages()
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        """Should have year and real_wage_index columns."""
        df = get_real_wages()
        assert "year" in df.columns
        assert "real_wage_index" in df.columns

    def test_default_year_range(self):
        """Default range should be 1780-2025."""
        df = get_real_wages()
        assert df["year"].min() == 1780
        assert df["year"].max() == 2025

    def test_custom_year_range(self):
        """Should respect custom year range."""
        df = get_real_wages(start_year=1900, end_year=2000)
        assert df["year"].min() == 1900
        assert df["year"].max() == 2000

    def test_annual_data(self):
        """Should have annual data (one row per year)."""
        df = get_real_wages(start_year=1900, end_year=1950)
        assert len(df) == 51  # 1900-1950 inclusive

    def test_1960_baseline(self):
        """1960 should be the baseline (index = 100)."""
        df = get_real_wages()
        val_1960 = df[df["year"] == 1960]["real_wage_index"].values[0]
        assert val_1960 == 100

    def test_wage_decline_after_1973(self):
        """Real wages should show decline after 1973 peak."""
        df = get_real_wages()
        val_1973 = df[df["year"] == 1973]["real_wage_index"].values[0]
        val_2020 = df[df["year"] == 2020]["real_wage_index"].values[0]
        assert val_2020 < val_1973

    def test_no_negative_values(self):
        """All values should be positive."""
        df = get_real_wages()
        assert (df["real_wage_index"] > 0).all()


class TestGetRelativeWages:
    """Tests for get_relative_wages function."""

    def test_returns_dataframe(self):
        """Should return a DataFrame."""
        df = get_relative_wages()
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        """Should have year and relative_wage_index columns."""
        df = get_relative_wages()
        assert "year" in df.columns
        assert "relative_wage_index" in df.columns

    def test_1960_baseline(self):
        """1960 should be the baseline (index = 100)."""
        df = get_relative_wages()
        val_1960 = df[df["year"] == 1960]["relative_wage_index"].values[0]
        assert val_1960 == 100

    def test_more_dramatic_decline(self):
        """Relative wages should decline more than real wages post-1970."""
        real = get_real_wages()
        relative = get_relative_wages()

        real_2020 = real[real["year"] == 2020]["real_wage_index"].values[0]
        rel_col = "relative_wage_index"
        relative_2020 = relative[relative["year"] == 2020][rel_col].values[0]

        # Relative wages should have fallen more
        assert relative_2020 < real_2020


class TestGetEliteIndicators:
    """Tests for get_elite_indicators function."""

    def test_returns_dataframe(self):
        """Should return a DataFrame."""
        df = get_elite_indicators()
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        """Should have expected columns."""
        df = get_elite_indicators()
        assert "year" in df.columns
        assert "lawyers_per_capita_index" in df.columns
        assert "combined_elite_index" in df.columns

    def test_elite_growth_post_1960(self):
        """Elite indicators should rise dramatically after 1960."""
        df = get_elite_indicators()
        val_1960 = df[df["year"] == 1960]["lawyers_per_capita_index"].values[0]
        val_2020 = df[df["year"] == 2020]["lawyers_per_capita_index"].values[0]
        assert val_2020 > val_1960 * 1.5  # At least 50% increase

    def test_first_gilded_age_peak(self):
        """Should show elevated values around 1900 (First Gilded Age)."""
        df = get_elite_indicators()
        val_1900 = df[df["year"] == 1900]["lawyers_per_capita_index"].values[0]
        val_1850 = df[df["year"] == 1850]["lawyers_per_capita_index"].values[0]
        assert val_1900 > val_1850


class TestGetWealthInequality:
    """Tests for get_wealth_inequality function."""

    def test_returns_dataframe(self):
        """Should return a DataFrame."""
        df = get_wealth_inequality()
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        """Should have expected columns."""
        df = get_wealth_inequality()
        assert "year" in df.columns
        assert "top_1pct_share" in df.columns
        assert "inequality_index" in df.columns

    def test_inequality_u_shape(self):
        """Inequality should show U-shape: high early, low mid-century, high now."""
        df = get_wealth_inequality()
        val_1900 = df[df["year"] == 1900]["top_1pct_share"].values[0]
        val_1960 = df[df["year"] == 1960]["top_1pct_share"].values[0]
        val_2020 = df[df["year"] == 2020]["top_1pct_share"].values[0]

        # Mid-century should be lowest
        assert val_1960 < val_1900
        assert val_1960 < val_2020


class TestGetInstabilityIndicators:
    """Tests for get_instability_indicators function."""

    def test_returns_dataframe(self):
        """Should return a DataFrame."""
        df = get_instability_indicators()
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        """Should have expected columns."""
        df = get_instability_indicators()
        assert "year" in df.columns
        assert "violence_index" in df.columns

    def test_civil_war_peak(self):
        """Violence index should peak around Civil War."""
        df = get_instability_indicators()
        val_1865 = df[df["year"] == 1865]["violence_index"].values[0]
        val_1850 = df[df["year"] == 1850]["violence_index"].values[0]
        assert val_1865 > val_1850

    def test_1960s_unrest(self):
        """Should show elevated violence in late 1960s."""
        df = get_instability_indicators()
        val_1968 = df[df["year"] == 1968]["violence_index"].values[0]
        val_1950 = df[df["year"] == 1950]["violence_index"].values[0]
        assert val_1968 > val_1950

    def test_2020_elevated(self):
        """2020 should show elevated instability."""
        df = get_instability_indicators()
        val_2020 = df[df["year"] == 2020]["violence_index"].values[0]
        val_2000 = df[df["year"] == 2000]["violence_index"].values[0]
        assert val_2020 > val_2000


class TestUSHistoricalData:
    """Tests for USHistoricalData class."""

    def test_initialization(self):
        """Should initialize with default parameters."""
        data = USHistoricalData()
        assert data.start_year == 1780
        assert data.end_year == 2025

    def test_custom_year_range(self):
        """Should accept custom year range."""
        data = USHistoricalData(start_year=1900, end_year=2000)
        assert data.start_year == 1900
        assert data.end_year == 2000

    def test_real_wages_property(self):
        """real_wages property should return DataFrame."""
        data = USHistoricalData()
        wages = data.real_wages
        assert isinstance(wages, pd.DataFrame)
        assert "real_wage_index" in wages.columns

    def test_relative_wages_property(self):
        """relative_wages property should return DataFrame."""
        data = USHistoricalData()
        wages = data.relative_wages
        assert isinstance(wages, pd.DataFrame)
        assert "relative_wage_index" in wages.columns

    def test_elite_indicators_property(self):
        """elite_indicators property should return DataFrame."""
        data = USHistoricalData()
        elites = data.elite_indicators
        assert isinstance(elites, pd.DataFrame)
        assert "lawyers_per_capita_index" in elites.columns

    def test_wealth_inequality_property(self):
        """wealth_inequality property should return DataFrame."""
        data = USHistoricalData()
        inequality = data.wealth_inequality
        assert isinstance(inequality, pd.DataFrame)
        assert "top_1pct_share" in inequality.columns

    def test_instability_indicators_property(self):
        """instability_indicators property should return DataFrame."""
        data = USHistoricalData()
        instability = data.instability_indicators
        assert isinstance(instability, pd.DataFrame)
        assert "violence_index" in instability.columns

    def test_get_combined_dataset(self):
        """get_combined_dataset should merge all data."""
        data = USHistoricalData()
        combined = data.get_combined_dataset()

        assert isinstance(combined, pd.DataFrame)
        assert "year" in combined.columns
        assert "real_wage_index" in combined.columns
        assert "lawyers_per_capita_index" in combined.columns
        assert "violence_index" in combined.columns

    def test_combined_dataset_consistent_years(self):
        """Combined dataset should have consistent year coverage."""
        data = USHistoricalData(start_year=1900, end_year=2000)
        combined = data.get_combined_dataset()

        # Should have all years in range
        assert combined["year"].min() == 1900
        assert combined["year"].max() == 2000

    def test_get_year_range(self):
        """get_year_range should return tuple."""
        data = USHistoricalData(start_year=1850, end_year=1950)
        start, end = data.get_year_range()
        assert start == 1850
        assert end == 1950

    def test_caching(self):
        """Data should be cached after first access."""
        data = USHistoricalData()

        # First access
        wages1 = data.real_wages
        # Second access should return same object
        wages2 = data.real_wages
        assert wages1 is wages2

    def test_clear_cache(self):
        """clear_cache should force reload."""
        data = USHistoricalData()

        # First access
        wages1 = data.real_wages
        # Clear cache
        data.clear_cache()
        # Second access should be new object
        wages2 = data.real_wages
        assert wages1 is not wages2


class TestDataConsistency:
    """Tests for data consistency across functions."""

    def test_all_functions_same_year_range(self):
        """All functions should respect same year range."""
        start, end = 1900, 2000

        real = get_real_wages(start, end)
        relative = get_relative_wages(start, end)
        elites = get_elite_indicators(start, end)
        inequality = get_wealth_inequality(start, end)
        instability = get_instability_indicators(start, end)

        # All should have same year range
        for df in [real, relative, elites, inequality, instability]:
            assert df["year"].min() == start
            assert df["year"].max() == end

    def test_interpolation_smoothness(self):
        """Interpolated data should be smooth (no large jumps)."""
        df = get_real_wages()
        values = df["real_wage_index"].values

        # Calculate year-over-year changes
        changes = np.abs(np.diff(values))

        # No change should be more than 10% of the value
        max_change = np.max(changes)
        mean_value = np.mean(values)
        assert max_change < 0.1 * mean_value
