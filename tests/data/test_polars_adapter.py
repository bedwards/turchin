"""
Unit tests for the cliodynamics.data.polars_adapter module.

Tests cover:
- Conversion utilities (pandas <-> polars)
- CSV/Excel reading functions
- PolarsAdapter data loading
- LazyPolarsAdapter lazy evaluation
- Edge cases and error handling
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

from cliodynamics.data.polars_adapter import (
    LazyPolarsAdapter,
    PolarsAdapter,
    PolarsAdapterConfig,
    get_adapter,
    pandas_to_polars,
    polars_to_pandas,
    read_csv_polars,
    read_excel_polars,
)


class TestPandasToPolars:
    """Tests for pandas_to_polars conversion function."""

    def test_simple_conversion(self):
        """Test basic DataFrame conversion."""
        df_pd = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "c": [1.0, 2.0, 3.0],
        })

        df_pl = pandas_to_polars(df_pd)

        assert isinstance(df_pl, pl.DataFrame)
        assert df_pl.shape == (3, 3)
        assert df_pl.columns == ["a", "b", "c"]
        assert df_pl["a"].to_list() == [1, 2, 3]
        assert df_pl["b"].to_list() == ["x", "y", "z"]

    def test_handles_nan_values(self):
        """Test that NaN values are converted to null."""
        df_pd = pd.DataFrame({
            "a": [1.0, float("nan"), 3.0],
            "b": [1, 2, 3],
        })

        df_pl = pandas_to_polars(df_pd, nan_to_null=True)

        # NaN should become null
        assert df_pl["a"].null_count() == 1
        assert df_pl["a"][0] == 1.0
        assert df_pl["a"][1] is None
        assert df_pl["a"][2] == 3.0

    def test_preserves_nan_when_requested(self):
        """Test that NaN is preserved when nan_to_null=False."""
        df_pd = pd.DataFrame({
            "a": [1.0, float("nan"), 3.0],
        })

        df_pl = pandas_to_polars(df_pd, nan_to_null=False)

        # NaN should still be NaN (not null)
        assert df_pl["a"][1] != df_pl["a"][1]  # NaN != NaN

    def test_schema_overrides(self):
        """Test applying schema overrides during conversion."""
        df_pd = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        })

        df_pl = pandas_to_polars(df_pd, schema_overrides={"a": pl.Float64})

        assert df_pl["a"].dtype == pl.Float64
        assert df_pl["a"].to_list() == [1.0, 2.0, 3.0]

    def test_empty_dataframe(self):
        """Test converting empty DataFrame."""
        df_pd = pd.DataFrame({"a": [], "b": []})

        df_pl = pandas_to_polars(df_pd)

        assert df_pl.shape == (0, 2)
        assert df_pl.columns == ["a", "b"]

    def test_categorical_column(self):
        """Test converting categorical columns."""
        df_pd = pd.DataFrame({
            "cat": pd.Categorical(["a", "b", "a", "c"]),
        })

        df_pl = pandas_to_polars(df_pd)

        assert df_pl.shape == (4, 1)
        # Polars converts to Enum or Categorical type

    def test_datetime_column(self):
        """Test converting datetime columns."""
        df_pd = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=3, freq="D"),
        })

        df_pl = pandas_to_polars(df_pd)

        assert df_pl.shape == (3, 1)
        assert df_pl["date"].dtype in (pl.Datetime("us"), pl.Datetime("ns"))

    def test_mixed_types(self):
        """Test DataFrame with various column types."""
        df_pd = pd.DataFrame({
            "int": [1, 2, 3],
            "float": [1.1, 2.2, 3.3],
            "str": ["a", "b", "c"],
            "bool": [True, False, True],
        })

        df_pl = pandas_to_polars(df_pd)

        assert df_pl["int"].dtype == pl.Int64
        assert df_pl["float"].dtype == pl.Float64
        assert df_pl["str"].dtype == pl.String
        assert df_pl["bool"].dtype == pl.Boolean


class TestPolarsToPandas:
    """Tests for polars_to_pandas conversion function."""

    def test_simple_conversion(self):
        """Test basic DataFrame conversion."""
        df_pl = pl.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        })

        df_pd = polars_to_pandas(df_pl)

        assert isinstance(df_pd, pd.DataFrame)
        assert df_pd.shape == (3, 2)
        assert list(df_pd.columns) == ["a", "b"]
        assert df_pd["a"].tolist() == [1, 2, 3]
        assert df_pd["b"].tolist() == ["x", "y", "z"]

    def test_handles_null_values(self):
        """Test that null values are converted appropriately."""
        df_pl = pl.DataFrame({
            "a": [1.0, None, 3.0],
        })

        df_pd = polars_to_pandas(df_pl)

        assert pd.isna(df_pd["a"].iloc[1])

    def test_lazyframe_collected(self):
        """Test that LazyFrame is collected before conversion."""
        lf = pl.DataFrame({"a": [1, 2, 3]}).lazy()

        df_pd = polars_to_pandas(lf)

        assert isinstance(df_pd, pd.DataFrame)
        assert df_pd["a"].tolist() == [1, 2, 3]

    def test_empty_dataframe(self):
        """Test converting empty DataFrame."""
        df_pl = pl.DataFrame({"a": [], "b": []}).cast({"a": pl.Int64, "b": pl.String})

        df_pd = polars_to_pandas(df_pl)

        assert df_pd.shape == (0, 2)
        assert list(df_pd.columns) == ["a", "b"]


class TestReadCsvPolars:
    """Tests for read_csv_polars function."""

    def test_read_simple_csv(self):
        """Test reading a simple CSV file."""
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("a,b,c\n1,x,1.0\n2,y,2.0\n3,z,3.0\n")

            df = read_csv_polars(csv_path)

            assert df.shape == (3, 3)
            assert df["a"].to_list() == [1, 2, 3]
            assert df["b"].to_list() == ["x", "y", "z"]

    def test_read_with_columns(self):
        """Test reading specific columns."""
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("a,b,c\n1,x,1.0\n2,y,2.0\n")

            df = read_csv_polars(csv_path, columns=["a", "c"])

            assert df.columns == ["a", "c"]
            assert "b" not in df.columns

    def test_read_with_dtypes(self):
        """Test reading with explicit dtypes."""
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("a,b\n1,2\n3,4\n")

            df = read_csv_polars(csv_path, schema_overrides={"a": pl.Float64})

            assert df["a"].dtype == pl.Float64

    def test_read_with_null_values(self):
        """Test reading with custom null values."""
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("a,b\n1,NA\n2,3\nN/A,4\n")

            df = read_csv_polars(csv_path, null_values=["NA", "N/A"])

            # Null values should be detected
            assert df["b"].null_count() == 1

    def test_read_with_skip_rows(self):
        """Test skipping rows at start."""
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("# Comment\na,b\n1,2\n3,4\n")

            df = read_csv_polars(csv_path, skip_rows=1)

            assert df.columns == ["a", "b"]
            assert df.shape == (2, 2)

    def test_read_with_n_rows(self):
        """Test limiting number of rows."""
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("a,b\n1,2\n3,4\n5,6\n7,8\n")

            df = read_csv_polars(csv_path, n_rows=2)

            assert df.shape == (2, 2)
            assert df["a"].to_list() == [1, 3]

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            read_csv_polars("/nonexistent/path.csv")

    def test_encoding_fallback(self):
        """Test that encoding fallback works."""
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            # Write with latin-1 encoding
            content = "a,b\n1,caf\xe9\n2,na\xefve\n"
            csv_path.write_bytes(content.encode("latin-1"))

            # Should succeed with fallback to latin-1
            df = read_csv_polars(csv_path, try_encodings=["utf-8", "latin-1"])

            assert df.shape == (2, 2)


class TestReadExcelPolars:
    """Tests for read_excel_polars function."""

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            read_excel_polars("/nonexistent/path.xlsx")

    def test_read_simple_excel(self):
        """Test reading a simple Excel file."""
        with TemporaryDirectory() as tmpdir:
            excel_path = Path(tmpdir) / "test.xlsx"
            # Create a simple Excel file using pandas
            df_pd = pd.DataFrame({
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
            })
            df_pd.to_excel(excel_path, index=False)

            df = read_excel_polars(excel_path)

            assert df.shape == (3, 2)
            assert df["a"].to_list() == [1, 2, 3]


class TestPolarsAdapter:
    """Tests for PolarsAdapter class."""

    def test_default_config(self):
        """Test adapter with default configuration."""
        adapter = PolarsAdapter()

        assert adapter.config.prefer_polaris is True
        assert adapter.config.cache_enabled is True

    def test_custom_config(self):
        """Test adapter with custom configuration."""
        config = PolarsAdapterConfig(
            prefer_polaris=False,
            cache_enabled=False,
        )
        adapter = PolarsAdapter(config=config)

        assert adapter.config.prefer_polaris is False
        assert adapter.config.cache_enabled is False

    def test_clear_cache(self):
        """Test cache clearing."""
        adapter = PolarsAdapter()
        # Manually add something to cache
        adapter._cache["test"] = pl.DataFrame({"a": [1, 2, 3]})

        adapter.clear_cache()

        assert len(adapter._cache) == 0

    @patch.object(PolarsAdapter, "_load_polaris_data")
    def test_load_seshat_returns_polars(self, mock_load):
        """Test that load_seshat returns Polars DataFrame by default."""
        mock_load.return_value = pl.DataFrame({
            "Polity": ["RomPrin", "RomDom"],
            "NGA": ["Italy", "Italy"],
        })

        adapter = PolarsAdapter()
        # Mock the directories to exist
        with patch.object(Path, "exists", return_value=True):
            df = adapter.load_seshat(format="polars")

        assert isinstance(df, pl.DataFrame)

    @patch.object(PolarsAdapter, "_load_polaris_data")
    def test_load_seshat_returns_pandas(self, mock_load):
        """Test that load_seshat can return pandas DataFrame."""
        mock_load.return_value = pl.DataFrame({
            "Polity": ["RomPrin", "RomDom"],
            "NGA": ["Italy", "Italy"],
        })

        adapter = PolarsAdapter()
        with patch.object(Path, "exists", return_value=True):
            df = adapter.load_seshat(format="pandas")

        assert isinstance(df, pd.DataFrame)

    def test_load_seshat_no_data(self):
        """Test error when no data is available."""
        config = PolarsAdapterConfig(
            seshat_data_dir="/nonexistent/seshat",
            polaris_data_dir="/nonexistent/polaris",
        )
        adapter = PolarsAdapter(config=config)

        with pytest.raises(FileNotFoundError):
            adapter.load_seshat()

    def test_load_seshat_specific_dataset(self):
        """Test loading specific dataset."""
        adapter = PolarsAdapter()

        with pytest.raises(FileNotFoundError, match="Polaris data not found"):
            adapter.load_seshat(dataset="polaris")

        with pytest.raises(FileNotFoundError, match="Equinox data not found"):
            adapter.load_seshat(dataset="equinox")

    def test_load_seshat_invalid_dataset(self):
        """Test error for invalid dataset name."""
        adapter = PolarsAdapter()

        with pytest.raises(ValueError, match="Unknown dataset"):
            adapter.load_seshat(dataset="invalid")  # type: ignore

    def test_load_us_historical(self):
        """Test loading US historical data."""
        adapter = PolarsAdapter()

        df = adapter.load_us_historical(start_year=1900, end_year=2000)

        assert isinstance(df, pl.DataFrame)
        assert "year" in df.columns
        assert "real_wage_index" in df.columns
        # Check year range
        years = df["year"].to_list()
        assert min(years) == 1900
        assert max(years) == 2000

    def test_load_us_historical_pandas_format(self):
        """Test loading US historical data as pandas."""
        adapter = PolarsAdapter()

        df = adapter.load_us_historical(format="pandas")

        assert isinstance(df, pd.DataFrame)
        assert "year" in df.columns

    def test_caching(self):
        """Test that data is cached."""
        adapter = PolarsAdapter()

        # Load data twice
        df1 = adapter.load_us_historical(start_year=1900, end_year=1950)
        df2 = adapter.load_us_historical(start_year=1900, end_year=1950)

        # Should be the same object (from cache)
        assert df1 is df2

    def test_cache_disabled(self):
        """Test that caching can be disabled."""
        config = PolarsAdapterConfig(cache_enabled=False)
        adapter = PolarsAdapter(config=config)

        _ = adapter.load_us_historical(start_year=1900, end_year=1950)
        _ = adapter.load_us_historical(start_year=1900, end_year=1950)

        # Should be different objects (not cached)
        # Note: We can't guarantee they're different objects since
        # the function returns the same result, but cache should be empty
        assert len(adapter._cache) == 0


class TestPolarsAdapterQueries:
    """Tests for PolarsAdapter query functionality."""

    @patch.object(PolarsAdapter, "load_seshat")
    def test_query_seshat_filter_by_region(self, mock_load):
        """Test filtering by region."""
        mock_load.return_value = pl.DataFrame({
            "Polity": ["RomPrin", "EgyptNew", "ChineseHan"],
            "NGA": ["Italy", "Egypt", "China"],
            "Start": [-27, -1550, -206],
            "End": [284, -1077, 220],
        })

        adapter = PolarsAdapter()
        df = adapter.query_seshat(regions=["Italy"])

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 1
        assert df["Polity"].to_list() == ["RomPrin"]

    @patch.object(PolarsAdapter, "load_seshat")
    def test_query_seshat_filter_by_time_range(self, mock_load):
        """Test filtering by time range."""
        mock_load.return_value = pl.DataFrame({
            "Polity": ["RomPrin", "RomDom", "Medieval"],
            "NGA": ["Italy", "Italy", "Italy"],
            "Start": [-27, 284, 800],
            "End": [284, 476, 1200],
        })

        adapter = PolarsAdapter()
        df = adapter.query_seshat(time_range=(0, 300))

        # Should include RomPrin (overlaps 0-300) and RomDom (starts at 284)
        assert len(df) == 2
        polities = df["Polity"].to_list()
        assert "RomPrin" in polities
        assert "RomDom" in polities
        assert "Medieval" not in polities

    @patch.object(PolarsAdapter, "load_seshat")
    def test_query_seshat_filter_by_polities(self, mock_load):
        """Test filtering by polity IDs."""
        mock_load.return_value = pl.DataFrame({
            "Polity": ["RomPrin", "RomDom", "EgyptNew"],
            "NGA": ["Italy", "Italy", "Egypt"],
            "Start": [-27, 284, -1550],
            "End": [284, 476, -1077],
        })

        adapter = PolarsAdapter()
        df = adapter.query_seshat(polities=["RomPrin", "EgyptNew"])

        assert len(df) == 2
        polities = df["Polity"].to_list()
        assert "RomPrin" in polities
        assert "EgyptNew" in polities
        assert "RomDom" not in polities

    @patch.object(PolarsAdapter, "load_seshat")
    def test_query_seshat_select_variables(self, mock_load):
        """Test selecting specific variables (columns)."""
        mock_load.return_value = pl.DataFrame({
            "Polity": ["RomPrin"],
            "Original_name": ["Roman Principate"],
            "NGA": ["Italy"],
            "Start": [-27],
            "End": [284],
            "PolPop": [50000000],
            "Polity_territory": [5000000],
            "OtherVar": [100],
        })

        adapter = PolarsAdapter()
        df = adapter.query_seshat(variables=["PolPop", "Polity_territory"])

        # Should include metadata columns plus requested variables
        assert "Polity" in df.columns
        assert "NGA" in df.columns
        assert "PolPop" in df.columns
        assert "Polity_territory" in df.columns
        # Should not include other variables
        assert "OtherVar" not in df.columns

    @patch.object(PolarsAdapter, "load_seshat")
    def test_query_seshat_combined_filters(self, mock_load):
        """Test combining multiple filters."""
        mock_load.return_value = pl.DataFrame({
            "Polity": ["RomPrin", "RomDom", "EgyptNew", "ChineseHan"],
            "NGA": ["Italy", "Italy", "Egypt", "China"],
            "Start": [-27, 284, -1550, -206],
            "End": [284, 476, -1077, 220],
            "PolPop": [50000000, 30000000, 3000000, 60000000],
        })

        adapter = PolarsAdapter()
        df = adapter.query_seshat(
            regions=["Italy"],
            time_range=(0, 300),
            variables=["PolPop"],
        )

        # Should only include RomPrin (Italy, overlaps 0-300)
        # RomDom starts at 284, so also overlaps
        assert len(df) == 2

    @patch.object(PolarsAdapter, "load_seshat")
    def test_query_seshat_pandas_format(self, mock_load):
        """Test query returning pandas format."""
        mock_load.return_value = pl.DataFrame({
            "Polity": ["RomPrin"],
            "NGA": ["Italy"],
            "Start": [-27],
            "End": [284],
        })

        adapter = PolarsAdapter()
        df = adapter.query_seshat(format="pandas")

        assert isinstance(df, pd.DataFrame)


class TestLazyPolarsAdapter:
    """Tests for LazyPolarsAdapter class."""

    def test_inherits_from_polars_adapter(self):
        """Test that LazyPolarsAdapter inherits from PolarsAdapter."""
        adapter = LazyPolarsAdapter()
        assert isinstance(adapter, PolarsAdapter)

    @patch.object(PolarsAdapter, "load_seshat")
    def test_load_seshat_lazy_returns_lazyframe(self, mock_load):
        """Test that load_seshat_lazy returns LazyFrame."""
        mock_load.return_value = pl.DataFrame({
            "Polity": ["RomPrin"],
            "NGA": ["Italy"],
        })

        adapter = LazyPolarsAdapter()
        lf = adapter.load_seshat_lazy()

        assert isinstance(lf, pl.LazyFrame)

    def test_load_us_historical_lazy(self):
        """Test lazy loading of US historical data."""
        adapter = LazyPolarsAdapter()

        lf = adapter.load_us_historical_lazy(start_year=1900, end_year=2000)

        assert isinstance(lf, pl.LazyFrame)

        # Collect and verify
        df = lf.collect()
        assert isinstance(df, pl.DataFrame)
        assert "year" in df.columns

    @patch.object(PolarsAdapter, "load_seshat")
    def test_lazy_query_optimization(self, mock_load):
        """Test that lazy queries can be chained and optimized."""
        mock_load.return_value = pl.DataFrame({
            "Polity": ["RomPrin", "RomDom", "EgyptNew"],
            "NGA": ["Italy", "Italy", "Egypt"],
            "Start": [-27, 284, -1550],
            "End": [284, 476, -1077],
            "PolPop": [50000000, 30000000, 3000000],
        })

        adapter = LazyPolarsAdapter()
        lf = adapter.load_seshat_lazy()

        # Build a complex query
        result = (
            lf
            .filter(pl.col("NGA") == "Italy")
            .select(["Polity", "PolPop"])
            .sort("PolPop", descending=True)
        )

        # Should still be lazy
        assert isinstance(result, pl.LazyFrame)

        # Collect to execute
        df = result.collect()
        assert len(df) == 2
        assert df["Polity"].to_list() == ["RomPrin", "RomDom"]


class TestGetAdapter:
    """Tests for get_adapter convenience function."""

    def test_returns_polars_adapter(self):
        """Test that default returns PolarsAdapter."""
        adapter = get_adapter()
        assert isinstance(adapter, PolarsAdapter)
        assert not isinstance(adapter, LazyPolarsAdapter)

    def test_returns_lazy_adapter(self):
        """Test that lazy=True returns LazyPolarsAdapter."""
        adapter = get_adapter(lazy=True)
        assert isinstance(adapter, LazyPolarsAdapter)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_convert_dataframe_with_index(self):
        """Test converting DataFrame with non-default index."""
        df_pd = pd.DataFrame(
            {"a": [1, 2, 3], "b": ["x", "y", "z"]},
            index=["row1", "row2", "row3"],
        )

        df_pl = pandas_to_polars(df_pd)

        # Index is lost in conversion (Polars doesn't have index)
        assert df_pl.shape == (3, 2)

    def test_convert_dataframe_with_multiindex(self):
        """Test converting DataFrame with MultiIndex columns."""
        df_pd = pd.DataFrame({
            ("A", "a"): [1, 2],
            ("A", "b"): [3, 4],
            ("B", "c"): [5, 6],
        })

        df_pl = pandas_to_polars(df_pd)

        # Multi-level columns become tuples in string form
        assert df_pl.shape == (2, 3)

    def test_convert_large_dataframe(self):
        """Test converting a reasonably large DataFrame."""
        n_rows = 100000
        df_pd = pd.DataFrame({
            "a": np.arange(n_rows),
            "b": np.random.randn(n_rows),
            "c": ["text"] * n_rows,
        })

        df_pl = pandas_to_polars(df_pd)

        assert df_pl.shape == (n_rows, 3)

    def test_roundtrip_conversion(self):
        """Test pandas -> polars -> pandas roundtrip."""
        original = pd.DataFrame({
            "int": [1, 2, 3],
            "float": [1.1, 2.2, 3.3],
            "str": ["a", "b", "c"],
        })

        polars_df = pandas_to_polars(original)
        final = polars_to_pandas(polars_df)

        pd.testing.assert_frame_equal(original, final)

    def test_column_name_normalization(self):
        """Test that column finding handles case variations."""
        adapter = PolarsAdapter()

        # Test lowercase
        df_lower = pl.DataFrame({"polity": ["A"], "nga": ["B"]})
        assert adapter._find_column(df_lower, ["Polity", "polity"]) == "polity"
        assert adapter._find_column(df_lower, ["NGA", "nga"]) == "nga"

        # Test uppercase
        df_upper = pl.DataFrame({"POLITY": ["A"], "NGA": ["B"]})
        result = adapter._find_column(df_upper, ["Polity", "polity", "POLITY"])
        assert result == "POLITY"

    def test_empty_query_result(self):
        """Test query that returns no results."""
        adapter = PolarsAdapter()

        with patch.object(PolarsAdapter, "load_seshat") as mock_load:
            mock_load.return_value = pl.DataFrame({
                "Polity": ["RomPrin"],
                "NGA": ["Italy"],
                "Start": [-27],
                "End": [284],
            })

            df = adapter.query_seshat(regions=["NonexistentRegion"])

            assert len(df) == 0


class TestModuleExports:
    """Tests for module exports from cliodynamics.data."""

    def test_polars_adapter_importable(self):
        """Test that PolarsAdapter can be imported from cliodynamics.data."""
        from cliodynamics.data import PolarsAdapter
        assert PolarsAdapter is not None

    def test_lazy_polars_adapter_importable(self):
        """Test that LazyPolarsAdapter can be imported from cliodynamics.data."""
        from cliodynamics.data import LazyPolarsAdapter
        assert LazyPolarsAdapter is not None

    def test_get_adapter_importable(self):
        """Test that get_adapter can be imported from cliodynamics.data."""
        from cliodynamics.data import get_adapter
        adapter = get_adapter()
        assert isinstance(adapter, PolarsAdapter)

    def test_conversion_functions_importable(self):
        """Test that conversion functions can be imported from cliodynamics.data."""
        from cliodynamics.data import pandas_to_polars, polars_to_pandas
        assert callable(pandas_to_polars)
        assert callable(polars_to_pandas)

    def test_read_functions_importable(self):
        """Test that read functions can be imported from cliodynamics.data."""
        from cliodynamics.data import read_csv_polars, read_excel_polars
        assert callable(read_csv_polars)
        assert callable(read_excel_polars)

    def test_config_importable(self):
        """Test that PolarsAdapterConfig can be imported from cliodynamics.data."""
        from cliodynamics.data import PolarsAdapterConfig
        config = PolarsAdapterConfig(prefer_polaris=False)
        assert config.prefer_polaris is False


class TestPolarsIntegration:
    """Integration tests for Polars functionality."""

    def test_adapter_with_us_data_polars_operations(self):
        """Test that US data can be loaded and manipulated with Polars."""
        adapter = PolarsAdapter()
        df = adapter.load_us_historical(start_year=1900, end_year=2000)

        # Test Polars operations
        filtered = df.filter(pl.col("year") >= 1950)
        assert len(filtered) < len(df)

        # Test aggregation
        mean_wage = df.select(pl.col("real_wage_index").mean()).item()
        assert mean_wage > 0

    def test_lazy_evaluation_chain(self):
        """Test chained lazy operations."""
        adapter = LazyPolarsAdapter()
        lf = adapter.load_us_historical_lazy(start_year=1800, end_year=2020)

        # Chain operations lazily
        result = (
            lf
            .filter(pl.col("year") >= 1900)
            .filter(pl.col("year") <= 2000)
            .select(["year", "real_wage_index"])
        )

        # Verify still lazy
        assert isinstance(result, pl.LazyFrame)

        # Collect and verify
        df = result.collect()
        assert len(df) == 101  # 1900-2000 inclusive
        assert df.columns == ["year", "real_wage_index"]

    def test_conversion_roundtrip_preserves_data(self):
        """Test that pandas<->polars roundtrip preserves data integrity."""
        import pandas as pd

        # Create test data with various types
        df_original = pd.DataFrame({
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
            "str_col": ["a", "b", "c", "d", "e"],
            "bool_col": [True, False, True, False, True],
        })

        # Convert to Polars and back
        df_polars = pandas_to_polars(df_original)
        df_roundtrip = polars_to_pandas(df_polars)

        # Verify data integrity
        pd.testing.assert_frame_equal(df_original, df_roundtrip)

    def test_null_handling_in_conversions(self):
        """Test that null values are handled correctly in conversions."""
        import pandas as pd

        df_pd = pd.DataFrame({
            "a": [1.0, None, 3.0, None, 5.0],
            "b": ["x", None, "z", None, "w"],
        })

        df_pl = pandas_to_polars(df_pd)

        # Verify nulls are preserved
        assert df_pl["a"].null_count() == 2
        assert df_pl["b"].null_count() == 2

    def test_schema_preservation(self):
        """Test that data types are preserved through conversion."""
        import pandas as pd

        df_pd = pd.DataFrame({
            "integers": pd.array([1, 2, 3], dtype="int64"),
            "floats": pd.array([1.0, 2.0, 3.0], dtype="float64"),
        })

        df_pl = pandas_to_polars(df_pd)

        assert df_pl["integers"].dtype == pl.Int64
        assert df_pl["floats"].dtype == pl.Float64
