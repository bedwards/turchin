"""
Polars adapter for the cliodynamics data layer.

This module provides Polars DataFrames from existing data sources and
utilities for converting between pandas and Polars during the migration
from pandas to Polars.

Polars offers significant performance advantages over pandas:
- Lazy evaluation enables query optimization
- Parallel execution on all available cores
- Zero-copy memory sharing
- More expressive and consistent API

Migration Strategy:
-------------------
1. This module provides a PolarsAdapter that wraps existing data sources
2. Backward compatibility is maintained via to_pandas() methods
3. New code should use Polars directly; old code can convert as needed
4. Eventually, pandas will become an optional dependency

Usage:
    >>> from cliodynamics.data.polars_adapter import PolarsAdapter
    >>> adapter = PolarsAdapter()
    >>>
    >>> # Get Seshat data as Polars DataFrame
    >>> df_pl = adapter.load_seshat()
    >>>
    >>> # Convert to pandas if needed (for legacy code)
    >>> df_pd = df_pl.to_pandas()
    >>>
    >>> # Use conversion utilities
    >>> from cliodynamics.data.polars_adapter import pandas_to_polars, polars_to_pandas
    >>> df_pl = pandas_to_polars(pandas_df)
    >>> df_pd = polars_to_pandas(polars_df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import polars as pl

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


# Type alias for DataFrame format selection
DataFrameFormat = Literal["polars", "pandas"]


def pandas_to_polars(
    df: "pd.DataFrame",
    *,
    schema_overrides: dict[str, pl.DataType] | None = None,
    rechunk: bool = True,
    nan_to_null: bool = True,
) -> pl.DataFrame:
    """
    Convert a pandas DataFrame to a Polars DataFrame.

    This function handles common conversion edge cases:
    - NaN values are converted to null (Polars doesn't have NaN for non-float types)
    - Categorical columns are preserved
    - Datetime columns are handled appropriately

    Args:
        df: The pandas DataFrame to convert.
        schema_overrides: Optional dictionary mapping column names to Polars data types.
            Use this to force specific types when automatic inference isn't ideal.
        rechunk: If True, ensure the resulting DataFrame is contiguous in memory.
            This can improve performance for subsequent operations.
        nan_to_null: If True, convert NaN values to null. This is usually what you want
            since Polars uses null (not NaN) for missing values.

    Returns:
        A Polars DataFrame with the same data.

    Example:
        >>> import pandas as pd
        >>> df_pd = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        >>> df_pl = pandas_to_polars(df_pd)
        >>> print(df_pl.schema)
        {'a': Int64, 'b': String}
    """
    # Use Polars' built-in conversion
    result = pl.from_pandas(df, nan_to_null=nan_to_null, rechunk=rechunk)

    # Apply schema overrides if provided
    if schema_overrides:
        casts = [
            pl.col(name).cast(dtype)
            for name, dtype in schema_overrides.items()
            if name in result.columns
        ]
        if casts:
            result = result.with_columns(casts)

    return result


def polars_to_pandas(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    use_pyarrow_extension_array: bool = False,
    date_as_object: bool = False,
) -> "pd.DataFrame":
    """
    Convert a Polars DataFrame (or LazyFrame) to a pandas DataFrame.

    This function is useful for interfacing with legacy code that expects
    pandas DataFrames, or for using pandas-specific functionality.

    Args:
        df: The Polars DataFrame or LazyFrame to convert.
            LazyFrames will be collected before conversion.
        use_pyarrow_extension_array: If True, use PyArrow-backed extension arrays
            in pandas. This can be more efficient for large datasets but may have
            compatibility issues with some pandas operations.
        date_as_object: If True, convert date columns to Python objects.
            By default, they're converted to pandas datetime64.

    Returns:
        A pandas DataFrame with the same data.

    Example:
        >>> df_pl = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        >>> df_pd = polars_to_pandas(df_pl)
        >>> print(type(df_pd))
        <class 'pandas.core.frame.DataFrame'>
    """
    # Collect LazyFrame if needed
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    return df.to_pandas(
        use_pyarrow_extension_array=use_pyarrow_extension_array,
        date_as_object=date_as_object,
    )


def read_csv_polars(
    path: Path | str,
    *,
    columns: Sequence[str] | None = None,
    schema_overrides: dict[str, pl.DataType] | None = None,
    encoding: str = "utf-8",
    null_values: str | list[str] | None = None,
    skip_rows: int = 0,
    n_rows: int | None = None,
    low_memory: bool = False,
    try_encodings: Sequence[str] | None = None,
) -> pl.DataFrame:
    """
    Read a CSV file into a Polars DataFrame.

    This function provides a consistent interface for reading CSV files
    with sensible defaults and automatic encoding detection fallback.

    Args:
        path: Path to the CSV file.
        columns: Specific columns to read. If None, read all columns.
        schema_overrides: Dictionary mapping column names to Polars data types.
        encoding: Character encoding (default: utf-8).
        null_values: Additional strings to interpret as null values.
        skip_rows: Number of rows to skip at the start of the file.
        n_rows: Maximum number of rows to read.
        low_memory: If True, reduce memory usage at the cost of performance.
        try_encodings: List of encodings to try if the primary encoding fails.
            Defaults to ["utf-8", "latin-1", "cp1252"].

    Returns:
        A Polars DataFrame with the CSV data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        pl.ComputeError: If the file cannot be parsed.

    Example:
        >>> df = read_csv_polars("data/seshat/polities.csv")
        >>> print(df.shape)
        (373, 45)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    if try_encodings is None:
        try_encodings = ["utf-8", "latin-1", "cp1252"]

    # Try primary encoding first, then fallbacks
    encodings_to_try = [encoding] + [e for e in try_encodings if e != encoding]
    last_error: Exception | None = None

    for enc in encodings_to_try:
        try:
            return pl.read_csv(
                path,
                columns=columns,
                schema_overrides=schema_overrides,
                encoding=enc,
                null_values=null_values or [],
                skip_rows=skip_rows,
                n_rows=n_rows,
                low_memory=low_memory,
            )
        except Exception as e:
            last_error = e
            logger.debug(f"Failed to read CSV with encoding {enc}: {e}")
            continue

    # If all encodings failed, raise the last error
    raise last_error or RuntimeError(f"Failed to read CSV file: {path}")


def read_excel_polars(
    path: Path | str,
    *,
    sheet_name: str | int = 0,
    columns: Sequence[str] | None = None,
    schema_overrides: dict[str, pl.DataType] | None = None,
    skip_rows: int = 0,
    n_rows: int | None = None,
) -> pl.DataFrame:
    """
    Read an Excel file into a Polars DataFrame.

    Args:
        path: Path to the Excel file.
        sheet_name: Sheet name (string) or index (int, 0-based).
        columns: Specific columns to read. If None, read all columns.
        schema_overrides: Dictionary mapping column names to Polars data types.
        skip_rows: Number of rows to skip at the start.
        n_rows: Maximum number of rows to read.

    Returns:
        A Polars DataFrame with the Excel data.

    Raises:
        FileNotFoundError: If the file doesn't exist.

    Example:
        >>> df = read_excel_polars("data/polaris2025/Polaris2025.xlsx")
        >>> print(df.shape)
        (500, 100)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    # Polars read_excel accepts sheet_id (1-indexed) or sheet_name
    if isinstance(sheet_name, int):
        # Convert 0-indexed to 1-indexed for Polars
        sheet_id = sheet_name + 1
        df = pl.read_excel(path, sheet_id=sheet_id)
    else:
        df = pl.read_excel(path, sheet_name=sheet_name)

    # Apply column selection
    if columns is not None:
        available_cols = [c for c in columns if c in df.columns]
        df = df.select(available_cols)

    # Apply row skipping and limiting
    if skip_rows > 0:
        df = df.slice(skip_rows)
    if n_rows is not None:
        df = df.head(n_rows)

    # Apply schema overrides
    if schema_overrides:
        casts = [
            pl.col(name).cast(dtype)
            for name, dtype in schema_overrides.items()
            if name in df.columns
        ]
        if casts:
            df = df.with_columns(casts)

    return df


@dataclass
class PolarsAdapterConfig:
    """
    Configuration for the PolarsAdapter.

    Attributes:
        seshat_data_dir: Path to Seshat data directory. If None, uses default.
        polaris_data_dir: Path to Polaris data directory. If None, uses default.
        prefer_polaris: If True, prefer Polaris-2025 over Equinox-2020.
        cache_enabled: If True, cache loaded DataFrames in memory.
    """

    seshat_data_dir: Path | str | None = None
    polaris_data_dir: Path | str | None = None
    prefer_polaris: bool = True
    cache_enabled: bool = True


class PolarsAdapter:
    """
    Adapter for loading data sources as Polars DataFrames.

    This class provides a unified interface for loading various data sources
    (Seshat, U.S. historical data, etc.) as Polars DataFrames, with optional
    conversion to pandas for backward compatibility.

    The adapter supports lazy loading and caching to minimize I/O and memory usage.

    Args:
        config: Configuration options for the adapter.
            If None, uses default configuration.

    Example:
        >>> adapter = PolarsAdapter()
        >>>
        >>> # Load Seshat data
        >>> seshat_df = adapter.load_seshat()
        >>> print(seshat_df.shape)
        (373, 50)
        >>>
        >>> # Load U.S. historical data
        >>> us_df = adapter.load_us_historical()
        >>> print(us_df.columns)
        ['year', 'real_wage_index', 'relative_wage_index', ...]
        >>>
        >>> # Convert to pandas for legacy code
        >>> seshat_pd = adapter.load_seshat(format="pandas")
    """

    def __init__(self, config: PolarsAdapterConfig | None = None) -> None:
        """Initialize the PolarsAdapter with optional configuration."""
        self.config = config or PolarsAdapterConfig()
        self._cache: dict[str, pl.DataFrame] = {}

        # Resolve data directories
        self._seshat_dir: Path | None = None
        self._polaris_dir: Path | None = None

        if self.config.seshat_data_dir:
            self._seshat_dir = Path(self.config.seshat_data_dir)
        if self.config.polaris_data_dir:
            self._polaris_dir = Path(self.config.polaris_data_dir)

    def _get_project_root(self) -> Path:
        """Get the project root directory."""
        # This module is at: src/cliodynamics/data/polars_adapter.py
        module_path = Path(__file__).resolve()
        src_path = module_path.parent.parent.parent  # up from data/ to src/
        return src_path.parent  # up from src/ to project root

    def _get_seshat_dir(self) -> Path:
        """Get the Seshat data directory."""
        if self._seshat_dir:
            return self._seshat_dir
        return self._get_project_root() / "data" / "seshat"

    def _get_polaris_dir(self) -> Path:
        """Get the Polaris data directory."""
        if self._polaris_dir:
            return self._polaris_dir
        return self._get_project_root() / "data" / "polaris2025"

    def _get_from_cache(self, key: str) -> pl.DataFrame | None:
        """Get a DataFrame from cache if available."""
        if self.config.cache_enabled and key in self._cache:
            logger.debug(f"Cache hit: {key}")
            return self._cache[key]
        return None

    def _put_in_cache(self, key: str, df: pl.DataFrame) -> None:
        """Store a DataFrame in cache."""
        if self.config.cache_enabled:
            self._cache[key] = df
            logger.debug(f"Cached: {key}")

    def clear_cache(self) -> None:
        """Clear all cached DataFrames."""
        self._cache.clear()
        logger.info("Cache cleared")

    def load_seshat(
        self,
        *,
        dataset: Literal["auto", "polaris", "equinox"] = "auto",
        format: DataFrameFormat = "polars",
    ) -> pl.DataFrame | "pd.DataFrame":
        """
        Load Seshat data as a DataFrame.

        Args:
            dataset: Which dataset to load:
                - "auto": Prefer Polaris-2025 if available, fall back to Equinox-2020
                - "polaris": Load Polaris-2025 (raises if not available)
                - "equinox": Load Equinox-2020 (raises if not available)
            format: Output format ("polars" or "pandas").

        Returns:
            DataFrame with Seshat data. Format depends on `format` argument.

        Raises:
            FileNotFoundError: If the requested dataset is not available.

        Example:
            >>> adapter = PolarsAdapter()
            >>> df = adapter.load_seshat()
            >>> print(df.columns[:5])
            ['Polity', 'Original_name', 'NGA', 'Start', 'End']
        """
        cache_key = f"seshat_{dataset}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return polars_to_pandas(cached) if format == "pandas" else cached

        # Determine which dataset to load
        polaris_dir = self._get_polaris_dir()
        seshat_dir = self._get_seshat_dir()

        df: pl.DataFrame | None = None

        if dataset == "auto":
            # Try Polaris first if preferred
            if self.config.prefer_polaris and polaris_dir.exists():
                try:
                    df = self._load_polaris_data(polaris_dir)
                except FileNotFoundError:
                    logger.debug("Polaris not found, trying Equinox")

            # Fall back to Equinox
            if df is None and seshat_dir.exists():
                try:
                    df = self._load_equinox_data(seshat_dir)
                except FileNotFoundError:
                    pass

            if df is None:
                raise FileNotFoundError(
                    "No Seshat data found. Run download_polaris or download first."
                )

        elif dataset == "polaris":
            if not polaris_dir.exists():
                raise FileNotFoundError(
                    f"Polaris data not found at {polaris_dir}. "
                    "Run 'python -m cliodynamics.data.download_polaris' first."
                )
            df = self._load_polaris_data(polaris_dir)

        elif dataset == "equinox":
            if not seshat_dir.exists():
                raise FileNotFoundError(
                    f"Equinox data not found at {seshat_dir}. "
                    "Run 'python -m cliodynamics.data.download' first."
                )
            df = self._load_equinox_data(seshat_dir)

        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        self._put_in_cache(cache_key, df)

        return polars_to_pandas(df) if format == "pandas" else df

    def _load_polaris_data(self, data_dir: Path) -> pl.DataFrame:
        """Load Polaris-2025 data from the specified directory."""
        excel_file = data_dir / "Polaris2025.xlsx"
        if excel_file.exists():
            logger.info(f"Loading Polaris data from {excel_file}")
            return read_excel_polars(excel_file)

        # Look for any Excel or CSV files
        excel_files = list(data_dir.glob("*.xlsx"))
        if excel_files:
            logger.info(f"Loading Polaris data from {excel_files[0]}")
            return read_excel_polars(excel_files[0])

        csv_files = [
            f for f in data_dir.glob("*.csv")
            if "threads" not in f.name.lower()
        ]
        if csv_files:
            logger.info(f"Loading Polaris data from {csv_files[0]}")
            return read_csv_polars(csv_files[0])

        raise FileNotFoundError(f"No data files found in {data_dir}")

    def _load_equinox_data(self, data_dir: Path) -> pl.DataFrame:
        """Load Equinox-2020 data from the specified directory."""
        excel_files = list(data_dir.glob("*.xlsx"))
        if excel_files:
            logger.info(f"Loading Equinox data from {excel_files[0]}")
            return read_excel_polars(excel_files[0])

        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            logger.info(f"Loading Equinox data from {csv_files[0]}")
            return read_csv_polars(csv_files[0])

        raise FileNotFoundError(f"No data files found in {data_dir}")

    def load_polaris_threads(
        self,
        *,
        format: DataFrameFormat = "polars",
    ) -> pl.DataFrame | "pd.DataFrame":
        """
        Load Polaris-2025 polity threads data.

        The polity threads file contains information about temporal continuity
        between polities, useful for tracking political succession.

        Args:
            format: Output format ("polars" or "pandas").

        Returns:
            DataFrame with polity thread data.

        Raises:
            FileNotFoundError: If the threads file is not available.
        """
        cache_key = "polaris_threads"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return polars_to_pandas(cached) if format == "pandas" else cached

        polaris_dir = self._get_polaris_dir()
        threads_file = polaris_dir / "polity_threads.csv"

        if not threads_file.exists():
            raise FileNotFoundError(
                f"Polity threads file not found at {threads_file}. "
                "Run 'python -m cliodynamics.data.download_polaris' first."
            )

        logger.info(f"Loading polity threads from {threads_file}")
        df = read_csv_polars(threads_file)

        self._put_in_cache(cache_key, df)

        return polars_to_pandas(df) if format == "pandas" else df

    def load_us_historical(
        self,
        *,
        start_year: int = 1780,
        end_year: int = 2025,
        format: DataFrameFormat = "polars",
    ) -> pl.DataFrame | "pd.DataFrame":
        """
        Load U.S. historical data as a DataFrame.

        This method loads data from the us.loader module and returns it
        as a Polars DataFrame. The data includes:
        - Real wages
        - Relative wages
        - Elite indicators (lawyers, PhDs per capita)
        - Wealth inequality
        - Political instability indicators

        Args:
            start_year: Start of time series (default: 1780).
            end_year: End of time series (default: 2025).
            format: Output format ("polars" or "pandas").

        Returns:
            DataFrame with all U.S. historical indicators merged on year.

        Example:
            >>> adapter = PolarsAdapter()
            >>> df = adapter.load_us_historical()
            >>> print(df.columns)
            ['year', 'real_wage_index', 'relative_wage_index', ...]
        """
        cache_key = f"us_historical_{start_year}_{end_year}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return polars_to_pandas(cached) if format == "pandas" else cached

        # Import and use the existing US data loader
        from cliodynamics.data.us.loader import USHistoricalData

        us_data = USHistoricalData(start_year=start_year, end_year=end_year)
        df_pandas = us_data.get_combined_dataset()

        # Convert to Polars
        df = pandas_to_polars(df_pandas)

        self._put_in_cache(cache_key, df)

        return polars_to_pandas(df) if format == "pandas" else df

    def query_seshat(
        self,
        *,
        variables: Sequence[str] | None = None,
        polities: Sequence[str] | None = None,
        time_range: tuple[int, int] | None = None,
        regions: Sequence[str] | None = None,
        format: DataFrameFormat = "polars",
    ) -> pl.DataFrame | "pd.DataFrame":
        """
        Query Seshat data with flexible filtering.

        This method loads Seshat data and applies filters using Polars'
        efficient lazy evaluation.

        Args:
            variables: List of variable (column) names to include.
                If None, include all columns.
            polities: List of polity IDs to include. If None, include all.
            time_range: Tuple of (start_year, end_year) to filter by.
                Uses CE years (negative = BCE).
            regions: List of NGA (Natural Geographic Area) names to filter by.
            format: Output format ("polars" or "pandas").

        Returns:
            Filtered DataFrame with Seshat data.

        Example:
            >>> adapter = PolarsAdapter()
            >>> df = adapter.query_seshat(
            ...     variables=["PolPop", "Polity_territory"],
            ...     time_range=(-500, 500),
            ...     regions=["Italy"],
            ... )
        """
        # Load the full dataset
        df = self.load_seshat(format="polars")
        assert isinstance(df, pl.DataFrame)

        # Build filter expressions
        filters: list[pl.Expr] = []

        # Filter by polities
        if polities is not None:
            polity_col = self._find_column(df, ["Polity", "polity"])
            if polity_col:
                filters.append(pl.col(polity_col).is_in(polities))

        # Filter by regions (NGA)
        if regions is not None:
            nga_col = self._find_column(df, ["NGA", "nga"])
            if nga_col:
                filters.append(pl.col(nga_col).is_in(regions))

        # Filter by time range
        if time_range is not None:
            start_year, end_year = time_range
            start_col = self._find_column(df, ["Start", "start", "start_year"])
            end_col = self._find_column(df, ["End", "end", "end_year"])

            if start_col and end_col:
                # Polity overlaps if it doesn't end before start or start after end
                filters.append(pl.col(end_col).cast(pl.Int64) >= start_year)
                filters.append(pl.col(start_col).cast(pl.Int64) <= end_year)

        # Apply filters
        if filters:
            combined_filter = filters[0]
            for f in filters[1:]:
                combined_filter = combined_filter & f
            df = df.filter(combined_filter)

        # Select specific variables (columns)
        if variables is not None:
            # Always include metadata columns
            metadata_cols = ["Polity", "polity", "Original_name", "original_name",
                           "NGA", "nga", "Start", "start", "End", "end"]
            cols_to_keep = [c for c in metadata_cols if c in df.columns]
            cols_to_keep.extend([v for v in variables if v in df.columns])
            df = df.select(cols_to_keep)

        return polars_to_pandas(df) if format == "pandas" else df

    def _find_column(self, df: pl.DataFrame, candidates: list[str]) -> str | None:
        """Find the first matching column name from candidates."""
        for c in candidates:
            if c in df.columns:
                return c
        return None


class LazyPolarsAdapter(PolarsAdapter):
    """
    Lazy-loading variant of PolarsAdapter.

    This adapter uses Polars' LazyFrame for deferred evaluation and
    query optimization. Operations are only executed when results
    are explicitly collected.

    This is particularly useful for:
    - Large datasets that don't fit in memory
    - Complex query chains where optimization matters
    - Exploratory analysis where you want to build queries incrementally

    Example:
        >>> adapter = LazyPolarsAdapter()
        >>>
        >>> # Build a lazy query
        >>> lf = adapter.load_seshat_lazy()
        >>> lf = lf.filter(pl.col("NGA") == "Italy")
        >>> lf = lf.select(["Polity", "Start", "End", "PolPop"])
        >>>
        >>> # Execute the optimized query
        >>> df = lf.collect()
    """

    def load_seshat_lazy(
        self,
        *,
        dataset: Literal["auto", "polaris", "equinox"] = "auto",
    ) -> pl.LazyFrame:
        """
        Load Seshat data as a LazyFrame for deferred execution.

        Args:
            dataset: Which dataset to load ("auto", "polaris", or "equinox").

        Returns:
            LazyFrame that can be further filtered/transformed before collecting.
        """
        df = self.load_seshat(dataset=dataset, format="polars")
        assert isinstance(df, pl.DataFrame)
        return df.lazy()

    def load_us_historical_lazy(
        self,
        *,
        start_year: int = 1780,
        end_year: int = 2025,
    ) -> pl.LazyFrame:
        """
        Load U.S. historical data as a LazyFrame.

        Args:
            start_year: Start of time series.
            end_year: End of time series.

        Returns:
            LazyFrame with U.S. historical data.
        """
        df = self.load_us_historical(
            start_year=start_year,
            end_year=end_year,
            format="polars",
        )
        assert isinstance(df, pl.DataFrame)
        return df.lazy()


# Convenience function for quick access
def get_adapter(lazy: bool = False) -> PolarsAdapter:
    """
    Get a PolarsAdapter instance.

    Args:
        lazy: If True, return a LazyPolarsAdapter for deferred execution.

    Returns:
        PolarsAdapter or LazyPolarsAdapter instance.

    Example:
        >>> adapter = get_adapter()
        >>> df = adapter.load_seshat()
    """
    if lazy:
        return LazyPolarsAdapter()
    return PolarsAdapter()
