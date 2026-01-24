"""
Data access module for Seshat and other historical datasets.

This module provides tools for downloading, parsing, and querying the Seshat
Global History Databank, supporting both local files (Equinox-2020) and the
live Seshat API (Polaris-2025+).

Key Classes:
    SeshatDB: High-level query interface for local Seshat data (pandas)
    SeshatAPIClient: Client for querying the live Seshat API
    PolarsAdapter: High-performance data access using Polars DataFrames
    LazyPolarsAdapter: Lazy-loading variant with deferred execution
    SeshatDataset: Container for parsed Seshat data
    Polity: A political entity with its variables
    ParsedValue: A parsed data value with uncertainty info
    DataQuality: Enum for data quality indicators

Key Functions:
    download_and_extract: Download Seshat dataset from Zenodo
    load_equinox: Load and parse the Seshat Equinox-2020 dataset
    parse_value: Parse individual Seshat data values
    pandas_to_polars: Convert pandas DataFrame to Polars
    polars_to_pandas: Convert Polars DataFrame to pandas
    get_adapter: Get a PolarsAdapter instance

Usage (local data with pandas - legacy):
    >>> from cliodynamics.data import SeshatDB
    >>> db = SeshatDB("data/seshat/")
    >>>
    >>> # Get all data for a polity
    >>> rome = db.get_polity("RomPrin")
    >>>
    >>> # Query by variable and time range
    >>> df = db.query(
    ...     variables=["population"],
    ...     time_range=(-500, 500)
    ... )

Usage (with Polars - recommended):
    >>> from cliodynamics.data import PolarsAdapter, get_adapter
    >>> adapter = get_adapter()
    >>>
    >>> # Load Seshat data as Polars DataFrame
    >>> df = adapter.load_seshat()
    >>>
    >>> # Load U.S. historical data
    >>> us_df = adapter.load_us_historical()
    >>>
    >>> # Query with filters
    >>> italy = adapter.query_seshat(regions=["Italy"], time_range=(0, 500))
    >>>
    >>> # Convert to pandas if needed
    >>> df_pd = df.to_pandas()

Usage (API client - requires seshat_api package):
    >>> from cliodynamics.data import SeshatAPIClient
    >>> client = SeshatAPIClient()
    >>>
    >>> # Query polities
    >>> polities = client.list_polities(region="Italy")
    >>>
    >>> # Get polity info
    >>> rome = client.get_polity("RomPrin")
"""

from cliodynamics.data.access import (
    DATASET_EQUINOX,
    DATASET_POLARIS,
    VARIABLE_ALIASES,
    VARIABLE_CATEGORIES,
    PolityTimeSeries,
    SeshatDB,
    TimeSeriesPoint,
)
from cliodynamics.data.api_client import (
    APICache,
    PolityInfo,
    SeshatAPIAuthenticationError,
    SeshatAPIClient,
    SeshatAPIConnectionError,
    SeshatAPIError,
    SeshatAPINotInstalledError,
)
from cliodynamics.data.download import download_and_extract, get_zenodo_download_url
from cliodynamics.data.download_polaris import download_polaris
from cliodynamics.data.parser import (
    DataQuality,
    ParsedValue,
    Polity,
    SeshatDataset,
    get_variable_summary,
    load_equinox,
    load_polaris,
    load_polaris_threads,
    load_seshat_csv,
    load_seshat_excel,
    parse_seshat_dataframe,
    parse_value,
)
from cliodynamics.data.polars_adapter import (
    DataFrameFormat,
    LazyPolarsAdapter,
    PolarsAdapter,
    PolarsAdapterConfig,
    get_adapter,
    pandas_to_polars,
    polars_to_pandas,
    read_csv_polars,
    read_excel_polars,
)

__all__ = [
    # Access layer classes (pandas-based, legacy)
    "SeshatDB",
    "PolityTimeSeries",
    "TimeSeriesPoint",
    "VARIABLE_ALIASES",
    "VARIABLE_CATEGORIES",
    "DATASET_EQUINOX",
    "DATASET_POLARIS",
    # Polars adapter classes (recommended)
    "PolarsAdapter",
    "LazyPolarsAdapter",
    "PolarsAdapterConfig",
    "DataFrameFormat",
    # Polars conversion utilities
    "pandas_to_polars",
    "polars_to_pandas",
    "read_csv_polars",
    "read_excel_polars",
    "get_adapter",
    # API client classes
    "SeshatAPIClient",
    "PolityInfo",
    "APICache",
    "SeshatAPIError",
    "SeshatAPINotInstalledError",
    "SeshatAPIAuthenticationError",
    "SeshatAPIConnectionError",
    # Download functions
    "download_and_extract",
    "get_zenodo_download_url",
    "download_polaris",
    # Parser classes
    "DataQuality",
    "ParsedValue",
    "Polity",
    "SeshatDataset",
    # Parser functions
    "get_variable_summary",
    "load_equinox",
    "load_polaris",
    "load_polaris_threads",
    "load_seshat_csv",
    "load_seshat_excel",
    "parse_seshat_dataframe",
    "parse_value",
]
