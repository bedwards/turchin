"""
Data access module for Seshat and other historical datasets.

This module provides tools for downloading, parsing, and querying the Seshat
Global History Databank, supporting both local files (Equinox-2020) and the
live Seshat API (Polaris-2025+).

Key Classes:
    SeshatDB: High-level query interface for local Seshat data
    SeshatAPIClient: Client for querying the live Seshat API
    SeshatDataset: Container for parsed Seshat data
    Polity: A political entity with its variables
    ParsedValue: A parsed data value with uncertainty info
    DataQuality: Enum for data quality indicators

Key Functions:
    download_and_extract: Download Seshat dataset from Zenodo
    load_equinox: Load and parse the Seshat Equinox-2020 dataset
    parse_value: Parse individual Seshat data values

Usage (local data):
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
    >>>
    >>> # List available polities
    >>> db.list_polities()

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
from cliodynamics.data.parser import (
    DataQuality,
    ParsedValue,
    Polity,
    SeshatDataset,
    get_variable_summary,
    load_equinox,
    load_seshat_csv,
    load_seshat_excel,
    parse_seshat_dataframe,
    parse_value,
)

__all__ = [
    # Access layer classes
    "SeshatDB",
    "PolityTimeSeries",
    "TimeSeriesPoint",
    "VARIABLE_ALIASES",
    "VARIABLE_CATEGORIES",
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
    # Parser classes
    "DataQuality",
    "ParsedValue",
    "Polity",
    "SeshatDataset",
    # Parser functions
    "get_variable_summary",
    "load_equinox",
    "load_seshat_csv",
    "load_seshat_excel",
    "parse_seshat_dataframe",
    "parse_value",
]
