"""
Data access module for Seshat and other historical datasets.

This module provides tools for downloading, parsing, and querying the Seshat
Global History Databank (Equinox-2020 release).

Key Classes:
    SeshatDB: High-level query interface for Seshat data
    SeshatDataset: Container for parsed Seshat data
    Polity: A political entity with its variables
    ParsedValue: A parsed data value with uncertainty info
    DataQuality: Enum for data quality indicators

Key Functions:
    download_and_extract: Download Seshat dataset from Zenodo
    load_equinox: Load and parse the Seshat Equinox-2020 dataset
    parse_value: Parse individual Seshat data values

Usage:
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
"""

from cliodynamics.data.access import (
    PolityTimeSeries,
    SeshatDB,
    TimeSeriesPoint,
    VARIABLE_ALIASES,
    VARIABLE_CATEGORIES,
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
