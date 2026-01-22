"""
Data access module for Seshat and other historical datasets.

This module provides tools for downloading and parsing the Seshat
Global History Databank (Equinox-2020 release).

Key Functions:
    download_and_extract: Download Seshat dataset from Zenodo
    load_equinox: Load and parse the Seshat Equinox-2020 dataset
    parse_value: Parse individual Seshat data values

Key Classes:
    SeshatDataset: Container for parsed Seshat data
    Polity: A political entity with its variables
    ParsedValue: A parsed data value with uncertainty info
    DataQuality: Enum for data quality indicators

Usage:
    >>> from cliodynamics.data import download_and_extract, load_equinox
    >>> # First, download the dataset
    >>> download_and_extract()
    >>> # Then load and parse it
    >>> dataset = load_equinox()
    >>> print(f"Loaded {len(dataset.polities)} polities")
"""

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
