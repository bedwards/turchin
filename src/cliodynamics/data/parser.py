"""
Parse Seshat Equinox-2020 dataset into Python data structures.

Schema Overview
---------------
The Seshat Global History Databank tracks ~100 variables for 373 polities
(societies) from 9600 BCE to 1900 CE.

Key Columns:
- NGA (Natural Geographic Area): Geographic region identifier
- Polity: Political entity identifier (e.g., "RomPrin" for Roman Principate)
- Original_name: Full polity name
- Start/End: Temporal bounds in CE years (negative = BCE)
- Variable columns: Various social complexity indicators

Temporal Encoding:
- Years are in CE format (negative values = BCE)
- E.g., -500 means 500 BCE, 1200 means 1200 CE
- Date ranges may be expressed as intervals: "300-600" meaning uncertain

Value Conventions:
- "present" / "absent": Binary presence/absence
- "inferred present" / "inferred absent": Expert inference
- "unknown" / "uncoded": Missing data
- Numeric values: Population sizes, territory area, etc.
- "[X-Y]": Range indicating uncertainty (e.g., "[100-500]")
- "suspected unknown": Searched but not found
- Empty cells: Not coded

Uncertainty Markers:
- "inferred": Value inferred from indirect evidence
- "disputed": Experts disagree on value
- "uncertain": Low confidence in value
- Range notation "[X-Y]" or "X-Y": Numeric uncertainty
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Quality indicators for Seshat data values."""

    PRESENT = "present"
    ABSENT = "absent"
    INFERRED_PRESENT = "inferred_present"
    INFERRED_ABSENT = "inferred_absent"
    UNKNOWN = "unknown"
    UNCODED = "uncoded"
    DISPUTED = "disputed"
    SUSPECTED_UNKNOWN = "suspected_unknown"


@dataclass
class ParsedValue:
    """
    A parsed Seshat data value with uncertainty information.

    Attributes:
        raw: Original string value from the dataset
        value: Parsed numeric value or None
        value_min: Lower bound of range (for uncertain values)
        value_max: Upper bound of range (for uncertain values)
        quality: Data quality indicator
        is_inferred: Whether value was inferred
        is_disputed: Whether value is disputed by experts
        notes: Any additional notes or qualifiers
    """

    raw: str
    value: float | None = None
    value_min: float | None = None
    value_max: float | None = None
    quality: DataQuality = DataQuality.PRESENT
    is_inferred: bool = False
    is_disputed: bool = False
    notes: str = ""


@dataclass
class Polity:
    """
    A Seshat polity (political entity).

    Attributes:
        id: Short identifier (e.g., "RomPrin")
        name: Full name (e.g., "Roman Principate")
        nga: Natural Geographic Area
        start_year: Start year (CE, negative for BCE)
        end_year: End year (CE, negative for BCE)
        variables: Dictionary of variable name to parsed values
    """

    id: str
    name: str
    nga: str
    start_year: int
    end_year: int
    variables: dict[str, list[ParsedValue]] = field(default_factory=dict)


@dataclass
class SeshatDataset:
    """
    Container for a parsed Seshat dataset.

    Attributes:
        polities: List of polity records
        variables: List of variable names
        df: Raw DataFrame for custom queries
        metadata: Dataset metadata
    """

    polities: list[Polity]
    variables: list[str]
    df: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


# Regex patterns for parsing values
RANGE_PATTERN = re.compile(r"\[?\s*(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)\s*\]?")
NUMBER_PATTERN = re.compile(r"^(-?\d+(?:\.\d+)?)\s*$")
INFERRED_PATTERN = re.compile(r"inferred\s+(present|absent)", re.IGNORECASE)


def parse_value(raw: str | float | None) -> ParsedValue:
    """
    Parse a raw Seshat value into a structured ParsedValue.

    Args:
        raw: The raw value from the dataset (string, number, or None)

    Returns:
        ParsedValue with parsed numeric value and quality indicators

    Examples:
        >>> parse_value("present")
        ParsedValue(raw='present', value=1.0, quality=DataQuality.PRESENT)

        >>> parse_value("[100-500]")
        ParsedValue(raw='[100-500]', value=300.0, value_min=100.0, value_max=500.0)

        >>> parse_value("unknown")
        ParsedValue(raw='unknown', value=None, quality=DataQuality.UNKNOWN)
    """
    # Handle None and NaN
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ParsedValue(raw="", quality=DataQuality.UNCODED)

    # Convert to string for processing
    if isinstance(raw, (int, float)):
        return ParsedValue(
            raw=str(raw),
            value=float(raw),
            quality=DataQuality.PRESENT,
        )

    raw_str = str(raw).strip()

    if not raw_str:
        return ParsedValue(raw="", quality=DataQuality.UNCODED)

    raw_lower = raw_str.lower()

    # Check for quality indicators
    if raw_lower == "present":
        return ParsedValue(raw=raw_str, value=1.0, quality=DataQuality.PRESENT)

    if raw_lower == "absent":
        return ParsedValue(raw=raw_str, value=0.0, quality=DataQuality.ABSENT)

    if raw_lower == "unknown":
        return ParsedValue(raw=raw_str, quality=DataQuality.UNKNOWN)

    if raw_lower == "uncoded":
        return ParsedValue(raw=raw_str, quality=DataQuality.UNCODED)

    if raw_lower == "suspected unknown":
        return ParsedValue(raw=raw_str, quality=DataQuality.SUSPECTED_UNKNOWN)

    # Check for inferred values
    inferred_match = INFERRED_PATTERN.search(raw_str)
    if inferred_match:
        status = inferred_match.group(1).lower()
        quality = (
            DataQuality.INFERRED_PRESENT
            if status == "present"
            else DataQuality.INFERRED_ABSENT
        )
        return ParsedValue(
            raw=raw_str,
            value=1.0 if status == "present" else 0.0,
            quality=quality,
            is_inferred=True,
        )

    # Check for range values
    range_match = RANGE_PATTERN.search(raw_str)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        midpoint = (low + high) / 2
        return ParsedValue(
            raw=raw_str,
            value=midpoint,
            value_min=low,
            value_max=high,
            quality=DataQuality.PRESENT,
        )

    # Check for simple numeric value
    number_match = NUMBER_PATTERN.match(raw_str)
    if number_match:
        return ParsedValue(
            raw=raw_str,
            value=float(number_match.group(1)),
            quality=DataQuality.PRESENT,
        )

    # Check for disputed values
    if "disputed" in raw_lower:
        return ParsedValue(
            raw=raw_str,
            quality=DataQuality.DISPUTED,
            is_disputed=True,
        )

    # Return as-is with notes
    return ParsedValue(
        raw=raw_str,
        quality=DataQuality.PRESENT,
        notes=raw_str,
    )


def load_seshat_excel(
    filepath: Path | str,
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    """
    Load a Seshat Excel file into a DataFrame.

    Args:
        filepath: Path to the Excel file
        sheet_name: Sheet name or index to load (default: first sheet)

    Returns:
        DataFrame with the raw Seshat data

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Seshat data file not found: {filepath}")

    logger.info(f"Loading Seshat data from: {filepath}")
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    return df


def load_seshat_csv(filepath: Path | str) -> pd.DataFrame:
    """
    Load a Seshat CSV file into a DataFrame.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame with the raw Seshat data

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Seshat data file not found: {filepath}")

    logger.info(f"Loading Seshat CSV from: {filepath}")

    # Try different encodings
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not read CSV file with any supported encoding: {filepath}")


def parse_seshat_dataframe(
    df: pd.DataFrame,
    polity_col: str = "Polity",
    name_col: str = "Original_name",
    nga_col: str = "NGA",
    start_col: str = "Start",
    end_col: str = "End",
) -> SeshatDataset:
    """
    Parse a Seshat DataFrame into structured Polity objects.

    Args:
        df: Raw DataFrame from load_seshat_excel or load_seshat_csv
        polity_col: Column name for polity ID
        name_col: Column name for polity name
        nga_col: Column name for Natural Geographic Area
        start_col: Column name for start year
        end_col: Column name for end year

    Returns:
        SeshatDataset with parsed polities and metadata

    Raises:
        ValueError: If required columns are missing
    """
    # Verify required columns exist
    required_cols = [polity_col, name_col, nga_col, start_col, end_col]

    # Try to find columns with similar names (case-insensitive)
    col_mapping = {}
    for req_col in required_cols:
        if req_col in df.columns:
            col_mapping[req_col] = req_col
        else:
            # Look for case-insensitive match
            for actual_col in df.columns:
                if actual_col.lower() == req_col.lower():
                    col_mapping[req_col] = actual_col
                    break

    # Check for still-missing columns
    still_missing = [col for col in required_cols if col not in col_mapping]
    if still_missing:
        available = ", ".join(df.columns[:10].tolist())
        raise ValueError(
            f"Required columns not found: {still_missing}. "
            f"Available columns (first 10): {available}..."
        )

    # Identify variable columns (everything except metadata columns)
    metadata_cols = set(col_mapping.values())
    variable_cols = [col for col in df.columns if col not in metadata_cols]

    logger.info(f"Found {len(variable_cols)} variable columns")

    # Parse each row into a Polity
    polities = []
    for _, row in df.iterrows():
        polity_id = str(row[col_mapping[polity_col]])
        name = str(row[col_mapping[name_col]])
        nga = str(row[col_mapping[nga_col]])

        # Parse years
        start_raw = row[col_mapping[start_col]]
        end_raw = row[col_mapping[end_col]]

        try:
            start_year = int(float(start_raw)) if pd.notna(start_raw) else 0
        except (ValueError, TypeError):
            start_year = 0

        try:
            end_year = int(float(end_raw)) if pd.notna(end_raw) else 0
        except (ValueError, TypeError):
            end_year = 0

        # Parse variables
        variables: dict[str, list[ParsedValue]] = {}
        for var_col in variable_cols:
            raw_value = row[var_col]
            parsed = parse_value(raw_value)
            variables[var_col] = [parsed]

        polity = Polity(
            id=polity_id,
            name=name,
            nga=nga,
            start_year=start_year,
            end_year=end_year,
            variables=variables,
        )
        polities.append(polity)

    logger.info(f"Parsed {len(polities)} polities")

    return SeshatDataset(
        polities=polities,
        variables=variable_cols,
        df=df,
        metadata={
            "source": "Seshat Equinox-2020",
            "doi": "10.5281/zenodo.6642229",
            "n_polities": len(polities),
            "n_variables": len(variable_cols),
        },
    )


def load_equinox(data_dir: Path | str | None = None) -> SeshatDataset:
    """
    Load the Seshat Equinox-2020 dataset.

    This is the main entry point for loading Seshat data. It looks for
    the downloaded Equinox dataset and parses it into structured format.

    Args:
        data_dir: Directory containing the downloaded data.
                  Defaults to ./data/seshat/

    Returns:
        SeshatDataset with all polities and variables

    Raises:
        FileNotFoundError: If dataset is not downloaded

    Example:
        >>> from cliodynamics.data.parser import load_equinox
        >>> dataset = load_equinox()
        >>> print(f"Loaded {len(dataset.polities)} polities")
        Loaded 373 polities
    """
    if data_dir is None:
        # Look in default location relative to project root
        module_path = Path(__file__).resolve()
        src_path = module_path.parent.parent.parent
        project_root = src_path.parent
        data_dir = project_root / "data" / "seshat"
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Seshat data directory not found: {data_dir}. "
            "Run 'python -m cliodynamics.data.download' first."
        )

    # Look for Excel file first, then CSV
    excel_files = list(data_dir.glob("*.xlsx"))
    csv_files = list(data_dir.glob("*.csv"))

    if excel_files:
        filepath = excel_files[0]
        df = load_seshat_excel(filepath)
    elif csv_files:
        filepath = csv_files[0]
        df = load_seshat_csv(filepath)
    else:
        raise FileNotFoundError(
            f"No Excel or CSV files found in {data_dir}. "
            "Run 'python -m cliodynamics.data.download' first."
        )

    return parse_seshat_dataframe(df)


def get_variable_summary(dataset: SeshatDataset, variable: str) -> dict[str, Any]:
    """
    Get summary statistics for a variable across all polities.

    Args:
        dataset: Loaded SeshatDataset
        variable: Variable name to summarize

    Returns:
        Dictionary with count, mean, min, max, and quality distribution

    Raises:
        KeyError: If variable is not in dataset
    """
    if variable not in dataset.variables:
        raise KeyError(f"Variable '{variable}' not found in dataset")

    values = []
    quality_counts: dict[str, int] = {}

    for polity in dataset.polities:
        if variable in polity.variables:
            for pv in polity.variables[variable]:
                quality_name = pv.quality.value
                quality_counts[quality_name] = quality_counts.get(quality_name, 0) + 1

                if pv.value is not None:
                    values.append(pv.value)

    return {
        "variable": variable,
        "n_values": len(values),
        "n_polities": len(dataset.polities),
        "mean": sum(values) / len(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "quality_distribution": quality_counts,
    }
