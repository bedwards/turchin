"""
Data access layer with query interface for Seshat data.

This module provides a clean API for querying Seshat data by polity,
time range, and variable category.

Example:
    >>> from cliodynamics.data import SeshatDB
    >>> db = SeshatDB("data/seshat/")
    >>>
    >>> # Get all data for a polity
    >>> rome = db.get_polity("RomPrin")
    >>>
    >>> # Get specific variables across polities
    >>> pop_data = db.query(
    ...     variables=["Population"],
    ...     time_range=(-500, 500),
    ...     regions=["Italy"]
    ... )
    >>>
    >>> # List available polities, variables
    >>> db.list_polities()
    >>> db.list_variables(category="social_complexity")
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from cliodynamics.data.parser import (
    DataQuality,
    Polity,
    SeshatDataset,
    load_equinox,
    load_seshat_csv,
    load_seshat_excel,
    parse_seshat_dataframe,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Variable categories based on Seshat codebook
# These group related variables for easier querying
VARIABLE_CATEGORIES: dict[str, list[str]] = {
    "social_complexity": [
        "PolPop",
        "Polity_territory",
        "Pop_of_largest_settlement",
        "Hierarchical_complexity",
        "Government_buildings",
        "Irrigation_systems",
        "Markets",
        "Roads",
        "Bridges",
        "Canals",
        "Ports",
    ],
    "military": [
        "Professional_military_officers",
        "Professional_soldiers",
        "Military_technology",
        "Fortifications",
        "Standing_army",
    ],
    "information": [
        "Writing_system",
        "Phonetic_writing",
        "Script",
        "Written_records",
        "Literature",
        "Lists",
        "Calendar",
        "Sacred_texts",
    ],
    "religion": [
        "High_gods",
        "Supernatural_enforcement_of_reciprocity",
        "Supernatural_enforcement_of_fairness",
        "Human_sacrifice",
        "Animal_sacrifice",
    ],
    "economy": [
        "Debt_and_credit_structures",
        "Stores_of_wealth",
        "Source_of_support",
        "Foreign_coins",
        "Indigenous_coins",
        "Paper_currency",
    ],
    "politics": [
        "Professional_lawyers",
        "Formal_legal_code",
        "Judges",
        "Courts",
        "Professional_bureaucrats",
        "Merit_promotion",
        "Examination_system",
    ],
}

# Normalized variable name mappings
# Maps common/simplified names to actual Seshat column names
VARIABLE_ALIASES: dict[str, str] = {
    "population": "PolPop",
    "territory": "Polity_territory",
    "largest_settlement": "Pop_of_largest_settlement",
    "hierarchy": "Hierarchical_complexity",
    "writing": "Writing_system",
    "military": "Professional_military_officers",
}


@dataclass
class TimeSeriesPoint:
    """
    A single point in a time series.

    Attributes:
        year: The year (CE, negative for BCE)
        value: The numeric value
        value_min: Lower bound of uncertainty range
        value_max: Upper bound of uncertainty range
        quality: Data quality indicator
        is_interpolated: Whether this point was interpolated
    """

    year: int
    value: float | None
    value_min: float | None = None
    value_max: float | None = None
    quality: DataQuality = DataQuality.PRESENT
    is_interpolated: bool = False


@dataclass
class PolityTimeSeries:
    """
    Time series data for a single polity.

    Attributes:
        polity_id: Short identifier
        polity_name: Full name
        nga: Natural Geographic Area
        start_year: Start year of polity
        end_year: End year of polity
        variables: Dictionary mapping variable names to time series
    """

    polity_id: str
    polity_name: str
    nga: str
    start_year: int
    end_year: int
    variables: dict[str, list[TimeSeriesPoint]]


class SeshatDB:
    """
    Query interface for Seshat Global History Databank.

    Provides methods for querying polities, variables, and time ranges,
    with support for time series interpolation and DataFrame export.

    Args:
        data_path: Path to directory containing Seshat data files,
                   or None to use default location.

    Example:
        >>> db = SeshatDB("data/seshat/")
        >>> rome = db.get_polity("RomPrin")
        >>> print(f"Roman Principate: {rome.start_year} to {rome.end_year}")
    """

    def __init__(self, data_path: Path | str | None = None) -> None:
        """Initialize SeshatDB with path to data directory."""
        self._data_path = Path(data_path) if data_path else None
        self._dataset: SeshatDataset | None = None
        self._polity_index: dict[str, Polity] = {}
        self._nga_index: dict[str, list[str]] = {}

    def _ensure_loaded(self) -> None:
        """Load dataset if not already loaded."""
        if self._dataset is not None:
            return

        logger.info("Loading Seshat dataset...")

        if self._data_path is not None:
            # Load from specified path
            excel_files = list(self._data_path.glob("*.xlsx"))
            csv_files = list(self._data_path.glob("*.csv"))

            if excel_files:
                df = load_seshat_excel(excel_files[0])
                self._dataset = parse_seshat_dataframe(df)
            elif csv_files:
                df = load_seshat_csv(csv_files[0])
                self._dataset = parse_seshat_dataframe(df)
            else:
                raise FileNotFoundError(
                    f"No Excel or CSV files found in {self._data_path}"
                )
        else:
            # Use default location via load_equinox
            self._dataset = load_equinox()

        # Build indices
        self._build_indices()
        logger.info(
            f"Loaded {len(self._dataset.polities)} polities, "
            f"{len(self._dataset.variables)} variables"
        )

    def _build_indices(self) -> None:
        """Build lookup indices for fast querying."""
        if self._dataset is None:
            return

        # Index by polity ID
        for polity in self._dataset.polities:
            self._polity_index[polity.id] = polity

            # Index by NGA (region)
            if polity.nga not in self._nga_index:
                self._nga_index[polity.nga] = []
            self._nga_index[polity.nga].append(polity.id)

    def _normalize_variable_name(self, name: str) -> str:
        """
        Normalize a variable name to its canonical form.

        Args:
            name: Variable name (possibly an alias)

        Returns:
            Canonical variable name
        """
        return VARIABLE_ALIASES.get(name.lower(), name)

    def _get_variables_for_category(self, category: str) -> list[str]:
        """
        Get list of variables for a category.

        Args:
            category: Category name (e.g., "social_complexity")

        Returns:
            List of variable names in that category

        Raises:
            ValueError: If category is not recognized
        """
        if category not in VARIABLE_CATEGORIES:
            available = ", ".join(VARIABLE_CATEGORIES.keys())
            raise ValueError(f"Unknown category '{category}'. Available: {available}")
        return VARIABLE_CATEGORIES[category]

    @property
    def dataset(self) -> SeshatDataset:
        """Get the underlying SeshatDataset."""
        self._ensure_loaded()
        assert self._dataset is not None
        return self._dataset

    def get_polity(self, polity_id: str) -> PolityTimeSeries:
        """
        Get all data for a specific polity.

        Args:
            polity_id: Short identifier (e.g., "RomPrin")

        Returns:
            PolityTimeSeries with all variables

        Raises:
            KeyError: If polity not found

        Example:
            >>> db = SeshatDB()
            >>> rome = db.get_polity("RomPrin")
            >>> print(f"{rome.polity_name}: {rome.start_year} to {rome.end_year}")
        """
        self._ensure_loaded()

        if polity_id not in self._polity_index:
            raise KeyError(f"Polity '{polity_id}' not found")

        polity = self._polity_index[polity_id]

        # Convert to time series format
        variables: dict[str, list[TimeSeriesPoint]] = {}
        for var_name, parsed_values in polity.variables.items():
            points = []
            for pv in parsed_values:
                # Use polity start year as the year for single values
                point = TimeSeriesPoint(
                    year=polity.start_year,
                    value=pv.value,
                    value_min=pv.value_min,
                    value_max=pv.value_max,
                    quality=pv.quality,
                    is_interpolated=False,
                )
                points.append(point)
            variables[var_name] = points

        return PolityTimeSeries(
            polity_id=polity.id,
            polity_name=polity.name,
            nga=polity.nga,
            start_year=polity.start_year,
            end_year=polity.end_year,
            variables=variables,
        )

    def query(
        self,
        variables: Sequence[str] | None = None,
        polities: Sequence[str] | None = None,
        time_range: tuple[int, int] | None = None,
        regions: Sequence[str] | None = None,
        category: str | None = None,
        interpolate: bool = False,
        interpolate_step: int = 100,
    ) -> pd.DataFrame:
        """
        Query Seshat data with flexible filtering.

        Args:
            variables: List of variable names to include.
                       Supports aliases (e.g., "population" -> "PolPop").
            polities: List of polity IDs to include. If None, include all.
            time_range: Tuple of (start_year, end_year) to filter by.
                        Uses CE years (negative = BCE).
            regions: List of NGA (Natural Geographic Area) names to filter by.
            category: Variable category (e.g., "social_complexity").
                      If specified, includes all variables in that category.
            interpolate: If True, interpolate values for missing periods.
            interpolate_step: Year step size for interpolation (default: 100).

        Returns:
            DataFrame with columns:
            - polity_id: Short identifier
            - polity_name: Full name
            - nga: Natural Geographic Area
            - start_year: Polity start year
            - end_year: Polity end year
            - year: Year of observation (if interpolated)
            - [variable columns]: One column per requested variable

        Raises:
            ValueError: If no matching data found

        Example:
            >>> db = SeshatDB()
            >>> df = db.query(
            ...     variables=["population", "territory"],
            ...     time_range=(-500, 500),
            ...     regions=["Italy"]
            ... )
        """
        self._ensure_loaded()
        assert self._dataset is not None

        # Resolve variable names
        resolved_variables: list[str] = []
        if category:
            resolved_variables.extend(self._get_variables_for_category(category))
        if variables:
            for var in variables:
                normalized = self._normalize_variable_name(var)
                if normalized not in resolved_variables:
                    resolved_variables.append(normalized)

        # If no variables specified, use all
        if not resolved_variables:
            resolved_variables = list(self._dataset.variables)

        # Filter polities
        matching_polities: list[Polity] = []

        for polity in self._dataset.polities:
            # Filter by polity ID
            if polities and polity.id not in polities:
                continue

            # Filter by region
            if regions and polity.nga not in regions:
                continue

            # Filter by time range
            if time_range:
                start, end = time_range
                # Check for overlap: polity overlaps if it doesn't end before
                # our start or start after our end
                if polity.end_year < start or polity.start_year > end:
                    continue

            matching_polities.append(polity)

        if not matching_polities:
            return pd.DataFrame()

        # Build result DataFrame
        rows: list[dict[str, Any]] = []

        for polity in matching_polities:
            if interpolate:
                # Generate time series with interpolation
                rows.extend(
                    self._interpolate_polity(
                        polity, resolved_variables, time_range, interpolate_step
                    )
                )
            else:
                # Single row per polity
                row: dict[str, Any] = {
                    "polity_id": polity.id,
                    "polity_name": polity.name,
                    "nga": polity.nga,
                    "start_year": polity.start_year,
                    "end_year": polity.end_year,
                }

                # Add variable values
                for var_name in resolved_variables:
                    if var_name in polity.variables:
                        pv_list = polity.variables[var_name]
                        if pv_list and pv_list[0].value is not None:
                            row[var_name] = pv_list[0].value
                        else:
                            row[var_name] = None
                    else:
                        row[var_name] = None

                rows.append(row)

        df = pd.DataFrame(rows)

        # Reorder columns
        base_cols = ["polity_id", "polity_name", "nga", "start_year", "end_year"]
        if interpolate:
            base_cols.append("year")
        var_cols = [c for c in df.columns if c not in base_cols]
        df = df[base_cols + sorted(var_cols)]

        return df

    def _interpolate_polity(
        self,
        polity: Polity,
        variables: list[str],
        time_range: tuple[int, int] | None,
        step: int,
    ) -> list[dict[str, Any]]:
        """
        Generate interpolated time series rows for a polity.

        Args:
            polity: The polity to interpolate
            variables: List of variable names
            time_range: Optional time range to constrain output
            step: Year step size

        Returns:
            List of row dictionaries
        """
        # Determine year range
        start_year = polity.start_year
        end_year = polity.end_year

        if time_range:
            start_year = max(start_year, time_range[0])
            end_year = min(end_year, time_range[1])

        if end_year < start_year:
            return []

        # Generate years
        years = list(range(start_year, end_year + 1, step))
        if years and years[-1] != end_year:
            years.append(end_year)

        rows = []
        for year in years:
            row: dict[str, Any] = {
                "polity_id": polity.id,
                "polity_name": polity.name,
                "nga": polity.nga,
                "start_year": polity.start_year,
                "end_year": polity.end_year,
                "year": year,
            }

            # For each variable, interpolate or use constant value
            for var_name in variables:
                if var_name in polity.variables:
                    pv_list = polity.variables[var_name]
                    if pv_list and pv_list[0].value is not None:
                        # Use constant value (Seshat typically has one value per polity)
                        row[var_name] = pv_list[0].value
                    else:
                        row[var_name] = None
                else:
                    row[var_name] = None

            rows.append(row)

        return rows

    def list_polities(
        self,
        region: str | None = None,
        time_range: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        """
        List available polities with metadata.

        Args:
            region: Filter by NGA (Natural Geographic Area)
            time_range: Filter by time range (start_year, end_year)

        Returns:
            DataFrame with columns: polity_id, polity_name, nga, start_year, end_year

        Example:
            >>> db = SeshatDB()
            >>> italian_polities = db.list_polities(region="Italy")
        """
        self._ensure_loaded()
        assert self._dataset is not None

        rows = []
        for polity in self._dataset.polities:
            # Filter by region
            if region and polity.nga != region:
                continue

            # Filter by time range
            if time_range:
                start, end = time_range
                if polity.end_year < start or polity.start_year > end:
                    continue

            rows.append(
                {
                    "polity_id": polity.id,
                    "polity_name": polity.name,
                    "nga": polity.nga,
                    "start_year": polity.start_year,
                    "end_year": polity.end_year,
                }
            )

        return pd.DataFrame(rows)

    def list_variables(
        self,
        category: str | None = None,
        search: str | None = None,
    ) -> pd.DataFrame:
        """
        List available variables with metadata.

        Args:
            category: Filter by category (e.g., "social_complexity")
            search: Search for variables containing this substring

        Returns:
            DataFrame with columns: variable_name, category

        Example:
            >>> db = SeshatDB()
            >>> social_vars = db.list_variables(category="social_complexity")
        """
        self._ensure_loaded()
        assert self._dataset is not None

        # Build category lookup
        var_to_category: dict[str, str] = {}
        for cat_name, var_list in VARIABLE_CATEGORIES.items():
            for var in var_list:
                var_to_category[var] = cat_name

        rows = []
        for var_name in self._dataset.variables:
            # Filter by category
            if category:
                if var_name not in VARIABLE_CATEGORIES.get(category, []):
                    continue

            # Filter by search string
            if search and search.lower() not in var_name.lower():
                continue

            rows.append(
                {
                    "variable_name": var_name,
                    "category": var_to_category.get(var_name, "other"),
                }
            )

        return pd.DataFrame(rows)

    def list_regions(self) -> list[str]:
        """
        List available Natural Geographic Areas (regions).

        Returns:
            List of NGA names

        Example:
            >>> db = SeshatDB()
            >>> regions = db.list_regions()
        """
        self._ensure_loaded()
        return sorted(self._nga_index.keys())

    def list_categories(self) -> list[str]:
        """
        List available variable categories.

        Returns:
            List of category names
        """
        return sorted(VARIABLE_CATEGORIES.keys())

    def get_cross_polity_comparison(
        self,
        variable: str,
        polities: Sequence[str] | None = None,
        regions: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get a variable across multiple polities for comparison.

        Args:
            variable: Variable name (supports aliases)
            polities: List of polity IDs to include
            regions: List of regions to include

        Returns:
            DataFrame with columns: polity_id, polity_name, nga, value,
            value_min, value_max

        Example:
            >>> db = SeshatDB()
            >>> pop_comparison = db.get_cross_polity_comparison("population")
        """
        normalized_var = self._normalize_variable_name(variable)
        df = self.query(
            variables=[normalized_var],
            polities=list(polities) if polities else None,
            regions=list(regions) if regions else None,
        )

        if df.empty:
            return df

        # Rename the variable column to 'value' for clarity
        if normalized_var in df.columns:
            df = df.rename(columns={normalized_var: "value"})

        # Keep only relevant columns
        result_cols = ["polity_id", "polity_name", "nga", "start_year", "end_year"]
        if "value" in df.columns:
            result_cols.append("value")

        return df[result_cols]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export entire dataset as a DataFrame.

        Returns:
            DataFrame with all polities and variables

        Example:
            >>> db = SeshatDB()
            >>> df = db.to_dataframe()
            >>> df.to_csv("seshat_export.csv")
        """
        self._ensure_loaded()
        assert self._dataset is not None
        return self._dataset.df.copy()

    def interpolate_time_series(
        self,
        polity_id: str,
        variable: str,
        start_year: int | None = None,
        end_year: int | None = None,
        method: str = "linear",
        step: int = 100,
    ) -> pd.DataFrame:
        """
        Generate interpolated time series for a polity variable.

        Args:
            polity_id: Polity identifier
            variable: Variable name
            start_year: Start of time series (default: polity start)
            end_year: End of time series (default: polity end)
            method: Interpolation method ("linear", "constant")
            step: Year step size

        Returns:
            DataFrame with columns: year, value, is_interpolated

        Example:
            >>> db = SeshatDB()
            >>> ts = db.interpolate_time_series("RomPrin", "population")
        """
        self._ensure_loaded()

        if polity_id not in self._polity_index:
            raise KeyError(f"Polity '{polity_id}' not found")

        polity = self._polity_index[polity_id]
        normalized_var = self._normalize_variable_name(variable)

        # Determine year range
        if start_year is None:
            start_year = polity.start_year
        if end_year is None:
            end_year = polity.end_year

        # Get the value
        value = None
        if normalized_var in polity.variables:
            pv_list = polity.variables[normalized_var]
            if pv_list and pv_list[0].value is not None:
                value = pv_list[0].value

        # Generate time series
        years = list(range(start_year, end_year + 1, step))
        if years and years[-1] != end_year:
            years.append(end_year)

        rows = []
        for year in years:
            rows.append(
                {
                    "year": year,
                    "value": value,
                    "is_interpolated": True,  # All points are interpolated for now
                }
            )

        return pd.DataFrame(rows)

    def get_summary_statistics(
        self,
        variable: str,
        polities: Sequence[str] | None = None,
        regions: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """
        Get summary statistics for a variable.

        Args:
            variable: Variable name
            polities: Optional list of polity IDs to include
            regions: Optional list of regions to include

        Returns:
            Dictionary with count, mean, std, min, max, median

        Example:
            >>> db = SeshatDB()
            >>> stats = db.get_summary_statistics("population")
        """
        df = self.query(
            variables=[variable],
            polities=list(polities) if polities else None,
            regions=list(regions) if regions else None,
        )

        normalized_var = self._normalize_variable_name(variable)

        if df.empty or normalized_var not in df.columns:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "median": None,
            }

        values = df[normalized_var].dropna()

        return {
            "count": len(values),
            "mean": float(values.mean()) if len(values) > 0 else None,
            "std": float(values.std()) if len(values) > 1 else None,
            "min": float(values.min()) if len(values) > 0 else None,
            "max": float(values.max()) if len(values) > 0 else None,
            "median": float(values.median()) if len(values) > 0 else None,
        }
