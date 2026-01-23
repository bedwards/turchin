"""
Seshat API client wrapper for accessing live Seshat database.

This module provides a Python client for the Seshat Global History Databank API,
offering a compatible interface with the local SeshatDB class for querying
polities, variables, and time ranges.

The official Seshat API client is available at:
https://github.com/Seshat-Global-History-Databank/seshat_api

Usage:
    >>> from cliodynamics.data import SeshatAPIClient
    >>>
    >>> # Connect to Seshat API (no authentication required for read-only access)
    >>> client = SeshatAPIClient()
    >>>
    >>> # Query polities
    >>> polities = client.list_polities(region="Europe")
    >>>
    >>> # Get full data for a polity
    >>> rome = client.get_polity("RomPrin")
    >>>
    >>> # Get specific variables
    >>> pop_data = client.query(
    ...     variables=["PolPop", "PolTerr"],
    ...     time_range=(-500, 500)
    ... )
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default Seshat API base URL
DEFAULT_BASE_URL = "https://seshatdata.com/api"

# Cache settings
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "cliodynamics" / "seshat_api"
DEFAULT_CACHE_TTL = 3600  # 1 hour in seconds


class SeshatAPIError(Exception):
    """Base exception for Seshat API errors."""

    pass


class SeshatAPINotInstalledError(SeshatAPIError):
    """Raised when seshat_api package is not installed."""

    pass


class SeshatAPIAuthenticationError(SeshatAPIError):
    """Raised when authentication fails."""

    pass


class SeshatAPIConnectionError(SeshatAPIError):
    """Raised when connection to API fails."""

    pass


@dataclass
class CacheEntry:
    """A cached API response with expiration."""

    data: Any
    timestamp: float
    ttl: int

    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        return time.time() - self.timestamp > self.ttl


@dataclass
class APICache:
    """
    Simple file-based cache for API responses.

    Stores JSON-serializable responses on disk with TTL expiration.
    """

    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    ttl: int = DEFAULT_CACHE_TTL
    enabled: bool = True

    def __post_init__(self) -> None:
        """Create cache directory if it doesn't exist."""
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        # Use SHA256 hash of key for filename
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{key_hash}.json"

    def get(self, key: str) -> Any | None:
        """
        Get a cached value if it exists and hasn't expired.

        Args:
            key: Cache key (typically the API endpoint + params)

        Returns:
            Cached value or None if not found/expired
        """
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        try:
            with cache_path.open("r") as f:
                cached = json.load(f)

            entry = CacheEntry(
                data=cached["data"],
                timestamp=cached["timestamp"],
                ttl=cached.get("ttl", self.ttl),
            )

            if entry.is_expired():
                cache_path.unlink(missing_ok=True)
                return None

            logger.debug(f"Cache hit for key: {key[:50]}...")
            return entry.data

        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Cache read error for {key}: {e}")
            cache_path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            value: JSON-serializable value to cache
        """
        if not self.enabled:
            return

        cache_path = self._get_cache_path(key)
        try:
            with cache_path.open("w") as f:
                json.dump(
                    {
                        "data": value,
                        "timestamp": time.time(),
                        "ttl": self.ttl,
                    },
                    f,
                )
            logger.debug(f"Cached value for key: {key[:50]}...")
        except (TypeError, OSError) as e:
            logger.warning(f"Cache write error for {key}: {e}")

    def clear(self) -> None:
        """Clear all cached entries."""
        if not self.cache_dir.exists():
            return

        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink(missing_ok=True)

        logger.info("Cache cleared")


@dataclass
class PolityInfo:
    """
    Basic polity information from the API.

    Attributes:
        id: Polity identifier
        name: Full polity name
        start_year: Start year (CE, negative for BCE)
        end_year: End year (CE, negative for BCE)
        nga: Natural Geographic Area
        region: Geographic region
    """

    id: str
    name: str
    start_year: int
    end_year: int
    nga: str = ""
    region: str = ""


class SeshatAPIClient:
    """
    Client for querying the Seshat Global History Databank API.

    Provides a similar interface to SeshatDB for compatibility, but queries
    the live Seshat API instead of local files.

    Args:
        base_url: Base URL for the Seshat API. Defaults to https://seshatdata.com/api
        username: Optional username for authenticated access
        password: Optional password for authenticated access
        cache_enabled: Whether to cache API responses (default: True)
        cache_ttl: Cache time-to-live in seconds (default: 3600)
        cache_dir: Directory for cache files (default: ~/.cache/cliodynamics/seshat_api)

    Example:
        >>> client = SeshatAPIClient()
        >>> polities = client.list_polities(region="Italy")
        >>> print(f"Found {len(polities)} polities in Italy")
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        username: str | None = None,
        password: str | None = None,
        cache_enabled: bool = True,
        cache_ttl: int = DEFAULT_CACHE_TTL,
        cache_dir: Path | str | None = None,
    ) -> None:
        """Initialize the Seshat API client."""
        self._base_url = base_url.rstrip("/")
        self._username = username
        self._password = password
        self._api_client: Any = None
        self._api_available: bool | None = None

        # Initialize cache
        cache_path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self._cache = APICache(
            cache_dir=cache_path,
            ttl=cache_ttl,
            enabled=cache_enabled,
        )

        # Polity and NGA indices (populated on first use)
        self._polity_index: dict[str, PolityInfo] = {}
        self._nga_index: dict[str, list[str]] = {}
        self._region_index: dict[str, list[str]] = {}
        self._indices_loaded = False

    def _ensure_api_client(self) -> Any:
        """
        Ensure the seshat_api client is available and authenticated.

        Returns:
            The SeshatAPI client instance

        Raises:
            SeshatAPINotInstalledError: If seshat_api package is not installed
            SeshatAPIAuthenticationError: If authentication fails
        """
        if self._api_client is not None:
            return self._api_client

        try:
            from seshat_api import SeshatAPI
        except ImportError as e:
            raise SeshatAPINotInstalledError(
                "The seshat_api package is not installed. "
                "Install it with: pip install git+https://github.com/"
                "Seshat-Global-History-Databank/seshat_api.git"
            ) from e

        try:
            self._api_client = SeshatAPI(
                base_url=self._base_url,
                username=self._username,
                password=self._password,
            )
            self._api_available = True
            logger.info(f"Connected to Seshat API at {self._base_url}")
            return self._api_client

        except Exception as e:
            self._api_available = False
            if "403" in str(e) or "401" in str(e):
                raise SeshatAPIAuthenticationError(f"Authentication failed: {e}") from e
            raise SeshatAPIConnectionError(
                f"Failed to connect to Seshat API: {e}"
            ) from e

    def _get_cache_key(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> str:
        """Generate a cache key for an API request."""
        key_parts = [self._base_url, endpoint]
        if params:
            # Sort params for consistent key generation
            sorted_params = sorted(params.items())
            key_parts.append(json.dumps(sorted_params))
        return "|".join(key_parts)

    def _fetch_endpoint(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Fetch data from an API endpoint with optional caching.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            use_cache: Whether to use cache for this request

        Returns:
            List of result dictionaries
        """
        cache_key = self._get_cache_key(endpoint, params)

        # Check cache first
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Fetch from API
        client = self._ensure_api_client()
        try:
            response = client.get(endpoint, params=params)

            # Handle paginated responses
            if isinstance(response, dict) and "results" in response:
                results = response["results"]
            elif isinstance(response, list):
                results = response
            else:
                results = [response] if response else []

            # Cache the results
            if use_cache:
                self._cache.set(cache_key, results)

            return results

        except Exception as e:
            logger.error(f"API request failed: {endpoint} - {e}")
            raise SeshatAPIConnectionError(f"API request failed: {e}") from e

    def _load_indices(self) -> None:
        """Load polity indices for efficient querying."""
        if self._indices_loaded:
            return

        logger.info("Loading polity indices from API...")

        try:
            # Try to import and use the Polities class from seshat_api
            from seshat_api.core import Polities

            client = self._ensure_api_client()
            polities_api = Polities(client)

            for polity_data in polities_api.get_all():
                polity_id = polity_data.get("name", "")
                if not polity_id:
                    continue

                polity = PolityInfo(
                    id=polity_id,
                    name=polity_data.get("long_name", polity_id),
                    start_year=polity_data.get("start_year", 0) or 0,
                    end_year=polity_data.get("end_year", 0) or 0,
                    nga=polity_data.get("home_nga", {}).get("name", "")
                    if polity_data.get("home_nga")
                    else "",
                    region=polity_data.get("home_nga", {}).get("subregion", "")
                    if polity_data.get("home_nga")
                    else "",
                )

                self._polity_index[polity_id] = polity

                # Index by NGA
                if polity.nga:
                    if polity.nga not in self._nga_index:
                        self._nga_index[polity.nga] = []
                    self._nga_index[polity.nga].append(polity_id)

                # Index by region
                if polity.region:
                    if polity.region not in self._region_index:
                        self._region_index[polity.region] = []
                    self._region_index[polity.region].append(polity_id)

            self._indices_loaded = True
            logger.info(f"Loaded {len(self._polity_index)} polities")

        except Exception as e:
            logger.warning(f"Failed to load polity indices: {e}")
            # Continue without indices - will use API queries

    @property
    def is_available(self) -> bool:
        """Check if the Seshat API is available."""
        if self._api_available is not None:
            return self._api_available

        try:
            self._ensure_api_client()
            return True
        except SeshatAPIError:
            return False

    def list_polities(
        self,
        region: str | None = None,
        time_range: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        """
        List available polities with metadata.

        Args:
            region: Filter by NGA or region name
            time_range: Filter by time range (start_year, end_year)

        Returns:
            DataFrame with columns: polity_id, polity_name, nga, start_year, end_year

        Example:
            >>> client = SeshatAPIClient()
            >>> italian_polities = client.list_polities(region="Italy")
        """
        self._load_indices()

        rows = []
        for polity_id, polity in self._polity_index.items():
            # Filter by region
            if region:
                region_lower = region.lower()
                if (
                    region_lower not in polity.nga.lower()
                    and region_lower not in polity.region.lower()
                ):
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

    def get_polity(self, polity_id: str) -> PolityInfo:
        """
        Get information about a specific polity.

        Args:
            polity_id: Polity identifier (e.g., "RomPrin")

        Returns:
            PolityInfo with polity metadata

        Raises:
            KeyError: If polity not found

        Example:
            >>> client = SeshatAPIClient()
            >>> rome = client.get_polity("RomPrin")
            >>> print(f"{rome.name}: {rome.start_year} to {rome.end_year}")
        """
        self._load_indices()

        if polity_id in self._polity_index:
            return self._polity_index[polity_id]

        # Try to fetch directly from API
        try:
            from seshat_api.core import Polities

            client = self._ensure_api_client()
            polities_api = Polities(client)
            polity_data = polities_api.get(polity_id)

            if polity_data:
                return PolityInfo(
                    id=polity_data.get("name", polity_id),
                    name=polity_data.get("long_name", polity_id),
                    start_year=polity_data.get("start_year", 0) or 0,
                    end_year=polity_data.get("end_year", 0) or 0,
                    nga=polity_data.get("home_nga", {}).get("name", "")
                    if polity_data.get("home_nga")
                    else "",
                    region=polity_data.get("home_nga", {}).get("subregion", "")
                    if polity_data.get("home_nga")
                    else "",
                )
        except Exception as e:
            logger.debug(f"Could not fetch polity {polity_id}: {e}")

        raise KeyError(f"Polity '{polity_id}' not found")

    def query(
        self,
        variables: Sequence[str] | None = None,
        polities: Sequence[str] | None = None,
        time_range: tuple[int, int] | None = None,
        regions: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Query Seshat data with flexible filtering.

        Args:
            variables: List of variable names to include
            polities: List of polity IDs to include. If None, include all.
            time_range: Tuple of (start_year, end_year) to filter by.
                        Uses CE years (negative = BCE).
            regions: List of NGA (Natural Geographic Area) names to filter by.

        Returns:
            DataFrame with columns:
            - polity_id: Short identifier
            - polity_name: Full name
            - nga: Natural Geographic Area
            - start_year: Polity start year
            - end_year: Polity end year
            - [variable columns]: One column per requested variable

        Example:
            >>> client = SeshatAPIClient()
            >>> df = client.query(
            ...     variables=["PolPop", "PolTerr"],
            ...     time_range=(-500, 500),
            ...     regions=["Italy"]
            ... )
        """
        self._load_indices()

        # Get matching polities
        matching_polities: list[PolityInfo] = []

        for polity_id, polity in self._polity_index.items():
            # Filter by polity ID
            if polities and polity_id not in polities:
                continue

            # Filter by region
            if regions:
                region_match = False
                for region in regions:
                    region_lower = region.lower()
                    if (
                        region_lower in polity.nga.lower()
                        or region_lower in polity.region.lower()
                    ):
                        region_match = True
                        break
                if not region_match:
                    continue

            # Filter by time range
            if time_range:
                start, end = time_range
                if polity.end_year < start or polity.start_year > end:
                    continue

            matching_polities.append(polity)

        if not matching_polities:
            return pd.DataFrame()

        # Build result DataFrame
        rows: list[dict[str, Any]] = []

        for polity in matching_polities:
            row: dict[str, Any] = {
                "polity_id": polity.id,
                "polity_name": polity.name,
                "nga": polity.nga,
                "start_year": polity.start_year,
                "end_year": polity.end_year,
            }

            # TODO: Fetch variable data from API
            # The seshat_api package has different modules for different variable types
            # (wf for warfare, sc for social complexity, etc.)
            # For now, we return polity metadata without variable values

            if variables:
                for var in variables:
                    row[var] = None  # Placeholder for variable values

            rows.append(row)

        df = pd.DataFrame(rows)

        # Reorder columns
        base_cols = ["polity_id", "polity_name", "nga", "start_year", "end_year"]
        var_cols = [c for c in df.columns if c not in base_cols]
        df = df[base_cols + sorted(var_cols)]

        return df

    def list_regions(self) -> list[str]:
        """
        List available Natural Geographic Areas (regions).

        Returns:
            List of NGA names

        Example:
            >>> client = SeshatAPIClient()
            >>> regions = client.list_regions()
        """
        self._load_indices()
        return sorted(self._nga_index.keys())

    def list_ngas(self) -> pd.DataFrame:
        """
        List available NGAs (Natural Geographic Areas) with metadata.

        Returns:
            DataFrame with NGA information
        """
        try:
            from seshat_api.core import NGAs

            client = self._ensure_api_client()
            ngas_api = NGAs(client)

            rows = []
            for nga_data in ngas_api.get_all():
                rows.append(
                    {
                        "name": nga_data.get("name", ""),
                        "subregion": nga_data.get("subregion", ""),
                        "world_region": nga_data.get("world_region", ""),
                    }
                )

            return pd.DataFrame(rows)

        except Exception as e:
            logger.warning(f"Could not fetch NGAs: {e}")
            # Fall back to indexed NGAs
            self._load_indices()
            return pd.DataFrame({"name": list(self._nga_index.keys())})

    def get_variable_data(
        self,
        variable_class: str,
        polity_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Get data for a specific variable type.

        The Seshat API organizes variables into modules (wf for warfare,
        sc for social complexity, etc.). This method fetches data from
        the appropriate module.

        Args:
            variable_class: Variable class name (e.g., "Crossbows", "PolPop")
            polity_id: Optional polity ID to filter by

        Returns:
            DataFrame with variable data

        Example:
            >>> client = SeshatAPIClient()
            >>> crossbow_data = client.get_variable_data("Crossbows")
        """
        try:
            from seshat_api import seshat_class_instance

            client = self._ensure_api_client()
            var_api = seshat_class_instance(client, variable_class)

            if var_api is None:
                raise ValueError(f"Unknown variable class: {variable_class}")

            rows = []
            for item in var_api.get_all():
                row = dict(item) if isinstance(item, dict) else item.__dict__
                rows.append(row)

            df = pd.DataFrame(rows)

            # Filter by polity if specified
            if polity_id and "polity" in df.columns:
                # Polity might be a dict or ID
                def matches_polity(p: Any) -> bool:
                    if isinstance(p, dict):
                        return p.get("name") == polity_id
                    return p == polity_id

                df = df[df["polity"].apply(matches_polity)]

            return df

        except ImportError as e:
            raise SeshatAPINotInstalledError(
                "The seshat_api package is required for this operation"
            ) from e

    def clear_cache(self) -> None:
        """Clear the API response cache."""
        self._cache.clear()

    def get_frequencies(
        self,
        variables: Sequence[str],
        start_year: int,
        end_year: int,
        step: int = 100,
    ) -> pd.DataFrame:
        """
        Get frequency data for variables over a time range.

        Uses the seshat_api get_frequencies function to aggregate
        variable presence across polities for each time period.

        Args:
            variables: List of variable class names
            start_year: Start year (CE, negative for BCE)
            end_year: End year (CE, negative for BCE)
            step: Year step size for aggregation

        Returns:
            DataFrame with year and variable frequency columns
        """
        try:
            from seshat_api import get_frequencies

            client = self._ensure_api_client()

            results = []
            for year in range(start_year, end_year + 1, step):
                row: dict[str, Any] = {"year": year}
                for var in variables:
                    try:
                        freq = get_frequencies(client, var, year, year)
                        row[var] = freq
                    except Exception:
                        row[var] = None
                results.append(row)

            return pd.DataFrame(results)

        except ImportError as e:
            raise SeshatAPINotInstalledError(
                "The seshat_api package is required for this operation"
            ) from e
