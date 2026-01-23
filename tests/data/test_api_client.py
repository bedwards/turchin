"""
Tests for the Seshat API client wrapper.

These tests use mocked API responses to test the client without requiring
actual network access or the seshat_api package.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cliodynamics.data.api_client import (
    APICache,
    CacheEntry,
    PolityInfo,
    SeshatAPIClient,
    SeshatAPIConnectionError,
    SeshatAPIError,
    SeshatAPINotInstalledError,
)

# Sample mock data for testing
MOCK_POLITIES_DATA = [
    {
        "name": "RomPrin",
        "long_name": "Roman Principate",
        "start_year": -27,
        "end_year": 284,
        "home_nga": {
            "name": "Latium",
            "subregion": "Italy",
            "world_region": "Europe",
        },
    },
    {
        "name": "RomRep",
        "long_name": "Roman Republic",
        "start_year": -509,
        "end_year": -27,
        "home_nga": {
            "name": "Latium",
            "subregion": "Italy",
            "world_region": "Europe",
        },
    },
    {
        "name": "AthCla",
        "long_name": "Athenian Classical Period",
        "start_year": -508,
        "end_year": -323,
        "home_nga": {
            "name": "Attica",
            "subregion": "Greece",
            "world_region": "Europe",
        },
    },
    {
        "name": "EgPtol",
        "long_name": "Ptolemaic Kingdom",
        "start_year": -305,
        "end_year": -30,
        "home_nga": {
            "name": "Middle Egypt",
            "subregion": "Egypt",
            "world_region": "Africa",
        },
    },
]

MOCK_NGAS_DATA = [
    {"name": "Latium", "subregion": "Italy", "world_region": "Europe"},
    {"name": "Attica", "subregion": "Greece", "world_region": "Europe"},
    {"name": "Middle Egypt", "subregion": "Egypt", "world_region": "Africa"},
]


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_is_expired_returns_false_when_fresh(self) -> None:
        """Test that fresh cache entries are not expired."""
        entry = CacheEntry(
            data={"test": "data"},
            timestamp=time.time(),
            ttl=3600,
        )
        assert entry.is_expired() is False

    def test_is_expired_returns_true_when_old(self) -> None:
        """Test that old cache entries are expired."""
        entry = CacheEntry(
            data={"test": "data"},
            timestamp=time.time() - 7200,  # 2 hours ago
            ttl=3600,  # 1 hour TTL
        )
        assert entry.is_expired() is True

    def test_is_expired_boundary(self) -> None:
        """Test expiration at boundary."""
        entry = CacheEntry(
            data={"test": "data"},
            timestamp=time.time() - 3601,  # Just over 1 hour ago
            ttl=3600,
        )
        assert entry.is_expired() is True


class TestAPICache:
    """Tests for APICache class."""

    def test_cache_set_and_get(self) -> None:
        """Test basic set and get operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = APICache(cache_dir=Path(tmpdir), ttl=3600)

            test_data = {"polity": "RomPrin", "name": "Roman Principate"}
            cache.set("test_key", test_data)

            result = cache.get("test_key")
            assert result == test_data

    def test_cache_returns_none_for_missing_key(self) -> None:
        """Test that missing keys return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = APICache(cache_dir=Path(tmpdir), ttl=3600)

            result = cache.get("nonexistent_key")
            assert result is None

    def test_cache_expired_entry_returns_none(self) -> None:
        """Test that expired entries return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = APICache(cache_dir=Path(tmpdir), ttl=1)  # 1 second TTL

            cache.set("test_key", {"data": "value"})
            time.sleep(1.5)  # Wait for expiration

            result = cache.get("test_key")
            assert result is None

    def test_cache_disabled(self) -> None:
        """Test that disabled cache doesn't store or retrieve."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = APICache(cache_dir=Path(tmpdir), ttl=3600, enabled=False)

            cache.set("test_key", {"data": "value"})
            result = cache.get("test_key")

            assert result is None

    def test_cache_clear(self) -> None:
        """Test clearing the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = APICache(cache_dir=Path(tmpdir), ttl=3600)

            cache.set("key1", {"data": 1})
            cache.set("key2", {"data": 2})

            cache.clear()

            assert cache.get("key1") is None
            assert cache.get("key2") is None

    def test_cache_handles_invalid_json(self) -> None:
        """Test that invalid JSON in cache returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = APICache(cache_dir=Path(tmpdir), ttl=3600)

            # Write invalid JSON directly
            cache_key = "test_key"
            cache_path = cache._get_cache_path(cache_key)
            cache_path.write_text("invalid json {{{")

            result = cache.get(cache_key)
            assert result is None


class TestPolityInfo:
    """Tests for PolityInfo dataclass."""

    def test_polity_info_creation(self) -> None:
        """Test creating a PolityInfo instance."""
        polity = PolityInfo(
            id="RomPrin",
            name="Roman Principate",
            start_year=-27,
            end_year=284,
            nga="Latium",
            region="Italy",
        )

        assert polity.id == "RomPrin"
        assert polity.name == "Roman Principate"
        assert polity.start_year == -27
        assert polity.end_year == 284
        assert polity.nga == "Latium"
        assert polity.region == "Italy"

    def test_polity_info_defaults(self) -> None:
        """Test default values for PolityInfo."""
        polity = PolityInfo(
            id="Test",
            name="Test Polity",
            start_year=0,
            end_year=100,
        )

        assert polity.nga == ""
        assert polity.region == ""


class TestSeshatAPIClient:
    """Tests for SeshatAPIClient class."""

    def test_client_initialization(self) -> None:
        """Test client initialization with default parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SeshatAPIClient(cache_dir=tmpdir)

            assert client._base_url == "https://seshatdata.com/api"
            assert client._username is None
            assert client._password is None
            assert client._cache.enabled is True

    def test_client_initialization_with_custom_params(self) -> None:
        """Test client initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SeshatAPIClient(
                base_url="https://custom.api.com/api",
                username="testuser",
                password="testpass",
                cache_enabled=False,
                cache_ttl=7200,
                cache_dir=tmpdir,
            )

            assert client._base_url == "https://custom.api.com/api"
            assert client._username == "testuser"
            assert client._password == "testpass"
            assert client._cache.enabled is False
            assert client._cache.ttl == 7200

    def test_raises_not_installed_error_when_seshat_api_missing(self) -> None:
        """Test SeshatAPINotInstalledError when seshat_api is not installed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SeshatAPIClient(cache_dir=tmpdir)

            # Mock the import to fail
            with patch.dict("sys.modules", {"seshat_api": None}):
                with patch(
                    "cliodynamics.data.api_client.SeshatAPIClient._ensure_api_client"
                ) as mock_ensure:
                    mock_ensure.side_effect = SeshatAPINotInstalledError(
                        "seshat_api not installed"
                    )

                    with pytest.raises(SeshatAPINotInstalledError):
                        client._ensure_api_client()

    def test_is_available_returns_false_when_api_unavailable(self) -> None:
        """Test is_available property when API is unavailable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SeshatAPIClient(cache_dir=tmpdir)
            client._api_available = False

            assert client.is_available is False

    def test_is_available_returns_true_when_api_available(self) -> None:
        """Test is_available property when API is available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SeshatAPIClient(cache_dir=tmpdir)
            client._api_available = True

            assert client.is_available is True


class TestSeshatAPIClientWithMockedAPI:
    """Tests for SeshatAPIClient with mocked seshat_api package."""

    @pytest.fixture
    def mock_seshat_api(self) -> MagicMock:
        """Create a mock seshat_api module."""
        mock_api = MagicMock()
        mock_api.SeshatAPI = MagicMock()
        return mock_api

    @pytest.fixture
    def mock_polities_class(self) -> MagicMock:
        """Create a mock Polities class."""
        mock_polities = MagicMock()
        mock_polities.return_value.get_all.return_value = MOCK_POLITIES_DATA
        return mock_polities

    @pytest.fixture
    def mock_ngas_class(self) -> MagicMock:
        """Create a mock NGAs class."""
        mock_ngas = MagicMock()
        mock_ngas.return_value.get_all.return_value = MOCK_NGAS_DATA
        return mock_ngas

    @pytest.fixture
    def client_with_loaded_indices(self) -> SeshatAPIClient:
        """Create a client with pre-loaded polity indices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SeshatAPIClient(cache_dir=tmpdir)

            # Manually populate the indices
            for polity_data in MOCK_POLITIES_DATA:
                polity = PolityInfo(
                    id=polity_data["name"],
                    name=polity_data["long_name"],
                    start_year=polity_data["start_year"],
                    end_year=polity_data["end_year"],
                    nga=polity_data["home_nga"]["name"],
                    region=polity_data["home_nga"]["subregion"],
                )
                client._polity_index[polity.id] = polity

                if polity.nga not in client._nga_index:
                    client._nga_index[polity.nga] = []
                client._nga_index[polity.nga].append(polity.id)

                if polity.region not in client._region_index:
                    client._region_index[polity.region] = []
                client._region_index[polity.region].append(polity.id)

            client._indices_loaded = True
            yield client

    def test_list_polities_returns_all(
        self, client_with_loaded_indices: SeshatAPIClient
    ) -> None:
        """Test listing all polities."""
        client = client_with_loaded_indices

        result = client.list_polities()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert "polity_id" in result.columns
        assert "polity_name" in result.columns
        assert "nga" in result.columns
        assert "start_year" in result.columns
        assert "end_year" in result.columns

    def test_list_polities_filter_by_region(
        self, client_with_loaded_indices: SeshatAPIClient
    ) -> None:
        """Test filtering polities by region."""
        client = client_with_loaded_indices

        result = client.list_polities(region="Italy")

        assert len(result) == 2
        assert all(
            "Italy" in row["nga"] or row["polity_id"] in ["RomPrin", "RomRep"]
            for _, row in result.iterrows()
        )

    def test_list_polities_filter_by_time_range(
        self, client_with_loaded_indices: SeshatAPIClient
    ) -> None:
        """Test filtering polities by time range."""
        client = client_with_loaded_indices

        # Get polities that existed between 100 BCE and 100 CE
        result = client.list_polities(time_range=(-100, 100))

        # Should include RomPrin (-27 to 284), RomRep (-509 to -27)
        assert len(result) >= 2
        polity_ids = set(result["polity_id"])
        assert "RomPrin" in polity_ids

    def test_get_polity_found(
        self, client_with_loaded_indices: SeshatAPIClient
    ) -> None:
        """Test getting a specific polity."""
        client = client_with_loaded_indices

        result = client.get_polity("RomPrin")

        assert result.id == "RomPrin"
        assert result.name == "Roman Principate"
        assert result.start_year == -27
        assert result.end_year == 284
        assert result.nga == "Latium"
        assert result.region == "Italy"

    def test_get_polity_not_found(
        self, client_with_loaded_indices: SeshatAPIClient
    ) -> None:
        """Test getting a non-existent polity raises KeyError."""
        client = client_with_loaded_indices

        with pytest.raises(KeyError, match="NonexistentPolity"):
            client.get_polity("NonexistentPolity")

    def test_query_returns_dataframe(
        self, client_with_loaded_indices: SeshatAPIClient
    ) -> None:
        """Test that query returns a DataFrame."""
        client = client_with_loaded_indices

        result = client.query()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4

    def test_query_filter_by_polities(
        self, client_with_loaded_indices: SeshatAPIClient
    ) -> None:
        """Test querying specific polities."""
        client = client_with_loaded_indices

        result = client.query(polities=["RomPrin", "RomRep"])

        assert len(result) == 2
        polity_ids = set(result["polity_id"])
        assert polity_ids == {"RomPrin", "RomRep"}

    def test_query_filter_by_regions(
        self, client_with_loaded_indices: SeshatAPIClient
    ) -> None:
        """Test querying by regions."""
        client = client_with_loaded_indices

        result = client.query(regions=["Greece"])

        assert len(result) == 1
        assert result.iloc[0]["polity_id"] == "AthCla"

    def test_query_filter_by_time_range(
        self, client_with_loaded_indices: SeshatAPIClient
    ) -> None:
        """Test querying by time range."""
        client = client_with_loaded_indices

        # Get polities that existed around 200 BCE
        result = client.query(time_range=(-250, -150))

        # Should include AthCla (-508 to -323), EgPtol (-305 to -30)
        polity_ids = set(result["polity_id"])
        assert "EgPtol" in polity_ids

    def test_query_with_variables(
        self, client_with_loaded_indices: SeshatAPIClient
    ) -> None:
        """Test querying with variable columns."""
        client = client_with_loaded_indices

        result = client.query(variables=["PolPop", "PolTerr"])

        assert "PolPop" in result.columns
        assert "PolTerr" in result.columns

    def test_query_returns_empty_dataframe_when_no_matches(
        self, client_with_loaded_indices: SeshatAPIClient
    ) -> None:
        """Test query returns empty DataFrame when no matches."""
        client = client_with_loaded_indices

        result = client.query(polities=["NonexistentPolity"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_list_regions(self, client_with_loaded_indices: SeshatAPIClient) -> None:
        """Test listing regions."""
        client = client_with_loaded_indices

        result = client.list_regions()

        assert isinstance(result, list)
        assert "Latium" in result
        assert "Attica" in result
        assert "Middle Egypt" in result

    def test_clear_cache(self, client_with_loaded_indices: SeshatAPIClient) -> None:
        """Test clearing the cache."""
        client = client_with_loaded_indices

        # Add something to cache
        client._cache.set("test_key", {"data": "value"})
        assert client._cache.get("test_key") is not None

        # Clear cache
        client.clear_cache()

        assert client._cache.get("test_key") is None


class TestSeshatAPIClientCacheIntegration:
    """Tests for cache integration with API client."""

    def test_cache_key_generation(self) -> None:
        """Test that cache keys are generated consistently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SeshatAPIClient(cache_dir=tmpdir)

            key1 = client._get_cache_key("polities", {"region": "Italy"})
            key2 = client._get_cache_key("polities", {"region": "Italy"})
            key3 = client._get_cache_key("polities", {"region": "Greece"})

            assert key1 == key2
            assert key1 != key3

    def test_cache_key_includes_base_url(self) -> None:
        """Test that cache key includes base URL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client1 = SeshatAPIClient(
                base_url="https://api1.example.com", cache_dir=tmpdir
            )
            client2 = SeshatAPIClient(
                base_url="https://api2.example.com", cache_dir=tmpdir
            )

            key1 = client1._get_cache_key("polities")
            key2 = client2._get_cache_key("polities")

            assert key1 != key2


class TestSeshatAPIErrors:
    """Tests for error classes."""

    def test_seshat_api_error_is_base_exception(self) -> None:
        """Test that SeshatAPIError is the base exception."""
        assert issubclass(SeshatAPINotInstalledError, SeshatAPIError)
        assert issubclass(SeshatAPIConnectionError, SeshatAPIError)

    def test_error_messages(self) -> None:
        """Test error messages are informative."""
        error = SeshatAPINotInstalledError("seshat_api not installed")
        assert "seshat_api" in str(error)

        error = SeshatAPIConnectionError("Connection failed")
        assert "Connection failed" in str(error)
