"""
Unit tests for the cliodynamics.data.access module.

Tests the SeshatDB query interface and related functionality.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from cliodynamics.data.access import (
    VARIABLE_ALIASES,
    VARIABLE_CATEGORIES,
    PolityTimeSeries,
    SeshatDB,
    TimeSeriesPoint,
)
from cliodynamics.data.parser import (
    DataQuality,
    ParsedValue,
    Polity,
    SeshatDataset,
)

# Test fixtures


@pytest.fixture
def sample_polities() -> list[Polity]:
    """Create sample polities for testing."""
    return [
        Polity(
            id="RomPrin",
            name="Roman Principate",
            nga="Italy",
            start_year=-27,
            end_year=284,
            variables={
                "PolPop": [ParsedValue(raw="50000000", value=50000000.0)],
                "Polity_territory": [ParsedValue(raw="5000000", value=5000000.0)],
                "Writing_system": [
                    ParsedValue(raw="present", value=1.0, quality=DataQuality.PRESENT)
                ],
            },
        ),
        Polity(
            id="RomDom",
            name="Roman Dominate",
            nga="Italy",
            start_year=284,
            end_year=476,
            variables={
                "PolPop": [ParsedValue(raw="30000000", value=30000000.0)],
                "Polity_territory": [ParsedValue(raw="4500000", value=4500000.0)],
                "Writing_system": [
                    ParsedValue(raw="present", value=1.0, quality=DataQuality.PRESENT)
                ],
            },
        ),
        Polity(
            id="AthensCl",
            name="Classical Athens",
            nga="Aegean",
            start_year=-500,
            end_year=-300,
            variables={
                "PolPop": [ParsedValue(raw="300000", value=300000.0)],
                "Polity_territory": [ParsedValue(raw="2650", value=2650.0)],
                "Writing_system": [
                    ParsedValue(raw="present", value=1.0, quality=DataQuality.PRESENT)
                ],
            },
        ),
        Polity(
            id="MaurEmp",
            name="Mauryan Empire",
            nga="Middle Ganga",
            start_year=-322,
            end_year=-185,
            variables={
                "PolPop": [ParsedValue(raw="50000000", value=50000000.0)],
                "Polity_territory": [ParsedValue(raw="5000000", value=5000000.0)],
                "Writing_system": [
                    ParsedValue(raw="present", value=1.0, quality=DataQuality.PRESENT)
                ],
            },
        ),
    ]


@pytest.fixture
def sample_dataset(sample_polities) -> SeshatDataset:
    """Create a sample dataset for testing."""
    # Create a simple DataFrame
    data = {
        "Polity": [p.id for p in sample_polities],
        "Original_name": [p.name for p in sample_polities],
        "NGA": [p.nga for p in sample_polities],
        "Start": [p.start_year for p in sample_polities],
        "End": [p.end_year for p in sample_polities],
        "PolPop": [50000000, 30000000, 300000, 50000000],
        "Polity_territory": [5000000, 4500000, 2650, 5000000],
        "Writing_system": ["present", "present", "present", "present"],
    }
    df = pd.DataFrame(data)

    return SeshatDataset(
        polities=sample_polities,
        variables=["PolPop", "Polity_territory", "Writing_system"],
        df=df,
        metadata={
            "source": "test",
            "n_polities": len(sample_polities),
            "n_variables": 3,
        },
    )


@pytest.fixture
def mock_db(sample_dataset) -> SeshatDB:
    """Create a SeshatDB with mocked dataset."""
    db = SeshatDB.__new__(SeshatDB)
    db._data_path = None
    db._dataset = sample_dataset
    db._polity_index = {p.id: p for p in sample_dataset.polities}
    db._nga_index = {}
    for p in sample_dataset.polities:
        if p.nga not in db._nga_index:
            db._nga_index[p.nga] = []
        db._nga_index[p.nga].append(p.id)
    return db


# Test TimeSeriesPoint


class TestTimeSeriesPoint:
    """Tests for TimeSeriesPoint dataclass."""

    def test_create_point(self):
        """Test creating a time series point."""
        point = TimeSeriesPoint(
            year=100,
            value=1000.0,
            quality=DataQuality.PRESENT,
        )
        assert point.year == 100
        assert point.value == 1000.0
        assert point.quality == DataQuality.PRESENT
        assert point.is_interpolated is False

    def test_point_with_range(self):
        """Test point with uncertainty range."""
        point = TimeSeriesPoint(
            year=100,
            value=300.0,
            value_min=100.0,
            value_max=500.0,
        )
        assert point.value == 300.0
        assert point.value_min == 100.0
        assert point.value_max == 500.0

    def test_interpolated_point(self):
        """Test interpolated point."""
        point = TimeSeriesPoint(
            year=100,
            value=1000.0,
            is_interpolated=True,
        )
        assert point.is_interpolated is True


# Test PolityTimeSeries


class TestPolityTimeSeries:
    """Tests for PolityTimeSeries dataclass."""

    def test_create_time_series(self):
        """Test creating a polity time series."""
        ts = PolityTimeSeries(
            polity_id="Test",
            polity_name="Test Polity",
            nga="Test Region",
            start_year=0,
            end_year=100,
            variables={
                "var1": [TimeSeriesPoint(year=0, value=100.0)],
            },
        )
        assert ts.polity_id == "Test"
        assert ts.polity_name == "Test Polity"
        assert len(ts.variables) == 1
        assert "var1" in ts.variables


# Test SeshatDB


class TestSeshatDBInit:
    """Tests for SeshatDB initialization."""

    def test_init_with_path(self):
        """Test initialization with data path."""
        db = SeshatDB("data/test/")
        assert db._data_path == Path("data/test/")
        assert db._dataset is None

    def test_init_without_path(self):
        """Test initialization without data path."""
        db = SeshatDB()
        assert db._data_path is None
        assert db._dataset is None


class TestSeshatDBGetPolity:
    """Tests for SeshatDB.get_polity method."""

    def test_get_existing_polity(self, mock_db):
        """Test getting an existing polity."""
        result = mock_db.get_polity("RomPrin")
        assert result.polity_id == "RomPrin"
        assert result.polity_name == "Roman Principate"
        assert result.nga == "Italy"
        assert result.start_year == -27
        assert result.end_year == 284

    def test_get_polity_variables(self, mock_db):
        """Test that polity variables are returned correctly."""
        result = mock_db.get_polity("RomPrin")
        assert "PolPop" in result.variables
        assert len(result.variables["PolPop"]) == 1
        assert result.variables["PolPop"][0].value == 50000000.0

    def test_get_nonexistent_polity(self, mock_db):
        """Test getting a nonexistent polity raises KeyError."""
        with pytest.raises(KeyError, match="Polity 'NonExistent' not found"):
            mock_db.get_polity("NonExistent")


class TestSeshatDBQuery:
    """Tests for SeshatDB.query method."""

    def test_query_all_polities(self, mock_db):
        """Test query returning all polities."""
        df = mock_db.query()
        assert len(df) == 4
        assert "polity_id" in df.columns
        assert "polity_name" in df.columns
        assert "nga" in df.columns

    def test_query_by_polity_ids(self, mock_db):
        """Test query filtering by polity IDs."""
        df = mock_db.query(polities=["RomPrin", "RomDom"])
        assert len(df) == 2
        assert set(df["polity_id"]) == {"RomPrin", "RomDom"}

    def test_query_by_region(self, mock_db):
        """Test query filtering by region."""
        df = mock_db.query(regions=["Italy"])
        assert len(df) == 2
        assert all(df["nga"] == "Italy")

    def test_query_by_time_range(self, mock_db):
        """Test query filtering by time range."""
        # RomPrin (-27 to 284) overlaps with 0-300
        # RomDom (284 to 476) overlaps with 0-300
        df = mock_db.query(time_range=(0, 300))
        assert len(df) == 2  # RomPrin and RomDom
        assert "RomPrin" in df["polity_id"].values
        assert "RomDom" in df["polity_id"].values

    def test_query_by_time_range_bce(self, mock_db):
        """Test query filtering by BCE time range."""
        # Athens and Maurya overlap with -400 to -200
        df = mock_db.query(time_range=(-400, -200))
        assert len(df) == 2
        assert "AthensCl" in df["polity_id"].values
        assert "MaurEmp" in df["polity_id"].values

    def test_query_specific_variables(self, mock_db):
        """Test query with specific variables."""
        df = mock_db.query(variables=["PolPop"])
        assert "PolPop" in df.columns
        # Other variables should still be excluded
        assert (
            len(
                [
                    c
                    for c in df.columns
                    if c
                    not in ["polity_id", "polity_name", "nga", "start_year", "end_year"]
                ]
            )
            == 1
        )

    def test_query_with_variable_alias(self, mock_db):
        """Test query with variable alias."""
        df = mock_db.query(variables=["population"])
        # Should normalize "population" to "PolPop"
        assert "PolPop" in df.columns

    def test_query_combined_filters(self, mock_db):
        """Test query with multiple filters."""
        df = mock_db.query(
            variables=["PolPop"],
            regions=["Italy"],
            time_range=(0, 300),
        )
        # Should get both Roman polities
        assert len(df) == 2

    def test_query_no_results(self, mock_db):
        """Test query with no matching results."""
        df = mock_db.query(regions=["Antarctica"])
        assert len(df) == 0

    def test_query_with_interpolation(self, mock_db):
        """Test query with time series interpolation."""
        df = mock_db.query(
            polities=["RomPrin"],
            interpolate=True,
            interpolate_step=100,
        )
        # Should have multiple rows for the polity
        assert len(df) > 1
        assert "year" in df.columns
        # All rows should be for RomPrin
        assert all(df["polity_id"] == "RomPrin")

    def test_query_by_category(self, mock_db):
        """Test query by variable category."""
        # Note: our sample data may not have all social_complexity variables
        df = mock_db.query(category="social_complexity")
        # Should include PolPop and Polity_territory
        assert "PolPop" in df.columns or "Polity_territory" in df.columns


class TestSeshatDBListMethods:
    """Tests for SeshatDB list methods."""

    def test_list_polities(self, mock_db):
        """Test listing all polities."""
        df = mock_db.list_polities()
        assert len(df) == 4
        assert "polity_id" in df.columns
        assert "polity_name" in df.columns

    def test_list_polities_by_region(self, mock_db):
        """Test listing polities filtered by region."""
        df = mock_db.list_polities(region="Italy")
        assert len(df) == 2
        assert all(df["nga"] == "Italy")

    def test_list_polities_by_time_range(self, mock_db):
        """Test listing polities filtered by time range."""
        # RomPrin (-27 to 284) and RomDom (284 to 476) overlap with 0-300
        df = mock_db.list_polities(time_range=(0, 300))
        assert len(df) == 2

    def test_list_variables(self, mock_db):
        """Test listing all variables."""
        df = mock_db.list_variables()
        assert len(df) == 3  # Our sample has 3 variables
        assert "variable_name" in df.columns
        assert "category" in df.columns

    def test_list_variables_by_category(self, mock_db):
        """Test listing variables filtered by category."""
        df = mock_db.list_variables(category="social_complexity")
        # Should only include variables in social_complexity
        for var in df["variable_name"]:
            assert var in VARIABLE_CATEGORIES["social_complexity"]

    def test_list_variables_by_search(self, mock_db):
        """Test listing variables filtered by search."""
        df = mock_db.list_variables(search="Pop")
        # Should include PolPop
        assert any("Pop" in v for v in df["variable_name"])

    def test_list_regions(self, mock_db):
        """Test listing all regions."""
        regions = mock_db.list_regions()
        assert "Italy" in regions
        assert "Aegean" in regions
        assert "Middle Ganga" in regions

    def test_list_categories(self, mock_db):
        """Test listing all categories."""
        categories = mock_db.list_categories()
        assert "social_complexity" in categories
        assert "military" in categories
        assert "information" in categories


class TestSeshatDBCrossPolityComparison:
    """Tests for SeshatDB.get_cross_polity_comparison method."""

    def test_cross_polity_comparison(self, mock_db):
        """Test cross-polity comparison."""
        df = mock_db.get_cross_polity_comparison("PolPop")
        assert len(df) == 4
        assert "value" in df.columns
        assert "polity_id" in df.columns

    def test_cross_polity_comparison_with_alias(self, mock_db):
        """Test cross-polity comparison with variable alias."""
        df = mock_db.get_cross_polity_comparison("population")
        assert len(df) == 4
        assert "value" in df.columns

    def test_cross_polity_comparison_filtered(self, mock_db):
        """Test cross-polity comparison with filters."""
        df = mock_db.get_cross_polity_comparison(
            "PolPop",
            regions=["Italy"],
        )
        assert len(df) == 2
        assert all(df["nga"] == "Italy")


class TestSeshatDBDataFrameExport:
    """Tests for SeshatDB.to_dataframe method."""

    def test_to_dataframe(self, mock_db):
        """Test exporting entire dataset as DataFrame."""
        df = mock_db.to_dataframe()
        assert len(df) == 4
        assert "Polity" in df.columns
        assert "NGA" in df.columns


class TestSeshatDBInterpolation:
    """Tests for SeshatDB.interpolate_time_series method."""

    def test_interpolate_time_series(self, mock_db):
        """Test time series interpolation."""
        df = mock_db.interpolate_time_series(
            "RomPrin",
            "PolPop",
            step=100,
        )
        assert len(df) > 0
        assert "year" in df.columns
        assert "value" in df.columns
        assert "is_interpolated" in df.columns

    def test_interpolate_with_custom_range(self, mock_db):
        """Test interpolation with custom year range."""
        df = mock_db.interpolate_time_series(
            "RomPrin",
            "PolPop",
            start_year=0,
            end_year=200,
            step=50,
        )
        assert df["year"].min() == 0
        assert df["year"].max() == 200

    def test_interpolate_nonexistent_polity(self, mock_db):
        """Test interpolation with nonexistent polity raises error."""
        with pytest.raises(KeyError):
            mock_db.interpolate_time_series("NonExistent", "PolPop")


class TestSeshatDBSummaryStatistics:
    """Tests for SeshatDB.get_summary_statistics method."""

    def test_summary_statistics(self, mock_db):
        """Test getting summary statistics."""
        stats = mock_db.get_summary_statistics("PolPop")
        assert stats["count"] == 4
        assert stats["mean"] is not None
        assert stats["min"] is not None
        assert stats["max"] is not None
        assert stats["median"] is not None

    def test_summary_statistics_filtered(self, mock_db):
        """Test summary statistics with filters."""
        stats = mock_db.get_summary_statistics("PolPop", regions=["Italy"])
        assert stats["count"] == 2

    def test_summary_statistics_empty(self, mock_db):
        """Test summary statistics with no matching data."""
        stats = mock_db.get_summary_statistics("PolPop", regions=["Antarctica"])
        assert stats["count"] == 0
        assert stats["mean"] is None


class TestVariableAliases:
    """Tests for variable aliases."""

    def test_aliases_defined(self):
        """Test that aliases are defined."""
        assert "population" in VARIABLE_ALIASES
        assert "territory" in VARIABLE_ALIASES
        assert VARIABLE_ALIASES["population"] == "PolPop"

    def test_normalize_variable_name(self, mock_db):
        """Test variable name normalization."""
        assert mock_db._normalize_variable_name("population") == "PolPop"
        assert mock_db._normalize_variable_name("territory") == "Polity_territory"
        # Unknown names should pass through
        assert mock_db._normalize_variable_name("UnknownVar") == "UnknownVar"


class TestVariableCategories:
    """Tests for variable categories."""

    def test_categories_defined(self):
        """Test that categories are defined."""
        assert "social_complexity" in VARIABLE_CATEGORIES
        assert "military" in VARIABLE_CATEGORIES
        assert "information" in VARIABLE_CATEGORIES
        assert "religion" in VARIABLE_CATEGORIES
        assert "economy" in VARIABLE_CATEGORIES
        assert "politics" in VARIABLE_CATEGORIES

    def test_social_complexity_variables(self):
        """Test social complexity category contains expected variables."""
        vars = VARIABLE_CATEGORIES["social_complexity"]
        assert "PolPop" in vars
        assert "Polity_territory" in vars

    def test_get_variables_for_category(self, mock_db):
        """Test getting variables for a category."""
        vars = mock_db._get_variables_for_category("social_complexity")
        assert "PolPop" in vars

    def test_get_variables_for_unknown_category(self, mock_db):
        """Test error for unknown category."""
        with pytest.raises(ValueError, match="Unknown category"):
            mock_db._get_variables_for_category("unknown_category")


class TestSeshatDBLazyLoading:
    """Tests for SeshatDB lazy loading behavior."""

    def test_lazy_loading(self):
        """Test that dataset is not loaded until needed."""
        db = SeshatDB("nonexistent/path/")
        # Dataset should not be loaded yet
        assert db._dataset is None

    @patch("cliodynamics.data.access.load_seshat_excel")
    @patch("pathlib.Path.glob")
    def test_loading_from_excel(self, mock_glob, mock_load_excel):
        """Test loading from Excel file."""
        # Setup mocks
        mock_glob.return_value = [Path("test.xlsx")]
        mock_df = pd.DataFrame(
            {
                "Polity": ["Test"],
                "Original_name": ["Test Polity"],
                "NGA": ["Region"],
                "Start": [0],
                "End": [100],
            }
        )
        mock_load_excel.return_value = mock_df

        db = SeshatDB(Path("/fake/path"))
        # This should trigger loading
        with patch.object(Path, "glob", return_value=[Path("test.xlsx")]):
            with patch(
                "cliodynamics.data.access.load_seshat_excel", return_value=mock_df
            ):
                with patch(
                    "cliodynamics.data.access.parse_seshat_dataframe"
                ) as mock_parse:
                    mock_parse.return_value = SeshatDataset(
                        polities=[],
                        variables=[],
                        df=mock_df,
                    )
                    # Access dataset property to trigger loading
                    try:
                        _ = db.dataset
                    except FileNotFoundError:
                        pass  # Expected in this test setup


class TestSeshatDBDatasetProperty:
    """Tests for SeshatDB.dataset property."""

    def test_dataset_property(self, mock_db):
        """Test accessing dataset property."""
        dataset = mock_db.dataset
        assert dataset is not None
        assert len(dataset.polities) == 4


class TestModuleImports:
    """Tests for module imports."""

    def test_import_from_package(self):
        """Test importing from cliodynamics.data."""
        from cliodynamics.data import (
            VARIABLE_ALIASES,
            VARIABLE_CATEGORIES,
            PolityTimeSeries,
            SeshatDB,
            TimeSeriesPoint,
        )

        assert SeshatDB is not None
        assert PolityTimeSeries is not None
        assert TimeSeriesPoint is not None
        assert VARIABLE_ALIASES is not None
        assert VARIABLE_CATEGORIES is not None

    def test_import_directly(self):
        """Test importing directly from access module."""
        from cliodynamics.data.access import SeshatDB

        assert SeshatDB is not None
