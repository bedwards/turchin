"""
Unit tests for the cliodynamics.data module.

Tests parsing logic, edge cases, and data quality indicators.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cliodynamics.data.parser import (
    DataQuality,
    ParsedValue,
    Polity,
    SeshatDataset,
    get_variable_summary,
    parse_seshat_dataframe,
    parse_value,
)


class TestParseValue:
    """Tests for the parse_value function."""

    def test_parse_none(self):
        """Test parsing None values."""
        result = parse_value(None)
        assert result.quality == DataQuality.UNCODED
        assert result.value is None
        assert result.raw == ""

    def test_parse_nan(self):
        """Test parsing NaN values."""
        result = parse_value(float("nan"))
        assert result.quality == DataQuality.UNCODED
        assert result.value is None

    def test_parse_empty_string(self):
        """Test parsing empty strings."""
        result = parse_value("")
        assert result.quality == DataQuality.UNCODED
        assert result.value is None

    def test_parse_whitespace_string(self):
        """Test parsing whitespace-only strings."""
        result = parse_value("   ")
        assert result.quality == DataQuality.UNCODED
        assert result.value is None

    def test_parse_present(self):
        """Test parsing 'present' values."""
        result = parse_value("present")
        assert result.quality == DataQuality.PRESENT
        assert result.value == 1.0
        assert result.raw == "present"

    def test_parse_present_case_insensitive(self):
        """Test parsing 'PRESENT' values (case insensitive)."""
        result = parse_value("PRESENT")
        assert result.quality == DataQuality.PRESENT
        assert result.value == 1.0

    def test_parse_absent(self):
        """Test parsing 'absent' values."""
        result = parse_value("absent")
        assert result.quality == DataQuality.ABSENT
        assert result.value == 0.0

    def test_parse_unknown(self):
        """Test parsing 'unknown' values."""
        result = parse_value("unknown")
        assert result.quality == DataQuality.UNKNOWN
        assert result.value is None

    def test_parse_uncoded(self):
        """Test parsing 'uncoded' values."""
        result = parse_value("uncoded")
        assert result.quality == DataQuality.UNCODED
        assert result.value is None

    def test_parse_suspected_unknown(self):
        """Test parsing 'suspected unknown' values."""
        result = parse_value("suspected unknown")
        assert result.quality == DataQuality.SUSPECTED_UNKNOWN
        assert result.value is None

    def test_parse_inferred_present(self):
        """Test parsing 'inferred present' values."""
        result = parse_value("inferred present")
        assert result.quality == DataQuality.INFERRED_PRESENT
        assert result.value == 1.0
        assert result.is_inferred is True

    def test_parse_inferred_absent(self):
        """Test parsing 'inferred absent' values."""
        result = parse_value("inferred absent")
        assert result.quality == DataQuality.INFERRED_ABSENT
        assert result.value == 0.0
        assert result.is_inferred is True

    def test_parse_integer(self):
        """Test parsing integer values."""
        result = parse_value(100)
        assert result.value == 100.0
        assert result.quality == DataQuality.PRESENT

    def test_parse_float(self):
        """Test parsing float values."""
        result = parse_value(3.14159)
        assert result.value == 3.14159
        assert result.quality == DataQuality.PRESENT

    def test_parse_negative_number(self):
        """Test parsing negative numbers (e.g., BCE years)."""
        result = parse_value(-500)
        assert result.value == -500.0
        assert result.quality == DataQuality.PRESENT

    def test_parse_string_number(self):
        """Test parsing numeric strings."""
        result = parse_value("1000")
        assert result.value == 1000.0
        assert result.quality == DataQuality.PRESENT

    def test_parse_string_negative_number(self):
        """Test parsing negative numeric strings."""
        result = parse_value("-500")
        assert result.value == -500.0

    def test_parse_string_float(self):
        """Test parsing float strings."""
        result = parse_value("123.456")
        assert result.value == 123.456

    def test_parse_range_brackets(self):
        """Test parsing range values with brackets [X-Y]."""
        result = parse_value("[100-500]")
        assert result.value == 300.0  # Midpoint
        assert result.value_min == 100.0
        assert result.value_max == 500.0
        assert result.quality == DataQuality.PRESENT

    def test_parse_range_no_brackets(self):
        """Test parsing range values without brackets."""
        result = parse_value("100-500")
        assert result.value == 300.0
        assert result.value_min == 100.0
        assert result.value_max == 500.0

    def test_parse_range_with_spaces(self):
        """Test parsing range values with spaces."""
        result = parse_value("[ 100 - 500 ]")
        assert result.value == 300.0
        assert result.value_min == 100.0
        assert result.value_max == 500.0

    def test_parse_range_floats(self):
        """Test parsing range with float values."""
        result = parse_value("[1.5-3.5]")
        assert result.value == 2.5
        assert result.value_min == 1.5
        assert result.value_max == 3.5

    def test_parse_range_negative(self):
        """Test parsing range with negative values."""
        result = parse_value("[-500--300]")
        assert result.value == -400.0
        assert result.value_min == -500.0
        assert result.value_max == -300.0

    def test_parse_disputed(self):
        """Test parsing 'disputed' values."""
        result = parse_value("disputed")
        assert result.quality == DataQuality.DISPUTED
        assert result.is_disputed is True

    def test_parse_text_with_disputed(self):
        """Test parsing text containing 'disputed'."""
        result = parse_value("value disputed by experts")
        assert result.quality == DataQuality.DISPUTED
        assert result.is_disputed is True

    def test_parse_arbitrary_text(self):
        """Test parsing arbitrary text values."""
        result = parse_value("some descriptive text")
        assert result.quality == DataQuality.PRESENT
        assert result.notes == "some descriptive text"
        assert result.value is None


class TestParsedValue:
    """Tests for the ParsedValue dataclass."""

    def test_defaults(self):
        """Test default values."""
        pv = ParsedValue(raw="test")
        assert pv.value is None
        assert pv.value_min is None
        assert pv.value_max is None
        assert pv.quality == DataQuality.PRESENT
        assert pv.is_inferred is False
        assert pv.is_disputed is False
        assert pv.notes == ""

    def test_with_range(self):
        """Test ParsedValue with range values."""
        pv = ParsedValue(
            raw="[100-500]",
            value=300.0,
            value_min=100.0,
            value_max=500.0,
        )
        assert pv.value == 300.0
        assert pv.value_min == 100.0
        assert pv.value_max == 500.0


class TestPolity:
    """Tests for the Polity dataclass."""

    def test_create_polity(self):
        """Test creating a basic polity."""
        polity = Polity(
            id="RomPrin",
            name="Roman Principate",
            nga="Italy",
            start_year=-27,
            end_year=284,
        )
        assert polity.id == "RomPrin"
        assert polity.name == "Roman Principate"
        assert polity.nga == "Italy"
        assert polity.start_year == -27
        assert polity.end_year == 284
        assert polity.variables == {}

    def test_polity_with_variables(self):
        """Test polity with variables."""
        polity = Polity(
            id="Test",
            name="Test Polity",
            nga="Test NGA",
            start_year=0,
            end_year=100,
            variables={
                "population": [ParsedValue(raw="1000000", value=1000000.0)],
            },
        )
        assert "population" in polity.variables
        assert polity.variables["population"][0].value == 1000000.0


class TestSeshatDataset:
    """Tests for the SeshatDataset dataclass."""

    def test_create_empty_dataset(self):
        """Test creating an empty dataset."""
        df = pd.DataFrame()
        dataset = SeshatDataset(
            polities=[],
            variables=[],
            df=df,
        )
        assert len(dataset.polities) == 0
        assert len(dataset.variables) == 0
        assert dataset.metadata == {}

    def test_dataset_with_metadata(self):
        """Test dataset with metadata."""
        df = pd.DataFrame()
        dataset = SeshatDataset(
            polities=[],
            variables=[],
            df=df,
            metadata={"source": "test", "version": "1.0"},
        )
        assert dataset.metadata["source"] == "test"
        assert dataset.metadata["version"] == "1.0"


class TestParseSeshatDataframe:
    """Tests for parse_seshat_dataframe function."""

    def test_parse_simple_dataframe(self):
        """Test parsing a simple DataFrame."""
        df = pd.DataFrame(
            {
                "Polity": ["RomPrin", "RomDom"],
                "Original_name": ["Roman Principate", "Roman Dominate"],
                "NGA": ["Italy", "Italy"],
                "Start": [-27, 284],
                "End": [284, 476],
                "Population": [50000000, 30000000],
            }
        )

        dataset = parse_seshat_dataframe(df)

        assert len(dataset.polities) == 2
        assert dataset.polities[0].id == "RomPrin"
        assert dataset.polities[0].name == "Roman Principate"
        assert dataset.polities[0].start_year == -27
        assert dataset.polities[0].end_year == 284
        assert "Population" in dataset.polities[0].variables

    def test_parse_with_missing_values(self):
        """Test parsing DataFrame with missing values."""
        df = pd.DataFrame(
            {
                "Polity": ["Test1", "Test2"],
                "Original_name": ["Test Polity 1", "Test Polity 2"],
                "NGA": ["Region 1", "Region 2"],
                "Start": [0, 100],
                "End": [100, 200],
                "Variable1": ["present", None],
                "Variable2": [100, "unknown"],
            }
        )

        dataset = parse_seshat_dataframe(df)

        assert len(dataset.polities) == 2
        # First polity has "present"
        v1_p1 = dataset.polities[0].variables["Variable1"][0]
        assert v1_p1.quality == DataQuality.PRESENT
        assert v1_p1.value == 1.0

        # Second polity has None (uncoded)
        v1_p2 = dataset.polities[1].variables["Variable1"][0]
        assert v1_p2.quality == DataQuality.UNCODED

        # Second polity has "unknown" for Variable2
        v2_p2 = dataset.polities[1].variables["Variable2"][0]
        assert v2_p2.quality == DataQuality.UNKNOWN

    def test_parse_case_insensitive_columns(self):
        """Test parsing with case-insensitive column matching."""
        df = pd.DataFrame(
            {
                "polity": ["Test"],  # lowercase
                "original_name": ["Test Polity"],  # lowercase
                "nga": ["Region"],  # lowercase
                "start": [0],  # lowercase
                "end": [100],  # lowercase
            }
        )

        dataset = parse_seshat_dataframe(df)
        assert len(dataset.polities) == 1
        assert dataset.polities[0].id == "Test"

    def test_parse_missing_required_columns(self):
        """Test error when required columns are missing."""
        df = pd.DataFrame(
            {
                "SomeOtherColumn": [1, 2, 3],
            }
        )

        with pytest.raises(ValueError, match="Required columns not found"):
            parse_seshat_dataframe(df)

    def test_parse_identifies_variables(self):
        """Test that variable columns are correctly identified."""
        df = pd.DataFrame(
            {
                "Polity": ["Test"],
                "Original_name": ["Test Polity"],
                "NGA": ["Region"],
                "Start": [0],
                "End": [100],
                "Var1": ["present"],
                "Var2": [100],
                "Var3": ["[10-20]"],
            }
        )

        dataset = parse_seshat_dataframe(df)

        # Should have 3 variables (not counting metadata columns)
        assert len(dataset.variables) == 3
        assert "Var1" in dataset.variables
        assert "Var2" in dataset.variables
        assert "Var3" in dataset.variables


class TestGetVariableSummary:
    """Tests for get_variable_summary function."""

    def test_summary_numeric_variable(self):
        """Test summary for a numeric variable."""
        polities = [
            Polity(
                id=f"P{i}",
                name=f"Polity {i}",
                nga="Region",
                start_year=0,
                end_year=100,
                variables={"Population": [ParsedValue(raw=str(v), value=float(v))]},
            )
            for i, v in enumerate([100, 200, 300, 400, 500])
        ]
        df = pd.DataFrame()
        dataset = SeshatDataset(
            polities=polities,
            variables=["Population"],
            df=df,
        )

        summary = get_variable_summary(dataset, "Population")

        assert summary["variable"] == "Population"
        assert summary["n_values"] == 5
        assert summary["mean"] == 300.0
        assert summary["min"] == 100.0
        assert summary["max"] == 500.0

    def test_summary_with_missing_values(self):
        """Test summary handles missing values correctly."""
        polities = [
            Polity(
                id="P1",
                name="Polity 1",
                nga="Region",
                start_year=0,
                end_year=100,
                variables={"Var": [ParsedValue(raw="100", value=100.0)]},
            ),
            Polity(
                id="P2",
                name="Polity 2",
                nga="Region",
                start_year=0,
                end_year=100,
                variables={
                    "Var": [ParsedValue(raw="unknown", quality=DataQuality.UNKNOWN)]
                },
            ),
        ]
        df = pd.DataFrame()
        dataset = SeshatDataset(
            polities=polities,
            variables=["Var"],
            df=df,
        )

        summary = get_variable_summary(dataset, "Var")

        # Only 1 numeric value
        assert summary["n_values"] == 1
        assert summary["n_polities"] == 2
        # Quality distribution should show both
        assert "present" in summary["quality_distribution"]
        assert "unknown" in summary["quality_distribution"]

    def test_summary_unknown_variable(self):
        """Test error for unknown variable."""
        df = pd.DataFrame()
        dataset = SeshatDataset(
            polities=[],
            variables=["KnownVar"],
            df=df,
        )

        with pytest.raises(KeyError, match="Variable 'UnknownVar' not found"):
            get_variable_summary(dataset, "UnknownVar")


class TestDataQuality:
    """Tests for DataQuality enum."""

    def test_quality_values(self):
        """Test all quality values exist."""
        assert DataQuality.PRESENT.value == "present"
        assert DataQuality.ABSENT.value == "absent"
        assert DataQuality.INFERRED_PRESENT.value == "inferred_present"
        assert DataQuality.INFERRED_ABSENT.value == "inferred_absent"
        assert DataQuality.UNKNOWN.value == "unknown"
        assert DataQuality.UNCODED.value == "uncoded"
        assert DataQuality.DISPUTED.value == "disputed"
        assert DataQuality.SUSPECTED_UNKNOWN.value == "suspected_unknown"


class TestDownloadModule:
    """Tests for the download module."""

    def test_import_download_functions(self):
        """Test that download functions can be imported."""
        from cliodynamics.data.download import (
            download_and_extract,
            get_zenodo_download_url,
        )

        assert callable(download_and_extract)
        assert callable(get_zenodo_download_url)

    @patch("cliodynamics.data.download.requests.get")
    def test_get_zenodo_download_url(self, mock_get):
        """Test fetching download URL from Zenodo API."""
        from cliodynamics.data.download import get_zenodo_download_url

        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "files": [
                {
                    "key": "seshatdb/Equinox_Data-v.1.zip",
                    "links": {"self": "https://zenodo.org/api/files/test.zip"},
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        url = get_zenodo_download_url()
        assert url == "https://zenodo.org/api/files/test.zip"

    @patch("cliodynamics.data.download.requests.get")
    def test_get_zenodo_download_url_no_files(self, mock_get):
        """Test error when no files in Zenodo record."""
        from cliodynamics.data.download import get_zenodo_download_url

        mock_response = MagicMock()
        mock_response.json.return_value = {"files": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="No files found"):
            get_zenodo_download_url()


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_parse_unicode_dash_in_range(self):
        """Test parsing range with unicode en-dash."""
        result = parse_value("[100\u2013500]")  # en-dash
        assert result.value == 300.0
        assert result.value_min == 100.0
        assert result.value_max == 500.0

    def test_parse_very_large_number(self):
        """Test parsing very large numbers."""
        result = parse_value(1e12)
        assert result.value == 1e12

    def test_parse_very_small_number(self):
        """Test parsing very small numbers."""
        result = parse_value(0.00001)
        assert result.value == 0.00001

    def test_parse_zero(self):
        """Test parsing zero."""
        result = parse_value(0)
        assert result.value == 0.0
        assert result.quality == DataQuality.PRESENT

    def test_parse_string_zero(self):
        """Test parsing string zero."""
        result = parse_value("0")
        assert result.value == 0.0

    def test_polity_negative_bce_years(self):
        """Test polity with BCE (negative) years."""
        df = pd.DataFrame(
            {
                "Polity": ["AncientCiv"],
                "Original_name": ["Ancient Civilization"],
                "NGA": ["Middle East"],
                "Start": [-3000],
                "End": [-1000],
            }
        )

        dataset = parse_seshat_dataframe(df)
        polity = dataset.polities[0]

        assert polity.start_year == -3000
        assert polity.end_year == -1000

    def test_inferred_present_various_formats(self):
        """Test various formats of 'inferred present'."""
        variants = [
            "inferred present",
            "Inferred Present",
            "INFERRED PRESENT",
            "inferred  present",  # extra space
        ]
        for variant in variants:
            result = parse_value(variant)
            assert result.is_inferred is True, f"Failed for: {variant}"
            assert result.value == 1.0, f"Failed for: {variant}"
