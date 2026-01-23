"""
Tests for Polaris-2025 dataset parsing and SeshatDB integration.

Tests the parser functions and SeshatDB dataset selection features.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from cliodynamics.data.access import (
    DATASET_EQUINOX,
    DATASET_POLARIS,
    SeshatDB,
)
from cliodynamics.data.parser import (
    load_polaris,
    load_polaris_threads,
    parse_seshat_dataframe,
)


class TestLoadPolaris:
    """Tests for the load_polaris function."""

    def test_load_polaris_file_not_found(self) -> None:
        """Test error when Polaris data directory does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Non-existent directory
            non_existent = Path(tmpdir) / "nonexistent"
            with pytest.raises(
                FileNotFoundError, match="Polaris data directory not found"
            ):
                load_polaris(non_existent)

    def test_load_polaris_no_files(self) -> None:
        """Test error when directory exists but has no data files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polaris_dir = Path(tmpdir) / "polaris2025"
            polaris_dir.mkdir()

            with pytest.raises(FileNotFoundError, match="No Excel or CSV files found"):
                load_polaris(polaris_dir)

    def test_load_polaris_with_excel(self) -> None:
        """Test loading Polaris from Excel file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polaris_dir = Path(tmpdir)

            # Create a mock Excel file with Polaris-style data
            df = pd.DataFrame(
                {
                    "Polity": ["TestPol1", "TestPol2"],
                    "Original_name": ["Test Polity 1", "Test Polity 2"],
                    "NGA": ["Region 1", "Region 2"],
                    "Start": [-100, 100],
                    "End": [100, 300],
                    "SomeVariable": [1000, 2000],
                }
            )

            excel_path = polaris_dir / "Polaris2025.xlsx"
            df.to_excel(excel_path, index=False)

            dataset = load_polaris(polaris_dir)

            assert len(dataset.polities) == 2
            assert dataset.metadata["source"] == "Seshat Polaris-2025"
            assert "repository" in dataset.metadata

    def test_load_polaris_with_csv(self) -> None:
        """Test loading Polaris from CSV file when no Excel present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polaris_dir = Path(tmpdir)

            df = pd.DataFrame(
                {
                    "Polity": ["TestPol1"],
                    "Original_name": ["Test Polity 1"],
                    "NGA": ["Region 1"],
                    "Start": [-100],
                    "End": [100],
                }
            )

            csv_path = polaris_dir / "main_data.csv"
            df.to_csv(csv_path, index=False)

            dataset = load_polaris(polaris_dir)

            assert len(dataset.polities) == 1

    def test_load_polaris_skips_threads_csv(self) -> None:
        """Test that load_polaris skips polity_threads.csv for main data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polaris_dir = Path(tmpdir)

            # Create threads file with different schema
            threads_df = pd.DataFrame(
                {
                    "thread_id": [1, 2],
                    "polity": ["Pol1", "Pol2"],
                }
            )
            threads_path = polaris_dir / "polity_threads.csv"
            threads_df.to_csv(threads_path, index=False)

            # Create main data file
            main_df = pd.DataFrame(
                {
                    "Polity": ["TestPol1"],
                    "Original_name": ["Test Polity 1"],
                    "NGA": ["Region 1"],
                    "Start": [-100],
                    "End": [100],
                }
            )
            main_path = polaris_dir / "main_data.csv"
            main_df.to_csv(main_path, index=False)

            dataset = load_polaris(polaris_dir)

            # Should load main_data.csv, not polity_threads.csv
            assert len(dataset.polities) == 1
            assert dataset.polities[0].id == "TestPol1"


class TestLoadPolarisThreads:
    """Tests for the load_polaris_threads function."""

    def test_load_threads_not_found(self) -> None:
        """Test error when threads file not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(
                FileNotFoundError, match="Polity threads file not found"
            ):
                load_polaris_threads(tmpdir)

    def test_load_threads_success(self) -> None:
        """Test successful loading of threads file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polaris_dir = Path(tmpdir)

            threads_df = pd.DataFrame(
                {
                    "thread_id": [1, 2, 3],
                    "polity_name": ["Pol1", "Pol2", "Pol3"],
                    "predecessor": ["", "Pol1", "Pol2"],
                }
            )
            threads_path = polaris_dir / "polity_threads.csv"
            threads_df.to_csv(threads_path, index=False)

            result = load_polaris_threads(polaris_dir)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert "thread_id" in result.columns


class TestSeshatDBDatasetSelection:
    """Tests for SeshatDB dataset selection."""

    def test_default_dataset_is_polaris(self) -> None:
        """Test that default dataset is Polaris-2025."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = SeshatDB(data_path=tmpdir, dataset=DATASET_POLARIS)
            assert db._dataset_name == DATASET_POLARIS

    def test_equinox_dataset_selection(self) -> None:
        """Test selecting Equinox-2020 dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = SeshatDB(data_path=tmpdir, dataset=DATASET_EQUINOX)
            assert db._dataset_name == DATASET_EQUINOX

    def test_invalid_dataset_raises_error(self) -> None:
        """Test that invalid dataset name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset"):
            SeshatDB(dataset="invalid_dataset")

    def test_case_insensitive_dataset_name(self) -> None:
        """Test that dataset name is case-insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_upper = SeshatDB(data_path=tmpdir, dataset="POLARIS2025")
            db_mixed = SeshatDB(data_path=tmpdir, dataset="Polaris2025")

            assert db_upper._dataset_name == DATASET_POLARIS
            assert db_mixed._dataset_name == DATASET_POLARIS

    def test_dataset_name_property(self) -> None:
        """Test the dataset_name property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock data
            df = pd.DataFrame(
                {
                    "Polity": ["TestPol"],
                    "Original_name": ["Test Polity"],
                    "NGA": ["Region"],
                    "Start": [0],
                    "End": [100],
                }
            )
            excel_path = Path(tmpdir) / "test.xlsx"
            df.to_excel(excel_path, index=False)

            db = SeshatDB(data_path=tmpdir, dataset=DATASET_EQUINOX)
            # Force loading
            db._ensure_loaded()

            assert db.dataset_name == DATASET_EQUINOX


class TestSeshatDBAutoDetect:
    """Tests for SeshatDB auto-detecting dataset from path."""

    def test_auto_detect_polaris_from_path(self) -> None:
        """Test auto-detecting Polaris dataset from path name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polaris_dir = Path(tmpdir) / "polaris2025"
            polaris_dir.mkdir()

            df = pd.DataFrame(
                {
                    "Polity": ["TestPol"],
                    "Original_name": ["Test Polity"],
                    "NGA": ["Region"],
                    "Start": [0],
                    "End": [100],
                }
            )
            excel_path = polaris_dir / "data.xlsx"
            df.to_excel(excel_path, index=False)

            db = SeshatDB(data_path=polaris_dir)
            db._ensure_loaded()

            # Should auto-detect as Polaris
            assert db._dataset_name == DATASET_POLARIS

    def test_auto_detect_equinox_from_path(self) -> None:
        """Test auto-detecting Equinox dataset from path name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seshat_dir = Path(tmpdir) / "seshat"
            seshat_dir.mkdir()

            df = pd.DataFrame(
                {
                    "Polity": ["TestPol"],
                    "Original_name": ["Test Polity"],
                    "NGA": ["Region"],
                    "Start": [0],
                    "End": [100],
                }
            )
            excel_path = seshat_dir / "data.xlsx"
            df.to_excel(excel_path, index=False)

            db = SeshatDB(data_path=seshat_dir)
            db._ensure_loaded()

            # Should auto-detect as Equinox
            assert db._dataset_name == DATASET_EQUINOX


class TestSeshatDBWithBothDatasets:
    """Tests for SeshatDB with both datasets."""

    def test_load_both_datasets_separately(self) -> None:
        """Test loading both datasets in separate SeshatDB instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create Equinox data
            equinox_dir = Path(tmpdir) / "equinox"
            equinox_dir.mkdir()
            equinox_df = pd.DataFrame(
                {
                    "Polity": ["EqPol1", "EqPol2"],
                    "Original_name": ["Equinox 1", "Equinox 2"],
                    "NGA": ["RegionA", "RegionB"],
                    "Start": [0, 100],
                    "End": [100, 200],
                }
            )
            equinox_df.to_excel(equinox_dir / "equinox.xlsx", index=False)

            # Create Polaris data
            polaris_dir = Path(tmpdir) / "polaris"
            polaris_dir.mkdir()
            polaris_df = pd.DataFrame(
                {
                    "Polity": ["PolPol1", "PolPol2", "PolPol3"],
                    "Original_name": ["Polaris 1", "Polaris 2", "Polaris 3"],
                    "NGA": ["RegionX", "RegionY", "RegionZ"],
                    "Start": [0, 100, 200],
                    "End": [100, 200, 300],
                }
            )
            polaris_df.to_excel(polaris_dir / "polaris.xlsx", index=False)

            # Load both
            db_equinox = SeshatDB(data_path=equinox_dir, dataset=DATASET_EQUINOX)
            db_polaris = SeshatDB(data_path=polaris_dir, dataset=DATASET_POLARIS)

            db_equinox._ensure_loaded()
            db_polaris._ensure_loaded()

            # Should have different data
            assert len(db_equinox._polity_index) == 2
            assert len(db_polaris._polity_index) == 3

            assert "EqPol1" in db_equinox._polity_index
            assert "PolPol1" in db_polaris._polity_index


class TestDatasetConstants:
    """Tests for dataset constants."""

    def test_dataset_equinox_value(self) -> None:
        """Test DATASET_EQUINOX constant value."""
        assert DATASET_EQUINOX == "equinox2020"

    def test_dataset_polaris_value(self) -> None:
        """Test DATASET_POLARIS constant value."""
        assert DATASET_POLARIS == "polaris2025"

    def test_constants_importable_from_package(self) -> None:
        """Test that constants are importable from package."""
        from cliodynamics.data import DATASET_EQUINOX, DATASET_POLARIS

        assert DATASET_EQUINOX == "equinox2020"
        assert DATASET_POLARIS == "polaris2025"


class TestPolarisParsingDifferences:
    """Tests for Polaris-specific parsing differences."""

    def test_lowercase_column_names(self) -> None:
        """Test that lowercase column names work (Polaris style)."""
        df = pd.DataFrame(
            {
                "polity": ["TestPol"],
                "original_name": ["Test Polity"],
                "nga": ["Region"],
                "start": [0],
                "end": [100],
            }
        )

        dataset = parse_seshat_dataframe(df)

        assert len(dataset.polities) == 1
        assert dataset.polities[0].id == "TestPol"

    def test_mixed_case_column_names(self) -> None:
        """Test that mixed case column names work (Equinox style)."""
        df = pd.DataFrame(
            {
                "Polity": ["TestPol"],
                "Original_name": ["Test Polity"],
                "NGA": ["Region"],
                "Start": [0],
                "End": [100],
            }
        )

        dataset = parse_seshat_dataframe(df)

        assert len(dataset.polities) == 1
        assert dataset.polities[0].id == "TestPol"


class TestModuleImports:
    """Tests for module imports."""

    def test_import_load_polaris(self) -> None:
        """Test that load_polaris is importable from package."""
        from cliodynamics.data import load_polaris

        assert callable(load_polaris)

    def test_import_load_polaris_threads(self) -> None:
        """Test that load_polaris_threads is importable from package."""
        from cliodynamics.data import load_polaris_threads

        assert callable(load_polaris_threads)

    def test_import_dataset_constants(self) -> None:
        """Test that dataset constants are importable."""
        from cliodynamics.data import DATASET_EQUINOX, DATASET_POLARIS

        assert DATASET_EQUINOX is not None
        assert DATASET_POLARIS is not None
