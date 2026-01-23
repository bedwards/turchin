"""
Tests for the Polaris-2025 download module.

Tests use mocking to avoid actual network requests during testing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cliodynamics.data.download_polaris import (
    EXPECTED_EXCEL_FILE,
    EXPECTED_THREADS_FILE,
    POLARIS_EXCEL_URL,
    POLARIS_THREADS_URL,
    download_file,
    download_polaris,
)


class TestDownloadFile:
    """Tests for the download_file function."""

    @patch("cliodynamics.data.download_polaris.requests.get")
    def test_download_file_success(self, mock_get: MagicMock) -> None:
        """Test successful file download."""
        # Mock response
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content.return_value = [b"test data"]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            dest_path = Path(tmpdir) / "test_file.txt"
            download_file("https://example.com/file.txt", dest_path)

            assert dest_path.exists()
            assert dest_path.read_bytes() == b"test data"

    @patch("cliodynamics.data.download_polaris.requests.get")
    def test_download_file_http_error(self, mock_get: MagicMock) -> None:
        """Test handling of HTTP errors."""
        import requests

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            dest_path = Path(tmpdir) / "test_file.txt"
            with pytest.raises(requests.HTTPError):
                download_file("https://example.com/notfound.txt", dest_path)


class TestDownloadPolaris:
    """Tests for the download_polaris function."""

    @patch("cliodynamics.data.download_polaris.download_file")
    def test_download_polaris_creates_directory(
        self, mock_download_file: MagicMock
    ) -> None:
        """Test that download_polaris creates the data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            result = download_polaris(data_dir)

            polaris_dir = data_dir / "polaris2025"
            assert polaris_dir.exists()
            assert result == polaris_dir

    @patch("cliodynamics.data.download_polaris.download_file")
    def test_download_polaris_downloads_files(
        self, mock_download_file: MagicMock
    ) -> None:
        """Test that download_polaris downloads both files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            download_polaris(data_dir, include_threads=True)

            # Should have called download_file twice
            assert mock_download_file.call_count == 2

            # Check correct URLs were called
            call_args = [call[0] for call in mock_download_file.call_args_list]
            urls = [args[0] for args in call_args]
            assert POLARIS_EXCEL_URL in urls
            assert POLARIS_THREADS_URL in urls

    @patch("cliodynamics.data.download_polaris.download_file")
    def test_download_polaris_skips_existing(
        self, mock_download_file: MagicMock
    ) -> None:
        """Test that download_polaris skips existing files without force."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            polaris_dir = data_dir / "polaris2025"
            polaris_dir.mkdir(parents=True)

            # Create existing files
            (polaris_dir / EXPECTED_EXCEL_FILE).write_bytes(b"existing data")
            (polaris_dir / EXPECTED_THREADS_FILE).write_bytes(b"existing threads")

            download_polaris(data_dir, force=False)

            # Should not have called download_file
            mock_download_file.assert_not_called()

    @patch("cliodynamics.data.download_polaris.download_file")
    def test_download_polaris_force_redownload(
        self, mock_download_file: MagicMock
    ) -> None:
        """Test that download_polaris redownloads with force=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            polaris_dir = data_dir / "polaris2025"
            polaris_dir.mkdir(parents=True)

            # Create existing files
            (polaris_dir / EXPECTED_EXCEL_FILE).write_bytes(b"existing data")
            (polaris_dir / EXPECTED_THREADS_FILE).write_bytes(b"existing threads")

            download_polaris(data_dir, force=True)

            # Should have called download_file twice
            assert mock_download_file.call_count == 2

    @patch("cliodynamics.data.download_polaris.download_file")
    def test_download_polaris_without_threads(
        self, mock_download_file: MagicMock
    ) -> None:
        """Test downloading without threads file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            download_polaris(data_dir, include_threads=False)

            # Should have called download_file once (Excel only)
            assert mock_download_file.call_count == 1

            # Check only Excel URL was called
            call_args = mock_download_file.call_args_list[0][0]
            assert call_args[0] == POLARIS_EXCEL_URL


class TestDownloadPolarisURLs:
    """Tests to verify correct URLs are used."""

    def test_excel_url_format(self) -> None:
        """Test that Excel URL is correctly formatted."""
        assert "Polaris2025.xlsx" in POLARIS_EXCEL_URL
        assert "github" in POLARIS_EXCEL_URL.lower()
        assert "raw" in POLARIS_EXCEL_URL

    def test_threads_url_format(self) -> None:
        """Test that threads URL is correctly formatted."""
        assert "polity_threads.csv" in POLARIS_THREADS_URL
        assert "github" in POLARIS_THREADS_URL.lower()
        assert "raw" in POLARIS_THREADS_URL


class TestExpectedFiles:
    """Tests for expected file constants."""

    def test_expected_excel_file(self) -> None:
        """Test expected Excel filename."""
        assert EXPECTED_EXCEL_FILE == "Polaris2025.xlsx"

    def test_expected_threads_file(self) -> None:
        """Test expected threads filename."""
        assert EXPECTED_THREADS_FILE == "polity_threads.csv"


class TestModuleImports:
    """Tests for module imports."""

    def test_import_from_package(self) -> None:
        """Test that download_polaris can be imported from package."""
        from cliodynamics.data import download_polaris as dp

        assert callable(dp)

    def test_import_download_function(self) -> None:
        """Test that download function is importable."""
        from cliodynamics.data.download_polaris import download_polaris

        assert callable(download_polaris)

    def test_import_download_file(self) -> None:
        """Test that download_file helper is importable."""
        from cliodynamics.data.download_polaris import download_file

        assert callable(download_file)
