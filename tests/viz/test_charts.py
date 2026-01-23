"""Tests for the charts module dimension validation and safeguards."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from cliodynamics.viz import charts


class TestChartValidationResult:
    """Tests for ChartValidationResult class."""

    def test_valid_result(self):
        """Test creating a valid result."""
        result = charts.ChartValidationResult(
            width=800,
            height=600,
            file_size=50000,
            is_valid=True,
            message="Dimensions OK",
        )
        assert result.width == 800
        assert result.height == 600
        assert result.file_size == 50000
        assert result.is_valid is True
        assert "Dimensions OK" in result.message

    def test_invalid_result(self):
        """Test creating an invalid result."""
        result = charts.ChartValidationResult(
            width=5000,
            height=3000,
            file_size=100000,
            is_valid=False,
            message="width exceeds max",
        )
        assert result.is_valid is False
        assert "width exceeds max" in result.message

    def test_repr(self):
        """Test string representation."""
        result = charts.ChartValidationResult(
            width=800,
            height=600,
            file_size=50000,
            is_valid=True,
            message="OK",
        )
        repr_str = repr(result)
        assert "VALID" in repr_str
        assert "800x600" in repr_str

        invalid_result = charts.ChartValidationResult(
            width=5000,
            height=3000,
            file_size=100000,
            is_valid=False,
            message="too large",
        )
        assert "INVALID" in repr(invalid_result)


class TestValidateChartDimensions:
    """Tests for validate_chart_dimensions function."""

    def test_file_not_found(self, tmp_path):
        """Test with non-existent file."""
        with pytest.raises(FileNotFoundError):
            charts.validate_chart_dimensions(tmp_path / "nonexistent.png")

    def test_file_too_small(self, tmp_path):
        """Test detection of too-small files."""
        small_file = tmp_path / "small.png"
        small_file.write_bytes(b"x" * 100)  # 100 bytes

        result = charts.validate_chart_dimensions(small_file)
        assert result.is_valid is False
        assert "too small" in result.message.lower()

    def test_file_too_large(self, tmp_path):
        """Test detection of too-large files."""
        large_file = tmp_path / "large.png"
        # Create a file larger than 50MB threshold
        large_file.write_bytes(b"x" * 51_000_000)

        result = charts.validate_chart_dimensions(large_file)
        assert result.is_valid is False
        assert "too large" in result.message.lower()

    @patch("cliodynamics.viz.charts.HAS_IMAGESIZE", False)
    def test_without_imagesize(self, tmp_path):
        """Test fallback when imagesize not installed."""
        # Create a file with reasonable size
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"x" * 10000)

        result = charts.validate_chart_dimensions(test_file)
        # Should return valid with warning message
        assert result.is_valid is True
        assert "skipped" in result.message.lower()

    @patch("cliodynamics.viz.charts.imagesize")
    def test_valid_dimensions(self, mock_imagesize, tmp_path):
        """Test with valid dimensions."""
        mock_imagesize.get.return_value = (800, 600)

        test_file = tmp_path / "valid.png"
        test_file.write_bytes(b"x" * 10000)

        result = charts.validate_chart_dimensions(test_file)
        assert result.is_valid is True
        assert result.width == 800
        assert result.height == 600

    @patch("cliodynamics.viz.charts.imagesize")
    def test_dimensions_exceed_max_width(self, mock_imagesize, tmp_path):
        """Test detection of width exceeding maximum."""
        mock_imagesize.get.return_value = (5000, 600)

        test_file = tmp_path / "wide.png"
        test_file.write_bytes(b"x" * 10000)

        result = charts.validate_chart_dimensions(test_file, max_width=3000)
        assert result.is_valid is False
        assert "width" in result.message.lower()
        assert "5000" in result.message

    @patch("cliodynamics.viz.charts.imagesize")
    def test_dimensions_exceed_max_height(self, mock_imagesize, tmp_path):
        """Test detection of height exceeding maximum."""
        mock_imagesize.get.return_value = (800, 53000)

        test_file = tmp_path / "tall.png"
        test_file.write_bytes(b"x" * 10000)

        result = charts.validate_chart_dimensions(test_file, max_height=3000)
        assert result.is_valid is False
        assert "height" in result.message.lower()
        assert "53000" in result.message

    @patch("cliodynamics.viz.charts.imagesize")
    def test_dimensions_exceed_both(self, mock_imagesize, tmp_path):
        """Test detection when both dimensions exceed max."""
        mock_imagesize.get.return_value = (5000, 6000)

        test_file = tmp_path / "huge.png"
        test_file.write_bytes(b"x" * 10000)

        result = charts.validate_chart_dimensions(
            test_file, max_width=3000, max_height=3000
        )
        assert result.is_valid is False
        assert "width" in result.message.lower()
        assert "height" in result.message.lower()

    @patch("cliodynamics.viz.charts.imagesize")
    def test_custom_max_dimensions(self, mock_imagesize, tmp_path):
        """Test with custom maximum dimensions."""
        mock_imagesize.get.return_value = (1000, 1000)

        test_file = tmp_path / "custom.png"
        test_file.write_bytes(b"x" * 10000)

        # Should pass with default max (3000)
        result = charts.validate_chart_dimensions(test_file)
        assert result.is_valid is True

        # Should fail with stricter max
        result = charts.validate_chart_dimensions(
            test_file, max_width=500, max_height=500
        )
        assert result.is_valid is False


class TestConfigureChart:
    """Tests for configure_chart function."""

    def test_height_clamping(self):
        """Test that height is clamped to max_height."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        import altair as alt

        chart = alt.Chart(df).mark_bar().encode(x="x", y="y")

        # Request height exceeding max
        configured = charts.configure_chart(
            chart, "Test", height=5000, max_height=2000
        )

        # Check that height was clamped (extract from chart spec)
        spec = configured.to_dict()
        assert spec["height"] == 2000

    def test_default_max_height(self):
        """Test default max height is applied."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        import altair as alt

        chart = alt.Chart(df).mark_bar().encode(x="x", y="y")

        configured = charts.configure_chart(chart, "Test", height=600)
        spec = configured.to_dict()
        assert spec["height"] == 600  # Should not be clamped


class TestSaveChart:
    """Tests for save_chart function."""

    def test_unsupported_format(self, tmp_path):
        """Test error on unsupported format."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        import altair as alt

        chart = alt.Chart(df).mark_bar().encode(x="x", y="y")

        with pytest.raises(ValueError, match="Unsupported format"):
            charts.save_chart(chart, tmp_path / "output.xyz")

    @patch("cliodynamics.viz.charts.validate_chart_dimensions")
    def test_validation_failure_raises_error(self, mock_validate, tmp_path):
        """Test that validation failure raises ChartDimensionError."""
        mock_validate.return_value = charts.ChartValidationResult(
            width=800,
            height=53000,
            file_size=10000,
            is_valid=False,
            message="height exceeds max",
        )

        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        import altair as alt

        chart = alt.Chart(df).mark_bar().encode(x="x", y="y")
        configured = charts.configure_chart(chart, "Test")

        with pytest.raises(charts.ChartDimensionError):
            charts.save_chart(configured, tmp_path / "output.png")

    def test_save_without_validation(self, tmp_path):
        """Test saving without validation."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        import altair as alt

        chart = alt.Chart(df).mark_bar().encode(x="x", y="y")
        configured = charts.configure_chart(chart, "Test")

        # Should not raise even without validation
        path = charts.save_chart(
            configured, tmp_path / "output.png", validate=False
        )
        assert path.exists()

    def test_save_html(self, tmp_path):
        """Test saving as HTML (no validation needed)."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        import altair as alt

        chart = alt.Chart(df).mark_bar().encode(x="x", y="y")
        configured = charts.configure_chart(chart, "Test")

        path = charts.save_chart(configured, tmp_path / "output.html")
        assert path.exists()
        assert path.suffix == ".html"

    def test_save_svg(self, tmp_path):
        """Test saving as SVG (no validation needed)."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        import altair as alt

        chart = alt.Chart(df).mark_bar().encode(x="x", y="y")
        configured = charts.configure_chart(chart, "Test")

        path = charts.save_chart(configured, tmp_path / "output.svg")
        assert path.exists()
        assert path.suffix == ".svg"


class TestCreateTimelineChart:
    """Tests for create_timeline_chart function."""

    def test_height_safeguard(self):
        """Test that timeline chart height is clamped."""
        # Create DataFrame with many categories
        n_categories = 200
        df = pd.DataFrame({
            "name": [f"item_{i}" for i in range(n_categories)],
            "start": list(range(n_categories)),
            "end": list(range(1, n_categories + 1)),
        })

        chart = charts.create_timeline_chart(
            df, "name", "start", "end", title="Test", max_height=2000
        )

        spec = chart.to_dict()
        assert spec["height"] <= 2000


class TestCreateBarChart:
    """Tests for create_bar_chart function."""

    def test_horizontal_bar_height_safeguard(self):
        """Test that horizontal bar chart height is clamped."""
        n_categories = 200
        df = pd.DataFrame({
            "category": [f"cat_{i}" for i in range(n_categories)],
            "value": list(range(n_categories)),
        })

        chart = charts.create_bar_chart(
            df, "value", "category", horizontal=True, max_height=1500
        )

        spec = chart.to_dict()
        assert spec["height"] <= 1500


class TestCreateHeatmap:
    """Tests for create_heatmap function."""

    def test_height_safeguard(self):
        """Test that heatmap height is clamped."""
        n_y_categories = 200
        df = pd.DataFrame({
            "x": ["A"] * n_y_categories,
            "y": [f"y_{i}" for i in range(n_y_categories)],
            "value": list(range(n_y_categories)),
        })

        chart = charts.create_heatmap(
            df, "x", "y", "value", title="Test", max_height=1500
        )

        spec = chart.to_dict()
        assert spec["height"] <= 1500


class TestVerifyChart:
    """Tests for verify_chart function."""

    def test_file_not_found(self, tmp_path, capsys):
        """Test with non-existent file."""
        result = charts.verify_chart(tmp_path / "nonexistent.png")
        assert result.is_valid is False
        assert "not found" in result.message.lower()

        captured = capsys.readouterr()
        assert "[FAIL]" in captured.out

    @patch("cliodynamics.viz.charts.validate_chart_dimensions")
    def test_valid_chart(self, mock_validate, tmp_path, capsys):
        """Test with valid chart."""
        mock_validate.return_value = charts.ChartValidationResult(
            width=800,
            height=600,
            file_size=50000,
            is_valid=True,
            message="Dimensions OK",
        )

        test_file = tmp_path / "valid.png"
        test_file.write_bytes(b"x" * 10000)

        result = charts.verify_chart(test_file)
        assert result.is_valid is True

        captured = capsys.readouterr()
        assert "[OK]" in captured.out
        assert charts.VERIFICATION_REMINDER in captured.out

    def test_print_info_disabled(self, tmp_path, capsys):
        """Test that print_info=False suppresses output."""
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"x" * 10000)

        with patch("cliodynamics.viz.charts.validate_chart_dimensions") as mock:
            mock.return_value = charts.ChartValidationResult(
                width=800, height=600, file_size=50000,
                is_valid=True, message="OK"
            )
            charts.verify_chart(test_file, print_info=False)

        captured = capsys.readouterr()
        assert captured.out == ""


class TestChartDimensionError:
    """Tests for ChartDimensionError exception."""

    def test_exception_message(self):
        """Test exception can be raised with message."""
        with pytest.raises(charts.ChartDimensionError, match="too large"):
            raise charts.ChartDimensionError("Chart is too large")


class TestDebugMode:
    """Tests for debug mode functionality."""

    def test_print_chart_debug_info(self, capsys):
        """Test debug info printing."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        import altair as alt

        chart = alt.Chart(df).mark_bar().encode(x="x", y="y")
        configured = charts.configure_chart(chart, "Test", width=800, height=600)

        charts._print_chart_debug_info(configured)

        captured = capsys.readouterr()
        assert "CHART DEBUG INFO" in captured.out
        assert "Width" in captured.out or "Height" in captured.out


class TestModuleConstants:
    """Tests for module constants."""

    def test_max_dimensions_defined(self):
        """Test that max dimension constants are defined."""
        assert charts.MAX_CHART_WIDTH == 3000
        assert charts.MAX_CHART_HEIGHT == 3000

    def test_chart_heights_defined(self):
        """Test that standard height constants are defined."""
        assert charts.CHART_HEIGHT_SMALL == 400
        assert charts.CHART_HEIGHT_MEDIUM == 600
        assert charts.CHART_HEIGHT_LARGE == 900
        assert charts.CHART_HEIGHT_XLARGE == 1200
