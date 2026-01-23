"""Tests for the visualization module.

Tests cover:
- Time series plotting
- Phase space diagrams (2D and 3D)
- Model vs data comparison plots
- Secular cycle detection
- Cycle visualization
- Export functionality
"""

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cliodynamics.viz import cycles, plots


def sample_simulation_df() -> pd.DataFrame:
    """Create sample simulation data for testing."""
    t = np.linspace(0, 200, 201)
    # Create oscillating dynamics
    N = 0.5 + 0.2 * np.sin(2 * np.pi * t / 100) + 0.1 * np.sin(2 * np.pi * t / 30)
    E = 0.1 + 0.05 * np.sin(2 * np.pi * t / 100 + np.pi / 4)
    W = 1.0 - 0.3 * np.sin(2 * np.pi * t / 100)
    S = 1.0 - 0.2 * np.sin(2 * np.pi * t / 100 + np.pi / 2)
    psi = 0.5 + 0.4 * np.sin(2 * np.pi * t / 100 + np.pi)

    return pd.DataFrame({"t": t, "N": N, "E": E, "W": W, "S": S, "psi": psi})


def sample_observed_df() -> pd.DataFrame:
    """Create sample observed data for comparison testing."""
    t = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])
    N = 0.5 + 0.2 * np.sin(2 * np.pi * t / 100) + np.random.normal(0, 0.05, len(t))
    psi = (
        0.5
        + 0.4 * np.sin(2 * np.pi * t / 100 + np.pi)
        + np.random.normal(0, 0.05, len(t))
    )

    return pd.DataFrame({"t": t, "N": N, "psi": psi})


class TestPlotTimeSeries:
    """Tests for plot_time_series function."""

    def test_basic_time_series(self) -> None:
        """Test basic time series plot creation."""
        df = sample_simulation_df()
        fig = plots.plot_time_series(df, variables=["N", "W"], title="Test Plot")

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_time_series_default_variables(self) -> None:
        """Test time series with default variables."""
        df = sample_simulation_df()
        fig = plots.plot_time_series(df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_time_series_subplot_layout(self) -> None:
        """Test time series with subplot layout."""
        df = sample_simulation_df()
        fig = plots.plot_time_series(
            df, variables=["N", "W", "psi"], subplot_layout=True
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_time_series_custom_labels(self) -> None:
        """Test time series with custom labels."""
        df = sample_simulation_df()
        fig = plots.plot_time_series(
            df,
            variables=["N", "W"],
            labels=["Population", "Wages"],
            title="Custom Labels",
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_time_series_custom_colors(self) -> None:
        """Test time series with custom colors."""
        df = sample_simulation_df()
        fig = plots.plot_time_series(
            df,
            variables=["N", "W"],
            colors=["red", "blue"],
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_time_series_missing_variable_raises(self) -> None:
        """Test that missing variable raises ValueError."""
        df = sample_simulation_df()

        with pytest.raises(ValueError, match="not found"):
            plots.plot_time_series(df, variables=["nonexistent"])

    def test_time_series_labels_length_mismatch_raises(self) -> None:
        """Test that mismatched labels length raises ValueError."""
        df = sample_simulation_df()

        with pytest.raises(ValueError, match="Length of labels"):
            plots.plot_time_series(df, variables=["N", "W"], labels=["Only One"])


class TestPlotPhaseSpace:
    """Tests for phase space diagram functions."""

    def test_basic_phase_space_2d(self) -> None:
        """Test basic 2D phase space plot."""
        df = sample_simulation_df()
        fig = plots.plot_phase_space(df, x="W", y="psi")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_phase_space_2d_no_color(self) -> None:
        """Test 2D phase space without color coding."""
        df = sample_simulation_df()
        fig = plots.plot_phase_space(df, x="W", y="psi", color_by=None)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_phase_space_2d_with_arrows(self) -> None:
        """Test 2D phase space with direction arrows."""
        df = sample_simulation_df()
        fig = plots.plot_phase_space(df, x="W", y="psi", arrow_interval=20)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_phase_space_2d_custom_colormap(self) -> None:
        """Test 2D phase space with custom colormap."""
        df = sample_simulation_df()
        fig = plots.plot_phase_space(df, x="W", y="psi", colormap="plasma")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_phase_space_2d_missing_variable_raises(self) -> None:
        """Test that missing variable raises ValueError."""
        df = sample_simulation_df()

        with pytest.raises(ValueError, match="not in"):
            plots.plot_phase_space(df, x="nonexistent", y="psi")

    def test_basic_phase_space_3d(self) -> None:
        """Test basic 3D phase space plot."""
        df = sample_simulation_df()
        fig = plots.plot_phase_space_3d(df, x="N", y="W", z="psi")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_phase_space_3d_no_color(self) -> None:
        """Test 3D phase space without color coding."""
        df = sample_simulation_df()
        fig = plots.plot_phase_space_3d(df, x="N", y="W", z="psi", color_by=None)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_phase_space_3d_custom_view(self) -> None:
        """Test 3D phase space with custom viewing angle."""
        df = sample_simulation_df()
        fig = plots.plot_phase_space_3d(
            df, x="N", y="W", z="psi", elevation=45, azimuth=135
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotComparison:
    """Tests for model vs data comparison plots."""

    def test_basic_comparison(self) -> None:
        """Test basic comparison plot."""
        model_df = sample_simulation_df()
        observed_df = sample_observed_df()

        fig = plots.plot_comparison(
            model_results=model_df,
            observed_data=observed_df,
            variables=["N", "psi"],
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_comparison_single_variable(self) -> None:
        """Test comparison plot with single variable."""
        model_df = sample_simulation_df()
        observed_df = sample_observed_df()

        fig = plots.plot_comparison(
            model_results=model_df,
            observed_data=observed_df,
            variables=["psi"],
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_comparison_with_confidence_bands(self) -> None:
        """Test comparison plot with confidence bands."""
        model_df = sample_simulation_df()
        # Add confidence band columns
        model_df["psi_lower"] = model_df["psi"] - 0.1
        model_df["psi_upper"] = model_df["psi"] + 0.1

        observed_df = sample_observed_df()

        fig = plots.plot_comparison(
            model_results=model_df,
            observed_data=observed_df,
            variables=["psi"],
            confidence_bands=True,
            confidence_columns={"psi": ("psi_lower", "psi_upper")},
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_comparison_confidence_bands_without_columns_raises(self) -> None:
        """Test that confidence_bands without columns raises ValueError."""
        model_df = sample_simulation_df()
        observed_df = sample_observed_df()

        with pytest.raises(ValueError, match="confidence_columns required"):
            plots.plot_comparison(
                model_results=model_df,
                observed_data=observed_df,
                variables=["psi"],
                confidence_bands=True,
            )


class TestSaveFigure:
    """Tests for save_figure function."""

    def test_save_png(self) -> None:
        """Test saving figure as PNG."""
        df = sample_simulation_df()
        fig = plots.plot_time_series(df, variables=["N"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            saved = plots.save_figure(fig, path)

            assert len(saved) == 1
            assert saved[0].exists()
            assert saved[0].suffix == ".png"

        plt.close(fig)

    def test_save_svg(self) -> None:
        """Test saving figure as SVG."""
        df = sample_simulation_df()
        fig = plots.plot_time_series(df, variables=["N"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.svg"
            saved = plots.save_figure(fig, path)

            assert len(saved) == 1
            assert saved[0].exists()
            assert saved[0].suffix == ".svg"

        plt.close(fig)

    def test_save_pdf(self) -> None:
        """Test saving figure as PDF."""
        df = sample_simulation_df()
        fig = plots.plot_time_series(df, variables=["N"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pdf"
            saved = plots.save_figure(fig, path)

            assert len(saved) == 1
            assert saved[0].exists()
            assert saved[0].suffix == ".pdf"

        plt.close(fig)

    def test_save_multiple_formats(self) -> None:
        """Test saving figure in multiple formats."""
        df = sample_simulation_df()
        fig = plots.plot_time_series(df, variables=["N"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test"
            saved = plots.save_figure(fig, path, formats=["png", "svg", "pdf"])

            assert len(saved) == 3
            assert all(p.exists() for p in saved)
            suffixes = {p.suffix for p in saved}
            assert suffixes == {".png", ".svg", ".pdf"}

        plt.close(fig)

    def test_save_unsupported_format_raises(self) -> None:
        """Test that unsupported format raises ValueError."""
        df = sample_simulation_df()
        fig = plots.plot_time_series(df, variables=["N"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gif"

            with pytest.raises(ValueError, match="Unsupported format"):
                plots.save_figure(fig, path)

        plt.close(fig)


class TestDetectPeaksAndTroughs:
    """Tests for peak and trough detection."""

    def test_detect_peaks_simple(self) -> None:
        """Test peak detection on simple sinusoid."""
        t = np.linspace(0, 4 * np.pi, 100)
        values = np.sin(t)

        peaks, troughs = cycles.detect_peaks_and_troughs(values, t)

        # Should find approximately 2 peaks and 2 troughs
        assert len(peaks) >= 1
        assert len(troughs) >= 1
        # Peaks should be near max value
        assert all(p.value > 0.5 for p in peaks)
        # Troughs should be near min value
        assert all(p.value < -0.5 for p in troughs)

    def test_detect_peaks_no_peaks(self) -> None:
        """Test detection on monotonic data (no peaks)."""
        values = np.linspace(0, 10, 50)

        peaks, troughs = cycles.detect_peaks_and_troughs(values)

        assert len(peaks) == 0
        assert len(troughs) == 0

    def test_detect_peaks_with_noise(self) -> None:
        """Test peak detection with noisy data."""
        np.random.seed(42)
        t = np.linspace(0, 4 * np.pi, 200)
        values = np.sin(t) + np.random.normal(0, 0.1, len(t))

        peaks, troughs = cycles.detect_peaks_and_troughs(values, t, min_prominence=0.3)

        assert len(peaks) >= 1
        assert len(troughs) >= 1


class TestDetectSecularCycles:
    """Tests for secular cycle detection."""

    def test_detect_cycles_simple(self) -> None:
        """Test cycle detection on simple oscillating data."""
        t = np.linspace(0, 300, 301)
        psi = 0.5 + 0.4 * np.sin(2 * np.pi * t / 100)

        result = cycles.detect_secular_cycles(psi, t)

        assert isinstance(result, cycles.CycleDetectionResult)
        assert len(result.cycles) >= 2
        # Periods should be approximately 100
        if result.mean_period is not None:
            assert 80 < result.mean_period < 120

    def test_detect_cycles_with_phases(self) -> None:
        """Test that cycles have phase information."""
        t = np.linspace(0, 300, 301)
        psi = 0.5 + 0.4 * np.sin(2 * np.pi * t / 100)

        result = cycles.detect_secular_cycles(psi, t)

        if result.cycles:
            cycle = result.cycles[0]
            assert len(cycle.phases) == 4
            phases = [p[2] for p in cycle.phases]
            assert cycles.CyclePhase.EXPANSION in phases
            assert cycles.CyclePhase.CRISIS in phases

    def test_detect_cycles_from_dataframe(self) -> None:
        """Test cycle detection from DataFrame column."""
        df = sample_simulation_df()

        result = cycles.detect_secular_cycles(df["psi"], df["t"])

        assert isinstance(result, cycles.CycleDetectionResult)

    def test_detect_cycles_minimum_length(self) -> None:
        """Test cycle detection with minimum length filter."""
        t = np.linspace(0, 300, 301)
        # Mix of short and long cycles
        psi = 0.5 + 0.4 * np.sin(2 * np.pi * t / 100)

        result = cycles.detect_secular_cycles(psi, t, min_cycle_length=150)

        # Only long cycles should pass filter
        for cycle in result.cycles:
            assert cycle.duration >= 150


class TestPlotWithCycles:
    """Tests for plot_with_cycles function."""

    def test_basic_cycle_plot(self) -> None:
        """Test basic cycle plotting."""
        df = sample_simulation_df()
        detected = cycles.detect_secular_cycles(df["psi"], df["t"])

        fig = cycles.plot_with_cycles(df, detected)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_cycle_plot_without_phases(self) -> None:
        """Test cycle plot without phase shading."""
        df = sample_simulation_df()
        detected = cycles.detect_secular_cycles(df["psi"], df["t"])

        fig = cycles.plot_with_cycles(df, detected, show_phases=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_cycle_plot_without_markers(self) -> None:
        """Test cycle plot without peak/trough markers."""
        df = sample_simulation_df()
        detected = cycles.detect_secular_cycles(df["psi"], df["t"])

        fig = cycles.plot_with_cycles(
            df, detected, show_peaks=False, show_troughs=False
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotCycleComparison:
    """Tests for plot_cycle_comparison function."""

    def test_cycle_comparison_plot(self) -> None:
        """Test cycle comparison plot."""
        df = sample_simulation_df()
        detected = cycles.detect_secular_cycles(df["psi"], df["t"])

        fig = cycles.plot_cycle_comparison(detected)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_cycle_comparison_no_cycles(self) -> None:
        """Test cycle comparison with no cycles detected."""
        # Monotonic data with no cycles
        result = cycles.CycleDetectionResult(cycles=[], peaks=[], troughs=[])

        fig = cycles.plot_cycle_comparison(result)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestComputeCycleStatistics:
    """Tests for compute_cycle_statistics function."""

    def test_statistics_with_cycles(self) -> None:
        """Test statistics computation with detected cycles."""
        df = sample_simulation_df()
        detected = cycles.detect_secular_cycles(df["psi"], df["t"])

        stats = cycles.compute_cycle_statistics(detected)

        assert "n_cycles" in stats
        assert "mean_period" in stats
        assert "mean_amplitude" in stats
        if detected.cycles:
            assert stats["n_cycles"] == len(detected.cycles)
            assert stats["mean_period"] is not None

    def test_statistics_no_cycles(self) -> None:
        """Test statistics with no cycles."""
        result = cycles.CycleDetectionResult(cycles=[], peaks=[], troughs=[])

        stats = cycles.compute_cycle_statistics(result)

        assert stats["n_cycles"] == 0
        assert stats["mean_period"] is None


class TestVariableLabels:
    """Tests for variable labeling."""

    def test_default_labels(self) -> None:
        """Test that default labels are used."""
        assert "N" in plots.VARIABLE_LABELS
        assert "psi" in plots.VARIABLE_LABELS
        assert "\u03c8" in plots.VARIABLE_LABELS["psi"]  # Greek psi

    def test_default_colors(self) -> None:
        """Test that default colors are defined."""
        assert len(plots.DEFAULT_COLORS) >= 5
        # Colors should be valid hex colors
        for color in plots.DEFAULT_COLORS:
            assert color.startswith("#")
            assert len(color) == 7
