"""Tests for the Plotly animations module.

Tests cover:
- Time series animation
- Phase space animation (2D and 3D)
- Secular cycles animation
- Scenario comparison animation
- Parameter sensitivity animation
- Animation export (HTML and images)
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from cliodynamics.viz import cycles, plotly_animations


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


def sample_counterfactual_df() -> pd.DataFrame:
    """Create sample counterfactual data for comparison testing."""
    t = np.linspace(0, 200, 201)
    # Different dynamics (e.g., policy intervention reduced instability)
    N = 0.5 + 0.15 * np.sin(2 * np.pi * t / 100)
    W = 1.0 - 0.2 * np.sin(2 * np.pi * t / 100)
    S = 1.0 - 0.1 * np.sin(2 * np.pi * t / 100)
    psi = 0.3 + 0.2 * np.sin(2 * np.pi * t / 100 + np.pi)

    return pd.DataFrame({"t": t, "N": N, "W": W, "S": S, "psi": psi})


class TestAnimateTimeSeries:
    """Tests for animate_time_series function."""

    def test_basic_time_series_animation(self) -> None:
        """Test basic time series animation creation."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_time_series(
            df,
            variables=["N", "W"],
            title="Test Animation",
            duration_ms=1000,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.frames) > 0
        assert fig.layout.title.text == "Test Animation"

    def test_time_series_default_variables(self) -> None:
        """Test time series animation with default variables."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_time_series(df, duration_ms=500)

        assert isinstance(fig, go.Figure)
        assert len(fig.frames) > 0

    def test_time_series_subplot_layout(self) -> None:
        """Test time series animation with subplot layout."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_time_series(
            df,
            variables=["N", "W", "psi"],
            subplot_layout=True,
            duration_ms=500,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.frames) > 0

    def test_time_series_custom_labels(self) -> None:
        """Test time series animation with custom labels."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_time_series(
            df,
            variables=["N", "W"],
            labels=["Population", "Wages"],
            duration_ms=500,
        )

        assert isinstance(fig, go.Figure)

    def test_time_series_custom_colors(self) -> None:
        """Test time series animation with custom colors."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_time_series(
            df,
            variables=["N", "W"],
            colors=["red", "blue"],
            duration_ms=500,
        )

        assert isinstance(fig, go.Figure)

    def test_time_series_missing_variable_raises(self) -> None:
        """Test that missing variable raises ValueError."""
        df = sample_simulation_df()

        with pytest.raises(ValueError, match="not found"):
            plotly_animations.animate_time_series(
                df, variables=["nonexistent"], duration_ms=500
            )

    def test_time_series_has_play_button(self) -> None:
        """Test that time series animation has play/pause button."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_time_series(
            df, variables=["N"], duration_ms=500, show_play_button=True
        )

        assert fig.layout.updatemenus is not None
        assert len(fig.layout.updatemenus) > 0

    def test_time_series_has_slider(self) -> None:
        """Test that time series animation has slider."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_time_series(
            df, variables=["N"], duration_ms=500
        )

        assert fig.layout.sliders is not None
        assert len(fig.layout.sliders) > 0


class TestAnimatePhaseSpace:
    """Tests for animate_phase_space function."""

    def test_basic_phase_space_animation(self) -> None:
        """Test basic 2D phase space animation."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_phase_space(df, x="W", y="psi", duration_ms=500)

        assert isinstance(fig, go.Figure)
        assert len(fig.frames) > 0

    def test_phase_space_custom_trail_length(self) -> None:
        """Test phase space animation with custom trail length."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_phase_space(
            df, x="W", y="psi", trail_length=20, duration_ms=500
        )

        assert isinstance(fig, go.Figure)

    def test_phase_space_custom_colorscale(self) -> None:
        """Test phase space animation with custom colorscale."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_phase_space(
            df, x="W", y="psi", colorscale="Plasma", duration_ms=500
        )

        assert isinstance(fig, go.Figure)

    def test_phase_space_no_start_marker(self) -> None:
        """Test phase space animation without start marker."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_phase_space(
            df, x="W", y="psi", show_start=False, duration_ms=500
        )

        assert isinstance(fig, go.Figure)

    def test_phase_space_missing_variable_raises(self) -> None:
        """Test that missing variable raises ValueError."""
        df = sample_simulation_df()

        with pytest.raises(ValueError, match="not in"):
            plotly_animations.animate_phase_space(
                df, x="nonexistent", y="psi", duration_ms=500
            )

    def test_phase_space_missing_color_variable_raises(self) -> None:
        """Test that missing color variable raises ValueError."""
        df = sample_simulation_df()

        with pytest.raises(ValueError, match="not in"):
            plotly_animations.animate_phase_space(
                df, x="W", y="psi", color_by="nonexistent", duration_ms=500
            )

    def test_phase_space_auto_title(self) -> None:
        """Test that phase space generates auto title."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_phase_space(df, x="W", y="psi", duration_ms=500)

        assert "Phase Space" in fig.layout.title.text


class TestAnimatePhaseSpace3D:
    """Tests for animate_phase_space_3d function."""

    def test_basic_3d_phase_space_animation(self) -> None:
        """Test basic 3D phase space animation."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_phase_space_3d(
            df, x="N", y="W", z="psi", duration_ms=500
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.frames) > 0

    def test_3d_phase_space_no_camera_orbit(self) -> None:
        """Test 3D phase space animation without camera orbit."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_phase_space_3d(
            df, x="N", y="W", z="psi", camera_orbit=False, duration_ms=500
        )

        assert isinstance(fig, go.Figure)

    def test_3d_phase_space_custom_camera(self) -> None:
        """Test 3D phase space animation with custom camera."""
        df = sample_simulation_df()
        custom_camera = dict(eye=dict(x=2, y=2, z=1))
        fig = plotly_animations.animate_phase_space_3d(
            df,
            x="N",
            y="W",
            z="psi",
            initial_camera=custom_camera,
            camera_orbit=False,
            duration_ms=500,
        )

        assert isinstance(fig, go.Figure)

    def test_3d_phase_space_custom_trail(self) -> None:
        """Test 3D phase space animation with custom trail length."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_phase_space_3d(
            df, x="N", y="W", z="psi", trail_length=30, duration_ms=500
        )

        assert isinstance(fig, go.Figure)

    def test_3d_phase_space_missing_variable_raises(self) -> None:
        """Test that missing variable raises ValueError."""
        df = sample_simulation_df()

        with pytest.raises(ValueError, match="not in"):
            plotly_animations.animate_phase_space_3d(
                df, x="N", y="W", z="nonexistent", duration_ms=500
            )

    def test_3d_phase_space_has_scene_axes(self) -> None:
        """Test that 3D animation has proper scene axes."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_phase_space_3d(
            df, x="N", y="W", z="psi", duration_ms=500
        )

        assert fig.layout.scene.xaxis.title.text is not None
        assert fig.layout.scene.yaxis.title.text is not None
        assert fig.layout.scene.zaxis.title.text is not None


class TestAnimateSecularCycles:
    """Tests for animate_secular_cycles function."""

    def test_basic_secular_cycles_animation(self) -> None:
        """Test basic secular cycles animation."""
        df = sample_simulation_df()
        detected = cycles.detect_secular_cycles(df["psi"], df["t"])

        fig = plotly_animations.animate_secular_cycles(df, detected, duration_ms=500)

        assert isinstance(fig, go.Figure)
        assert len(fig.frames) > 0

    def test_secular_cycles_without_phases(self) -> None:
        """Test secular cycles animation without phase highlighting."""
        df = sample_simulation_df()
        detected = cycles.detect_secular_cycles(df["psi"], df["t"])

        fig = plotly_animations.animate_secular_cycles(
            df, detected, show_phase_transitions=False, duration_ms=500
        )

        assert isinstance(fig, go.Figure)

    def test_secular_cycles_custom_variable(self) -> None:
        """Test secular cycles animation with different variable."""
        df = sample_simulation_df()
        detected = cycles.detect_secular_cycles(df["W"], df["t"])

        fig = plotly_animations.animate_secular_cycles(
            df, detected, variable="W", duration_ms=500
        )

        assert isinstance(fig, go.Figure)

    def test_secular_cycles_missing_variable_raises(self) -> None:
        """Test that missing variable raises ValueError."""
        df = sample_simulation_df()
        detected = cycles.detect_secular_cycles(df["psi"], df["t"])

        with pytest.raises(ValueError, match="not in"):
            plotly_animations.animate_secular_cycles(
                df, detected, variable="nonexistent", duration_ms=500
            )


class TestAnimateComparison:
    """Tests for animate_comparison function."""

    def test_basic_comparison_animation_overlay(self) -> None:
        """Test basic comparison animation with overlay layout."""
        baseline = sample_simulation_df()
        counterfactual = sample_counterfactual_df()

        fig = plotly_animations.animate_comparison(
            baseline,
            counterfactual,
            variables=["N", "psi"],
            layout="overlay",
            duration_ms=500,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.frames) > 0

    def test_comparison_animation_side_by_side(self) -> None:
        """Test comparison animation with side-by-side layout."""
        baseline = sample_simulation_df()
        counterfactual = sample_counterfactual_df()

        fig = plotly_animations.animate_comparison(
            baseline,
            counterfactual,
            variables=["N", "psi"],
            layout="side_by_side",
            duration_ms=500,
        )

        assert isinstance(fig, go.Figure)

    def test_comparison_custom_labels(self) -> None:
        """Test comparison animation with custom labels."""
        baseline = sample_simulation_df()
        counterfactual = sample_counterfactual_df()

        fig = plotly_animations.animate_comparison(
            baseline,
            counterfactual,
            variables=["psi"],
            labels=("Baseline", "With Policy"),
            duration_ms=500,
        )

        assert isinstance(fig, go.Figure)

    def test_comparison_custom_colors(self) -> None:
        """Test comparison animation with custom colors."""
        baseline = sample_simulation_df()
        counterfactual = sample_counterfactual_df()

        fig = plotly_animations.animate_comparison(
            baseline,
            counterfactual,
            variables=["psi"],
            colors=("#FF0000", "#00FF00"),
            duration_ms=500,
        )

        assert isinstance(fig, go.Figure)

    def test_comparison_single_variable(self) -> None:
        """Test comparison animation with single variable."""
        baseline = sample_simulation_df()
        counterfactual = sample_counterfactual_df()

        fig = plotly_animations.animate_comparison(
            baseline,
            counterfactual,
            variables=["psi"],
            duration_ms=500,
        )

        assert isinstance(fig, go.Figure)

    def test_comparison_missing_variable_raises(self) -> None:
        """Test that missing variable raises ValueError."""
        baseline = sample_simulation_df()
        counterfactual = sample_counterfactual_df()

        with pytest.raises(ValueError, match="not in baseline"):
            plotly_animations.animate_comparison(
                baseline,
                counterfactual,
                variables=["nonexistent"],
                duration_ms=500,
            )


class TestAnimateParameterSensitivity:
    """Tests for animate_parameter_sensitivity function."""

    def test_basic_parameter_sensitivity_animation(self) -> None:
        """Test basic parameter sensitivity animation."""
        # Create results for different parameter values
        results = {}
        for r in [0.01, 0.02, 0.03]:
            t = np.linspace(0, 100, 101)
            psi = 0.5 + 0.3 * r * 10 * np.sin(2 * np.pi * t / 50)
            results[f"r={r}"] = pd.DataFrame({"t": t, "psi": psi})

        fig = plotly_animations.animate_parameter_sensitivity(
            results, variable="psi", parameter_name="Growth Rate (r)", duration_ms=500
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 3  # One frame per parameter value

    def test_parameter_sensitivity_custom_colorscale(self) -> None:
        """Test parameter sensitivity with custom colorscale."""
        results = {}
        for i, r in enumerate([0.01, 0.02]):
            t = np.linspace(0, 100, 101)
            psi = 0.5 + 0.3 * (i + 1) * np.sin(2 * np.pi * t / 50)
            results[f"r={r}"] = pd.DataFrame({"t": t, "psi": psi})

        fig = plotly_animations.animate_parameter_sensitivity(
            results, variable="psi", colorscale="Plasma", duration_ms=500
        )

        assert isinstance(fig, go.Figure)

    def test_parameter_sensitivity_empty_dict_raises(self) -> None:
        """Test that empty results dict raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            plotly_animations.animate_parameter_sensitivity(
                {}, variable="psi", duration_ms=500
            )

    def test_parameter_sensitivity_missing_variable_raises(self) -> None:
        """Test that missing variable raises ValueError."""
        results = {"r=0.01": pd.DataFrame({"t": [0, 1], "N": [0.5, 0.6]})}

        with pytest.raises(ValueError, match="not in"):
            plotly_animations.animate_parameter_sensitivity(
                results, variable="psi", duration_ms=500
            )


class TestSaveAnimation:
    """Tests for save_animation function."""

    def test_save_html(self) -> None:
        """Test saving animation as HTML."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_time_series(
            df, variables=["N"], duration_ms=500
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.html"
            saved = plotly_animations.save_animation(fig, path)

            assert saved.exists()
            assert saved.suffix == ".html"
            # Check file is not empty
            assert saved.stat().st_size > 0

    def test_save_html_inferred_format(self) -> None:
        """Test saving animation with format inferred from path."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_time_series(
            df, variables=["N"], duration_ms=500
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.html"
            saved = plotly_animations.save_animation(fig, path)

            assert saved.exists()
            assert saved.suffix == ".html"

    def test_save_creates_parent_directories(self) -> None:
        """Test that save creates parent directories if needed."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_time_series(
            df, variables=["N"], duration_ms=500
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "test.html"
            saved = plotly_animations.save_animation(fig, path)

            assert saved.exists()
            assert saved.parent.exists()

    def test_save_adds_extension_if_missing(self) -> None:
        """Test that save adds correct extension if missing."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_time_series(
            df, variables=["N"], duration_ms=500
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test"
            saved = plotly_animations.save_animation(fig, path, format="html")

            assert saved.exists()
            assert saved.suffix == ".html"

    def test_save_unsupported_format_raises(self) -> None:
        """Test that unsupported format raises ValueError."""
        df = sample_simulation_df()
        fig = plotly_animations.animate_time_series(
            df, variables=["N"], duration_ms=500
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.avi"

            with pytest.raises(ValueError, match="Unsupported format"):
                plotly_animations.save_animation(fig, path, format="avi")


class TestVerificationReminder:
    """Tests for verification reminder."""

    def test_verification_reminder_exists(self) -> None:
        """Test that verification reminder is defined."""
        assert hasattr(plotly_animations, "VERIFICATION_REMINDER")
        assert "IMPORTANT" in plotly_animations.VERIFICATION_REMINDER
        assert "verify" in plotly_animations.VERIFICATION_REMINDER.lower()


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_defined(self) -> None:
        """Test that all expected exports are in __all__."""
        expected_exports = [
            "animate_time_series",
            "animate_phase_space",
            "animate_phase_space_3d",
            "animate_secular_cycles",
            "animate_comparison",
            "animate_parameter_sensitivity",
            "save_animation",
            "VERIFICATION_REMINDER",
            "DEFAULT_COLORS",
            "VARIABLE_LABELS",
        ]

        for export in expected_exports:
            assert export in plotly_animations.__all__
            assert hasattr(plotly_animations, export)

    def test_default_colors_defined(self) -> None:
        """Test that default colors are defined."""
        assert len(plotly_animations.DEFAULT_COLORS) >= 7
        # Check colors are valid hex
        for color in plotly_animations.DEFAULT_COLORS:
            assert color.startswith("#")
            assert len(color) == 7

    def test_variable_labels_defined(self) -> None:
        """Test that variable labels are defined."""
        expected_vars = ["N", "E", "W", "S", "psi", "t"]
        for var in expected_vars:
            assert var in plotly_animations.VARIABLE_LABELS
