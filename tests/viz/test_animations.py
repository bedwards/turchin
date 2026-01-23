"""Tests for the animations module.

Tests cover:
- Time series animation
- Phase space animation (2D and 3D)
- Secular cycles animation
- Scenario comparison animation
- Animation export (GIF)
"""

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.animation import FuncAnimation

from cliodynamics.viz import animations, cycles


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
        anim = animations.animate_time_series(
            df,
            variables=["N", "W"],
            title="Test Animation",
            duration_seconds=1.0,
            fps=10,
        )

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_time_series_default_variables(self) -> None:
        """Test time series animation with default variables."""
        df = sample_simulation_df()
        anim = animations.animate_time_series(df, duration_seconds=0.5, fps=10)

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_time_series_subplot_layout(self) -> None:
        """Test time series animation with subplot layout."""
        df = sample_simulation_df()
        anim = animations.animate_time_series(
            df,
            variables=["N", "W", "psi"],
            subplot_layout=True,
            duration_seconds=0.5,
            fps=10,
        )

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_time_series_custom_labels(self) -> None:
        """Test time series animation with custom labels."""
        df = sample_simulation_df()
        anim = animations.animate_time_series(
            df,
            variables=["N", "W"],
            labels=["Population", "Wages"],
            duration_seconds=0.5,
            fps=10,
        )

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_time_series_custom_colors(self) -> None:
        """Test time series animation with custom colors."""
        df = sample_simulation_df()
        anim = animations.animate_time_series(
            df,
            variables=["N", "W"],
            colors=["red", "blue"],
            duration_seconds=0.5,
            fps=10,
        )

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_time_series_missing_variable_raises(self) -> None:
        """Test that missing variable raises ValueError."""
        df = sample_simulation_df()

        with pytest.raises(ValueError, match="not found"):
            animations.animate_time_series(
                df, variables=["nonexistent"], duration_seconds=0.5, fps=10
            )


class TestAnimatePhaseSpace:
    """Tests for animate_phase_space function."""

    def test_basic_phase_space_animation(self) -> None:
        """Test basic 2D phase space animation."""
        df = sample_simulation_df()
        anim = animations.animate_phase_space(
            df, x="W", y="psi", duration_seconds=0.5, fps=10
        )

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_phase_space_custom_trail_length(self) -> None:
        """Test phase space animation with custom trail length."""
        df = sample_simulation_df()
        anim = animations.animate_phase_space(
            df, x="W", y="psi", trail_length=20, duration_seconds=0.5, fps=10
        )

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_phase_space_custom_colormap(self) -> None:
        """Test phase space animation with custom colormap."""
        df = sample_simulation_df()
        anim = animations.animate_phase_space(
            df, x="W", y="psi", colormap="plasma", duration_seconds=0.5, fps=10
        )

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_phase_space_no_start_marker(self) -> None:
        """Test phase space animation without start marker."""
        df = sample_simulation_df()
        anim = animations.animate_phase_space(
            df, x="W", y="psi", show_start=False, duration_seconds=0.5, fps=10
        )

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_phase_space_missing_variable_raises(self) -> None:
        """Test that missing variable raises ValueError."""
        df = sample_simulation_df()

        with pytest.raises(ValueError, match="not in"):
            animations.animate_phase_space(
                df, x="nonexistent", y="psi", duration_seconds=0.5, fps=10
            )

    def test_phase_space_missing_color_variable_raises(self) -> None:
        """Test that missing color variable raises ValueError."""
        df = sample_simulation_df()

        with pytest.raises(ValueError, match="not in"):
            animations.animate_phase_space(
                df, x="W", y="psi", color_by="nonexistent", duration_seconds=0.5, fps=10
            )


class TestAnimatePhaseSpace3D:
    """Tests for animate_phase_space_3d function."""

    def test_basic_3d_phase_space_animation(self) -> None:
        """Test basic 3D phase space animation."""
        df = sample_simulation_df()
        anim = animations.animate_phase_space_3d(
            df, x="N", y="W", z="psi", duration_seconds=0.5, fps=10
        )

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_3d_phase_space_no_rotation(self) -> None:
        """Test 3D phase space animation without view rotation."""
        df = sample_simulation_df()
        anim = animations.animate_phase_space_3d(
            df, x="N", y="W", z="psi", rotate_view=False, duration_seconds=0.5, fps=10
        )

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_3d_phase_space_custom_view(self) -> None:
        """Test 3D phase space animation with custom view angles."""
        df = sample_simulation_df()
        anim = animations.animate_phase_space_3d(
            df,
            x="N",
            y="W",
            z="psi",
            elevation=45,
            initial_azimuth=90,
            rotation_speed=2.0,
            duration_seconds=0.5,
            fps=10,
        )

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_3d_phase_space_custom_trail(self) -> None:
        """Test 3D phase space animation with custom trail length."""
        df = sample_simulation_df()
        anim = animations.animate_phase_space_3d(
            df, x="N", y="W", z="psi", trail_length=30, duration_seconds=0.5, fps=10
        )

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_3d_phase_space_missing_variable_raises(self) -> None:
        """Test that missing variable raises ValueError."""
        df = sample_simulation_df()

        with pytest.raises(ValueError, match="not in"):
            animations.animate_phase_space_3d(
                df, x="N", y="W", z="nonexistent", duration_seconds=0.5, fps=10
            )


class TestAnimateSecularCycles:
    """Tests for animate_secular_cycles function."""

    def test_basic_secular_cycles_animation(self) -> None:
        """Test basic secular cycles animation."""
        df = sample_simulation_df()
        detected = cycles.detect_secular_cycles(df["psi"], df["t"])

        anim = animations.animate_secular_cycles(
            df, detected, duration_seconds=0.5, fps=10
        )

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_secular_cycles_without_phases(self) -> None:
        """Test secular cycles animation without phase highlighting."""
        df = sample_simulation_df()
        detected = cycles.detect_secular_cycles(df["psi"], df["t"])

        anim = animations.animate_secular_cycles(
            df, detected, show_phase_transitions=False, duration_seconds=0.5, fps=10
        )

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_secular_cycles_custom_variable(self) -> None:
        """Test secular cycles animation with different variable."""
        df = sample_simulation_df()
        detected = cycles.detect_secular_cycles(df["W"], df["t"])

        anim = animations.animate_secular_cycles(
            df, detected, variable="W", duration_seconds=0.5, fps=10
        )

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_secular_cycles_missing_variable_raises(self) -> None:
        """Test that missing variable raises ValueError."""
        df = sample_simulation_df()
        detected = cycles.detect_secular_cycles(df["psi"], df["t"])

        with pytest.raises(ValueError, match="not in"):
            animations.animate_secular_cycles(
                df, detected, variable="nonexistent", duration_seconds=0.5, fps=10
            )


class TestAnimateComparison:
    """Tests for animate_comparison function."""

    def test_basic_comparison_animation_overlay(self) -> None:
        """Test basic comparison animation with overlay layout."""
        baseline = sample_simulation_df()
        counterfactual = sample_counterfactual_df()

        anim = animations.animate_comparison(
            baseline,
            counterfactual,
            variables=["N", "psi"],
            layout="overlay",
            duration_seconds=0.5,
            fps=10,
        )

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_comparison_animation_side_by_side(self) -> None:
        """Test comparison animation with side-by-side layout."""
        baseline = sample_simulation_df()
        counterfactual = sample_counterfactual_df()

        anim = animations.animate_comparison(
            baseline,
            counterfactual,
            variables=["N", "psi"],
            layout="side_by_side",
            duration_seconds=0.5,
            fps=10,
        )

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_comparison_custom_labels(self) -> None:
        """Test comparison animation with custom labels."""
        baseline = sample_simulation_df()
        counterfactual = sample_counterfactual_df()

        anim = animations.animate_comparison(
            baseline,
            counterfactual,
            variables=["psi"],
            labels=("Baseline", "With Policy"),
            duration_seconds=0.5,
            fps=10,
        )

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_comparison_custom_colors(self) -> None:
        """Test comparison animation with custom colors."""
        baseline = sample_simulation_df()
        counterfactual = sample_counterfactual_df()

        anim = animations.animate_comparison(
            baseline,
            counterfactual,
            variables=["psi"],
            colors=("#FF0000", "#00FF00"),
            duration_seconds=0.5,
            fps=10,
        )

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_comparison_single_variable(self) -> None:
        """Test comparison animation with single variable."""
        baseline = sample_simulation_df()
        counterfactual = sample_counterfactual_df()

        anim = animations.animate_comparison(
            baseline,
            counterfactual,
            variables=["psi"],
            duration_seconds=0.5,
            fps=10,
        )

        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_comparison_missing_variable_raises(self) -> None:
        """Test that missing variable raises ValueError."""
        baseline = sample_simulation_df()
        counterfactual = sample_counterfactual_df()

        with pytest.raises(ValueError, match="not in baseline"):
            animations.animate_comparison(
                baseline,
                counterfactual,
                variables=["nonexistent"],
                duration_seconds=0.5,
                fps=10,
            )


class TestSaveAnimation:
    """Tests for save_animation function."""

    def test_save_gif(self) -> None:
        """Test saving animation as GIF."""
        df = sample_simulation_df()
        anim = animations.animate_time_series(
            df, variables=["N"], duration_seconds=0.5, fps=5
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gif"
            saved = animations.save_animation(anim, path, fps=5)

            assert saved.exists()
            assert saved.suffix == ".gif"
            # Check file is not empty
            assert saved.stat().st_size > 0

        plt.close("all")

    def test_save_gif_inferred_format(self) -> None:
        """Test saving animation with format inferred from path."""
        df = sample_simulation_df()
        anim = animations.animate_time_series(
            df, variables=["N"], duration_seconds=0.5, fps=5
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gif"
            saved = animations.save_animation(anim, path, fps=5)

            assert saved.exists()
            assert saved.suffix == ".gif"

        plt.close("all")

    def test_save_creates_parent_directories(self) -> None:
        """Test that save creates parent directories if needed."""
        df = sample_simulation_df()
        anim = animations.animate_time_series(
            df, variables=["N"], duration_seconds=0.5, fps=5
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "test.gif"
            saved = animations.save_animation(anim, path, fps=5)

            assert saved.exists()
            assert saved.parent.exists()

        plt.close("all")

    def test_save_adds_extension_if_missing(self) -> None:
        """Test that save adds correct extension if missing."""
        df = sample_simulation_df()
        anim = animations.animate_time_series(
            df, variables=["N"], duration_seconds=0.5, fps=5
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test"
            saved = animations.save_animation(anim, path, format="gif", fps=5)

            assert saved.exists()
            assert saved.suffix == ".gif"

        plt.close("all")

    def test_save_unsupported_format_raises(self) -> None:
        """Test that unsupported format raises ValueError."""
        df = sample_simulation_df()
        anim = animations.animate_time_series(
            df, variables=["N"], duration_seconds=0.5, fps=5
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.avi"

            with pytest.raises(ValueError, match="Unsupported format"):
                animations.save_animation(anim, path, format="avi", fps=5)

        plt.close("all")


class TestVerificationReminder:
    """Tests for verification reminder."""

    def test_verification_reminder_exists(self) -> None:
        """Test that verification reminder is defined."""
        assert hasattr(animations, "VERIFICATION_REMINDER")
        assert "IMPORTANT" in animations.VERIFICATION_REMINDER
        assert "verify" in animations.VERIFICATION_REMINDER.lower()
