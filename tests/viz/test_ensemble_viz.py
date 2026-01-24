"""Tests for ensemble visualization module.

Tests cover:
- Phase diagram generation
- Stability map visualization
- Bifurcation diagrams
- Stability boundary plots
- Parameter sensitivity grids
"""

import numpy as np
import pandas as pd
import pytest

from cliodynamics.simulation.ensemble import (
    EnsembleResults,
    SimulationOutcome,
    StabilityClassification,
)
from cliodynamics.viz import ensemble as ens_viz


@pytest.fixture
def sample_results() -> EnsembleResults:
    """Create sample EnsembleResults for testing visualizations."""
    # Create synthetic outcomes with varying stability
    outcomes = []
    alpha_values = np.array([0.005, 0.010, 0.015, 0.020])
    gamma_values = np.array([0.01, 0.02, 0.03])

    for i, alpha in enumerate(alpha_values):
        for j, gamma in enumerate(gamma_values):
            # Create varied max_psi values for interesting patterns
            # Higher alpha and gamma lead to higher instability
            max_psi = alpha * 80 + gamma * 30 + np.random.uniform(0, 0.2)
            mean_psi = max_psi * 0.5

            if max_psi >= 2.0:
                classification = StabilityClassification.COLLAPSE
            elif max_psi >= 1.0:
                classification = StabilityClassification.UNSTABLE
            else:
                classification = StabilityClassification.STABLE

            outcome = SimulationOutcome(
                parameters={"alpha": alpha, "gamma": gamma},
                final_state={
                    "N": 0.6 - alpha * 10,
                    "E": 0.08 + alpha * 2,
                    "W": 0.9 - gamma * 5,
                    "S": 0.8 - alpha * 5,
                    "psi": max_psi * 0.8,
                },
                max_psi=max_psi,
                mean_psi=mean_psi,
                classification=classification,
                metadata={"psi_std": 0.1, "psi_final": max_psi * 0.8},
            )
            outcomes.append(outcome)

    return EnsembleResults(
        outcomes=outcomes,
        parameter_grid={
            "alpha": alpha_values,
            "gamma": gamma_values,
        },
        grid_shape=(4, 3),
        parameter_names=["alpha", "gamma"],
        parameter_values={
            "alpha": alpha_values,
            "gamma": gamma_values,
        },
        initial_conditions={
            "N": 0.5,
            "E": 0.05,
            "W": 1.0,
            "S": 1.0,
            "psi": 0.0,
        },
        time_span=(0, 100),
        n_simulations=12,
        n_successful=12,
        n_failed=0,
    )


class TestPhaseDiagram:
    """Tests for phase diagram visualization."""

    def test_plot_phase_diagram(self, sample_results: EnsembleResults) -> None:
        """Test phase diagram generation."""
        chart = ens_viz.plot_phase_diagram(
            sample_results,
            x_param="alpha",
            y_param="gamma",
            metric="max_psi",
        )

        # Check chart structure
        assert chart is not None
        # Altair LayerChart or Chart
        assert hasattr(chart, "to_dict")

    def test_phase_diagram_with_boundary(self, sample_results: EnsembleResults) -> None:
        """Test phase diagram with stability boundary overlay."""
        chart = ens_viz.plot_phase_diagram(
            sample_results,
            x_param="alpha",
            y_param="gamma",
            show_boundary=True,
            psi_threshold=1.0,
        )

        assert chart is not None

    def test_phase_diagram_custom_colormap(
        self, sample_results: EnsembleResults
    ) -> None:
        """Test phase diagram with custom colormap."""
        chart = ens_viz.plot_phase_diagram(
            sample_results,
            x_param="alpha",
            y_param="gamma",
            colormap="plasma",
            title="Custom Colormap Test",
        )

        assert chart is not None


class TestStabilityMap:
    """Tests for stability map visualization."""

    def test_plot_stability_map(self, sample_results: EnsembleResults) -> None:
        """Test stability map generation."""
        chart = ens_viz.plot_stability_map(
            sample_results,
            x_param="alpha",
            y_param="gamma",
            psi_threshold=1.0,
        )

        assert chart is not None
        assert hasattr(chart, "to_dict")

    def test_stability_map_with_thresholds(
        self, sample_results: EnsembleResults
    ) -> None:
        """Test stability map with custom thresholds."""
        chart = ens_viz.plot_stability_map(
            sample_results,
            x_param="alpha",
            y_param="gamma",
            psi_threshold=0.8,
            collapse_threshold=2.5,
        )

        assert chart is not None


class TestBifurcationDiagram:
    """Tests for bifurcation diagram visualization."""

    def test_plot_bifurcation_diagram(self, sample_results: EnsembleResults) -> None:
        """Test bifurcation diagram generation."""
        chart = ens_viz.plot_bifurcation_diagram(
            sample_results,
            parameter="alpha",
            metric="max_psi",
        )

        assert chart is not None
        assert hasattr(chart, "to_dict")

    def test_bifurcation_with_fixed_params(
        self, sample_results: EnsembleResults
    ) -> None:
        """Test bifurcation diagram with fixed parameters."""
        chart = ens_viz.plot_bifurcation_diagram(
            sample_results,
            parameter="alpha",
            fixed_params={"gamma": 0.02},
            show_threshold=True,
            psi_threshold=1.0,
        )

        assert chart is not None

    def test_bifurcation_without_threshold(
        self, sample_results: EnsembleResults
    ) -> None:
        """Test bifurcation diagram without threshold line."""
        chart = ens_viz.plot_bifurcation_diagram(
            sample_results,
            parameter="alpha",
            show_threshold=False,
        )

        assert chart is not None


class TestStabilityBoundary:
    """Tests for stability boundary visualization."""

    def test_plot_stability_boundary(self, sample_results: EnsembleResults) -> None:
        """Test stability boundary plot."""
        chart = ens_viz.plot_stability_boundary(
            sample_results,
            x_param="alpha",
            y_param="gamma",
            psi_threshold=1.0,
        )

        assert chart is not None
        assert hasattr(chart, "to_dict")


class TestParameterSensitivity:
    """Tests for parameter sensitivity visualization."""

    def test_plot_parameter_sensitivity_grid(
        self, sample_results: EnsembleResults
    ) -> None:
        """Test parameter sensitivity grid plot."""
        chart = ens_viz.plot_parameter_sensitivity_grid(
            sample_results,
            metric="max_psi",
        )

        assert chart is not None


class TestStabilityFraction:
    """Tests for stability fraction visualization."""

    def test_plot_stability_fraction(self, sample_results: EnsembleResults) -> None:
        """Test stability fraction plot."""
        chart = ens_viz.plot_stability_fraction(
            sample_results,
            parameter="alpha",
            psi_threshold=1.0,
        )

        assert chart is not None
        assert hasattr(chart, "to_dict")


class TestOutcomeDistribution:
    """Tests for outcome distribution visualization."""

    def test_plot_outcome_distribution(self, sample_results: EnsembleResults) -> None:
        """Test outcome distribution histogram."""
        chart = ens_viz.plot_outcome_distribution(
            sample_results,
            metric="max_psi",
            bins=20,
        )

        assert chart is not None
        assert hasattr(chart, "to_dict")


class TestLabels:
    """Tests for label functions."""

    def test_parameter_labels(self) -> None:
        """Test that common parameters have labels."""
        assert "alpha" in ens_viz.PARAMETER_LABELS
        assert "gamma" in ens_viz.PARAMETER_LABELS
        assert "lambda_psi" in ens_viz.PARAMETER_LABELS

    def test_metric_labels(self) -> None:
        """Test that common metrics have labels."""
        assert "max_psi" in ens_viz.METRIC_LABELS
        assert "mean_psi" in ens_viz.METRIC_LABELS

    def test_stability_colors(self) -> None:
        """Test stability color mapping."""
        assert "stable" in ens_viz.STABILITY_COLORS
        assert "unstable" in ens_viz.STABILITY_COLORS
        assert "collapse" in ens_viz.STABILITY_COLORS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
