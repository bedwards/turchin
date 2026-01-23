"""Secular cycle detection and visualization for cliodynamics analysis.

This module provides tools for detecting and visualizing secular cycles
(long-term periodic patterns) in SDT model output, following the methodology
in Turchin & Nefedov (2009) Secular Cycles.

Secular cycles are multi-generational oscillations (typically 100-300 years)
in societal dynamics driven by demographic-structural pressures. This module
identifies cycle phases:
- Expansion: Growth, low instability
- Stagflation: Stagnation, rising pressures
- Crisis: Peak instability, potential collapse
- Depression/Intercycle: Recovery, low population

IMPORTANT: After generating any plot, visually verify it looks correct
before committing.

Example:
    >>> from cliodynamics.viz.cycles import detect_secular_cycles, plot_with_cycles
    >>> cycles = detect_secular_cycles(results['psi'])
    >>> fig = plot_with_cycles(results, cycles)
    >>> fig.savefig('cycles.png')
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from pandas import DataFrame, Series

    from cliodynamics.simulation import SimulationResult

logger = logging.getLogger(__name__)


class CyclePhase(Enum):
    """Phase of a secular cycle.

    Following Turchin & Nefedov (2009), secular cycles consist of:
    - EXPANSION: Population growth, economic development, state formation
    - STAGFLATION: Stagnation, rising inequality, elite overproduction
    - CRISIS: Instability peaks, potential state collapse
    - DEPRESSION: Population decline, recovery begins
    """

    EXPANSION = "expansion"
    STAGFLATION = "stagflation"
    CRISIS = "crisis"
    DEPRESSION = "depression"


# Colors for cycle phases
PHASE_COLORS = {
    CyclePhase.EXPANSION: "#90EE90",  # Light green
    CyclePhase.STAGFLATION: "#FFD700",  # Gold
    CyclePhase.CRISIS: "#FF6B6B",  # Light red
    CyclePhase.DEPRESSION: "#87CEEB",  # Sky blue
}


@dataclass
class CyclePoint:
    """A significant point in a secular cycle (peak or trough).

    Attributes:
        time: Time at which the point occurs.
        value: Value of the variable at the point.
        point_type: Type of point ('peak' or 'trough').
        index: Index in the original time series.
    """

    time: float
    value: float
    point_type: str  # 'peak' or 'trough'
    index: int


@dataclass
class SecularCycle:
    """Represents a detected secular cycle.

    Attributes:
        start_time: Beginning of the cycle (typically a trough).
        end_time: End of the cycle (next trough).
        peak_time: Time of peak instability.
        peak_value: Maximum instability value in cycle.
        trough_value: Minimum instability at cycle start.
        duration: Length of cycle in time units.
        amplitude: Peak-to-trough difference.
        phases: List of (start_time, end_time, phase) tuples.
    """

    start_time: float
    end_time: float
    peak_time: float
    peak_value: float
    trough_value: float
    duration: float = field(init=False)
    amplitude: float = field(init=False)
    phases: list[tuple[float, float, CyclePhase]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Compute derived quantities."""
        self.duration = self.end_time - self.start_time
        self.amplitude = self.peak_value - self.trough_value


@dataclass
class CycleDetectionResult:
    """Results from secular cycle detection.

    Attributes:
        cycles: List of detected SecularCycle objects.
        peaks: List of CyclePoint objects for peaks.
        troughs: List of CyclePoint objects for troughs.
        mean_period: Average cycle period (time between troughs).
        mean_amplitude: Average peak-to-trough amplitude.
    """

    cycles: list[SecularCycle]
    peaks: list[CyclePoint]
    troughs: list[CyclePoint]
    mean_period: float | None = None
    mean_amplitude: float | None = None

    def __post_init__(self) -> None:
        """Compute summary statistics."""
        if self.cycles:
            self.mean_period = np.mean([c.duration for c in self.cycles])
            self.mean_amplitude = np.mean([c.amplitude for c in self.cycles])


def detect_peaks_and_troughs(
    values: ArrayLike,
    times: ArrayLike | None = None,
    min_prominence: float | None = None,
    min_distance: int = 1,
) -> tuple[list[CyclePoint], list[CyclePoint]]:
    """Detect peaks and troughs in a time series.

    Uses scipy's find_peaks for robust peak detection with prominence filtering.

    Args:
        values: 1D array of values to analyze.
        times: Optional time values corresponding to values. If None, uses indices.
        min_prominence: Minimum prominence for peak detection. If None, uses
            10% of value range.
        min_distance: Minimum number of points between peaks.

    Returns:
        Tuple of (peaks, troughs) where each is a list of CyclePoint objects.

    Example:
        >>> peaks, troughs = detect_peaks_and_troughs(results['psi'].values)
    """
    from scipy.signal import find_peaks

    values = np.asarray(values)
    if times is None:
        times = np.arange(len(values))
    else:
        times = np.asarray(times)

    # Determine prominence threshold
    if min_prominence is None:
        value_range = values.max() - values.min()
        min_prominence = value_range * 0.1 if value_range > 0 else 0.0

    # Find peaks
    peak_indices, peak_props = find_peaks(
        values, prominence=min_prominence, distance=min_distance
    )

    peaks = [
        CyclePoint(
            time=float(times[i]),
            value=float(values[i]),
            point_type="peak",
            index=int(i),
        )
        for i in peak_indices
    ]

    # Find troughs (peaks of negated signal)
    trough_indices, trough_props = find_peaks(
        -values, prominence=min_prominence, distance=min_distance
    )

    troughs = [
        CyclePoint(
            time=float(times[i]),
            value=float(values[i]),
            point_type="trough",
            index=int(i),
        )
        for i in trough_indices
    ]

    return peaks, troughs


def detect_secular_cycles(
    values: ArrayLike | Series,
    times: ArrayLike | None = None,
    min_prominence: float | None = None,
    min_cycle_length: float | None = None,
    phase_thresholds: dict[str, float] | None = None,
) -> CycleDetectionResult:
    """Detect secular cycles in a time series.

    Identifies complete cycles from trough to trough, with peak detection
    and optional phase classification.

    Args:
        values: Time series values (typically Political Stress Index psi).
        times: Optional time values. If None, uses integer indices.
        min_prominence: Minimum prominence for peak/trough detection.
        min_cycle_length: Minimum duration for a valid cycle.
        phase_thresholds: Dict with 'expansion_end', 'stagflation_end' fractions
            (relative to cycle duration) for phase boundaries.

    Returns:
        CycleDetectionResult with detected cycles and summary statistics.

    Example:
        >>> cycles = detect_secular_cycles(results['psi'])
        >>> print(f"Found {len(cycles.cycles)} cycles")
        >>> print(f"Mean period: {cycles.mean_period:.1f} years")
    """
    # Handle pandas Series
    if hasattr(values, "values"):
        values = values.values
    values = np.asarray(values)

    if times is None:
        times = np.arange(len(values))
    elif hasattr(times, "values"):
        times = times.values
    times = np.asarray(times)

    # Default phase thresholds (relative to cycle duration)
    if phase_thresholds is None:
        phase_thresholds = {
            "expansion_end": 0.3,  # First 30% is expansion
            "stagflation_end": 0.6,  # 30-60% is stagflation
            # 60-peak is crisis, peak-end is depression
        }

    # Detect peaks and troughs
    peaks, troughs = detect_peaks_and_troughs(values, times, min_prominence)

    # Build cycles from trough to trough
    cycles: list[SecularCycle] = []

    for i in range(len(troughs) - 1):
        trough1 = troughs[i]
        trough2 = troughs[i + 1]

        # Find peak between these troughs
        cycle_peaks = [p for p in peaks if trough1.time < p.time < trough2.time]

        if not cycle_peaks:
            continue  # No peak found, skip this cycle

        # Use the highest peak in this interval
        main_peak = max(cycle_peaks, key=lambda p: p.value)

        # Check minimum cycle length
        duration = trough2.time - trough1.time
        if min_cycle_length is not None and duration < min_cycle_length:
            continue

        # Classify phases
        phases = _classify_phases(
            trough1.time, trough2.time, main_peak.time, phase_thresholds
        )

        cycle = SecularCycle(
            start_time=trough1.time,
            end_time=trough2.time,
            peak_time=main_peak.time,
            peak_value=main_peak.value,
            trough_value=trough1.value,
            phases=phases,
        )
        cycles.append(cycle)

    return CycleDetectionResult(cycles=cycles, peaks=peaks, troughs=troughs)


def _classify_phases(
    start: float,
    end: float,
    peak: float,
    thresholds: dict[str, float],
) -> list[tuple[float, float, CyclePhase]]:
    """Classify phases within a secular cycle.

    Args:
        start: Cycle start time (trough).
        end: Cycle end time (next trough).
        peak: Peak time.
        thresholds: Phase boundary thresholds.

    Returns:
        List of (start, end, phase) tuples.
    """
    duration = end - start
    expansion_end = start + duration * thresholds.get("expansion_end", 0.3)
    stagflation_end = start + duration * thresholds.get("stagflation_end", 0.6)

    # Ensure phases don't overlap with peak
    expansion_end = min(expansion_end, peak - 0.01 * duration)
    stagflation_end = min(stagflation_end, peak - 0.001 * duration)
    stagflation_end = max(stagflation_end, expansion_end + 0.01 * duration)

    phases = [
        (start, expansion_end, CyclePhase.EXPANSION),
        (expansion_end, stagflation_end, CyclePhase.STAGFLATION),
        (stagflation_end, peak, CyclePhase.CRISIS),
        (peak, end, CyclePhase.DEPRESSION),
    ]

    return phases


def plot_with_cycles(
    results: SimulationResult | DataFrame,
    cycles: CycleDetectionResult,
    variable: str = "psi",
    time_column: str = "t",
    title: str = "Political Stress Index with Secular Cycles",
    figsize: tuple[float, float] = (12, 6),
    show_phases: bool = True,
    show_peaks: bool = True,
    show_troughs: bool = True,
    annotate_cycles: bool = True,
) -> Figure:
    """Plot time series with secular cycles highlighted.

    Creates a time series plot with cycle phases shown as shaded regions
    and peaks/troughs marked.

    Args:
        results: SimulationResult or DataFrame with time series data.
        cycles: CycleDetectionResult from detect_secular_cycles.
        variable: Variable to plot (typically 'psi').
        time_column: Name of time column.
        title: Plot title.
        figsize: Figure size (width, height) in inches.
        show_phases: If True, shade cycle phases with colors.
        show_peaks: If True, mark peak points.
        show_troughs: If True, mark trough points.
        annotate_cycles: If True, add cycle number labels.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> cycles = detect_secular_cycles(results['psi'])
        >>> fig = plot_with_cycles(results, cycles)
        >>> fig.savefig('cycles.png')
    """
    # Get DataFrame from results
    if hasattr(results, "df"):
        df = results.df
    else:
        df = results

    t = df[time_column].values
    y = df[variable].values

    fig, ax = plt.subplots(figsize=figsize)

    # Set y limits first
    y_margin = 0.1 * (y.max() - y.min())
    ax.set_ylim(y.min() - y_margin, y.max() + y_margin)

    # Draw phase backgrounds with correct limits
    if show_phases:
        for cycle in cycles.cycles:
            for start, end, phase in cycle.phases:
                rect = Rectangle(
                    (start, y.min() - y_margin),
                    end - start,
                    y.max() - y.min() + 2 * y_margin,
                    facecolor=PHASE_COLORS[phase],
                    alpha=0.3,
                    edgecolor="none",
                    zorder=1,
                )
                ax.add_patch(rect)

    # Plot main line
    ax.plot(t, y, color="#0072B2", linewidth=1.5, label=variable, zorder=3)

    # Mark peaks
    if show_peaks and cycles.peaks:
        peak_times = [p.time for p in cycles.peaks]
        peak_values = [p.value for p in cycles.peaks]
        ax.scatter(
            peak_times,
            peak_values,
            color="red",
            marker="v",
            s=80,
            zorder=5,
            label="Peaks",
        )

    # Mark troughs
    if show_troughs and cycles.troughs:
        trough_times = [p.time for p in cycles.troughs]
        trough_values = [p.value for p in cycles.troughs]
        ax.scatter(
            trough_times,
            trough_values,
            color="green",
            marker="^",
            s=80,
            zorder=5,
            label="Troughs",
        )

    # Annotate cycles
    if annotate_cycles:
        for i, cycle in enumerate(cycles.cycles):
            mid_time = (cycle.start_time + cycle.end_time) / 2
            ax.annotate(
                f"Cycle {i + 1}",
                xy=(mid_time, y.max() + 0.05 * (y.max() - y.min())),
                ha="center",
                fontsize=9,
                color="gray",
            )

    ax.set_xlabel("Time")
    ax.set_ylabel(_get_variable_label(variable))
    ax.set_title(title, fontsize=14)
    ax.set_xlim(t[0], t[-1])

    # Create legend with phase colors
    if show_phases:
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(
                facecolor=PHASE_COLORS[CyclePhase.EXPANSION],
                alpha=0.3,
                label="Expansion",
            ),
            Patch(
                facecolor=PHASE_COLORS[CyclePhase.STAGFLATION],
                alpha=0.3,
                label="Stagflation",
            ),
            Patch(facecolor=PHASE_COLORS[CyclePhase.CRISIS], alpha=0.3, label="Crisis"),
            Patch(
                facecolor=PHASE_COLORS[CyclePhase.DEPRESSION],
                alpha=0.3,
                label="Depression",
            ),
        ]
        if show_peaks:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="v",
                    color="w",
                    markerfacecolor="red",
                    markersize=10,
                    label="Peak",
                )
            )
        if show_troughs:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="w",
                    markerfacecolor="green",
                    markersize=10,
                    label="Trough",
                )
            )
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    else:
        ax.legend(loc="upper right")

    ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
    fig.tight_layout()

    return fig


def plot_cycle_comparison(
    cycles: CycleDetectionResult,
    figsize: tuple[float, float] = (10, 8),
    title: str = "Secular Cycle Comparison",
) -> Figure:
    """Create comparison plot of multiple secular cycles.

    Aligns cycles by phase to compare patterns across cycles.

    Args:
        cycles: CycleDetectionResult with detected cycles.
        figsize: Figure size (width, height) in inches.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    if not cycles.cycles:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No cycles detected", ha="center", va="center", fontsize=12)
        ax.set_title(title)
        return fig

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Duration histogram
    ax = axes[0, 0]
    durations = [c.duration for c in cycles.cycles]
    ax.hist(
        durations, bins=max(5, len(durations) // 2), color="#0072B2", edgecolor="white"
    )
    ax.axvline(
        np.mean(durations),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(durations):.1f}",
    )
    ax.set_xlabel("Cycle Duration")
    ax.set_ylabel("Count")
    ax.set_title("Cycle Duration Distribution")
    ax.legend()

    # 2. Amplitude histogram
    ax = axes[0, 1]
    amplitudes = [c.amplitude for c in cycles.cycles]
    ax.hist(
        amplitudes,
        bins=max(5, len(amplitudes) // 2),
        color="#D55E00",
        edgecolor="white",
    )
    ax.axvline(
        np.mean(amplitudes),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(amplitudes):.2f}",
    )
    ax.set_xlabel("Peak Amplitude")
    ax.set_ylabel("Count")
    ax.set_title("Cycle Amplitude Distribution")
    ax.legend()

    # 3. Duration vs amplitude scatter
    ax = axes[1, 0]
    ax.scatter(durations, amplitudes, color="#009E73", s=60)
    ax.set_xlabel("Duration")
    ax.set_ylabel("Amplitude")
    ax.set_title("Duration vs Amplitude")
    # Add correlation
    if len(durations) > 2:
        corr = np.corrcoef(durations, amplitudes)[0, 1]
        ax.annotate(
            f"r = {corr:.2f}", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=10
        )

    # 4. Cycle timeline
    ax = axes[1, 1]
    for i, cycle in enumerate(cycles.cycles):
        y = len(cycles.cycles) - i
        ax.barh(y, cycle.duration, left=cycle.start_time, color="#0072B2", alpha=0.7)
        ax.scatter([cycle.peak_time], [y], color="red", marker="v", s=50, zorder=5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cycle Number")
    ax.set_title("Cycle Timeline")
    ax.set_yticks(range(1, len(cycles.cycles) + 1))
    ax.set_yticklabels([f"Cycle {i}" for i in range(len(cycles.cycles), 0, -1)])

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    return fig


def _get_variable_label(variable: str) -> str:
    """Get display label for a variable."""
    labels = {
        "N": "Population (N)",
        "E": "Elite Population (E)",
        "W": "Real Wages (W)",
        "S": "State Fiscal Health (S)",
        "psi": "Political Stress Index (\u03c8)",
        "t": "Time",
    }
    return labels.get(variable, variable)


def compute_cycle_statistics(cycles: CycleDetectionResult) -> dict:
    """Compute summary statistics for detected cycles.

    Args:
        cycles: CycleDetectionResult from detect_secular_cycles.

    Returns:
        Dictionary with cycle statistics.
    """
    if not cycles.cycles:
        return {
            "n_cycles": 0,
            "mean_period": None,
            "std_period": None,
            "mean_amplitude": None,
            "std_amplitude": None,
        }

    durations = [c.duration for c in cycles.cycles]
    amplitudes = [c.amplitude for c in cycles.cycles]

    return {
        "n_cycles": len(cycles.cycles),
        "mean_period": float(np.mean(durations)),
        "std_period": float(np.std(durations)) if len(durations) > 1 else 0.0,
        "min_period": float(np.min(durations)),
        "max_period": float(np.max(durations)),
        "mean_amplitude": float(np.mean(amplitudes)),
        "std_amplitude": float(np.std(amplitudes)) if len(amplitudes) > 1 else 0.0,
        "min_amplitude": float(np.min(amplitudes)),
        "max_amplitude": float(np.max(amplitudes)),
    }


__all__ = [
    "CyclePhase",
    "CyclePoint",
    "SecularCycle",
    "CycleDetectionResult",
    "detect_secular_cycles",
    "detect_peaks_and_troughs",
    "plot_with_cycles",
    "plot_cycle_comparison",
    "compute_cycle_statistics",
    "PHASE_COLORS",
]
