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

All visualizations are built with Altair for consistency with the project's
viz stack.

IMPORTANT: After generating any plot, visually verify it looks correct
before committing.

Example:
    >>> from cliodynamics.viz.cycles import detect_secular_cycles, plot_with_cycles
    >>> cycles = detect_secular_cycles(results['psi'])
    >>> chart = plot_with_cycles(results, cycles)
    >>> save_chart(chart, 'cycles.png')
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import altair as alt
import numpy as np
import pandas as pd

from cliodynamics.viz.charts import (
    CHART_HEIGHT_MEDIUM,
    CHART_WIDTH,
    FONT_SIZE_AXIS_LABEL,
    FONT_SIZE_AXIS_TITLE,
    FONT_SIZE_LEGEND_LABEL,
    FONT_SIZE_LEGEND_TITLE,
    FONT_SIZE_TITLE,
    configure_chart,
    save_chart,
)

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


def _get_variable_label(variable: str) -> str:
    """Get display label for a variable."""
    labels = {
        "N": "Population (N)",
        "E": "Elite Population (E)",
        "W": "Real Wages (W)",
        "S": "State Fiscal Health (S)",
        "psi": "Political Stress Index (ψ)",
        "t": "Time",
    }
    return labels.get(variable, variable)


def _to_dataframe(
    results: "SimulationResult | pd.DataFrame",
) -> pd.DataFrame:
    """Convert SimulationResult or DataFrame to pandas DataFrame."""
    if hasattr(results, "df"):
        df = results.df
    else:
        df = results

    # Convert polars to pandas if needed
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()

    return df


def plot_with_cycles(
    results: "SimulationResult | DataFrame",
    cycles: CycleDetectionResult,
    variable: str = "psi",
    time_column: str = "t",
    title: str = "Political Stress Index with Secular Cycles",
    figsize: tuple[float, float] | None = None,
    show_phases: bool = True,
    show_peaks: bool = True,
    show_troughs: bool = True,
    annotate_cycles: bool = True,
) -> alt.Chart:
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
        Altair Chart object.

    Example:
        >>> cycles = detect_secular_cycles(results['psi'])
        >>> chart = plot_with_cycles(results, cycles)
        >>> save_chart(chart, 'cycles.png')
    """
    df = _to_dataframe(results)

    t = df[time_column].values
    y = df[variable].values

    # Calculate dimensions
    if figsize is not None:
        width = int(figsize[0] * 100)
        height = int(figsize[1] * 100)
    else:
        width = CHART_WIDTH + 200  # Extra width for legend
        height = CHART_HEIGHT_MEDIUM

    # Calculate y domain with margin
    y_min = float(y.min())
    y_max = float(y.max())
    y_margin = 0.1 * (y_max - y_min)
    y_domain = [y_min - y_margin, y_max + y_margin]

    layers = []

    # Draw phase backgrounds
    if show_phases and cycles.cycles:
        phase_data = []
        for cycle in cycles.cycles:
            for start, end, phase in cycle.phases:
                phase_data.append(
                    {
                        "start": start,
                        "end": end,
                        "phase": phase.value.title(),
                        "y_min": y_domain[0],
                        "y_max": y_domain[1],
                    }
                )

        if phase_data:
            phase_df = pd.DataFrame(phase_data)
            phase_color_map = {
                "Expansion": PHASE_COLORS[CyclePhase.EXPANSION],
                "Stagflation": PHASE_COLORS[CyclePhase.STAGFLATION],
                "Crisis": PHASE_COLORS[CyclePhase.CRISIS],
                "Depression": PHASE_COLORS[CyclePhase.DEPRESSION],
            }

            phase_rects = (
                alt.Chart(phase_df)
                .mark_rect(opacity=0.3)
                .encode(
                    x=alt.X("start:Q"),
                    x2=alt.X2("end:Q"),
                    y=alt.Y("y_min:Q"),
                    y2=alt.Y2("y_max:Q"),
                    color=alt.Color(
                        "phase:N",
                        scale=alt.Scale(
                            domain=list(phase_color_map.keys()),
                            range=list(phase_color_map.values()),
                        ),
                        legend=alt.Legend(
                            title="Phase",
                            labelFontSize=FONT_SIZE_LEGEND_LABEL,
                            titleFontSize=FONT_SIZE_LEGEND_TITLE,
                        ),
                    ),
                )
            )
            layers.append(phase_rects)

    # Main line
    line = (
        alt.Chart(df)
        .mark_line(strokeWidth=1.5, color="#0072B2")
        .encode(
            x=alt.X(
                f"{time_column}:Q",
                title="Time",
                scale=alt.Scale(domain=[float(t[0]), float(t[-1])]),
            ),
            y=alt.Y(
                f"{variable}:Q",
                title=_get_variable_label(variable),
                scale=alt.Scale(domain=y_domain),
            ),
        )
    )
    layers.append(line)

    # Peak markers
    if show_peaks and cycles.peaks:
        peak_df = pd.DataFrame(
            [{"time": p.time, "value": p.value} for p in cycles.peaks]
        )
        peaks_chart = (
            alt.Chart(peak_df)
            .mark_point(
                size=80,
                color="red",
                shape="triangle-down",
                filled=True,
            )
            .encode(
                x=alt.X("time:Q"),
                y=alt.Y("value:Q"),
            )
        )
        layers.append(peaks_chart)

    # Trough markers
    if show_troughs and cycles.troughs:
        trough_df = pd.DataFrame(
            [{"time": p.time, "value": p.value} for p in cycles.troughs]
        )
        troughs_chart = (
            alt.Chart(trough_df)
            .mark_point(
                size=80,
                color="green",
                shape="triangle-up",
                filled=True,
            )
            .encode(
                x=alt.X("time:Q"),
                y=alt.Y("value:Q"),
            )
        )
        layers.append(troughs_chart)

    # Cycle annotations
    if annotate_cycles and cycles.cycles:
        annotation_data = []
        for i, cycle in enumerate(cycles.cycles):
            mid_time = (cycle.start_time + cycle.end_time) / 2
            annotation_data.append(
                {
                    "time": mid_time,
                    "value": y_max + 0.05 * (y_max - y_min),
                    "label": f"Cycle {i + 1}",
                }
            )

        if annotation_data:
            annot_df = pd.DataFrame(annotation_data)
            annotations = (
                alt.Chart(annot_df)
                .mark_text(
                    fontSize=9,
                    color="gray",
                    align="center",
                )
                .encode(
                    x=alt.X("time:Q"),
                    y=alt.Y("value:Q"),
                    text="label:N",
                )
            )
            layers.append(annotations)

    chart = alt.layer(*layers)
    chart = configure_chart(chart, title, width=width, height=height)

    return chart


def plot_cycle_comparison(
    cycles: CycleDetectionResult,
    figsize: tuple[float, float] | None = None,
    title: str = "Secular Cycle Comparison",
) -> alt.Chart:
    """Create comparison plot of multiple secular cycles.

    Creates a 2x2 grid showing:
    - Duration histogram
    - Amplitude histogram
    - Duration vs amplitude scatter
    - Cycle timeline

    Args:
        cycles: CycleDetectionResult with detected cycles.
        figsize: Figure size (width, height) in inches.
        title: Plot title.

    Returns:
        Altair Chart object.
    """
    if figsize is not None:
        total_width = int(figsize[0] * 100)
        total_height = int(figsize[1] * 100)
    else:
        total_width = CHART_WIDTH
        total_height = CHART_HEIGHT_MEDIUM + 200

    panel_width = total_width // 2 - 50
    panel_height = total_height // 2 - 50

    if not cycles.cycles:
        # Return empty chart with message
        empty_df = pd.DataFrame(
            {"text": ["No cycles detected"], "x": [0.5], "y": [0.5]}
        )
        chart = (
            alt.Chart(empty_df)
            .mark_text(fontSize=12)
            .encode(
                x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[0, 1])),
                text="text:N",
            )
            .properties(width=total_width, height=total_height, title=title)
        )
        return chart

    # Extract cycle data
    durations = [c.duration for c in cycles.cycles]
    amplitudes = [c.amplitude for c in cycles.cycles]

    cycle_df = pd.DataFrame(
        {
            "duration": durations,
            "amplitude": amplitudes,
            "cycle": [f"Cycle {i + 1}" for i in range(len(cycles.cycles))],
            "start_time": [c.start_time for c in cycles.cycles],
            "end_time": [c.end_time for c in cycles.cycles],
            "peak_time": [c.peak_time for c in cycles.cycles],
        }
    )

    # 1. Duration histogram
    duration_hist = (
        alt.Chart(cycle_df)
        .mark_bar(color="#0072B2")
        .encode(
            x=alt.X(
                "duration:Q",
                bin=alt.Bin(maxbins=max(5, len(durations) // 2)),
                title="Cycle Duration",
            ),
            y=alt.Y("count()", title="Count"),
        )
        .properties(
            width=panel_width, height=panel_height, title="Cycle Duration Distribution"
        )
    )

    # Add mean line
    mean_duration = np.mean(durations)
    duration_rule = (
        alt.Chart(pd.DataFrame({"mean": [mean_duration]}))
        .mark_rule(color="red", strokeDash=[5, 5])
        .encode(x=alt.X("mean:Q"))
    )

    duration_chart = alt.layer(duration_hist, duration_rule)

    # 2. Amplitude histogram
    amplitude_hist = (
        alt.Chart(cycle_df)
        .mark_bar(color="#D55E00")
        .encode(
            x=alt.X(
                "amplitude:Q",
                bin=alt.Bin(maxbins=max(5, len(amplitudes) // 2)),
                title="Peak Amplitude",
            ),
            y=alt.Y("count()", title="Count"),
        )
        .properties(
            width=panel_width, height=panel_height, title="Cycle Amplitude Distribution"
        )
    )

    mean_amplitude = np.mean(amplitudes)
    amplitude_rule = (
        alt.Chart(pd.DataFrame({"mean": [mean_amplitude]}))
        .mark_rule(color="red", strokeDash=[5, 5])
        .encode(x=alt.X("mean:Q"))
    )

    amplitude_chart = alt.layer(amplitude_hist, amplitude_rule)

    # 3. Duration vs Amplitude scatter
    scatter = (
        alt.Chart(cycle_df)
        .mark_point(size=60, color="#009E73", filled=True)
        .encode(
            x=alt.X("duration:Q", title="Duration"),
            y=alt.Y("amplitude:Q", title="Amplitude"),
        )
        .properties(
            width=panel_width, height=panel_height, title="Duration vs Amplitude"
        )
    )

    # Add correlation annotation if enough data
    if len(durations) > 2:
        corr = np.corrcoef(durations, amplitudes)[0, 1]
        corr_df = pd.DataFrame(
            {"text": [f"r = {corr:.2f}"], "x": [min(durations)], "y": [max(amplitudes)]}
        )
        corr_text = (
            alt.Chart(corr_df)
            .mark_text(align="left", baseline="top", fontSize=10)
            .encode(x="x:Q", y="y:Q", text="text:N")
        )
        scatter = alt.layer(scatter, corr_text)

    # 4. Cycle timeline
    timeline = (
        alt.Chart(cycle_df)
        .mark_bar(color="#0072B2", opacity=0.7)
        .encode(
            x=alt.X("start_time:Q", title="Time"),
            x2=alt.X2("end_time:Q"),
            y=alt.Y(
                "cycle:N",
                title="Cycle Number",
                sort=alt.SortField("start_time", order="descending"),
            ),
        )
        .properties(width=panel_width, height=panel_height, title="Cycle Timeline")
    )

    # Add peak markers
    peak_markers = (
        alt.Chart(cycle_df)
        .mark_point(color="red", shape="triangle-down", size=50)
        .encode(
            x=alt.X("peak_time:Q"),
            y=alt.Y("cycle:N", sort=alt.SortField("start_time", order="descending")),
        )
    )

    timeline_chart = alt.layer(timeline, peak_markers)

    # Combine into 2x2 grid
    top_row = alt.hconcat(duration_chart, amplitude_chart)
    bottom_row = alt.hconcat(scatter, timeline_chart)

    chart = (
        alt.vconcat(top_row, bottom_row)
        .properties(
            title=alt.TitleParams(text=title, fontSize=FONT_SIZE_TITLE, anchor="start")
        )
        .configure_axis(
            labelFontSize=FONT_SIZE_AXIS_LABEL,
            titleFontSize=FONT_SIZE_AXIS_TITLE,
        )
        .configure_legend(
            labelFontSize=FONT_SIZE_LEGEND_LABEL,
            titleFontSize=FONT_SIZE_LEGEND_TITLE,
        )
    )

    return chart


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
    "save_chart",
    "PHASE_COLORS",
]
