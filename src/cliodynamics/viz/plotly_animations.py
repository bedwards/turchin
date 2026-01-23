"""Plotly-based animated visualizations for SDT model simulations.

This module provides interactive animated visualizations using Plotly,
bringing secular cycles and model dynamics to life for essays and presentations.
Animations show how societies evolve through demographic-structural pressures
over time.

Key advantages over matplotlib animations:
- Native animation support with frames
- Interactive HTML export (play/pause, scrub)
- 3D animations work smoothly
- No ffmpeg dependency for video
- Embeddable in GitHub Pages

IMPORTANT: After generating any animation, visually verify it looks correct
before committing. Check that:
1. Animation plays smoothly without artifacts
2. Labels and legends are readable
3. Trajectories are correctly rendered
4. Phase transitions (if shown) are accurate

Example:
    >>> from cliodynamics.viz import plotly_animations
    >>> from cliodynamics.simulation import SimulationResult
    >>>
    >>> # Animate time series
    >>> fig = plotly_animations.animate_time_series(
    ...     results,
    ...     variables=['N', 'W', 'psi'],
    ...     title="Roman Empire Dynamics"
    ... )
    >>> fig.write_html('rome_timeseries.html')
    >>>
    >>> # Animate 3D phase space with camera orbit
    >>> fig = plotly_animations.animate_phase_space_3d(
    ...     results, x='N', y='W', z='psi',
    ...     trail_length=50, camera_orbit=True
    ... )
    >>> fig.write_html('phase_3d.html')
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from pandas import DataFrame

    from cliodynamics.simulation import SimulationResult
    from cliodynamics.viz.cycles import CycleDetectionResult

logger = logging.getLogger(__name__)

# Verification reminder for animations
VERIFICATION_REMINDER = """
=====================================================================
  IMPORTANT: Visually verify all animations before committing!

  Check that:
  1. Animation plays smoothly without artifacts
  2. Labels and legends are readable throughout
  3. Trajectories are correctly rendered
  4. Phase transitions (if shown) are accurate
  5. Interactive controls work as expected
=====================================================================
"""

# Default color palette (colorblind-friendly, consistent with plots.py)
DEFAULT_COLORS = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#009E73",  # Green
    "#CC79A7",  # Pink
    "#F0E442",  # Yellow
    "#56B4E9",  # Light blue
    "#E69F00",  # Orange
]

# Variable display names
VARIABLE_LABELS = {
    "N": "Population (N)",
    "E": "Elite Population (E)",
    "W": "Real Wages (W)",
    "S": "State Fiscal Health (S)",
    "psi": "Political Stress Index (\u03c8)",
    "t": "Time",
}

# Phase colors for secular cycles (consistent with cycles.py)
PHASE_COLORS_HEX = {
    "expansion": "#90EE90",  # Light green
    "stagflation": "#FFD700",  # Gold
    "crisis": "#FF6B6B",  # Light red
    "depression": "#87CEEB",  # Sky blue
}


def _get_label(variable: str, custom_labels: dict[str, str] | None = None) -> str:
    """Get display label for a variable.

    Args:
        variable: Variable name.
        custom_labels: Optional custom label mapping.

    Returns:
        Display label for the variable.
    """
    if custom_labels and variable in custom_labels:
        return custom_labels[variable]
    return VARIABLE_LABELS.get(variable, variable)


def _get_dataframe(results: "SimulationResult | DataFrame") -> "DataFrame":
    """Extract DataFrame from results.

    Args:
        results: SimulationResult object or DataFrame.

    Returns:
        DataFrame with time series data.
    """
    if hasattr(results, "df"):
        return results.df
    return results


def animate_time_series(
    results: "SimulationResult | DataFrame",
    variables: list[str] | None = None,
    title: str = "SDT Model Evolution",
    duration_ms: int = 10000,
    time_column: str = "t",
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    subplot_layout: bool = False,
    width: int = 1000,
    height: int = 600,
    show_play_button: bool = True,
    frame_duration: int = 50,
) -> go.Figure:
    """Animate time series with progressive reveal.

    Creates an interactive animation that progressively reveals time series data,
    showing how variables evolve over the simulation period.

    Args:
        results: SimulationResult object or DataFrame with time series data.
        variables: List of variable names to plot. Defaults to ['N', 'W', 'psi'].
        title: Plot title.
        duration_ms: Total duration of animation in milliseconds.
        time_column: Name of time column in data.
        labels: Display labels for variables.
        colors: Custom color list for variables.
        subplot_layout: If True, use separate subplots for each variable.
        width: Figure width in pixels.
        height: Figure height in pixels.
        show_play_button: If True, show play/pause button.
        frame_duration: Duration of each frame in milliseconds.

    Returns:
        Plotly Figure object with animation frames.

    Example:
        >>> fig = animate_time_series(
        ...     results,
        ...     variables=['N', 'W', 'psi'],
        ...     title="Roman Empire 500 BCE - 500 CE",
        ...     duration_ms=15000
        ... )
        >>> fig.write_html('rome_evolution.html')
    """
    df = _get_dataframe(results)

    # Default variables
    if variables is None:
        variables = ["N", "W", "psi"]
        variables = [v for v in variables if v in df.columns]

    # Validate variables
    for var in variables:
        if var not in df.columns:
            raise ValueError(
                f"Variable '{var}' not found in data columns: {df.columns.tolist()}"
            )

    # Default labels
    if labels is None:
        labels = [_get_label(v) for v in variables]

    # Colors
    if colors is None:
        colors = DEFAULT_COLORS[: len(variables)]

    # Get data
    t = df[time_column].values
    data = {var: df[var].values for var in variables}
    n_points = len(t)

    # Calculate number of frames
    n_frames = max(50, min(200, duration_ms // frame_duration))
    indices = np.linspace(0, n_points - 1, n_frames).astype(int)

    if subplot_layout:
        fig = make_subplots(
            rows=len(variables),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[_get_label(v) for v in variables],
        )

        # Add initial traces (empty)
        for i, (var, label, color) in enumerate(zip(variables, labels, colors)):
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=2),
                    showlegend=False,
                ),
                row=i + 1,
                col=1,
            )
            y_data = data[var]
            y_margin = 0.1 * (y_data.max() - y_data.min())
            fig.update_yaxes(
                range=[y_data.min() - y_margin, y_data.max() + y_margin],
                row=i + 1,
                col=1,
            )

        fig.update_xaxes(range=[t[0], t[-1]])
        fig.update_xaxes(title_text=_get_label(time_column), row=len(variables), col=1)

        # Create frames
        frames = []
        for frame_idx, data_idx in enumerate(indices):
            frame_data = []
            for i, var in enumerate(variables):
                frame_data.append(
                    go.Scatter(
                        x=t[: data_idx + 1],
                        y=data[var][: data_idx + 1],
                    )
                )
            frames.append(
                go.Frame(
                    data=frame_data,
                    name=str(frame_idx),
                    traces=list(range(len(variables))),
                )
            )

    else:
        fig = go.Figure()

        # Add initial traces (empty)
        for var, label, color in zip(variables, labels, colors):
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=2),
                )
            )

        # Set axis limits
        all_values = np.concatenate([data[v] for v in variables])
        y_margin = 0.1 * (all_values.max() - all_values.min())
        fig.update_xaxes(range=[t[0], t[-1]], title=_get_label(time_column))
        fig.update_yaxes(
            range=[all_values.min() - y_margin, all_values.max() + y_margin],
            title="Value",
        )

        # Create frames
        frames = []
        for frame_idx, data_idx in enumerate(indices):
            frame_data = []
            for var in variables:
                frame_data.append(
                    go.Scatter(
                        x=t[: data_idx + 1],
                        y=data[var][: data_idx + 1],
                    )
                )
            frames.append(
                go.Frame(
                    data=frame_data,
                    name=str(frame_idx),
                    traces=list(range(len(variables))),
                )
            )

    fig.frames = frames

    # Add animation controls
    sliders = [
        {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Progress: ",
                "visible": True,
                "xanchor": "right",
            },
            "transition": {"duration": frame_duration, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [str(i)],
                        {
                            "frame": {"duration": frame_duration, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": frame_duration},
                        },
                    ],
                    "label": f"{int(100 * i / n_frames)}%",
                    "method": "animate",
                }
                for i in range(0, n_frames, max(1, n_frames // 20))
            ],
        }
    ]

    updatemenus = []
    if show_play_button:
        updatemenus = [
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": frame_duration},
                            },
                        ],
                        "label": "\u25b6 Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "\u23f8 Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ]

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        width=width,
        height=height,
        sliders=sliders,
        updatemenus=updatemenus,
        template="plotly_white",
    )

    return fig


def animate_phase_space(
    results: "SimulationResult | DataFrame",
    x: str,
    y: str,
    color_by: str = "t",
    trail_length: int = 50,
    duration_ms: int = 10000,
    title: str | None = None,
    width: int = 800,
    height: int = 600,
    colorscale: str = "Viridis",
    show_start: bool = True,
    frame_duration: int = 50,
) -> go.Figure:
    """Animate phase space trajectory with trailing effect.

    Creates an interactive animation showing the trajectory through phase space
    with a fading trail effect to visualize direction of motion.

    Args:
        results: SimulationResult object or DataFrame with time series data.
        x: Variable name for x-axis.
        y: Variable name for y-axis.
        color_by: Variable to color trajectory by (e.g., 't' for time).
        trail_length: Number of points in the trailing effect.
        duration_ms: Total duration of animation in milliseconds.
        title: Plot title. Defaults to auto-generated title.
        width: Figure width in pixels.
        height: Figure height in pixels.
        colorscale: Plotly colorscale name for trajectory color.
        show_start: If True, mark starting point.
        frame_duration: Duration of each frame in milliseconds.

    Returns:
        Plotly Figure object with animation frames.

    Example:
        >>> fig = animate_phase_space(
        ...     results,
        ...     x='W', y='psi',
        ...     trail_length=100,
        ...     duration_ms=12000
        ... )
        >>> fig.write_html('phase_trajectory.html')
    """
    df = _get_dataframe(results)

    # Validate variables
    for var in [x, y]:
        if var not in df.columns:
            raise ValueError(
                f"Variable '{var}' not in data columns: {df.columns.tolist()}"
            )

    if color_by not in df.columns:
        raise ValueError(
            f"Color variable '{color_by}' not in columns: {df.columns.tolist()}"
        )

    x_vals = df[x].values
    y_vals = df[y].values
    c_vals = df[color_by].values

    n_points = len(x_vals)

    # Calculate number of frames
    n_frames = max(50, min(200, duration_ms // frame_duration))
    indices = np.linspace(0, n_points - 1, n_frames).astype(int)

    # Set up figure
    fig = go.Figure()

    # Set axis limits
    x_margin = 0.1 * (x_vals.max() - x_vals.min())
    y_margin = 0.1 * (y_vals.max() - y_vals.min())

    # Add starting point marker
    if show_start:
        fig.add_trace(
            go.Scatter(
                x=[x_vals[0]],
                y=[y_vals[0]],
                mode="markers",
                marker=dict(color="green", size=15, symbol="circle"),
                name="Start",
                showlegend=True,
            )
        )

    # Add trail trace (will be updated in frames)
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines+markers",
            marker=dict(
                color=[],
                colorscale=colorscale,
                size=8,
                showscale=True,
                colorbar=dict(title=_get_label(color_by)),
            ),
            line=dict(width=2, color="rgba(100,100,100,0.3)"),
            name="Trajectory",
        )
    )

    # Add current position marker (head)
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(color="red", size=12, symbol="circle"),
            name="Current",
            showlegend=True,
        )
    )

    # Create frames
    frames = []
    for frame_idx, data_idx in enumerate(indices):
        start_idx = max(0, data_idx - trail_length)

        frame_data = []

        # Start marker (unchanged)
        if show_start:
            frame_data.append(
                go.Scatter(
                    x=[x_vals[0]],
                    y=[y_vals[0]],
                )
            )

        # Trail
        frame_data.append(
            go.Scatter(
                x=x_vals[start_idx : data_idx + 1],
                y=y_vals[start_idx : data_idx + 1],
                marker=dict(color=c_vals[start_idx : data_idx + 1]),
            )
        )

        # Head marker
        frame_data.append(
            go.Scatter(
                x=[x_vals[data_idx]],
                y=[y_vals[data_idx]],
            )
        )

        list(range(3 if show_start else 2))
        if not show_start:
            frame_data = frame_data[1:]

        frames.append(
            go.Frame(
                data=frame_data,
                name=str(frame_idx),
                traces=[i for i in range(len(frame_data) + (1 if show_start else 0))],
            )
        )

    fig.frames = frames

    if title is None:
        title = f"Phase Space: {_get_label(x)} vs {_get_label(y)}"

    # Animation controls
    sliders = [
        {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Progress: ",
                "visible": True,
                "xanchor": "right",
            },
            "transition": {"duration": frame_duration},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [str(i)],
                        {
                            "frame": {"duration": frame_duration, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": frame_duration},
                        },
                    ],
                    "label": f"{int(100 * i / n_frames)}%",
                    "method": "animate",
                }
                for i in range(0, n_frames, max(1, n_frames // 20))
            ],
        }
    ]

    updatemenus = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": frame_duration, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": frame_duration},
                        },
                    ],
                    "label": "\u25b6 Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "\u23f8 Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }
    ]

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(
            title=_get_label(x),
            range=[x_vals.min() - x_margin, x_vals.max() + x_margin],
        ),
        yaxis=dict(
            title=_get_label(y),
            range=[y_vals.min() - y_margin, y_vals.max() + y_margin],
        ),
        width=width,
        height=height,
        sliders=sliders,
        updatemenus=updatemenus,
        template="plotly_white",
    )

    return fig


def animate_phase_space_3d(
    results: "SimulationResult | DataFrame",
    x: str,
    y: str,
    z: str,
    camera_orbit: bool = True,
    trail_length: int = 50,
    duration_ms: int = 10000,
    title: str | None = None,
    width: int = 900,
    height: int = 700,
    colorscale: str = "Viridis",
    initial_camera: dict | None = None,
    frame_duration: int = 50,
) -> go.Figure:
    """Animate 3D phase space with optional camera orbit.

    Creates an interactive animation showing the trajectory through 3D phase space
    with optional camera rotation to reveal the full structure.

    Args:
        results: SimulationResult object or DataFrame with time series data.
        x: Variable name for x-axis.
        y: Variable name for y-axis.
        z: Variable name for z-axis.
        camera_orbit: If True, rotate camera during animation.
        trail_length: Number of points in the trailing effect.
        duration_ms: Total duration of animation in milliseconds.
        title: Plot title. Defaults to auto-generated title.
        width: Figure width in pixels.
        height: Figure height in pixels.
        colorscale: Plotly colorscale name for trajectory color.
        initial_camera: Initial camera position dict.
        frame_duration: Duration of each frame in milliseconds.

    Returns:
        Plotly Figure object with animation frames.

    Example:
        >>> fig = animate_phase_space_3d(
        ...     results,
        ...     x='N', y='W', z='psi',
        ...     camera_orbit=True,
        ...     duration_ms=15000
        ... )
        >>> fig.write_html('phase_3d.html')
    """
    df = _get_dataframe(results)

    # Validate variables
    for var in [x, y, z]:
        if var not in df.columns:
            raise ValueError(
                f"Variable '{var}' not in data columns: {df.columns.tolist()}"
            )

    x_vals = df[x].values
    y_vals = df[y].values
    z_vals = df[z].values

    n_points = len(x_vals)

    # Calculate number of frames
    n_frames = max(50, min(200, duration_ms // frame_duration))
    indices = np.linspace(0, n_points - 1, n_frames).astype(int)

    # Normalize values for color
    color_vals = np.arange(n_points)

    # Set up figure
    fig = go.Figure()

    # Add starting point marker
    fig.add_trace(
        go.Scatter3d(
            x=[x_vals[0]],
            y=[y_vals[0]],
            z=[z_vals[0]],
            mode="markers",
            marker=dict(color="green", size=10, symbol="circle"),
            name="Start",
            showlegend=True,
        )
    )

    # Add trail trace
    fig.add_trace(
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="lines+markers",
            marker=dict(
                color=[],
                colorscale=colorscale,
                size=4,
                showscale=True,
                colorbar=dict(title="Time"),
            ),
            line=dict(width=3, color="rgba(100,100,100,0.5)"),
            name="Trajectory",
        )
    )

    # Add current position marker (head)
    fig.add_trace(
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="markers",
            marker=dict(color="red", size=8, symbol="circle"),
            name="Current",
            showlegend=True,
        )
    )

    # Create frames with camera orbit
    frames = []
    for frame_idx, data_idx in enumerate(indices):
        start_idx = max(0, data_idx - trail_length)

        frame_data = [
            # Start marker (unchanged)
            go.Scatter3d(
                x=[x_vals[0]],
                y=[y_vals[0]],
                z=[z_vals[0]],
            ),
            # Trail
            go.Scatter3d(
                x=x_vals[start_idx : data_idx + 1],
                y=y_vals[start_idx : data_idx + 1],
                z=z_vals[start_idx : data_idx + 1],
                marker=dict(color=color_vals[start_idx : data_idx + 1]),
            ),
            # Head marker
            go.Scatter3d(
                x=[x_vals[data_idx]],
                y=[y_vals[data_idx]],
                z=[z_vals[data_idx]],
            ),
        ]

        # Calculate camera position for orbit
        if camera_orbit:
            angle = 2 * np.pi * frame_idx / n_frames
            camera_eye = dict(
                x=1.5 * np.cos(angle),
                y=1.5 * np.sin(angle),
                z=0.8,
            )
        else:
            camera_eye = initial_camera.get("eye") if initial_camera else None

        # Create frame with layout update for camera
        frame = go.Frame(
            data=frame_data,
            name=str(frame_idx),
            traces=[0, 1, 2],
        )

        if camera_orbit:
            frame.layout = go.Layout(scene=dict(camera=dict(eye=camera_eye)))

        frames.append(frame)

    fig.frames = frames

    if title is None:
        title = f"3D Phase Space: {x}, {y}, {z}"

    # Set initial camera
    if initial_camera is None:
        initial_camera = dict(
            eye=dict(x=1.5, y=1.5, z=0.8),
            up=dict(x=0, y=0, z=1),
        )

    # Animation controls
    sliders = [
        {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Progress: ",
                "visible": True,
                "xanchor": "right",
            },
            "transition": {"duration": frame_duration},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [str(i)],
                        {
                            "frame": {"duration": frame_duration, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": frame_duration},
                        },
                    ],
                    "label": f"{int(100 * i / n_frames)}%",
                    "method": "animate",
                }
                for i in range(0, n_frames, max(1, n_frames // 20))
            ],
        }
    ]

    updatemenus = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": frame_duration, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": frame_duration},
                        },
                    ],
                    "label": "\u25b6 Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "\u23f8 Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }
    ]

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        scene=dict(
            xaxis_title=_get_label(x),
            yaxis_title=_get_label(y),
            zaxis_title=_get_label(z),
            camera=initial_camera,
        ),
        width=width,
        height=height,
        sliders=sliders,
        updatemenus=updatemenus,
    )

    return fig


def animate_secular_cycles(
    results: "SimulationResult | DataFrame",
    cycles: "CycleDetectionResult",
    variable: str = "psi",
    show_phase_transitions: bool = True,
    duration_ms: int = 10000,
    time_column: str = "t",
    title: str = "Political Stress Index with Secular Cycles",
    width: int = 1000,
    height: int = 500,
    frame_duration: int = 50,
) -> go.Figure:
    """Animate time series with cycle phase transitions.

    Creates an interactive animation that progressively reveals the time series
    while highlighting secular cycle phases with color-coded backgrounds.

    Args:
        results: SimulationResult object or DataFrame with time series data.
        cycles: CycleDetectionResult from detect_secular_cycles.
        variable: Variable to plot (typically 'psi').
        show_phase_transitions: If True, show phase backgrounds.
        duration_ms: Total duration of animation in milliseconds.
        time_column: Name of time column.
        title: Plot title.
        width: Figure width in pixels.
        height: Figure height in pixels.
        frame_duration: Duration of each frame in milliseconds.

    Returns:
        Plotly Figure object with animation frames.

    Example:
        >>> from cliodynamics.viz.cycles import detect_secular_cycles
        >>> detected = detect_secular_cycles(results['psi'], results['t'])
        >>> fig = animate_secular_cycles(results, detected)
        >>> fig.write_html('secular_cycles.html')
    """
    df = _get_dataframe(results)

    if variable not in df.columns:
        raise ValueError(
            f"Variable '{variable}' not in data columns: {df.columns.tolist()}"
        )

    t = df[time_column].values
    y = df[variable].values
    n_points = len(t)

    # Calculate number of frames
    n_frames = max(50, min(200, duration_ms // frame_duration))
    indices = np.linspace(0, n_points - 1, n_frames).astype(int)

    # Set up figure
    fig = go.Figure()

    y_margin = 0.1 * (y.max() - y.min())
    y_min, y_max = y.min() - y_margin, y.max() + y_margin

    # Add phase backgrounds as shapes (will be controlled via visibility)
    shapes = []
    if show_phase_transitions:
        for cycle in cycles.cycles:
            for start, end, phase in cycle.phases:
                phase_name = phase.value if hasattr(phase, "value") else str(phase)
                color = PHASE_COLORS_HEX.get(phase_name, "#CCCCCC")
                shapes.append(
                    dict(
                        type="rect",
                        x0=start,
                        x1=end,
                        y0=y_min,
                        y1=y_max,
                        fillcolor=color,
                        opacity=0.3,
                        line_width=0,
                        layer="below",
                    )
                )

    # Add main line trace
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            name=_get_label(variable),
            line=dict(color="#0072B2", width=2),
        )
    )

    # Create frames
    frames = []
    for frame_idx, data_idx in enumerate(indices):
        current_time = t[data_idx]

        # Filter shapes visible up to current time
        visible_shapes = []
        if show_phase_transitions:
            for cycle in cycles.cycles:
                for start, end, phase in cycle.phases:
                    if start < current_time:
                        visible_end = min(end, current_time)
                        if visible_end > start:
                            phase_name = (
                                phase.value if hasattr(phase, "value") else str(phase)
                            )
                            color = PHASE_COLORS_HEX.get(phase_name, "#CCCCCC")
                            visible_shapes.append(
                                dict(
                                    type="rect",
                                    x0=start,
                                    x1=visible_end,
                                    y0=y_min,
                                    y1=y_max,
                                    fillcolor=color,
                                    opacity=0.3,
                                    line_width=0,
                                    layer="below",
                                )
                            )

        frame = go.Frame(
            data=[
                go.Scatter(
                    x=t[: data_idx + 1],
                    y=y[: data_idx + 1],
                )
            ],
            name=str(frame_idx),
            traces=[0],
            layout=go.Layout(shapes=visible_shapes),
        )
        frames.append(frame)

    fig.frames = frames

    # Animation controls
    sliders = [
        {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Time: ",
                "visible": True,
                "xanchor": "right",
            },
            "transition": {"duration": frame_duration},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [str(i)],
                        {
                            "frame": {"duration": frame_duration, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": frame_duration},
                        },
                    ],
                    "label": f"{t[indices[i]]:.0f}",
                    "method": "animate",
                }
                for i in range(0, n_frames, max(1, n_frames // 20))
            ],
        }
    ]

    updatemenus = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": frame_duration, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": frame_duration},
                        },
                    ],
                    "label": "\u25b6 Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "\u23f8 Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }
    ]

    # Add legend for phases
    if show_phase_transitions:
        for phase_name, color in PHASE_COLORS_HEX.items():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color),
                    name=phase_name.capitalize(),
                    showlegend=True,
                )
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title=_get_label(time_column), range=[t[0], t[-1]]),
        yaxis=dict(title=_get_label(variable), range=[y_min, y_max]),
        width=width,
        height=height,
        sliders=sliders,
        updatemenus=updatemenus,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def animate_comparison(
    baseline: "SimulationResult | DataFrame",
    counterfactual: "SimulationResult | DataFrame",
    variables: list[str],
    labels: tuple[str, str] = ("Historical", "Counterfactual"),
    layout: str = "overlay",
    duration_ms: int = 10000,
    time_column: str = "t",
    title: str = "Scenario Comparison",
    width: int = 1000,
    height: int = 600,
    colors: tuple[str, str] | None = None,
    frame_duration: int = 50,
) -> go.Figure:
    """Animate comparison of two simulation scenarios.

    Creates an interactive animation comparing baseline and counterfactual scenarios,
    useful for policy analysis and exploring alternative histories.

    Args:
        baseline: SimulationResult or DataFrame for baseline scenario.
        counterfactual: SimulationResult or DataFrame for counterfactual.
        variables: List of variable names to compare.
        labels: Display labels for (baseline, counterfactual).
        layout: 'overlay' for same axes, 'side_by_side' for separate panels.
        duration_ms: Total duration of animation in milliseconds.
        time_column: Name of time column.
        title: Plot title.
        width: Figure width in pixels.
        height: Figure height in pixels.
        colors: Custom colors for (baseline, counterfactual).
        frame_duration: Duration of each frame in milliseconds.

    Returns:
        Plotly Figure object with animation frames.

    Example:
        >>> fig = animate_comparison(
        ...     baseline=historical_results,
        ...     counterfactual=policy_results,
        ...     variables=['psi', 'S'],
        ...     labels=('Historical', 'With Reform'),
        ...     layout='side_by_side'
        ... )
        >>> fig.write_html('comparison.html')
    """
    df_base = _get_dataframe(baseline)
    df_counter = _get_dataframe(counterfactual)

    # Validate variables
    for var in variables:
        if var not in df_base.columns:
            raise ValueError(f"Variable '{var}' not in baseline data")
        if var not in df_counter.columns:
            raise ValueError(f"Variable '{var}' not in counterfactual data")

    # Default colors
    if colors is None:
        colors = (DEFAULT_COLORS[0], DEFAULT_COLORS[1])

    t_base = df_base[time_column].values
    t_counter = df_counter[time_column].values

    n_points_base = len(t_base)
    n_points_counter = len(t_counter)

    # Calculate number of frames
    n_frames = max(50, min(200, duration_ms // frame_duration))
    indices_base = np.linspace(0, n_points_base - 1, n_frames).astype(int)
    indices_counter = np.linspace(0, n_points_counter - 1, n_frames).astype(int)

    # Get combined time range
    t_min = min(t_base[0], t_counter[0])
    t_max = max(t_base[-1], t_counter[-1])

    if layout == "side_by_side":
        fig = make_subplots(
            rows=len(variables),
            cols=2,
            shared_xaxes=True,
            horizontal_spacing=0.1,
            vertical_spacing=0.05,
            subplot_titles=[f"{labels[0]} - {_get_label(v)}" for v in variables]
            + [f"{labels[1]} - {_get_label(v)}" for v in variables],
        )

        # Reorder subplot titles
        new_titles = []
        for i, var in enumerate(variables):
            new_titles.append(f"{labels[0]} - {_get_label(var)}")
        for i, var in enumerate(variables):
            new_titles.append(f"{labels[1]} - {_get_label(var)}")

        # Add traces for each variable
        for i, var in enumerate(variables):
            # Baseline
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    name=f"{labels[0]} - {_get_label(var)}",
                    line=dict(color=colors[0], width=2),
                    showlegend=(i == 0),
                ),
                row=i + 1,
                col=1,
            )
            # Counterfactual
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    name=f"{labels[1]} - {_get_label(var)}",
                    line=dict(color=colors[1], width=2),
                    showlegend=(i == 0),
                ),
                row=i + 1,
                col=2,
            )

            # Set y-axis limits
            all_vals = np.concatenate([df_base[var].values, df_counter[var].values])
            y_margin = 0.1 * (all_vals.max() - all_vals.min())
            y_range = [all_vals.min() - y_margin, all_vals.max() + y_margin]
            fig.update_yaxes(range=y_range, row=i + 1, col=1)
            fig.update_yaxes(range=y_range, row=i + 1, col=2)

        fig.update_xaxes(range=[t_min, t_max])

        # Create frames
        frames = []
        for frame_idx in range(n_frames):
            idx_base = indices_base[frame_idx]
            idx_counter = indices_counter[frame_idx]

            frame_data = []
            for i, var in enumerate(variables):
                # Baseline
                frame_data.append(
                    go.Scatter(
                        x=t_base[: idx_base + 1],
                        y=df_base[var].values[: idx_base + 1],
                    )
                )
                # Counterfactual
                frame_data.append(
                    go.Scatter(
                        x=t_counter[: idx_counter + 1],
                        y=df_counter[var].values[: idx_counter + 1],
                    )
                )

            frames.append(
                go.Frame(
                    data=frame_data,
                    name=str(frame_idx),
                    traces=list(range(len(variables) * 2)),
                )
            )

    else:  # overlay layout
        if len(variables) > 1:
            fig = make_subplots(
                rows=len(variables),
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=[_get_label(v) for v in variables],
            )
        else:
            fig = go.Figure()

        # Add traces for each variable
        for i, var in enumerate(variables):
            row = i + 1 if len(variables) > 1 else None
            col = 1 if len(variables) > 1 else None

            # Baseline
            trace_kwargs = dict(
                x=[],
                y=[],
                mode="lines",
                name=labels[0] if i == 0 else None,
                line=dict(color=colors[0], width=2),
                showlegend=(i == 0),
                legendgroup="baseline",
            )
            if len(variables) > 1:
                fig.add_trace(go.Scatter(**trace_kwargs), row=row, col=col)
            else:
                fig.add_trace(go.Scatter(**trace_kwargs))

            # Counterfactual
            trace_kwargs = dict(
                x=[],
                y=[],
                mode="lines",
                name=labels[1] if i == 0 else None,
                line=dict(color=colors[1], width=2, dash="dash"),
                showlegend=(i == 0),
                legendgroup="counterfactual",
            )
            if len(variables) > 1:
                fig.add_trace(go.Scatter(**trace_kwargs), row=row, col=col)
            else:
                fig.add_trace(go.Scatter(**trace_kwargs))

            # Set y-axis limits
            all_vals = np.concatenate([df_base[var].values, df_counter[var].values])
            y_margin = 0.1 * (all_vals.max() - all_vals.min())
            y_range = [all_vals.min() - y_margin, all_vals.max() + y_margin]
            if len(variables) > 1:
                fig.update_yaxes(range=y_range, row=i + 1, col=1)
            else:
                fig.update_yaxes(range=y_range)

        fig.update_xaxes(range=[t_min, t_max])

        # Create frames
        frames = []
        for frame_idx in range(n_frames):
            idx_base = indices_base[frame_idx]
            idx_counter = indices_counter[frame_idx]

            frame_data = []
            for var in variables:
                # Baseline
                frame_data.append(
                    go.Scatter(
                        x=t_base[: idx_base + 1],
                        y=df_base[var].values[: idx_base + 1],
                    )
                )
                # Counterfactual
                frame_data.append(
                    go.Scatter(
                        x=t_counter[: idx_counter + 1],
                        y=df_counter[var].values[: idx_counter + 1],
                    )
                )

            frames.append(
                go.Frame(
                    data=frame_data,
                    name=str(frame_idx),
                    traces=list(range(len(variables) * 2)),
                )
            )

    fig.frames = frames

    # Animation controls
    sliders = [
        {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Progress: ",
                "visible": True,
                "xanchor": "right",
            },
            "transition": {"duration": frame_duration},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [str(i)],
                        {
                            "frame": {"duration": frame_duration, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": frame_duration},
                        },
                    ],
                    "label": f"{int(100 * i / n_frames)}%",
                    "method": "animate",
                }
                for i in range(0, n_frames, max(1, n_frames // 20))
            ],
        }
    ]

    updatemenus = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": frame_duration, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": frame_duration},
                        },
                    ],
                    "label": "\u25b6 Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "\u23f8 Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }
    ]

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        width=width,
        height=height,
        sliders=sliders,
        updatemenus=updatemenus,
        template="plotly_white",
    )

    return fig


def animate_parameter_sensitivity(
    results_dict: dict[str, "SimulationResult | DataFrame"],
    variable: str = "psi",
    parameter_name: str = "Parameter",
    duration_ms: int = 10000,
    time_column: str = "t",
    title: str | None = None,
    width: int = 1000,
    height: int = 600,
    colorscale: str = "Viridis",
    frame_duration: int = 50,
) -> go.Figure:
    """Animate parameter sensitivity analysis.

    Shows how changing a parameter affects the model output by animating
    through different parameter values.

    Args:
        results_dict: Dict mapping parameter value labels to simulation results.
        variable: Variable to plot.
        parameter_name: Name of parameter being varied.
        duration_ms: Total duration of animation in milliseconds.
        time_column: Name of time column.
        title: Plot title.
        width: Figure width in pixels.
        height: Figure height in pixels.
        colorscale: Plotly colorscale name.
        frame_duration: Duration of each frame in milliseconds.

    Returns:
        Plotly Figure object with animation frames.

    Example:
        >>> results = {
        ...     "r=0.01": sim_r001,
        ...     "r=0.02": sim_r002,
        ...     "r=0.03": sim_r003,
        ... }
        >>> fig = animate_parameter_sensitivity(
        ...     results, variable='psi', parameter_name='Growth Rate (r)'
        ... )
        >>> fig.write_html('sensitivity.html')
    """
    param_values = list(results_dict.keys())
    n_params = len(param_values)

    if n_params == 0:
        raise ValueError("results_dict must contain at least one result")

    # Get first result to determine dimensions
    first_df = _get_dataframe(list(results_dict.values())[0])
    if variable not in first_df.columns:
        raise ValueError(
            f"Variable '{variable}' not in data columns: {first_df.columns.tolist()}"
        )

    # Set up figure
    fig = go.Figure()

    # Get color scale
    import plotly.colors

    colors = plotly.colors.sample_colorscale(colorscale, n_params)

    # Calculate global y-axis range
    all_y = []
    all_t = []
    for df in results_dict.values():
        df = _get_dataframe(df)
        all_y.extend(df[variable].values)
        all_t.extend(df[time_column].values)
    y_margin = 0.1 * (max(all_y) - min(all_y))
    y_range = [min(all_y) - y_margin, max(all_y) + y_margin]
    t_range = [min(all_t), max(all_t)]

    # Add traces for each parameter value
    for i, (param_val, result) in enumerate(results_dict.items()):
        df = _get_dataframe(result)
        fig.add_trace(
            go.Scatter(
                x=df[time_column].values,
                y=df[variable].values,
                mode="lines",
                name=param_val,
                line=dict(color=colors[i], width=2),
                visible=(i == 0),  # Only first trace visible initially
            )
        )

    # Create frames that show each parameter value
    frames = []
    for i, param_val in enumerate(param_values):
        # Create visibility array
        visibility = [False] * n_params
        visibility[i] = True

        frame = go.Frame(
            data=[go.Scatter(visible=vis) for vis in visibility],
            name=param_val,
            traces=list(range(n_params)),
        )
        frames.append(frame)

    fig.frames = frames

    if title is None:
        title = f"Parameter Sensitivity: {parameter_name}"

    # Create dropdown for parameter selection
    buttons = [
        {
            "args": [
                {"visible": [j == i for j in range(n_params)]},
            ],
            "label": param_val,
            "method": "update",
        }
        for i, param_val in enumerate(param_values)
    ]

    # Animation controls (cycles through all parameters)
    sliders = [
        {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": f"{parameter_name}: ",
                "visible": True,
                "xanchor": "right",
            },
            "transition": {"duration": frame_duration * 5},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [param_val],
                        {
                            "frame": {"duration": frame_duration * 5, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": frame_duration * 5},
                        },
                    ],
                    "label": param_val,
                    "method": "animate",
                }
                for param_val in param_values
            ],
        }
    ]

    updatemenus = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": frame_duration * 5, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": frame_duration * 5},
                        },
                    ],
                    "label": "\u25b6 Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "\u23f8 Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        },
        {
            "buttons": buttons,
            "direction": "down",
            "pad": {"r": 10, "t": 10},
            "showactive": True,
            "x": 1.0,
            "xanchor": "right",
            "y": 1.15,
            "yanchor": "top",
        },
    ]

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title=_get_label(time_column), range=t_range),
        yaxis=dict(title=_get_label(variable), range=y_range),
        width=width,
        height=height,
        sliders=sliders,
        updatemenus=updatemenus,
        template="plotly_white",
    )

    return fig


def save_animation(
    fig: go.Figure,
    path: str | Path,
    format: str | None = None,
    width: int | None = None,
    height: int | None = None,
    scale: float = 2.0,
) -> Path:
    """Save animation to file.

    Supports HTML export (interactive) and static image export via kaleido.

    Args:
        fig: Plotly Figure object to save.
        path: Output file path.
        format: Output format ('html', 'png', 'gif', 'svg', 'pdf').
            If None, inferred from path extension.
        width: Width for image export (uses figure width if None).
        height: Height for image export (uses figure height if None).
        scale: Scale factor for image export (default 2x for crisp images).

    Returns:
        Path to saved animation file.

    Raises:
        ValueError: If format is not supported.

    Example:
        >>> save_animation(fig, 'output.html')  # Interactive
        PosixPath('output.html')
        >>> save_animation(fig, 'output.png', format='png')  # Static
        PosixPath('output.png')
    """
    path = Path(path)

    # Infer format from path if not specified
    if format is None:
        format = path.suffix.lstrip(".").lower()
        if not format:
            format = "html"

    format = format.lower()

    supported_formats = ("html", "png", "gif", "svg", "pdf", "jpeg", "webp")
    if format not in supported_formats:
        raise ValueError(
            f"Unsupported format: {format}. Supported: {supported_formats}"
        )

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure correct extension
    if path.suffix.lower() != f".{format}":
        path = path.with_suffix(f".{format}")

    logger.info(f"Saving animation to {path} (format={format})")

    if format == "html":
        fig.write_html(str(path), include_plotlyjs=True, full_html=True)
    else:
        # Static image export
        fig.write_image(
            str(path),
            format=format,
            width=width,
            height=height,
            scale=scale,
        )

    logger.info(f"Animation saved to {path}")

    # Verification reminder
    print(VERIFICATION_REMINDER)

    return path


__all__ = [
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
