#!/usr/bin/env python3
"""Generate Plotly animations for Essay 070: Monte Carlo Methods.

This script creates interactive animations demonstrating Monte Carlo
simulation concepts for the essay.

Output directory: docs/assets/animations/essay-070/
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import plotly.graph_objects as go

from cliodynamics.models import SDTModel, SDTParams
from cliodynamics.viz.plotly_animations import save_animation

OUTPUT_DIR = "docs/assets/animations/essay-070"


def create_calibrated_us_model() -> SDTModel:
    """Create SDT model calibrated for U.S. dynamics."""
    params = SDTParams(
        r_max=0.012,
        K_0=1.0,
        beta=0.9,
        mu=0.15,
        alpha=0.006,
        delta_e=0.012,
        gamma=1.2,
        eta=0.6,
        rho=0.15,
        sigma=0.06,
        epsilon=0.03,
        lambda_psi=0.04,
        theta_w=0.8,
        theta_e=1.2,
        theta_s=0.6,
        psi_decay=0.08,
        W_0=1.0,
        E_0=0.08,
        S_0=1.0,
    )
    return SDTModel(params)


def get_us_state_2020() -> dict[str, float]:
    """Current U.S. state circa 2020."""
    return {
        "N": 0.85,
        "E": 0.11,
        "W": 0.82,
        "S": 0.88,
        "psi": 0.30,
    }


def generate_monte_carlo_accumulation_animation(output_path: str) -> None:
    """Create animation showing trajectories accumulating over simulation runs.

    This animation demonstrates how Monte Carlo builds up a probability
    distribution from individual random simulations.
    """
    print("  Generating Monte Carlo accumulation animation...")

    np.random.seed(42)

    # Generate synthetic trajectories for visualization
    n_trajectories = 100
    n_times = 51
    years = np.arange(n_times)

    trajectories = []
    for i in range(n_trajectories):
        # Base trajectory with random variation
        alpha_var = 0.006 + np.random.normal(0, 0.002)
        alpha_var = np.clip(alpha_var, 0.002, 0.012)

        # Simulate simplified dynamics
        psi = [0.30 + np.random.normal(0, 0.03)]
        for t in range(1, n_times):
            # Simplified accumulation with variation
            dpsi = alpha_var * 2 + np.random.normal(0, 0.01)
            new_psi = psi[-1] + dpsi
            psi.append(max(0, new_psi))

        trajectories.append(psi)

    trajectories = np.array(trajectories)

    # Create figure with frames
    fig = go.Figure()

    # Add traces that will be updated in frames
    # Empty traces for initial state
    for i in range(n_trajectories):
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="rgba(31, 119, 180, 0.2)", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Add percentile lines (initially hidden)
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            name="90th percentile",
            line=dict(color="#d62728", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            name="Median",
            line=dict(color="#d62728", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            name="10th percentile",
            line=dict(color="#d62728", width=2, dash="dash"),
        )
    )

    # Counter text
    fig.add_trace(
        go.Scatter(
            x=[40],
            y=[1.2],
            mode="text",
            text=["Simulations: 0"],
            textfont=dict(size=18, color="black"),
            showlegend=False,
        )
    )

    # Create frames
    frames = []

    # Frames 1-100: Add trajectories one by one
    for frame_num in range(1, n_trajectories + 1):
        frame_data = []

        # Add trajectories up to current count
        for i in range(n_trajectories):
            if i < frame_num:
                frame_data.append(
                    go.Scatter(
                        x=years,
                        y=trajectories[i],
                    )
                )
            else:
                frame_data.append(go.Scatter(x=[], y=[]))

        # Percentile lines (only show after we have enough data)
        if frame_num >= 10:
            current_p10 = np.percentile(trajectories[:frame_num], 10, axis=0)
            current_p50 = np.percentile(trajectories[:frame_num], 50, axis=0)
            current_p90 = np.percentile(trajectories[:frame_num], 90, axis=0)

            frame_data.append(go.Scatter(x=years, y=current_p90))
            frame_data.append(go.Scatter(x=years, y=current_p50))
            frame_data.append(go.Scatter(x=years, y=current_p10))
        else:
            frame_data.append(go.Scatter(x=[], y=[]))
            frame_data.append(go.Scatter(x=[], y=[]))
            frame_data.append(go.Scatter(x=[], y=[]))

        # Counter text
        frame_data.append(
            go.Scatter(
                x=[40],
                y=[1.2],
                text=[f"Simulations: {frame_num}"],
            )
        )

        frames.append(
            go.Frame(
                data=frame_data,
                name=str(frame_num),
                traces=list(range(n_trajectories + 4)),
            )
        )

    fig.frames = frames

    # Animation controls
    updatemenus = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 50, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                        },
                    ],
                    "label": "Pause",
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

    sliders = [
        {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Simulations: ",
                "visible": True,
                "xanchor": "right",
            },
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [str(i)],
                        {
                            "frame": {"duration": 50, "redraw": True},
                            "mode": "immediate",
                        },
                    ],
                    "label": str(i),
                    "method": "animate",
                }
                for i in range(1, n_trajectories + 1, 5)
            ],
        }
    ]

    fig.update_layout(
        title=dict(
            text="Monte Carlo Simulation: Building a Probability Distribution",
            font=dict(size=20),
        ),
        xaxis=dict(
            title="Years from 2020",
            range=[0, 50],
        ),
        yaxis=dict(
            title="Political Stress Index (PSI)",
            range=[0, 1.5],
        ),
        width=900,
        height=600,
        updatemenus=updatemenus,
        sliders=sliders,
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )

    save_animation(fig, output_path)
    print(f"    Saved: {output_path}")


def generate_parameter_sensitivity_animation(output_path: str) -> None:
    """Create animation showing how parameter changes affect trajectories.

    This demonstrates sensitivity analysis by animating through
    different parameter values and showing the resulting trajectories.
    """
    print("  Generating parameter sensitivity animation...")

    np.random.seed(42)

    # Alpha values to sweep through
    alpha_values = np.linspace(0.003, 0.012, 20)
    n_times = 51
    years = np.arange(n_times)

    # Generate trajectories for each alpha value
    trajectories = {}
    for alpha in alpha_values:
        # Simplified simulation with this alpha
        psi = [0.30]
        for t in range(1, n_times):
            dpsi = alpha * 2.5 - 0.002  # Simplified dynamics
            new_psi = psi[-1] + dpsi + np.random.normal(0, 0.005)
            psi.append(max(0, new_psi))
        trajectories[alpha] = psi

    # Create figure
    fig = go.Figure()

    # Add initial trajectory
    fig.add_trace(
        go.Scatter(
            x=years,
            y=trajectories[alpha_values[0]],
            mode="lines",
            name="PSI Trajectory",
            line=dict(color="#1f77b4", width=3),
        )
    )

    # Add threshold line
    fig.add_trace(
        go.Scatter(
            x=[0, 50],
            y=[0.8, 0.8],
            mode="lines",
            name="Crisis Threshold",
            line=dict(color="#d62728", width=2, dash="dash"),
        )
    )

    # Add alpha value text
    fig.add_trace(
        go.Scatter(
            x=[5],
            y=[1.3],
            mode="text",
            text=[f"Alpha = {alpha_values[0]:.4f}"],
            textfont=dict(size=18, color="black"),
            showlegend=False,
        )
    )

    # Create frames
    frames = []
    for i, alpha in enumerate(alpha_values):
        frame_data = [
            go.Scatter(
                x=years,
                y=trajectories[alpha],
            ),
            go.Scatter(),  # Threshold line unchanged
            go.Scatter(
                x=[5],
                y=[1.3],
                text=[f"Elite Growth Rate (alpha) = {alpha:.4f}"],
            ),
        ]
        frames.append(
            go.Frame(
                data=frame_data,
                name=f"{alpha:.4f}",
                traces=[0, 1, 2],
            )
        )

    fig.frames = frames

    # Animation controls
    updatemenus = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 200, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 100},
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                        },
                    ],
                    "label": "Pause",
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

    sliders = [
        {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 14},
                "prefix": "Alpha: ",
                "visible": True,
                "xanchor": "right",
            },
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [f"{alpha:.4f}"],
                        {
                            "frame": {"duration": 200, "redraw": True},
                            "mode": "immediate",
                        },
                    ],
                    "label": f"{alpha:.4f}",
                    "method": "animate",
                }
                for alpha in alpha_values
            ],
        }
    ]

    fig.update_layout(
        title=dict(
            text="Parameter Sensitivity: How Elite Growth Rate Shapes the Future",
            font=dict(size=20),
        ),
        xaxis=dict(
            title="Years from 2020",
            range=[0, 50],
        ),
        yaxis=dict(
            title="Political Stress Index (PSI)",
            range=[0, 1.5],
        ),
        width=900,
        height=600,
        updatemenus=updatemenus,
        sliders=sliders,
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
    )

    save_animation(fig, output_path)
    print(f"    Saved: {output_path}")


def generate_uncertainty_growth_animation(output_path: str) -> None:
    """Create animation showing how uncertainty grows over forecast horizon.

    This demonstrates the fundamental challenge of long-range forecasting:
    uncertainty compounds over time.
    """
    print("  Generating uncertainty growth animation...")

    np.random.seed(42)

    n_sims = 200
    n_times = 51
    years = np.arange(n_times)

    # Generate ensemble with growing uncertainty
    base_trajectory = 0.30 + 0.015 * years
    trajectories = []

    for _ in range(n_sims):
        # Random walk component that grows with time
        noise = np.cumsum(np.random.normal(0, 0.02, n_times))
        traj = base_trajectory + noise
        trajectories.append(np.maximum(0, traj))

    trajectories = np.array(trajectories)

    # Calculate percentiles at each time
    p5 = np.percentile(trajectories, 5, axis=0)
    p25 = np.percentile(trajectories, 25, axis=0)
    p50 = np.percentile(trajectories, 50, axis=0)
    p75 = np.percentile(trajectories, 75, axis=0)
    p95 = np.percentile(trajectories, 95, axis=0)

    # Create figure
    fig = go.Figure()

    # Outer band (5-95%)
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            fill=None,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            fill="tonexty",
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(31, 119, 180, 0.2)",
            name="90% confidence",
        )
    )

    # Inner band (25-75%)
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            fill=None,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            fill="tonexty",
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(31, 119, 180, 0.4)",
            name="50% confidence",
        )
    )

    # Median line
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            name="Median forecast",
            line=dict(color="#d62728", width=3),
        )
    )

    # Uncertainty width annotation
    fig.add_trace(
        go.Scatter(
            x=[25],
            y=[1.3],
            mode="text",
            text=["Year 0"],
            textfont=dict(size=16, color="black"),
            showlegend=False,
        )
    )

    # Create frames - progressively reveal the forecast
    frames = []
    for t_end in range(1, n_times + 1):
        t_range = years[:t_end]

        frame_data = [
            go.Scatter(x=t_range, y=p5[:t_end]),  # Lower bound outer
            go.Scatter(x=t_range, y=p95[:t_end]),  # Upper bound outer
            go.Scatter(x=t_range, y=p25[:t_end]),  # Lower bound inner
            go.Scatter(x=t_range, y=p75[:t_end]),  # Upper bound inner
            go.Scatter(x=t_range, y=p50[:t_end]),  # Median
            go.Scatter(
                x=[25],
                y=[1.3],
                text=[
                    f"Year {t_end - 1} | Width: {p95[t_end - 1] - p5[t_end - 1]:.2f}"
                ],
            ),
        ]

        frames.append(
            go.Frame(
                data=frame_data,
                name=str(t_end),
                traces=[0, 1, 2, 3, 4, 5],
            )
        )

    fig.frames = frames

    # Animation controls
    updatemenus = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 100, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 50},
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                        },
                    ],
                    "label": "Pause",
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

    sliders = [
        {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 14},
                "prefix": "Year: ",
                "visible": True,
                "xanchor": "right",
            },
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [str(t)],
                        {
                            "frame": {"duration": 100, "redraw": True},
                            "mode": "immediate",
                        },
                    ],
                    "label": str(t - 1),
                    "method": "animate",
                }
                for t in range(1, n_times + 1, 2)
            ],
        }
    ]

    fig.update_layout(
        title=dict(
            text="Uncertainty Grows Over Time: The Forecast Cone",
            font=dict(size=20),
        ),
        xaxis=dict(
            title="Years from 2020",
            range=[0, 50],
        ),
        yaxis=dict(
            title="Political Stress Index (PSI)",
            range=[0, 1.6],
        ),
        width=900,
        height=600,
        updatemenus=updatemenus,
        sliders=sliders,
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )

    save_animation(fig, output_path)
    print(f"    Saved: {output_path}")


def main():
    """Generate all animations for Essay 070."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Generating Essay 070 Animations: Monte Carlo Methods")
    print("=" * 60)

    print("\n1. Monte Carlo accumulation animation...")
    generate_monte_carlo_accumulation_animation(
        os.path.join(OUTPUT_DIR, "monte-carlo-accumulation.html")
    )

    print("\n2. Parameter sensitivity animation...")
    generate_parameter_sensitivity_animation(
        os.path.join(OUTPUT_DIR, "parameter-sensitivity.html")
    )

    print("\n3. Uncertainty growth animation...")
    generate_uncertainty_growth_animation(
        os.path.join(OUTPUT_DIR, "uncertainty-growth.html")
    )

    print("\n" + "=" * 60)
    print(f"All animations saved to: {OUTPUT_DIR}")
    print("=" * 60)
    print("\nIMPORTANT: Visually verify all animations before committing!")
    print("Open each .html file in a browser and check:")
    print("  1. Animation plays smoothly")
    print("  2. Labels are readable")
    print("  3. Play/pause controls work")


if __name__ == "__main__":
    main()
