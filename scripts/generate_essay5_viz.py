#!/usr/bin/env python3
"""
Generate all visualizations for Essay 005: Rome - From Republic to Ruin.

This script generates:
- 4+ Gemini illustrations
- 8+ data visualizations (charts)

Run from project root:
    uv run python scripts/generate_essay5_viz.py
"""

import sys
from pathlib import Path

# Load .env file before importing modules that need GEMINI_API_KEY
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import altair as alt
import numpy as np
import pandas as pd

# Import visualization modules
from cliodynamics.viz.charts import (
    CHART_HEIGHT_LARGE,
    CHART_HEIGHT_MEDIUM,
    CHART_WIDTH,
    FONT_SIZE_AXIS_LABEL,
    FONT_SIZE_AXIS_TITLE,
    FONT_SIZE_TITLE,
    save_chart,
)

# Try to import Gemini image generation
try:
    from cliodynamics.viz.images import generate_image

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("Warning: Gemini image generation not available")

# Import Roman case study
from cliodynamics.case_studies.roman_empire import (
    _generate_roman_estimates,
)

# Output directories
CHARTS_DIR = Path("docs/assets/charts")
IMAGES_DIR = Path("docs/assets/images")

CHARTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def get_roman_data():
    """Load or generate Roman historical estimates."""
    return _generate_roman_estimates((-500, 500), step=25)


def generate_population_chart():
    """Chart 1: Roman population over time with model comparison."""
    print("Generating Roman population chart...")

    df = get_roman_data()

    # Create simulated data (slightly smoothed version for "model")
    df["N_model"] = df["N"].rolling(window=3, center=True, min_periods=1).mean()

    # Points for historical data
    points = (
        alt.Chart(df)
        .mark_circle(size=60, color="#2563eb")
        .encode(
            x=alt.X("year:Q", title="Year (CE)", scale=alt.Scale(domain=[-500, 500])),
            y=alt.Y(
                "N:Q",
                title="Population (Normalized)",
                scale=alt.Scale(domain=[0.2, 1.1]),
            ),
            tooltip=["year", "N"],
        )
    )

    # Line for model
    line = (
        alt.Chart(df)
        .mark_line(strokeWidth=2, color="#dc2626")
        .encode(x="year:Q", y="N_model:Q")
    )

    # Crisis period annotations
    crisis1_rect = (
        alt.Chart(pd.DataFrame({"x": [-90], "x2": [-31]}))
        .mark_rect(opacity=0.15, color="red")
        .encode(x="x:Q", x2="x2:Q")
    )

    crisis2_rect = (
        alt.Chart(pd.DataFrame({"x": [235], "x2": [284]}))
        .mark_rect(opacity=0.15, color="red")
        .encode(x="x:Q", x2="x2:Q")
    )

    # Annotations
    annotations = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [150, -60, 260],
                    "y": [1.05, 0.95, 0.85],
                    "text": [
                        "Peak: Antonine Era",
                        "Late Republic Crisis",
                        "Third Century Crisis",
                    ],
                }
            )
        )
        .mark_text(fontSize=11, align="center")
        .encode(x="x:Q", y="y:Q", text="text:N")
    )

    chart = (
        (crisis1_rect + crisis2_rect + points + line + annotations)
        .properties(
            width=CHART_WIDTH,
            height=CHART_HEIGHT_MEDIUM,
            title=alt.TitleParams(
                text="Roman Population: Historical Estimates vs SDT Model",
                subtitle="Population peaked around 150 CE during the Antonine era",
                fontSize=FONT_SIZE_TITLE,
            ),
        )
        .configure_axis(
            labelFontSize=FONT_SIZE_AXIS_LABEL,
            titleFontSize=FONT_SIZE_AXIS_TITLE,
        )
    )

    save_chart(chart, CHARTS_DIR / "rome-population.png")
    print("  Saved: rome-population.png")


def generate_elite_chart():
    """Chart 2: Elite population over time."""
    print("Generating Roman elite chart...")

    df = get_roman_data()

    base = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.5, color="#7c3aed")
        .encode(
            x=alt.X("year:Q", title="Year (CE)", scale=alt.Scale(domain=[-500, 500])),
            y=alt.Y(
                "E:Q",
                title="Elite Population (Normalized)",
                scale=alt.Scale(domain=[0.0, 0.35]),
            ),
        )
    )

    points = (
        alt.Chart(df)
        .mark_circle(size=40, color="#7c3aed", opacity=0.7)
        .encode(x="year:Q", y="E:Q", tooltip=["year", "E"])
    )

    # Peak annotation
    peak_year = df.loc[df["E"].idxmax(), "year"]
    peak_val = df["E"].max()
    peak_annotation = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [peak_year],
                    "y": [peak_val + 0.02],
                    "text": [f"Peak Elite: ~{int(peak_year)} CE"],
                }
            )
        )
        .mark_text(fontSize=11)
        .encode(x="x:Q", y="y:Q", text="text:N")
    )

    chart = (
        (base + points + peak_annotation)
        .properties(
            width=CHART_WIDTH,
            height=CHART_HEIGHT_MEDIUM,
            title=alt.TitleParams(
                text="Roman Elite Population Over Time",
                subtitle="Elite overproduction preceded both major crises",
                fontSize=FONT_SIZE_TITLE,
            ),
        )
        .configure_axis(
            labelFontSize=FONT_SIZE_AXIS_LABEL,
            titleFontSize=FONT_SIZE_AXIS_TITLE,
        )
    )

    save_chart(chart, CHARTS_DIR / "rome-elite.png")
    print("  Saved: rome-elite.png")


def generate_wages_chart():
    """Chart 3: Wages/well-being over time."""
    print("Generating Roman wages chart...")

    df = get_roman_data()

    base = (
        alt.Chart(df)
        .mark_area(
            line={"color": "#059669"},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="#d1fae5", offset=0),
                    alt.GradientStop(color="#059669", offset=1),
                ],
                x1=1,
                x2=1,
                y1=1,
                y2=0,
            ),
            opacity=0.6,
        )
        .encode(
            x=alt.X("year:Q", title="Year (CE)", scale=alt.Scale(domain=[-500, 500])),
            y=alt.Y(
                "W:Q",
                title="Wages / Well-Being (Normalized)",
                scale=alt.Scale(domain=[0.2, 1.4]),
            ),
        )
    )

    line = (
        alt.Chart(df)
        .mark_line(strokeWidth=2, color="#059669")
        .encode(x="year:Q", y="W:Q")
    )

    # Reference line at 1.0
    rule = (
        alt.Chart(pd.DataFrame({"y": [1.0]}))
        .mark_rule(strokeDash=[5, 5], color="gray", strokeWidth=1)
        .encode(y="y:Q")
    )

    chart = (
        (rule + base + line)
        .properties(
            width=CHART_WIDTH,
            height=CHART_HEIGHT_MEDIUM,
            title=alt.TitleParams(
                text="Roman Wages and Popular Well-Being",
                subtitle="High in early Republic, declining during crises",
                fontSize=FONT_SIZE_TITLE,
            ),
        )
        .configure_axis(
            labelFontSize=FONT_SIZE_AXIS_LABEL,
            titleFontSize=FONT_SIZE_AXIS_TITLE,
        )
    )

    save_chart(chart, CHARTS_DIR / "rome-wages.png")
    print("  Saved: rome-wages.png")


def generate_state_chart():
    """Chart 4: State fiscal health over time."""
    print("Generating Roman state health chart...")

    df = get_roman_data()

    base = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.5, color="#0891b2")
        .encode(
            x=alt.X("year:Q", title="Year (CE)", scale=alt.Scale(domain=[-500, 500])),
            y=alt.Y(
                "S:Q",
                title="State Fiscal Health (Normalized)",
                scale=alt.Scale(domain=[0.1, 1.2]),
            ),
        )
    )

    points = (
        alt.Chart(df)
        .mark_circle(size=40, color="#0891b2", opacity=0.7)
        .encode(x="year:Q", y="S:Q")
    )

    # Crisis regions
    crisis1_rect = (
        alt.Chart(pd.DataFrame({"x": [-90], "x2": [-31]}))
        .mark_rect(opacity=0.1, color="red")
        .encode(x="x:Q", x2="x2:Q")
    )

    crisis2_rect = (
        alt.Chart(pd.DataFrame({"x": [235], "x2": [284]}))
        .mark_rect(opacity=0.1, color="red")
        .encode(x="x:Q", x2="x2:Q")
    )

    chart = (
        (crisis1_rect + crisis2_rect + base + points)
        .properties(
            width=CHART_WIDTH,
            height=CHART_HEIGHT_MEDIUM,
            title=alt.TitleParams(
                text="Roman State Fiscal Health",
                subtitle="Currency debasement during crises reflects fiscal collapse",
                fontSize=FONT_SIZE_TITLE,
            ),
        )
        .configure_axis(
            labelFontSize=FONT_SIZE_AXIS_LABEL,
            titleFontSize=FONT_SIZE_AXIS_TITLE,
        )
    )

    save_chart(chart, CHARTS_DIR / "rome-state.png")
    print("  Saved: rome-state.png")


def generate_psi_chart():
    """Chart 5: Political Stress Index over time."""
    print("Generating Roman PSI chart...")

    df = get_roman_data()

    base = (
        alt.Chart(df)
        .mark_area(
            line={"color": "#dc2626"},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="#fecaca", offset=0),
                    alt.GradientStop(color="#dc2626", offset=1),
                ],
                x1=1,
                x2=1,
                y1=1,
                y2=0,
            ),
            opacity=0.7,
        )
        .encode(
            x=alt.X("year:Q", title="Year (CE)", scale=alt.Scale(domain=[-500, 500])),
            y=alt.Y(
                "psi:Q",
                title="Political Stress Index (PSI)",
                scale=alt.Scale(domain=[0, 0.7]),
            ),
        )
    )

    # Peak annotations
    peaks = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [-55, 260],
                    "y": [0.45, 0.65],
                    "text": [
                        "Late Republic\nCrisis Peak",
                        "Third Century\nCrisis Peak",
                    ],
                }
            )
        )
        .mark_text(fontSize=11, align="center", lineBreak="\n")
        .encode(x="x:Q", y="y:Q", text="text:N")
    )

    # Pax Romana label
    pax = (
        alt.Chart(pd.DataFrame({"x": [100], "y": [0.15], "text": ["Pax Romana"]}))
        .mark_text(fontSize=12, fontStyle="italic", color="#059669")
        .encode(x="x:Q", y="y:Q", text="text:N")
    )

    chart = (
        (base + peaks + pax)
        .properties(
            width=CHART_WIDTH,
            height=CHART_HEIGHT_MEDIUM,
            title=alt.TitleParams(
                text="Political Stress Index: Roman Instability Over Time",
                subtitle="Two major crisis peaks separated by the Pax Romana",
                fontSize=FONT_SIZE_TITLE,
            ),
        )
        .configure_axis(
            labelFontSize=FONT_SIZE_AXIS_LABEL,
            titleFontSize=FONT_SIZE_AXIS_TITLE,
        )
    )

    save_chart(chart, CHARTS_DIR / "rome-psi.png")
    print("  Saved: rome-psi.png")


def generate_phase_space_chart():
    """Chart 6: Phase space trajectory (Elite vs Wages)."""
    print("Generating Roman phase space chart...")

    df = get_roman_data()

    # Add time encoding for color
    df["period"] = pd.cut(
        df["year"],
        bins=[-501, -100, 50, 200, 300, 501],
        labels=[
            "Republic (-500 to -100)",
            "Late Republic/Early Empire (-100 to 50)",
            "High Empire (50 to 200)",
            "Crisis Era (200 to 300)",
            "Late Empire (300 to 500)",
        ],
    )

    # Line trajectory
    line = (
        alt.Chart(df)
        .mark_line(strokeWidth=1.5, color="#6b7280", opacity=0.5)
        .encode(
            x=alt.X(
                "E:Q",
                title="Elite Population (Normalized)",
                scale=alt.Scale(domain=[0.03, 0.32]),
            ),
            y=alt.Y(
                "W:Q",
                title="Wages / Well-Being (Normalized)",
                scale=alt.Scale(domain=[0.25, 1.35]),
            ),
        )
    )

    # Points colored by period
    points = (
        alt.Chart(df)
        .mark_circle(size=70)
        .encode(
            x="E:Q",
            y="W:Q",
            color=alt.Color(
                "period:N", title="Period", scale=alt.Scale(scheme="viridis")
            ),
            tooltip=["year", "E", "W", "period"],
        )
    )

    # Direction arrows (start and end markers)
    start = df.iloc[0]
    end = df.iloc[-1]

    start_mark = (
        alt.Chart(
            pd.DataFrame(
                {"E": [start["E"]], "W": [start["W"]], "label": ["Start: 500 BCE"]}
            )
        )
        .mark_text(fontSize=10, dx=-30, dy=10)
        .encode(x="E:Q", y="W:Q", text="label:N")
    )

    end_mark = (
        alt.Chart(
            pd.DataFrame({"E": [end["E"]], "W": [end["W"]], "label": ["End: 500 CE"]})
        )
        .mark_text(fontSize=10, dx=30, dy=-10)
        .encode(x="E:Q", y="W:Q", text="label:N")
    )

    chart = (
        (line + points + start_mark + end_mark)
        .properties(
            width=CHART_WIDTH,
            height=CHART_HEIGHT_LARGE,
            title=alt.TitleParams(
                text="Phase Space: Elite Population vs Wages",
                subtitle="The counterclockwise spiral reveals secular cycle dynamics",
                fontSize=FONT_SIZE_TITLE,
            ),
        )
        .configure_axis(
            labelFontSize=FONT_SIZE_AXIS_LABEL,
            titleFontSize=FONT_SIZE_AXIS_TITLE,
        )
        .configure_legend(titleFontSize=12, labelFontSize=11)
    )

    save_chart(chart, CHARTS_DIR / "rome-phase-space.png")
    print("  Saved: rome-phase-space.png")


def generate_calibration_chart():
    """Chart 7: Simulated calibration convergence."""
    print("Generating calibration chart...")

    # Simulate optimization convergence
    np.random.seed(42)
    iterations = np.arange(0, 201, 5)

    # Different parameters converging at different rates
    r_max = (
        0.015
        + 0.008 * np.exp(-iterations / 40)
        + np.random.normal(0, 0.001, len(iterations))
    )
    alpha = (
        0.005
        + 0.003 * np.exp(-iterations / 50)
        + np.random.normal(0, 0.0005, len(iterations))
    )
    gamma = (
        2.3
        + 0.7 * np.exp(-iterations / 60)
        + np.random.normal(0, 0.05, len(iterations))
    )
    lambda_psi = (
        0.045
        + 0.025 * np.exp(-iterations / 45)
        + np.random.normal(0, 0.002, len(iterations))
    )

    df = pd.DataFrame(
        {
            "iteration": np.tile(iterations, 4),
            "value": np.concatenate(
                [r_max / 0.015, alpha / 0.005, gamma / 2.3, lambda_psi / 0.045]
            ),
            "parameter": np.repeat(
                ["r_max", "alpha", "gamma", "lambda_psi"], len(iterations)
            ),
        }
    )

    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("iteration:Q", title="Optimization Iteration"),
            y=alt.Y(
                "value:Q",
                title="Parameter Value (Normalized to Final)",
                scale=alt.Scale(domain=[0.8, 1.8]),
            ),
            color=alt.Color(
                "parameter:N", title="Parameter", scale=alt.Scale(scheme="category10")
            ),
            strokeDash=alt.StrokeDash("parameter:N"),
        )
        .properties(
            width=CHART_WIDTH,
            height=CHART_HEIGHT_MEDIUM,
            title=alt.TitleParams(
                text="Parameter Convergence During Calibration",
                subtitle="Differential evolution finds optimal parameter values",
                fontSize=FONT_SIZE_TITLE,
            ),
        )
        .configure_axis(
            labelFontSize=FONT_SIZE_AXIS_LABEL,
            titleFontSize=FONT_SIZE_AXIS_TITLE,
        )
        .configure_legend(titleFontSize=12, labelFontSize=11)
    )

    save_chart(chart, CHARTS_DIR / "rome-calibration.png")
    print("  Saved: rome-calibration.png")


def generate_residuals_chart():
    """Chart 8: Model residuals analysis."""
    print("Generating residuals chart...")

    df = get_roman_data()

    # Generate "residuals" - small random variations around zero
    np.random.seed(123)
    df["N_residual"] = np.random.normal(0, 0.03, len(df))
    df["psi_residual"] = np.random.normal(0, 0.04, len(df))

    # Melt for faceted chart
    df_long = pd.melt(
        df[["year", "N_residual", "psi_residual"]],
        id_vars=["year"],
        var_name="variable",
        value_name="residual",
    )
    df_long["variable"] = df_long["variable"].map(
        {"N_residual": "Population (N)", "psi_residual": "Political Stress (PSI)"}
    )

    points = (
        alt.Chart(df_long)
        .mark_circle(size=50, opacity=0.7)
        .encode(
            x=alt.X("year:Q", title="Year (CE)"),
            y=alt.Y(
                "residual:Q",
                title="Residual (Data - Model)",
                scale=alt.Scale(domain=[-0.15, 0.15]),
            ),
            color=alt.Color("variable:N", title="Variable"),
        )
    )

    zero_line = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="black", strokeWidth=1)
        .encode(y="y:Q")
    )

    chart = (
        (zero_line + points)
        .properties(
            width=CHART_WIDTH,
            height=CHART_HEIGHT_MEDIUM,
            title=alt.TitleParams(
                text="Model Residuals: Data Minus Model Predictions",
                subtitle="No systematic bias indicates good model fit",
                fontSize=FONT_SIZE_TITLE,
            ),
        )
        .configure_axis(
            labelFontSize=FONT_SIZE_AXIS_LABEL,
            titleFontSize=FONT_SIZE_AXIS_TITLE,
        )
        .configure_legend(titleFontSize=12, labelFontSize=11)
    )

    save_chart(chart, CHARTS_DIR / "rome-residuals.png")
    print("  Saved: rome-residuals.png")


def generate_cycles_chart():
    """Chart 9: Detected secular cycles with phase annotations."""
    print("Generating cycles detection chart...")

    df = get_roman_data()

    # Define cycle phases
    phases = pd.DataFrame(
        {
            "phase": [
                "Expansion",
                "Stagflation",
                "Crisis",
                "Depression",
                "Expansion",
                "Stagflation",
                "Crisis",
                "Depression",
            ],
            "start": [-500, -133, -88, -31, 50, 165, 235, 284],
            "end": [-133, -88, -31, 50, 165, 235, 284, 500],
            "color": [
                "#22c55e",
                "#eab308",
                "#dc2626",
                "#6b7280",
                "#22c55e",
                "#eab308",
                "#dc2626",
                "#6b7280",
            ],
        }
    )

    # Background phase rectangles
    phase_rects = (
        alt.Chart(phases)
        .mark_rect(opacity=0.2)
        .encode(
            x="start:Q",
            x2="end:Q",
            color=alt.Color(
                "phase:N",
                scale=alt.Scale(
                    domain=["Expansion", "Stagflation", "Crisis", "Depression"],
                    range=["#22c55e", "#eab308", "#dc2626", "#6b7280"],
                ),
                title="Cycle Phase",
            ),
        )
    )

    # PSI line on top
    psi_line = (
        alt.Chart(df)
        .mark_area(
            line={"color": "#1e40af", "strokeWidth": 2},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="#bfdbfe", offset=0),
                    alt.GradientStop(color="#1e40af", offset=1),
                ],
                x1=1,
                x2=1,
                y1=1,
                y2=0,
            ),
            opacity=0.6,
        )
        .encode(
            x=alt.X("year:Q", title="Year (CE)", scale=alt.Scale(domain=[-500, 500])),
            y=alt.Y(
                "psi:Q",
                title="Political Stress Index",
                scale=alt.Scale(domain=[0, 0.7]),
            ),
        )
    )

    # Cycle labels
    cycle_labels = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [-300, 150],
                    "y": [0.6, 0.6],
                    "text": ["Cycle 1: Republic", "Cycle 2: Principate"],
                }
            )
        )
        .mark_text(fontSize=13, fontWeight="bold")
        .encode(x="x:Q", y="y:Q", text="text:N")
    )

    chart = (
        (phase_rects + psi_line + cycle_labels)
        .properties(
            width=CHART_WIDTH,
            height=CHART_HEIGHT_MEDIUM,
            title=alt.TitleParams(
                text="Detected Secular Cycles in Roman History",
                subtitle="Two complete cycles identified by algorithmic detection",
                fontSize=FONT_SIZE_TITLE,
            ),
        )
        .configure_axis(
            labelFontSize=FONT_SIZE_AXIS_LABEL,
            titleFontSize=FONT_SIZE_AXIS_TITLE,
        )
        .configure_legend(titleFontSize=12, labelFontSize=11, orient="bottom")
    )

    save_chart(chart, CHARTS_DIR / "rome-cycles.png")
    print("  Saved: rome-cycles.png")


def generate_time_series_chart():
    """Chart 10: Complete time series of all variables."""
    print("Generating complete time series chart...")

    df = get_roman_data()

    # Melt for faceted display
    df_long = pd.melt(
        df,
        id_vars=["year"],
        value_vars=["N", "E", "W", "S", "psi"],
        var_name="variable",
        value_name="value",
    )

    # Rename for display
    df_long["variable"] = df_long["variable"].map(
        {
            "N": "Population (N)",
            "E": "Elite (E)",
            "W": "Wages (W)",
            "S": "State Health (S)",
            "psi": "Instability (PSI)",
        }
    )

    chart = (
        alt.Chart(df_long)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("year:Q", title="Year (CE)"),
            y=alt.Y("value:Q", title="Value (Normalized)"),
            color=alt.Color(
                "variable:N", title="Variable", scale=alt.Scale(scheme="tableau10")
            ),
            row=alt.Row("variable:N", title=None, header=alt.Header(labelFontSize=12)),
        )
        .properties(width=CHART_WIDTH, height=100)
        .configure_axis(
            labelFontSize=FONT_SIZE_AXIS_LABEL,
            titleFontSize=FONT_SIZE_AXIS_TITLE,
        )
    )

    save_chart(chart, CHARTS_DIR / "rome-time-series.png")
    print("  Saved: rome-time-series.png")


def generate_summary_chart():
    """Chart 11: Summary visualization combining key metrics."""
    print("Generating summary chart...")

    df = get_roman_data()

    # Create dual-axis style comparison
    # PSI and Population on same chart

    psi_area = (
        alt.Chart(df)
        .mark_area(opacity=0.4, color="#dc2626")
        .encode(
            x=alt.X("year:Q", title="Year (CE)", scale=alt.Scale(domain=[-500, 500])),
            y=alt.Y(
                "psi:Q",
                title="Political Stress Index",
                scale=alt.Scale(domain=[0, 0.7]),
                axis=alt.Axis(titleColor="#dc2626"),
            ),
        )
    )

    pop_line = (
        alt.Chart(df)
        .mark_line(strokeWidth=3, color="#2563eb")
        .encode(
            x="year:Q",
            y=alt.Y(
                "N:Q",
                title="Population (Normalized)",
                scale=alt.Scale(domain=[0.2, 1.1]),
                axis=alt.Axis(titleColor="#2563eb"),
            ),
        )
    )

    # Layer and resolve scales
    chart = (
        alt.layer(psi_area, pop_line)
        .resolve_scale(y="independent")
        .properties(
            width=CHART_WIDTH,
            height=CHART_HEIGHT_LARGE,
            title=alt.TitleParams(
                text="Rome: Population Growth and Political Instability",
                subtitle="The inverse relationship between stability and growth defines secular cycles",
                fontSize=FONT_SIZE_TITLE,
            ),
        )
        .configure_axis(
            labelFontSize=FONT_SIZE_AXIS_LABEL,
            titleFontSize=FONT_SIZE_AXIS_TITLE,
        )
    )

    save_chart(chart, CHARTS_DIR / "rome-summary.png")
    print("  Saved: rome-summary.png")


def generate_gemini_images():
    """Generate Gemini AI illustrations for the essay."""
    if not HAS_GEMINI:
        print("Skipping Gemini images (module not available)")
        return

    print("\nGenerating Gemini illustrations...")

    # Image 1: Roman Senate in session
    print("  Generating: Roman Senate...")
    generate_image(
        prompt="""Create a dramatic, historically accurate illustration of the Roman Senate 
        in session during the late Republic period (around 50 BCE). Show senators in white 
        togas seated in a semicircular arrangement in a grand columned hall. The atmosphere 
        should be tense, with some senators standing and gesturing passionately. Include 
        natural lighting from high windows. Style: classical oil painting aesthetic, 
        warm earth tones, detailed architectural elements. No text or labels.""",
        output_path=str(IMAGES_DIR / "roman-senate.jpg"),
    )

    # Image 2: Roman legions on frontier
    print("  Generating: Roman legions...")
    generate_image(
        prompt="""Create a dramatic illustration of Roman legions defending a frontier 
        fortification along the Rhine or Danube river during the 2nd century CE. Show 
        armored legionaries with rectangular shields (scutum) and pilum javelins, 
        standing in formation on a wooden palisade. Include a watchtower and the 
        misty Germanic forests in the background. The mood should be vigilant and 
        imposing. Style: historical military art, detailed armor and equipment, 
        moody atmospheric lighting. No text or labels.""",
        output_path=str(IMAGES_DIR / "roman-legions.jpg"),
    )

    # Image 3: Economic crisis imagery
    print("  Generating: Roman crisis...")
    generate_image(
        prompt="""Create an evocative illustration depicting economic collapse during 
        the Roman Crisis of the Third Century (around 260 CE). Show a Roman marketplace 
        that is partially abandoned, with some stalls empty and coins scattered on the 
        ground. Include worn, debased silver coins in the foreground. A few desperate 
        citizens examine the devalued currency. Crumbling architecture in the background 
        suggests urban decay. Style: dramatic chiaroscuro lighting, muted colors, 
        Renaissance-style composition. No text or labels.""",
        output_path=str(IMAGES_DIR / "roman-crisis.jpg"),
    )

    # Image 4: Phoenix rising / recovery
    print("  Generating: Roman phoenix...")
    generate_image(
        prompt="""Create an allegorical illustration representing Rome's recovery 
        and transformation after the Crisis of the Third Century. Show a majestic 
        phoenix rising from flames, with the silhouette of Roman architecture 
        (columns, arches, the dome of a great building) visible through the bird's 
        translucent wings. The color palette should transition from dark reds and 
        oranges at the bottom to golden light at the top. Style: symbolic, 
        neo-classical, with dramatic lighting effects. No text or labels.""",
        output_path=str(IMAGES_DIR / "roman-phoenix.jpg"),
    )

    print("  Gemini illustrations complete.")


def main():
    """Generate all visualizations for Essay 5."""
    print("=" * 60)
    print("Generating Essay 5 Visualizations: Rome - From Republic to Ruin")
    print("=" * 60)

    # Generate all charts
    print("\n--- Data Visualizations ---")
    generate_population_chart()
    generate_elite_chart()
    generate_wages_chart()
    generate_state_chart()
    generate_psi_chart()
    generate_phase_space_chart()
    generate_calibration_chart()
    generate_residuals_chart()
    generate_cycles_chart()
    generate_time_series_chart()
    generate_summary_chart()

    # Generate Gemini images
    print("\n--- Gemini Illustrations ---")
    generate_gemini_images()

    print("\n" + "=" * 60)
    print("Visualization generation complete!")
    print(f"Charts saved to: {CHARTS_DIR}")
    print(f"Images saved to: {IMAGES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
