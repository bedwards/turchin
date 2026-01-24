#!/usr/bin/env python3
"""Generate Altair charts for Essay 019: Seeing the Invisible.

This script creates all the static visualizations demonstrating
visualization concepts and techniques for cliodynamics.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd
import altair as alt

from cliodynamics.models import SDTModel, SDTParams
from cliodynamics.simulation import Simulator
from cliodynamics.viz.charts import (
    configure_chart, save_chart, year_axis,
    CHART_WIDTH, CHART_HEIGHT_MEDIUM, FONT_SIZE_TITLE,
    FONT_SIZE_AXIS_LABEL, FONT_SIZE_AXIS_TITLE,
)
from cliodynamics.viz.plots import DEFAULT_COLORS

CHARTS_DIR = project_root / "docs" / "assets" / "charts" / "essay-019"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_simulation_data():
    """Generate SDT simulation data for visualizations."""
    params = SDTParams()
    model = SDTModel(params)
    simulator = Simulator(model)
    initial = {'N': 0.5, 'E': 0.02, 'W': 1.0, 'S': 1.0, 'psi': 0.1}
    results = simulator.run(initial, time_span=(0, 500), dt=1.0)
    return results


def main():
    print("Generating charts for Essay 019: Visualization...")
    print("=" * 60)

    results = generate_simulation_data()
    df = results.df
    print(f"Generated simulation with {len(df)} time points")

    # ==================================================
    # 1. Basic Time Series - Single Variable
    # ==================================================
    print("\n1. Basic time series (single variable)...")
    chart = alt.Chart(df).mark_line(
        color=DEFAULT_COLORS[0], strokeWidth=2
    ).encode(
        x=alt.X('t:Q', title='Time (years)', axis=alt.Axis(format='d')),
        y=alt.Y('psi:Q', title='Political Stress Index (psi)')
    )
    chart = configure_chart(chart, 'Political Stress Over Time', height=400)
    save_chart(chart, CHARTS_DIR / "timeseries-single.png")
    print("  Saved timeseries-single.png")

    # ==================================================
    # 2. Multi-Variable Time Series (Overlaid)
    # ==================================================
    print("\n2. Multi-variable time series (overlaid)...")
    melted = df[['t', 'N', 'W', 'psi']].melt(
        id_vars=['t'], var_name='Variable', value_name='Value'
    )
    label_map = {'N': 'Population (N)', 'W': 'Wages (W)', 'psi': 'Stress (psi)'}
    melted['Label'] = melted['Variable'].map(label_map)

    chart = alt.Chart(melted).mark_line(strokeWidth=2).encode(
        x=alt.X('t:Q', title='Time (years)', axis=alt.Axis(format='d')),
        y=alt.Y('Value:Q', title='Normalized Value'),
        color=alt.Color('Label:N', title='Variable', scale=alt.Scale(
            domain=list(label_map.values()),
            range=DEFAULT_COLORS[:3]
        ))
    )
    chart = configure_chart(chart, 'SDT Model Core Variables', height=450)
    save_chart(chart, CHARTS_DIR / "timeseries-multi.png")
    print("  Saved timeseries-multi.png")

    # ==================================================
    # 3. Faceted Time Series (Small Multiples)
    # ==================================================
    print("\n3. Faceted time series (small multiples)...")
    melted_all = df[['t', 'N', 'E', 'W', 'S', 'psi']].melt(
        id_vars=['t'], var_name='Variable', value_name='Value'
    )
    var_order = ['N', 'E', 'W', 'S', 'psi']
    full_labels = {
        'N': 'Population (N)',
        'E': 'Elites (E)',
        'W': 'Wages (W)',
        'S': 'State Strength (S)',
        'psi': 'Stress Index (psi)'
    }
    melted_all['Label'] = melted_all['Variable'].map(full_labels)

    chart = alt.Chart(melted_all).mark_line(
        color=DEFAULT_COLORS[0], strokeWidth=2
    ).encode(
        x=alt.X('t:Q', title='Time (years)', axis=alt.Axis(format='d')),
        y=alt.Y('Value:Q', title='')
    ).properties(
        width=700, height=100
    ).facet(
        row=alt.Row('Label:N', title=None, sort=[full_labels[v] for v in var_order],
                    header=alt.Header(labelFontSize=14, labelAngle=0, labelAlign='left'))
    ).configure_axis(
        labelFontSize=FONT_SIZE_AXIS_LABEL,
        titleFontSize=FONT_SIZE_AXIS_TITLE
    ).properties(
        title=alt.TitleParams('All Five SDT Variables', fontSize=FONT_SIZE_TITLE)
    )
    save_chart(chart, CHARTS_DIR / "timeseries-faceted.png")
    print("  Saved timeseries-faceted.png")

    # ==================================================
    # 4. Phase Space 2D (W vs psi)
    # ==================================================
    print("\n4. Phase space 2D (W vs psi)...")
    chart = alt.Chart(df).mark_trail(strokeWidth=2).encode(
        x=alt.X('W:Q', title='Real Wages (W)'),
        y=alt.Y('psi:Q', title='Political Stress Index (psi)'),
        color=alt.Color('t:Q', title='Time', scale=alt.Scale(scheme='viridis')),
        size=alt.value(2),
        order='t:Q'
    )
    # Add start and end markers
    start_pt = alt.Chart(df.head(1)).mark_point(
        color='green', size=100, filled=True
    ).encode(x='W:Q', y='psi:Q')
    end_pt = alt.Chart(df.tail(1)).mark_point(
        color='red', size=100, shape='square', filled=True
    ).encode(x='W:Q', y='psi:Q')
    combined = chart + start_pt + end_pt
    combined = configure_chart(combined, 'Phase Space: Wages vs Instability', height=500)
    save_chart(combined, CHARTS_DIR / "phase-space-2d.png")
    print("  Saved phase-space-2d.png")

    # ==================================================
    # 5. Phase Space Comparison (N vs psi)
    # ==================================================
    print("\n5. Phase space 2D (N vs psi)...")
    chart = alt.Chart(df).mark_trail(strokeWidth=2).encode(
        x=alt.X('N:Q', title='Population (N)'),
        y=alt.Y('psi:Q', title='Political Stress Index (psi)'),
        color=alt.Color('t:Q', title='Time', scale=alt.Scale(scheme='plasma')),
        size=alt.value(2),
        order='t:Q'
    )
    start_pt = alt.Chart(df.head(1)).mark_point(
        color='green', size=100, filled=True
    ).encode(x='N:Q', y='psi:Q')
    end_pt = alt.Chart(df.tail(1)).mark_point(
        color='red', size=100, shape='square', filled=True
    ).encode(x='N:Q', y='psi:Q')
    combined = chart + start_pt + end_pt
    combined = configure_chart(combined, 'Phase Space: Population vs Instability', height=500)
    save_chart(combined, CHARTS_DIR / "phase-space-n-psi.png")
    print("  Saved phase-space-n-psi.png")

    # ==================================================
    # 6. Bad Design Example (Before)
    # ==================================================
    print("\n6. Bad design example (before)...")
    np.random.seed(42)
    t = np.linspace(0, 100, 200)
    psi = 0.2 + 0.3 * np.sin(2 * np.pi * t / 50) + 0.1 * np.random.randn(200)
    design_df = pd.DataFrame({'t': t, 'psi': psi})

    # Intentionally poor design
    bad_chart = alt.Chart(design_df).mark_line(strokeWidth=1).encode(
        x='t:Q',
        y='psi:Q'
    ).properties(width=400, height=200)
    # No title, tiny fonts, no labels
    bad_chart.save(str(CHARTS_DIR / "design-bad.png"), scale_factor=1)
    print("  Saved design-bad.png")

    # ==================================================
    # 7. Good Design Example (After)
    # ==================================================
    print("\n7. Good design example (after)...")
    good_chart = alt.Chart(design_df).mark_line(
        color=DEFAULT_COLORS[0], strokeWidth=2
    ).encode(
        x=alt.X('t:Q', title='Time (years)', axis=alt.Axis(format='d')),
        y=alt.Y('psi:Q', title='Political Stress Index')
    )
    good_chart = configure_chart(good_chart, 'Political Instability Over Time', height=400)
    save_chart(good_chart, CHARTS_DIR / "design-good.png")
    print("  Saved design-good.png")

    # ==================================================
    # 8. Colorblind-Friendly Palette Demonstration
    # ==================================================
    print("\n8. Colorblind-friendly palette demo...")
    colors = DEFAULT_COLORS[:7]
    names = ['Blue', 'Vermillion', 'Green', 'Pink', 'Yellow', 'Light Blue', 'Orange']
    palette_data = []
    for i, (name, color) in enumerate(zip(names, colors)):
        for x in range(20):
            palette_data.append({
                'x': x,
                'y': np.sin(x / 3 + i * 0.5) + i * 0.4,
                'color_name': name
            })
    palette_df = pd.DataFrame(palette_data)

    chart = alt.Chart(palette_df).mark_line(strokeWidth=3).encode(
        x=alt.X('x:Q', title=''),
        y=alt.Y('y:Q', title=''),
        color=alt.Color('color_name:N', title='Color',
                        scale=alt.Scale(domain=names, range=colors))
    )
    chart = configure_chart(chart, 'Colorblind-Friendly Palette (Wong 2011)', height=400)
    save_chart(chart, CHARTS_DIR / "color-palette.png")
    print("  Saved color-palette.png")

    # ==================================================
    # 9. Raw Data Scatter (Noisy)
    # ==================================================
    print("\n9. Raw data scatter plot...")
    np.random.seed(42)
    t = np.linspace(0, 200, 500)
    signal = 0.5 + 0.3 * np.sin(2 * np.pi * t / 80) + 0.2 * np.sin(2 * np.pi * t / 30)
    noise = 0.15 * np.random.randn(500)
    raw = signal + noise
    pattern_df = pd.DataFrame({'t': t, 'raw': raw, 'pattern': signal})

    chart = alt.Chart(pattern_df).mark_point(size=10, opacity=0.4, color='gray').encode(
        x=alt.X('t:Q', title='Time (years)', axis=alt.Axis(format='d')),
        y=alt.Y('raw:Q', title='Observed Value')
    )
    chart = configure_chart(chart, 'Raw Data: Noise Obscures the Signal', height=350)
    save_chart(chart, CHARTS_DIR / "raw-data-scatter.png")
    print("  Saved raw-data-scatter.png")

    # ==================================================
    # 10. Pattern Revealed (Smoothed)
    # ==================================================
    print("\n10. Pattern revealed (smoothed)...")
    chart = alt.Chart(pattern_df).mark_line(
        color=DEFAULT_COLORS[0], strokeWidth=3
    ).encode(
        x=alt.X('t:Q', title='Time (years)', axis=alt.Axis(format='d')),
        y=alt.Y('pattern:Q', title='Underlying Pattern')
    )
    chart = configure_chart(chart, 'Pattern Revealed: Two Overlapping Cycles', height=350)
    save_chart(chart, CHARTS_DIR / "pattern-revealed.png")
    print("  Saved pattern-revealed.png")

    # ==================================================
    # 11. Uncertainty Bands
    # ==================================================
    print("\n11. Uncertainty bands (ensemble forecast)...")
    np.random.seed(42)
    n_runs = 50
    psi_ensemble = []
    for i in range(n_runs):
        noise_walk = np.cumsum(np.random.normal(0, 0.01, len(df)))
        psi_run = df['psi'].values + noise_walk
        psi_ensemble.append(psi_run)
    psi_ensemble = np.array(psi_ensemble)

    ensemble_df = pd.DataFrame({
        't': df['t'].values,
        'mean': np.mean(psi_ensemble, axis=0),
        'p10': np.percentile(psi_ensemble, 10, axis=0),
        'p90': np.percentile(psi_ensemble, 90, axis=0),
        'p25': np.percentile(psi_ensemble, 25, axis=0),
        'p75': np.percentile(psi_ensemble, 75, axis=0),
    })

    band_outer = alt.Chart(ensemble_df).mark_area(
        opacity=0.2, color=DEFAULT_COLORS[0]
    ).encode(
        x=alt.X('t:Q', title='Time (years)', axis=alt.Axis(format='d')),
        y='p10:Q', y2='p90:Q'
    )
    band_inner = alt.Chart(ensemble_df).mark_area(
        opacity=0.4, color=DEFAULT_COLORS[0]
    ).encode(x='t:Q', y='p25:Q', y2='p75:Q')
    line = alt.Chart(ensemble_df).mark_line(
        color=DEFAULT_COLORS[0], strokeWidth=2
    ).encode(
        x='t:Q',
        y=alt.Y('mean:Q', title='Political Stress Index')
    )
    combined = band_outer + band_inner + line
    combined = configure_chart(combined, 'Forecast with Uncertainty Bands', height=400)
    save_chart(combined, CHARTS_DIR / "uncertainty-bands.png")
    print("  Saved uncertainty-bands.png")

    # ==================================================
    # 12. Model vs Observed Comparison
    # ==================================================
    print("\n12. Model vs observed comparison...")
    np.random.seed(42)
    observed_t = df['t'].values[::20]  # Sparse observations
    observed_psi = df['psi'].values[::20] + np.random.normal(0, 0.08, len(observed_t))

    obs_df = pd.DataFrame({'t': observed_t, 'psi': observed_psi})

    model_line = alt.Chart(df).mark_line(
        color=DEFAULT_COLORS[0], strokeWidth=2
    ).encode(
        x=alt.X('t:Q', title='Time (years)', axis=alt.Axis(format='d')),
        y=alt.Y('psi:Q', title='Political Stress Index')
    )
    obs_points = alt.Chart(obs_df).mark_point(
        color=DEFAULT_COLORS[1], size=60, filled=True
    ).encode(x='t:Q', y='psi:Q')

    # Legend hack
    legend_df = pd.DataFrame({
        'source': ['Model', 'Observed'],
        'x': [0, 0], 'y': [0, 0]
    })
    legend = alt.Chart(legend_df).mark_point(size=0).encode(
        color=alt.Color('source:N', scale=alt.Scale(
            domain=['Model', 'Observed'],
            range=[DEFAULT_COLORS[0], DEFAULT_COLORS[1]]
        ), title='Source')
    )

    combined = (model_line + obs_points + legend)
    combined = configure_chart(combined, 'Model Predictions vs Historical Data', height=400)
    save_chart(combined, CHARTS_DIR / "model-vs-observed.png")
    print("  Saved model-vs-observed.png")

    # ==================================================
    # 13. Small Multiples - Parameter Sensitivity
    # ==================================================
    print("\n13. Small multiples (parameter sensitivity)...")
    scenarios = []
    for name, r_max in [('Low Growth (r=0.01)', 0.01),
                         ('Medium Growth (r=0.025)', 0.025),
                         ('High Growth (r=0.04)', 0.04)]:
        params = SDTParams(r_max=r_max)
        model = SDTModel(params)
        simulator = Simulator(model)
        initial = {'N': 0.5, 'E': 0.02, 'W': 1.0, 'S': 1.0, 'psi': 0.1}
        res = simulator.run(initial, time_span=(0, 300), dt=1.0)
        res_df = res.df.copy()
        res_df['scenario'] = name
        scenarios.append(res_df)
    combined_df = pd.concat(scenarios, ignore_index=True)

    chart = alt.Chart(combined_df).mark_line(
        color=DEFAULT_COLORS[1], strokeWidth=2
    ).encode(
        x=alt.X('t:Q', title='Time (years)', axis=alt.Axis(format='d')),
        y=alt.Y('psi:Q', title='Stress Index')
    ).properties(width=700, height=150).facet(
        row=alt.Row('scenario:N', title=None,
                    header=alt.Header(labelFontSize=14, labelAngle=0, labelAlign='left'))
    ).configure_axis(
        labelFontSize=FONT_SIZE_AXIS_LABEL,
        titleFontSize=FONT_SIZE_AXIS_TITLE
    ).properties(
        title=alt.TitleParams('Parameter Sensitivity: Growth Rate Effects',
                              fontSize=FONT_SIZE_TITLE)
    )
    save_chart(chart, CHARTS_DIR / "small-multiples-sensitivity.png")
    print("  Saved small-multiples-sensitivity.png")

    # ==================================================
    # 14. Cycle Detection with Annotations
    # ==================================================
    print("\n14. Cycle detection with phase annotations...")
    from cliodynamics.viz.cycles import detect_secular_cycles, CyclePhase, PHASE_COLORS

    detected = detect_secular_cycles(df['psi'].values, df['t'].values)

    # Create phase rectangles
    phase_data = []
    y_min = df['psi'].min() - 0.1 * (df['psi'].max() - df['psi'].min())
    y_max = df['psi'].max() + 0.1 * (df['psi'].max() - df['psi'].min())

    for cycle in detected.cycles:
        for start, end, phase in cycle.phases:
            phase_name = phase.value.title()
            phase_data.append({
                'start': start, 'end': end,
                'phase': phase_name,
                'y_min': y_min, 'y_max': y_max
            })

    phase_df = pd.DataFrame(phase_data)
    phase_color_map = {
        'Expansion': PHASE_COLORS[CyclePhase.EXPANSION],
        'Stagflation': PHASE_COLORS[CyclePhase.STAGFLATION],
        'Crisis': PHASE_COLORS[CyclePhase.CRISIS],
        'Depression': PHASE_COLORS[CyclePhase.DEPRESSION],
    }

    phase_rects = alt.Chart(phase_df).mark_rect(opacity=0.3).encode(
        x=alt.X('start:Q'),
        x2='end:Q',
        y=alt.Y('y_min:Q'),
        y2='y_max:Q',
        color=alt.Color('phase:N', title='Phase', scale=alt.Scale(
            domain=list(phase_color_map.keys()),
            range=list(phase_color_map.values())
        ))
    )

    main_line = alt.Chart(df).mark_line(
        color=DEFAULT_COLORS[0], strokeWidth=2
    ).encode(
        x=alt.X('t:Q', title='Time (years)', axis=alt.Axis(format='d')),
        y=alt.Y('psi:Q', title='Political Stress Index')
    )

    # Peak markers
    peak_df = pd.DataFrame([{'t': p.time, 'psi': p.value} for p in detected.peaks])
    peaks = alt.Chart(peak_df).mark_point(
        color='red', size=80, shape='triangle-down', filled=True
    ).encode(x='t:Q', y='psi:Q')

    combined = phase_rects + main_line + peaks
    combined = configure_chart(combined, 'Secular Cycles with Phase Annotations', height=450)
    save_chart(combined, CHARTS_DIR / "cycle-detection.png")
    print("  Saved cycle-detection.png")

    # ==================================================
    # 15. Dimension Progression - 1D view
    # ==================================================
    print("\n15. Dimension progression - 1D...")
    chart = alt.Chart(df).mark_line(
        color=DEFAULT_COLORS[1], strokeWidth=2
    ).encode(
        x=alt.X('t:Q', title='Time (years)', axis=alt.Axis(format='d')),
        y=alt.Y('psi:Q', title='Political Stress Index')
    )
    chart = configure_chart(chart, '1D View: Instability Over Time', height=300)
    save_chart(chart, CHARTS_DIR / "dimension-1d.png")
    print("  Saved dimension-1d.png")

    # ==================================================
    # Summary
    # ==================================================
    print("\n" + "=" * 60)
    print(f"Done! All charts saved to: {CHARTS_DIR}")
    print("=" * 60)
    print("\nGenerated 15 charts for Essay 019.")
    print("\nIMPORTANT: Visually verify all charts before committing!")


if __name__ == "__main__":
    main()
