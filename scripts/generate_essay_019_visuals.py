#!/usr/bin/env python3
"""Generate visualizations for Essay 019: Seeing the Invisible."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import altair as alt

CHARTS_DIR = Path(__file__).parent.parent / 'docs' / 'assets' / 'charts' / 'essay-019'
ANIMATIONS_DIR = Path(__file__).parent.parent / 'docs' / 'assets' / 'animations' / 'essay-019'

CHARTS_DIR.mkdir(parents=True, exist_ok=True)
ANIMATIONS_DIR.mkdir(parents=True, exist_ok=True)


def generate_sdt_like_data(t_max=500, dt=1.0):
    """Generate synthetic SDT-like data with realistic dynamics."""
    t = np.arange(0, t_max + dt, dt)
    n = len(t)
    
    # Create realistic-looking oscillatory dynamics
    primary = 0.3 * np.sin(2 * np.pi * t / 120)
    secondary = 0.15 * np.sin(2 * np.pi * t / 50)
    trend = 0.0001 * t
    noise = 0.02 * np.random.randn(n)
    
    N = 0.5 + 0.3 * (1 - np.exp(-t/100)) + 0.1 * np.sin(2 * np.pi * t / 150) + 0.02 * np.random.randn(n)
    N = np.clip(N, 0.1, 1.0)
    
    W = 1.0 - 0.3 * (N - 0.5) + 0.15 * np.sin(2 * np.pi * t / 80 + 1) + 0.02 * np.random.randn(n)
    W = np.clip(W, 0.3, 1.5)
    
    E = 0.02 + 0.08 * (1 - np.exp(-t/200)) + 0.02 * np.sin(2 * np.pi * t / 100) + 0.005 * np.random.randn(n)
    E = np.clip(E, 0.01, 0.15)
    
    S = 1.0 - 0.2 * (E / 0.1) + 0.1 * np.sin(2 * np.pi * t / 90 + 2) + 0.03 * np.random.randn(n)
    S = np.clip(S, 0.2, 1.2)
    
    psi = 0.1 + primary + secondary + trend + noise
    for peak_t in [100, 220, 340, 460]:
        if peak_t < t_max:
            psi += 0.2 * np.exp(-((t - peak_t) / 15)**2)
    psi = np.clip(psi, 0, 1.2)
    
    return pd.DataFrame({'t': t, 'N': N, 'E': E, 'W': W, 'S': S, 'psi': psi})


def main():
    print("Generating visualizations for Essay 019...")
    np.random.seed(42)
    
    df = generate_sdt_like_data()
    print(f"Generated dataset with {len(df)} time points")
    
    # All static charts
    charts_to_generate = [
        ('timeseries-multi.png', lambda: alt.Chart(
            df[['t', 'N', 'W', 'psi']].melt(id_vars=['t'], var_name='variable', value_name='value')
        ).mark_line(strokeWidth=2).encode(
            x=alt.X('t:Q', title='Time (years)'),
            y=alt.Y('value:Q', title='Value'),
            color=alt.Color('variable:N', scale=alt.Scale(
                domain=['N', 'W', 'psi'], range=['#0072B2', '#D55E00', '#009E73']), title='Variable')
        ).properties(width=800, height=400, title='SDT Model Core Variables')),
        
        ('timeseries-full.png', lambda: alt.Chart(
            df[['t', 'N', 'E', 'W', 'S', 'psi']].melt(id_vars=['t'], var_name='variable', value_name='value')
        ).mark_line(color='#0072B2', strokeWidth=2).encode(
            x=alt.X('t:Q', title='Time'),
            y=alt.Y('value:Q', title='Value'),
            facet=alt.Facet('variable:N', columns=1, title=None)
        ).properties(width=800, height=100)),
        
        ('phase-2d-w-psi.png', lambda: alt.Chart(df).mark_line(strokeWidth=2, opacity=0.8).encode(
            x=alt.X('W:Q', title='Wages (W)'),
            y=alt.Y('psi:Q', title='Instability (psi)'),
            color=alt.Color('t:Q', scale=alt.Scale(scheme='viridis'), title='Time')
        ).properties(width=600, height=500, title='Phase Space: Wages vs. Instability')),
        
        ('phase-2d-n-psi.png', lambda: alt.Chart(df).mark_line(strokeWidth=2, opacity=0.8).encode(
            x=alt.X('N:Q', title='Population (N)'),
            y=alt.Y('psi:Q', title='Instability (psi)'),
            color=alt.Color('t:Q', scale=alt.Scale(scheme='viridis'), title='Time')
        ).properties(width=600, height=500, title='Phase Space: Population vs. Instability')),
        
        ('dimension-1d.png', lambda: alt.Chart(df).mark_line(color='#D55E00', strokeWidth=2).encode(
            x=alt.X('t:Q', title='Time'),
            y=alt.Y('psi:Q', title='Instability (psi)')
        ).properties(width=800, height=300, title='1D View: Instability Over Time')),
        
        ('dimension-2d.png', lambda: alt.Chart(df).mark_line(strokeWidth=2, opacity=0.8).encode(
            x=alt.X('W:Q', title='Wages (W)'),
            y=alt.Y('psi:Q', title='Instability (psi)'),
            color=alt.Color('t:Q', scale=alt.Scale(scheme='viridis'), title='Time')
        ).properties(width=600, height=500, title='2D View: Phase Space Trajectory')),
    ]
    
    for name, chart_fn in charts_to_generate:
        print(f"Creating {name}...")
        chart_fn().save(str(CHARTS_DIR / name), scale_factor=2)
    
    # Design comparison
    print("Creating design-before.png...")
    t = np.linspace(0, 100, 200)
    psi = 0.2 + 0.3 * np.sin(2 * np.pi * t / 50) + 0.1 * np.random.randn(200)
    design_df = pd.DataFrame({'t': t, 'psi': psi})
    alt.Chart(design_df).mark_line().encode(x='t:Q', y='psi:Q').properties(width=400, height=200).save(
        str(CHARTS_DIR / 'design-before.png'), scale_factor=1)
    
    print("Creating design-after.png...")
    alt.Chart(design_df).mark_line(color='#0072B2', strokeWidth=2).encode(
        x=alt.X('t:Q', title='Time (years)'),
        y=alt.Y('psi:Q', title='Political Stress Index')
    ).properties(width=800, height=400, title='Instability Over Time').save(
        str(CHARTS_DIR / 'design-after.png'), scale_factor=2)
    
    # Color palette
    print("Creating color-palette.png...")
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00']
    names = ['Blue', 'Vermillion', 'Green', 'Pink', 'Yellow', 'Light Blue', 'Orange']
    data = []
    for i, name in enumerate(names):
        for x in range(10):
            data.append({'x': x, 'y': np.sin(x / 2 + i) + i * 0.3, 'color': name})
    alt.Chart(pd.DataFrame(data)).mark_line(strokeWidth=3).encode(
        x=alt.X('x:Q', title=''),
        y=alt.Y('y:Q', title=''),
        color=alt.Color('color:N', scale=alt.Scale(domain=names, range=colors), title='Color Palette')
    ).properties(width=800, height=400, title='Colorblind-Friendly Palette (Wong 2011)').save(
        str(CHARTS_DIR / 'color-palette.png'), scale_factor=2)
    
    # Raw vs pattern
    print("Creating raw-data.png and pattern-revealed.png...")
    t = np.linspace(0, 200, 500)
    signal = 0.5 + 0.3 * np.sin(2 * np.pi * t / 80) + 0.2 * np.sin(2 * np.pi * t / 30)
    noise = 0.15 * np.random.randn(500)
    pattern_df = pd.DataFrame({'t': t, 'raw': signal + noise, 'pattern': signal})
    
    alt.Chart(pattern_df).mark_point(size=5, opacity=0.5, color='gray').encode(
        x=alt.X('t:Q', title='Time'), y=alt.Y('raw:Q', title='Observed Value')
    ).properties(width=800, height=300, title='Raw Data: Chaos or Pattern?').save(
        str(CHARTS_DIR / 'raw-data.png'), scale_factor=2)
    
    alt.Chart(pattern_df).mark_line(color='#0072B2', strokeWidth=3).encode(
        x=alt.X('t:Q', title='Time'), y=alt.Y('pattern:Q', title='Underlying Pattern')
    ).properties(width=800, height=300, title='Pattern Revealed: Two Overlapping Cycles').save(
        str(CHARTS_DIR / 'pattern-revealed.png'), scale_factor=2)
    
    # Uncertainty bands
    print("Creating uncertainty.png...")
    psi_ensemble = np.array([df['psi'].values + np.cumsum(np.random.normal(0, 0.01, len(df))) for _ in range(50)])
    ensemble_df = pd.DataFrame({
        't': df['t'].values,
        'mean': np.mean(psi_ensemble, axis=0),
        'p10': np.percentile(psi_ensemble, 10, axis=0),
        'p90': np.percentile(psi_ensemble, 90, axis=0),
        'p25': np.percentile(psi_ensemble, 25, axis=0),
        'p75': np.percentile(psi_ensemble, 75, axis=0),
    })
    base = alt.Chart(ensemble_df)
    (base.mark_area(opacity=0.2, color='#0072B2').encode(x='t:Q', y='p10:Q', y2='p90:Q') +
     base.mark_area(opacity=0.4, color='#0072B2').encode(x='t:Q', y='p25:Q', y2='p75:Q') +
     base.mark_line(color='#0072B2', strokeWidth=2).encode(
         x=alt.X('t:Q', title='Time (years)'), y=alt.Y('mean:Q', title='Political Stress Index'))
    ).properties(width=800, height=400, title='Instability Forecast with Uncertainty').save(
        str(CHARTS_DIR / 'uncertainty.png'), scale_factor=2)
    
    # Comparison
    print("Creating comparison.png...")
    comparison_df = pd.DataFrame({
        't': df['t'].values, 'Model': df['psi'].values,
        'Observed': np.clip(df['psi'].values + np.random.normal(0, 0.05, len(df)), 0, None)
    })
    alt.Chart(comparison_df.melt(id_vars=['t'], var_name='source', value_name='psi')).mark_line(strokeWidth=2).encode(
        x=alt.X('t:Q', title='Time (years)'), y=alt.Y('psi:Q', title='Political Stress Index'),
        color=alt.Color('source:N', title='Source', scale=alt.Scale(domain=['Model', 'Observed'], range=['#0072B2', '#D55E00'])),
        strokeDash=alt.StrokeDash('source:N', scale=alt.Scale(domain=['Model', 'Observed'], range=[[0], [5, 3]]))
    ).properties(width=800, height=400, title='Model vs. Observed Data').save(
        str(CHARTS_DIR / 'comparison.png'), scale_factor=2)
    
    # Small multiples
    print("Creating small-multiples.png...")
    scenarios = []
    for name, amplitude in [('Stable', 0.15), ('Growing', 0.30), ('Volatile', 0.50)]:
        t = np.arange(0, 301, 1.0)
        scenarios.append(pd.DataFrame({
            't': t, 'psi': 0.2 + amplitude * np.sin(2 * np.pi * t / 80) + 0.03 * np.random.randn(len(t)),
            'scenario': name
        }))
    alt.Chart(pd.concat(scenarios, ignore_index=True)).mark_line(color='#D55E00', strokeWidth=2).encode(
        x=alt.X('t:Q', title='Time (years)'), y=alt.Y('psi:Q', title='Political Stress Index'),
        facet=alt.Facet('scenario:N', columns=1, title=None)
    ).properties(width=800, height=150).save(str(CHARTS_DIR / 'small-multiples.png'), scale_factor=2)
    
    # Annotated crises
    print("Creating annotated-crises.png...")
    psi_vals = df['psi'].values
    t_vals = df['t'].values
    peaks = []
    for i in range(5, len(psi_vals) - 5):
        if psi_vals[i] > psi_vals[i-1] and psi_vals[i] > psi_vals[i+1] and psi_vals[i] > 0.5:
            if not peaks or t_vals[i] - peaks[-1]['t'] > 50:
                peaks.append({'t': t_vals[i], 'psi': psi_vals[i], 'label': f'Crisis {len(peaks)+1}'})
    peaks_df = pd.DataFrame(peaks[:4])
    
    base = alt.Chart(df).mark_line(color='#D55E00', strokeWidth=2).encode(
        x=alt.X('t:Q', title='Time (years)'), y=alt.Y('psi:Q', title='Political Stress Index (psi)')
    ).properties(width=800, height=400, title='Annotated Crisis Events')
    points = alt.Chart(peaks_df).mark_circle(size=100, color='#0072B2').encode(x='t:Q', y='psi:Q')
    labels = alt.Chart(peaks_df).mark_text(align='center', baseline='bottom', dy=-10, fontSize=12).encode(
        x='t:Q', y='psi:Q', text='label:N')
    (base + points + labels).save(str(CHARTS_DIR / 'annotated-crises.png'), scale_factor=2)
    
    print(f"\nAll static charts saved to: {CHARTS_DIR}")
    
    # Animations
    print("\nGenerating animations...")
    from cliodynamics.viz import plotly_animations
    
    class MockResult:
        def __init__(self, df): self.df = df
    results = MockResult(df)
    
    print("Creating timeseries-animation.html...")
    plotly_animations.animate_time_series(results, variables=['N', 'W', 'psi'],
        title='SDT Model Evolution Over Time', duration_ms=15000).write_html(
        str(ANIMATIONS_DIR / 'timeseries-animation.html'))
    
    print("Creating phase-2d-animation.html...")
    plotly_animations.animate_phase_space(results, x='W', y='psi',
        title='Phase Space Trajectory').write_html(str(ANIMATIONS_DIR / 'phase-2d-animation.html'))
    
    print("Creating phase-3d-animation.html...")
    plotly_animations.animate_phase_space_3d(results, x='N', y='W', z='psi', trail_length=50,
        camera_orbit=True, title='3D Phase Space with Camera Orbit').write_html(
        str(ANIMATIONS_DIR / 'phase-3d-animation.html'))
    
    print(f"\nAnimations saved to: {ANIMATIONS_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    main()
