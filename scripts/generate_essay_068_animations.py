"""Generate Plotly animations for Essay 068: The Claude Code Experiment.

This script creates interactive HTML animations showing the project workflow
and development progression.
"""

import sys
sys.path.insert(0, '/Users/bedwards/turchin/src')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

OUTPUT_DIR = Path('/Users/bedwards/turchin/docs/assets/animations/essay-068')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Generating Plotly animations for Essay 068...")
print("="*60)

# Animation 1: Project Evolution Timeline
print("\n1. Project Evolution Animation...")

# Timeline data representing project growth
timeline_events = [
    {'hour': 0, 'code_lines': 0, 'tests': 0, 'essays': 0, 'prs': 0, 'phase': 'Setup'},
    {'hour': 1, 'code_lines': 500, 'tests': 200, 'essays': 1, 'prs': 2, 'phase': 'Foundation'},
    {'hour': 2, 'code_lines': 2000, 'tests': 800, 'essays': 2, 'prs': 6, 'phase': 'Core'},
    {'hour': 4, 'code_lines': 4000, 'tests': 1800, 'essays': 2, 'prs': 9, 'phase': 'Calibration'},
    {'hour': 8, 'code_lines': 6000, 'tests': 2800, 'essays': 3, 'prs': 12, 'phase': 'Case Studies'},
    {'hour': 12, 'code_lines': 8000, 'tests': 4000, 'essays': 4, 'prs': 16, 'phase': 'Visualization'},
    {'hour': 16, 'code_lines': 10000, 'tests': 5500, 'essays': 5, 'prs': 20, 'phase': 'Essays'},
    {'hour': 20, 'code_lines': 12000, 'tests': 7000, 'essays': 6, 'prs': 24, 'phase': 'Polish'},
    {'hour': 24, 'code_lines': 14000, 'tests': 8500, 'essays': 7, 'prs': 26, 'phase': 'Explorer'},
    {'hour': 32, 'code_lines': 15939, 'tests': 9502, 'essays': 8, 'prs': 27, 'phase': 'Final'},
]

df = pd.DataFrame(timeline_events)

# Create figure with animation
fig = go.Figure()

# Add traces for each metric
colors = {'code_lines': '#1f77b4', 'tests': '#ff7f0e', 'essays': '#2ca02c'}
names = {'code_lines': 'Source Code (lines)', 'tests': 'Test Code (lines)', 'essays': 'Essays Published'}

# Initial traces (empty)
for metric, color in colors.items():
    if metric == 'essays':
        fig.add_trace(go.Scatter(
            x=[], y=[], mode='lines+markers',
            name=names[metric], line=dict(color=color, width=3),
            marker=dict(size=10)
        ))
    else:
        fig.add_trace(go.Scatter(
            x=[], y=[], mode='lines',
            name=names[metric], line=dict(color=color, width=2),
            fill='tozeroy', fillcolor=color.replace(')', ', 0.3)').replace('rgb', 'rgba')
        ))

# Create frames
frames = []
for i in range(1, len(df) + 1):
    frame_data = []
    subset = df.iloc[:i]
    
    for j, metric in enumerate(colors.keys()):
        if metric == 'essays':
            frame_data.append(go.Scatter(
                x=subset['hour'], y=subset[metric] * 1000,  # Scale essays for visibility
                mode='lines+markers', name=names[metric]
            ))
        else:
            frame_data.append(go.Scatter(
                x=subset['hour'], y=subset[metric],
                mode='lines', name=names[metric],
                fill='tozeroy'
            ))
    
    frames.append(go.Frame(
        data=frame_data,
        name=str(i),
        layout=go.Layout(
            title=f"Project Growth - Hour {df.iloc[i-1]['hour']}: {df.iloc[i-1]['phase']}"
        )
    ))

fig.frames = frames

# Animation settings
fig.update_layout(
    title="Cliodynamics Project Evolution Over 32 Hours",
    xaxis=dict(title="Hours Since Project Start", range=[0, 35]),
    yaxis=dict(title="Lines of Code / Essays x1000", range=[0, 18000]),
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'y': 1.15,
        'x': 0.5,
        'xanchor': 'center',
        'buttons': [
            {'label': 'Play', 'method': 'animate',
             'args': [None, {'frame': {'duration': 500, 'redraw': True},
                           'fromcurrent': True, 'transition': {'duration': 300}}]},
            {'label': 'Pause', 'method': 'animate',
             'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                             'mode': 'immediate', 'transition': {'duration': 0}}]}
        ]
    }],
    sliders=[{
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 16},
            'prefix': 'Time Point: ',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': [
            {'args': [[str(i)], {'frame': {'duration': 300, 'redraw': True},
                                'mode': 'immediate', 'transition': {'duration': 300}}],
             'label': f"Hour {df.iloc[i-1]['hour']}",
             'method': 'animate'}
            for i in range(1, len(df) + 1)
        ]
    }],
    height=600,
    width=1000,
    legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
)

fig.write_html(OUTPUT_DIR / 'project-evolution.html')
print(f"  Saved: {OUTPUT_DIR / 'project-evolution.html'}")


# Animation 2: Worker Spawning Visualization
print("\n2. Worker Spawning Animation...")

# Worker wave data
worker_waves = [
    {'wave': 1, 'hour': 0, 'workers': ['A', 'B', 'C', 'D'], 
     'issues': ['#28-29 Images', '#15 Math Essay', '#16 Calibration', '#26 API']},
    {'wave': 2, 'hour': 9, 'workers': ['E', 'F', 'G', 'H', 'I'],
     'issues': ['#27 Polaris', '#8 US Data', '#9 Viz', '#7 Rome', '#10 Forecast']},
    {'wave': 3, 'hour': 16, 'workers': ['J', 'K', 'L', 'M'],
     'issues': ['#17 Rome Essay', '#18 America Essay', '#41 Policy', '#44 Animation']},
    {'wave': 4, 'hour': 20, 'workers': ['N', 'O', 'P'],
     'issues': ['#42 Policy Essay', '#20 Forecast Essay', '#47 Chart Fix']},
    {'wave': 5, 'hour': 22, 'workers': ['Q', 'R', 'S'],
     'issues': ['#49 Dimension', '#53 Explorer', '#55 Altair']},
    {'wave': 6, 'hour': 30, 'workers': ['T', 'U'],
     'issues': ['#56 Plotly', '#63 Essay']},
]

# Create animated figure showing worker spawning
fig2 = go.Figure()

# Calculate worker positions
max_workers = 5
y_positions = np.linspace(0.2, 0.8, max_workers)

# Create frames for each wave
frames2 = []
cumulative_workers = []

for wave_idx, wave in enumerate(worker_waves):
    # Add workers from this wave
    for w_idx, worker in enumerate(wave['workers']):
        y_pos = y_positions[w_idx % max_workers]
        cumulative_workers.append({
            'wave': wave['wave'],
            'hour': wave['hour'],
            'worker': worker,
            'issue': wave['issues'][w_idx],
            'y': y_pos,
            'x': wave['hour']
        })
    
    # Create frame
    df_workers = pd.DataFrame(cumulative_workers)
    
    frame_data = [
        go.Scatter(
            x=df_workers['x'],
            y=df_workers['y'],
            mode='markers+text',
            marker=dict(
                size=40,
                color=df_workers['wave'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Wave')
            ),
            text=df_workers['worker'],
            textposition='middle center',
            textfont=dict(size=14, color='white'),
            hovertext=df_workers['issue'],
            hoverinfo='text'
        )
    ]
    
    frames2.append(go.Frame(
        data=frame_data,
        name=str(wave_idx),
        layout=go.Layout(title=f"Worker Wave {wave['wave']} - Hour {wave['hour']}")
    ))

fig2.frames = frames2

# Initial empty state
fig2.add_trace(go.Scatter(
    x=[], y=[],
    mode='markers+text',
    marker=dict(size=40, colorscale='Viridis', showscale=True),
    textposition='middle center'
))

fig2.update_layout(
    title="Worker Spawning Pattern: 6 Waves, 21 Workers",
    xaxis=dict(title="Hour", range=[-2, 35]),
    yaxis=dict(title="", range=[0, 1], showticklabels=False),
    height=500,
    width=1000,
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'y': 1.15,
        'x': 0.5,
        'xanchor': 'center',
        'buttons': [
            {'label': 'Play Waves', 'method': 'animate',
             'args': [None, {'frame': {'duration': 1500, 'redraw': True},
                           'fromcurrent': True, 'transition': {'duration': 500}}]},
            {'label': 'Pause', 'method': 'animate',
             'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                             'mode': 'immediate'}]}
        ]
    }],
    sliders=[{
        'active': 0,
        'currentvalue': {'prefix': 'Wave: ', 'visible': True},
        'steps': [
            {'args': [[str(i)], {'frame': {'duration': 500, 'redraw': True}}],
             'label': f"Wave {i+1}",
             'method': 'animate'}
            for i in range(len(worker_waves))
        ]
    }]
)

fig2.write_html(OUTPUT_DIR / 'worker-spawning.html')
print(f"  Saved: {OUTPUT_DIR / 'worker-spawning.html'}")


# Animation 3: Context Utilization Over Session
print("\n3. Context Utilization Animation...")

# Simulated context utilization data
np.random.seed(42)
n_points = 50
messages = np.arange(n_points)
context_used = np.zeros(n_points)
context_used[0] = 5

for i in range(1, n_points):
    # Context grows with each message, occasionally jumps or resets
    if i % 15 == 0:  # New session
        context_used[i] = 5
    else:
        growth = np.random.uniform(1, 5)
        context_used[i] = min(context_used[i-1] + growth, 100)

fig3 = go.Figure()

# Add gauge
fig3.add_trace(go.Indicator(
    mode="gauge+number",
    value=context_used[0],
    title={'text': "Context Window Utilization (%)"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 50], 'color': "lightgreen"},
            {'range': [50, 80], 'color': "yellow"},
            {'range': [80, 100], 'color': "red"}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 90
        }
    }
))

# Create frames
frames3 = []
for i in range(n_points):
    frames3.append(go.Frame(
        data=[go.Indicator(
            mode="gauge+number+delta",
            value=context_used[i],
            delta={'reference': context_used[max(0, i-1)]},
            title={'text': f"Message {i+1}: Context Utilization"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ]
            }
        )],
        name=str(i)
    ))

fig3.frames = frames3

fig3.update_layout(
    height=500,
    width=700,
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'y': 1.1,
        'x': 0.5,
        'xanchor': 'center',
        'buttons': [
            {'label': 'Simulate Session', 'method': 'animate',
             'args': [None, {'frame': {'duration': 200, 'redraw': True},
                           'fromcurrent': True, 'transition': {'duration': 100}}]},
            {'label': 'Pause', 'method': 'animate',
             'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
        ]
    }]
)

fig3.write_html(OUTPUT_DIR / 'context-utilization.html')
print(f"  Saved: {OUTPUT_DIR / 'context-utilization.html'}")

print("\n" + "="*60)
print("Animation generation complete!")
print(f"Output directory: {OUTPUT_DIR}")
print("="*60)
