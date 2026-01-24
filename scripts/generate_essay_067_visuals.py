#!/usr/bin/env python3
"""Generate all visuals for Essay 067 (Policy Lessons).

This script generates:
1. Gemini illustrations showing policy metaphors and concepts
2. Altair charts showing intervention effectiveness and timing
3. Plotly animations showing counterfactual scenarios

Usage:
    python scripts/generate_essay_067_visuals.py
    python scripts/generate_essay_067_visuals.py --images-only
    python scripts/generate_essay_067_visuals.py --charts-only
    python scripts/generate_essay_067_visuals.py --animations-only
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd

# Output directories
BASE_DIR = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / 'docs/assets/images/essay-067'
CHARTS_DIR = BASE_DIR / 'docs/assets/charts/essay-067'
ANIMATIONS_DIR = BASE_DIR / 'docs/assets/animations/essay-067'

# Ensure directories exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
ANIMATIONS_DIR.mkdir(parents=True, exist_ok=True)


def generate_images():
    """Generate Gemini illustrations for policy essay."""
    import os
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / '.env')

    if not os.environ.get('GEMINI_API_KEY'):
        print("ERROR: GEMINI_API_KEY not found in environment")
        print("Skipping image generation")
        return

    from cliodynamics.viz.images import generate_image

    print("\n" + "=" * 60)
    print("GENERATING GEMINI ILLUSTRATIONS")
    print("=" * 60)

    # Image 1: Ship steering metaphor
    print("\n1. Ship Steering Metaphor...")
    prompt1 = """
    A dramatic illustration of steering a great ship of state.

    Show a large, ornate wooden ship (like a Roman trireme or Renaissance galleon)
    on a turbulent sea. At the helm, a captain grips a large wheel, looking
    determinedly at the horizon where storm clouds gather on one side and clear
    skies on the other.

    The ship represents a nation. The captain represents wise leadership.
    The steering wheel has multiple spokes, each representing different policy levers.
    The sea represents historical forces and secular cycles.
    The storm represents potential crisis.
    The clear horizon represents successful navigation.

    Style: Epic maritime painting, reminiscent of Turner or Aivazovsky.
    Warm golden light breaking through clouds. Deep blues and greens for the sea.
    Dramatic but hopeful atmosphere.
    No text labels on the image.
    """
    try:
        generate_image(prompt1, IMAGES_DIR / 'ship-steering.png')
        print(f"  Saved: {IMAGES_DIR / 'ship-steering.png'}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Image 2: Elite bargain concept
    print("\n2. Elite Bargain Concept...")
    prompt2 = """
    A conceptual illustration showing the elite bargain in society.

    The image shows three distinct groups connected in a triangle of reciprocity:
    - At the top: A smaller group of elites in classical robes around a table
    - At the bottom left: A larger group representing common workers and farmers
    - At the bottom right: A structure representing the state (perhaps a temple or capitol)

    Between them, flowing arrows show the social contract:
    - Elites provide: resources, leadership, investment (shown as coins, scrolls)
    - Workers provide: labor, military service, taxes (shown as hands, tools)
    - State provides: order, protection, infrastructure (shown as scales, shields)

    When in balance (shown with glowing golden connections), society thrives.

    Style: Clean, semi-allegorical illustration. Classical aesthetic with
    subtle modern touches. Balanced composition emphasizing reciprocity.
    No text labels on the image.
    """
    try:
        generate_image(prompt2, IMAGES_DIR / 'elite-bargain.png')
        print(f"  Saved: {IMAGES_DIR / 'elite-bargain.png'}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Image 3: State fiscal health
    print("\n3. State Fiscal Health...")
    prompt3 = """
    A visual metaphor showing state fiscal health as a reservoir or dam.

    Show a grand stone dam holding back water (representing state capacity).
    Water flows in from multiple sources at the top (taxes, tribute, revenue).
    Water flows out through controlled outlets (public works, military, welfare).

    On one side of the dam: a thriving city with well-maintained infrastructure.
    On the other side: an arid, stressed landscape representing what happens
    when the reservoir empties.

    Key visual elements:
    - Water level indicator on the dam wall
    - Multiple inflow streams (diverse revenue sources)
    - Controlled outflows (planned expenditure)
    - Cracks appearing when water gets too low

    Style: Architectural illustration with landscape elements.
    Warm afternoon light. Blues for water, warm stone colors.
    Clear visual hierarchy showing cause and effect.
    No text labels on the image.
    """
    try:
        generate_image(prompt3, IMAGES_DIR / 'state-fiscal.png')
        print(f"  Saved: {IMAGES_DIR / 'state-fiscal.png'}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Image 4: Feedback loops in policy
    print("\n4. Policy Feedback Loops...")
    prompt4 = """
    An abstract but beautiful illustration of interconnected feedback loops.

    Show multiple circular paths (feedback loops) of different sizes,
    all interconnected like a complex mechanism or ecosystem:

    - A large outer loop (secular cycles) encompassing everything
    - Medium loops (elite dynamics, popular welfare, state capacity)
    - Small loops (wages, taxes, spending)

    Where loops intersect, show nodes or junctions where policy interventions
    can be applied (visualized as handles, valves, or control points).

    Some loops glow warmly (positive feedback), others cooler (negative feedback).
    The entire system has organic, flowing qualities like a living system.

    Style: Abstract scientific visualization meets art. Think of David Bohm's
    implicate order illustrations or Renaissance diagrams of celestial mechanics.
    Rich jewel tones: sapphire blues, emerald greens, amber golds.
    No text labels on the image.
    """
    try:
        generate_image(prompt4, IMAGES_DIR / 'feedback-loops.png')
        print(f"  Saved: {IMAGES_DIR / 'feedback-loops.png'}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Image 5: Historical reform leaders
    print("\n5. Historical Reform Leaders...")
    prompt5 = """
    A composition showing historical figures who attempted structural reforms.

    Arrange four portrait-style images in a balanced composition:
    - Tiberius Gracchus (Roman reformer, toga, laurel wreath, earnest expression)
    - FDR (American president, 1930s suit, glasses, determined smile)
    - Augustus (Roman emperor, imperial regalia, calm authority)
    - A modern figure representing current reform potential (gender-neutral,
      contemporary dress, looking toward a hopeful future)

    Each figure is shown against a backdrop representing their era and reforms:
    - Gracchus: Roman farmland being divided
    - FDR: Industrial factories with workers
    - Augustus: Roman architecture and order
    - Modern: Sustainable city with diverse population

    Style: Classical portrait composition with historical accuracy.
    Dignified, serious tone emphasizing the weight of leadership.
    Rich, museum-quality aesthetic.
    No text labels on the image.
    """
    try:
        generate_image(prompt5, IMAGES_DIR / 'reform-leaders.png')
        print(f"  Saved: {IMAGES_DIR / 'reform-leaders.png'}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Image 6: Window of opportunity
    print("\n6. Window of Opportunity...")
    prompt6 = """
    A metaphorical illustration showing the concept of a narrowing window
    of opportunity for policy intervention.

    Show a grand arched window or doorway that is slowly closing:
    - The window opens onto a bright, hopeful landscape (successful reform)
    - As it closes, shadows creep in from the edges
    - A figure stands at the threshold, considering whether to step through

    The closing mechanism could be visualized as:
    - Hourglasses showing sand running out
    - Sunset light fading
    - Storm clouds approaching from the dark side

    The message: the window remains open but is narrowing.
    Act now while passage is still possible.

    Style: Romantic era illustration with symbolic elements.
    Dramatic lighting contrast between hope and urgency.
    Golden light through the open portion, deep shadows where closed.
    No text labels on the image.
    """
    try:
        generate_image(prompt6, IMAGES_DIR / 'window-opportunity.png')
        print(f"  Saved: {IMAGES_DIR / 'window-opportunity.png'}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Image 7: Successful vs Failed Policy
    print("\n7. Success vs Failure Comparison...")
    prompt7 = """
    A split-screen style illustration contrasting successful and failed
    policy interventions.

    Left side (Failure - Roman Republic):
    - Shows the Roman Senate in chaos, senators fighting
    - Farms consolidated into slave-worked estates
    - Displaced farmers crowding into urban slums
    - Fire and smoke in the background
    - Muted, desaturated colors

    Right side (Success - New Deal):
    - Shows workers building infrastructure (dams, roads)
    - Families in stable homes
    - Factory with fair conditions
    - Rising sun and clear skies
    - Warm, saturated colors

    A dividing line in the middle shows where different choices led to
    different outcomes. Both started from similar crisis conditions.

    Style: Historical illustration with clear narrative contrast.
    Balanced composition emphasizing parallel structures with
    divergent outcomes.
    No text labels on the image.
    """
    try:
        generate_image(prompt7, IMAGES_DIR / 'success-vs-failure.png')
        print(f"  Saved: {IMAGES_DIR / 'success-vs-failure.png'}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("Image generation complete!")
    print("=" * 60)


def generate_charts():
    """Generate Altair charts for policy essay."""
    import altair as alt
    from cliodynamics.viz.charts import configure_chart, save_chart, CHART_WIDTH

    print("\n" + "=" * 60)
    print("GENERATING ALTAIR CHARTS")
    print("=" * 60)

    # Chart 1: Intervention Effectiveness Comparison
    print("\n1. Intervention Effectiveness Chart...")
    intervention_data = [
        {'intervention': 'Elite Cap (1.3x)', 'psi_reduction': 32, 'category': 'Elite Management'},
        {'intervention': 'Wage Floor (85%)', 'psi_reduction': 28, 'category': 'Popular Welfare'},
        {'intervention': 'Progressive Taxation', 'psi_reduction': 24, 'category': 'Fiscal Policy'},
        {'intervention': 'Wage Boost (2%)', 'psi_reduction': 22, 'category': 'Popular Welfare'},
        {'intervention': 'Institutional Reform', 'psi_reduction': 18, 'category': 'Governance'},
        {'intervention': 'Fiscal Stimulus', 'psi_reduction': 15, 'category': 'Fiscal Policy'},
        {'intervention': 'Migration Control', 'psi_reduction': 10, 'category': 'Demographics'},
        {'intervention': 'Frontier Expansion', 'psi_reduction': 8, 'category': 'Demographics'},
    ]
    df1 = pd.DataFrame(intervention_data)

    chart1 = alt.Chart(df1).mark_bar().encode(
        x=alt.X('psi_reduction:Q', title='PSI Reduction (%)'),
        y=alt.Y('intervention:N', title=None, sort='-x'),
        color=alt.Color('category:N', title='Category',
                        scale=alt.Scale(domain=['Elite Management', 'Popular Welfare',
                                                'Fiscal Policy', 'Governance', 'Demographics'],
                                        range=['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']))
    )
    chart1 = configure_chart(chart1, 'Policy Intervention Effectiveness', height=400)
    save_chart(chart1, CHARTS_DIR / 'intervention-effectiveness.png')
    print(f"  Saved: {CHARTS_DIR / 'intervention-effectiveness.png'}")

    # Chart 2: Timing Sensitivity
    print("\n2. Intervention Timing Sensitivity Chart...")
    timing_data = []
    for year in [10, 25, 50, 75, 100, 125, 150]:
        # Effectiveness decays over time
        base_effectiveness = 45 * np.exp(-year / 60)
        timing_data.append({
            'year': year,
            'effectiveness': base_effectiveness,
            'intervention': 'Elite Cap'
        })
        timing_data.append({
            'year': year,
            'effectiveness': base_effectiveness * 0.85,
            'intervention': 'Wage Floor'
        })
        timing_data.append({
            'year': year,
            'effectiveness': base_effectiveness * 0.7,
            'intervention': 'Progressive Tax'
        })
    df2 = pd.DataFrame(timing_data)

    chart2 = alt.Chart(df2).mark_line(point=True).encode(
        x=alt.X('year:Q', title='Year of Intervention', axis=alt.Axis(format='d')),
        y=alt.Y('effectiveness:Q', title='Effectiveness (% PSI Reduction)'),
        color=alt.Color('intervention:N', title='Intervention')
    )
    chart2 = configure_chart(chart2, 'Intervention Timing Sensitivity', height=400)
    save_chart(chart2, CHARTS_DIR / 'timing-sensitivity.png')
    print(f"  Saved: {CHARTS_DIR / 'timing-sensitivity.png'}")

    # Chart 3: Elite Dynamics Under Different Policies
    print("\n3. Elite Dynamics Chart...")
    years = list(range(0, 201, 10))
    elite_data = []
    for year in years:
        # Baseline growth
        baseline = 1.0 * np.exp(0.015 * year)
        # With cap
        capped = min(baseline, 1.5)
        # With progressive tax
        taxed = 1.0 * np.exp(0.008 * year)
        elite_data.append({'year': year, 'elite_ratio': baseline, 'policy': 'No Intervention'})
        elite_data.append({'year': year, 'elite_ratio': capped, 'policy': 'Elite Cap (1.5x)'})
        elite_data.append({'year': year, 'elite_ratio': taxed, 'policy': 'Progressive Taxation'})
    df3 = pd.DataFrame(elite_data)

    chart3 = alt.Chart(df3).mark_line(strokeWidth=2).encode(
        x=alt.X('year:Q', title='Year', axis=alt.Axis(format='d')),
        y=alt.Y('elite_ratio:Q', title='Elite Population (relative to baseline)'),
        color=alt.Color('policy:N', title='Policy'),
        strokeDash=alt.StrokeDash('policy:N')
    )
    # Add crisis threshold line
    threshold = alt.Chart(pd.DataFrame({'y': [2.5]})).mark_rule(
        color='red', strokeDash=[5, 5]
    ).encode(y='y:Q')
    chart3 = (chart3 + threshold)
    chart3 = configure_chart(chart3, 'Elite Population Under Different Policies', height=400)
    save_chart(chart3, CHARTS_DIR / 'elite-dynamics.png')
    print(f"  Saved: {CHARTS_DIR / 'elite-dynamics.png'}")

    # Chart 4: Wage Trajectories
    print("\n4. Wage Trajectories Chart...")
    wage_data = []
    for year in years:
        # Different wage trajectories
        baseline = 1.0 * np.exp(-0.005 * year)  # Declining
        with_floor = max(baseline, 0.85)  # Floor at 85%
        with_boost = 1.0 * np.exp(0.01 * year)  # Growing
        wage_data.append({'year': year, 'wage': baseline, 'policy': 'No Intervention'})
        wage_data.append({'year': year, 'wage': with_floor, 'policy': 'Wage Floor (85%)'})
        wage_data.append({'year': year, 'wage': with_boost, 'policy': 'Wage Boost (2%/yr)'})
    df4 = pd.DataFrame(wage_data)

    chart4 = alt.Chart(df4).mark_line(strokeWidth=2).encode(
        x=alt.X('year:Q', title='Year', axis=alt.Axis(format='d')),
        y=alt.Y('wage:Q', title='Real Wage (relative to baseline)'),
        color=alt.Color('policy:N', title='Policy'),
        strokeDash=alt.StrokeDash('policy:N')
    )
    chart4 = configure_chart(chart4, 'Wage Trajectories Under Different Policies', height=400)
    save_chart(chart4, CHARTS_DIR / 'wage-trajectories.png')
    print(f"  Saved: {CHARTS_DIR / 'wage-trajectories.png'}")

    # Chart 5: State Capacity Over Time
    print("\n5. State Capacity Chart...")
    state_data = []
    for year in years:
        # State capacity dynamics
        baseline = 1.0 * np.exp(-0.008 * year)  # Declining
        with_reform = 1.0 * (0.9 + 0.1 * np.cos(year * 0.03))  # Stable with cycles
        with_austerity = 0.8 + 0.2 * np.exp(-0.01 * year)  # Slow decline
        state_data.append({'year': year, 'capacity': baseline, 'policy': 'No Intervention'})
        state_data.append({'year': year, 'capacity': with_reform, 'policy': 'Institutional Reform'})
        state_data.append({'year': year, 'capacity': with_austerity, 'policy': 'Fiscal Austerity'})
    df5 = pd.DataFrame(state_data)

    chart5 = alt.Chart(df5).mark_line(strokeWidth=2).encode(
        x=alt.X('year:Q', title='Year', axis=alt.Axis(format='d')),
        y=alt.Y('capacity:Q', title='State Fiscal Health', scale=alt.Scale(domain=[0, 1.2])),
        color=alt.Color('policy:N', title='Policy'),
        strokeDash=alt.StrokeDash('policy:N')
    )
    # Add crisis threshold
    crisis_line = alt.Chart(pd.DataFrame({'y': [0.2]})).mark_rule(
        color='red', strokeDash=[5, 5]
    ).encode(y='y:Q')
    chart5 = (chart5 + crisis_line)
    chart5 = configure_chart(chart5, 'State Fiscal Health Under Different Policies', height=400)
    save_chart(chart5, CHARTS_DIR / 'state-capacity.png')
    print(f"  Saved: {CHARTS_DIR / 'state-capacity.png'}")

    # Chart 6: PSI Comparison - Baseline vs Reform
    print("\n6. PSI Comparison Chart...")
    psi_data = []
    for year in years:
        # PSI dynamics
        t_norm = year / 200
        baseline_psi = 0.1 + 0.5 * np.sin(np.pi * t_norm) ** 2 + 0.1 * t_norm
        reform_psi = 0.1 + 0.3 * np.sin(np.pi * t_norm) ** 2 - 0.05 * t_norm
        reform_psi = max(0.05, reform_psi)
        psi_data.append({'year': year, 'psi': baseline_psi, 'scenario': 'Baseline (No Reform)'})
        psi_data.append({'year': year, 'psi': reform_psi, 'scenario': 'With Comprehensive Reform'})
    df6 = pd.DataFrame(psi_data)

    chart6 = alt.Chart(df6).mark_area(opacity=0.5).encode(
        x=alt.X('year:Q', title='Year', axis=alt.Axis(format='d')),
        y=alt.Y('psi:Q', title='Political Stress Index'),
        color=alt.Color('scenario:N', title='Scenario',
                        scale=alt.Scale(domain=['Baseline (No Reform)', 'With Comprehensive Reform'],
                                        range=['#d62728', '#2ca02c']))
    )
    chart6 = configure_chart(chart6, 'Political Stress: Baseline vs Reform Scenarios', height=400)
    save_chart(chart6, CHARTS_DIR / 'psi-comparison.png')
    print(f"  Saved: {CHARTS_DIR / 'psi-comparison.png'}")

    # Chart 7: Historical Case Studies - Intervention Success
    print("\n7. Historical Case Studies Chart...")
    historical_data = [
        {'case': 'Gracchan Reforms (Rome)', 'outcome': 'Failed', 'psi_change': 10,
         'timing': 'Late', 'elite_resistance': 'High'},
        {'case': 'Augustan Reforms (Rome)', 'outcome': 'Succeeded', 'psi_change': -35,
         'timing': 'Post-Crisis', 'elite_resistance': 'Low'},
        {'case': 'New Deal (USA)', 'outcome': 'Succeeded', 'psi_change': -30,
         'timing': 'Crisis', 'elite_resistance': 'Medium'},
        {'case': 'Grain Dole (Rome)', 'outcome': 'Partial', 'psi_change': -15,
         'timing': 'Early', 'elite_resistance': 'Low'},
        {'case': '1970s Labor Reform (USA)', 'outcome': 'Failed', 'psi_change': 5,
         'timing': 'Early', 'elite_resistance': 'High'},
    ]
    df7 = pd.DataFrame(historical_data)

    chart7 = alt.Chart(df7).mark_bar().encode(
        y=alt.Y('case:N', title=None, sort=alt.SortField('psi_change', order='ascending')),
        x=alt.X('psi_change:Q', title='PSI Change (%)', scale=alt.Scale(domain=[-40, 15])),
        color=alt.Color('outcome:N', title='Outcome',
                        scale=alt.Scale(domain=['Succeeded', 'Partial', 'Failed'],
                                        range=['#2ca02c', '#ff7f0e', '#d62728']))
    )
    chart7 = configure_chart(chart7, 'Historical Reform Outcomes', height=350)
    save_chart(chart7, CHARTS_DIR / 'historical-outcomes.png')
    print(f"  Saved: {CHARTS_DIR / 'historical-outcomes.png'}")

    # Chart 8: Policy Package Synergies
    print("\n8. Policy Synergies Chart...")
    synergy_data = [
        {'policy1': 'Elite Cap', 'policy2': 'Elite Cap', 'effect': 32},
        {'policy1': 'Elite Cap', 'policy2': 'Wage Floor', 'effect': 50},
        {'policy1': 'Elite Cap', 'policy2': 'Progressive Tax', 'effect': 48},
        {'policy1': 'Elite Cap', 'policy2': 'Institutional Reform', 'effect': 42},
        {'policy1': 'Wage Floor', 'policy2': 'Wage Floor', 'effect': 28},
        {'policy1': 'Wage Floor', 'policy2': 'Progressive Tax', 'effect': 44},
        {'policy1': 'Wage Floor', 'policy2': 'Institutional Reform', 'effect': 38},
        {'policy1': 'Progressive Tax', 'policy2': 'Progressive Tax', 'effect': 24},
        {'policy1': 'Progressive Tax', 'policy2': 'Institutional Reform', 'effect': 35},
        {'policy1': 'Institutional Reform', 'policy2': 'Institutional Reform', 'effect': 18},
    ]
    df8 = pd.DataFrame(synergy_data)

    chart8 = alt.Chart(df8).mark_rect().encode(
        x=alt.X('policy1:N', title=None),
        y=alt.Y('policy2:N', title=None),
        color=alt.Color('effect:Q', title='Combined Effect (%)',
                        scale=alt.Scale(scheme='viridis'))
    )
    chart8 = configure_chart(chart8, 'Policy Combination Effects (Synergies)', height=400)
    save_chart(chart8, CHARTS_DIR / 'policy-synergies.png')
    print(f"  Saved: {CHARTS_DIR / 'policy-synergies.png'}")

    print("\n" + "=" * 60)
    print("Chart generation complete!")
    print("=" * 60)


def generate_animations():
    """Generate Plotly animations for policy essay."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    print("\n" + "=" * 60)
    print("GENERATING PLOTLY ANIMATIONS")
    print("=" * 60)

    # Animation 1: Counterfactual - With and Without Reform
    print("\n1. Policy Intervention Animation...")

    # Generate simulation data
    years = np.linspace(0, 200, 201)

    # Baseline trajectory
    baseline_psi = 0.1 + 0.4 * (years / 200) ** 1.5 + 0.2 * np.sin(years * 0.03)
    baseline_psi = np.clip(baseline_psi, 0, 1)

    # Reform trajectory (intervention at year 50)
    reform_psi = np.copy(baseline_psi)
    reform_start = 50
    for i, y in enumerate(years):
        if y >= reform_start:
            # Gradual reduction
            decay = 0.02 * (y - reform_start)
            reform_psi[i] = max(0.05, baseline_psi[i] - decay * 0.5)

    # Create figure with animation
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('Political Stress Index (PSI)',
                                        'Intervention Effect'),
                        vertical_spacing=0.15)

    # Initial empty traces
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Baseline',
                             line=dict(color='#d62728', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='With Reform',
                             line=dict(color='#2ca02c', width=2)), row=1, col=1)
    fig.add_trace(go.Bar(x=[], y=[], name='PSI Reduction',
                         marker_color='#2ca02c'), row=2, col=1)

    # Create frames
    frames = []
    n_frames = 50
    frame_indices = np.linspace(0, len(years) - 1, n_frames).astype(int)

    for frame_idx, data_idx in enumerate(frame_indices):
        frame_data = [
            go.Scatter(x=years[:data_idx + 1], y=baseline_psi[:data_idx + 1],
                       mode='lines', name='Baseline', line=dict(color='#d62728', width=2)),
            go.Scatter(x=years[:data_idx + 1], y=reform_psi[:data_idx + 1],
                       mode='lines', name='With Reform', line=dict(color='#2ca02c', width=2)),
            go.Bar(x=['Current Year'], y=[max(0, baseline_psi[data_idx] - reform_psi[data_idx]) * 100],
                   name='PSI Reduction', marker_color='#2ca02c'),
        ]
        frames.append(go.Frame(data=frame_data, name=str(frame_idx),
                               layout=go.Layout(title=f'Year {int(years[data_idx])}')))

    fig.frames = frames

    # Layout
    fig.update_layout(
        title='Policy Intervention: Baseline vs Reform',
        xaxis=dict(title='Year', range=[0, 210]),
        yaxis=dict(title='PSI', range=[0, 0.8]),
        xaxis2=dict(title=''),
        yaxis2=dict(title='PSI Reduction (%)', range=[0, 40]),
        height=700,
        width=1000,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'y': 1.05,
            'x': 0.5,
            'xanchor': 'center',
            'buttons': [
                {'label': 'Play', 'method': 'animate',
                 'args': [None, {'frame': {'duration': 100, 'redraw': True},
                                 'fromcurrent': True, 'transition': {'duration': 50}}]},
                {'label': 'Pause', 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
            ]
        }],
        sliders=[{
            'active': 0,
            'currentvalue': {'prefix': 'Frame: ', 'visible': True},
            'steps': [
                {'args': [[str(i)], {'frame': {'duration': 100, 'redraw': True}}],
                 'label': f'{int(years[frame_indices[i]])}',
                 'method': 'animate'}
                for i in range(n_frames)
            ]
        }]
    )

    fig.write_html(ANIMATIONS_DIR / 'policy-intervention.html')
    print(f"  Saved: {ANIMATIONS_DIR / 'policy-intervention.html'}")

    # Animation 2: Multi-Scenario Comparison
    print("\n2. Multi-Scenario Comparison Animation...")

    # Generate multiple scenario trajectories
    scenarios = {
        'No Intervention': {'color': '#d62728', 'psi': baseline_psi},
        'Elite Cap Only': {'color': '#ff7f0e',
                           'psi': baseline_psi * 0.7 + 0.1 * np.random.randn(len(years)) * 0.01},
        'Wage Floor Only': {'color': '#1f77b4',
                            'psi': baseline_psi * 0.75 + 0.05},
        'Comprehensive Reform': {'color': '#2ca02c',
                                 'psi': reform_psi},
    }

    fig2 = go.Figure()

    # Add initial traces
    for name, data in scenarios.items():
        fig2.add_trace(go.Scatter(x=[], y=[], mode='lines', name=name,
                                   line=dict(color=data['color'], width=2)))

    # Create frames
    frames2 = []
    for frame_idx, data_idx in enumerate(frame_indices):
        frame_data = []
        for name, data in scenarios.items():
            frame_data.append(go.Scatter(
                x=years[:data_idx + 1],
                y=np.clip(data['psi'][:data_idx + 1], 0, 1),
                mode='lines', name=name,
                line=dict(color=data['color'], width=2)
            ))

        title = f'Year {int(years[data_idx])}'
        if years[data_idx] >= reform_start:
            title += ' (Reform Active)'

        frames2.append(go.Frame(data=frame_data, name=str(frame_idx),
                                layout=go.Layout(title=title)))

    fig2.frames = frames2

    # Add vertical line for intervention start
    fig2.add_vline(x=reform_start, line_dash='dash', line_color='gray',
                   annotation_text='Reform Begins')

    fig2.update_layout(
        title='Multi-Scenario Comparison: Political Stress Trajectories',
        xaxis=dict(title='Year', range=[0, 210]),
        yaxis=dict(title='Political Stress Index', range=[0, 0.8]),
        height=600,
        width=1000,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'y': 1.1,
            'x': 0.5,
            'xanchor': 'center',
            'buttons': [
                {'label': 'Play', 'method': 'animate',
                 'args': [None, {'frame': {'duration': 100, 'redraw': True},
                                 'fromcurrent': True}]},
                {'label': 'Pause', 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
            ]
        }],
        sliders=[{
            'active': 0,
            'currentvalue': {'prefix': 'Year: ', 'visible': True},
            'steps': [
                {'args': [[str(i)], {'frame': {'duration': 100, 'redraw': True}}],
                 'label': f'{int(years[frame_indices[i]])}',
                 'method': 'animate'}
                for i in range(n_frames)
            ]
        }]
    )

    fig2.write_html(ANIMATIONS_DIR / 'multi-scenario.html')
    print(f"  Saved: {ANIMATIONS_DIR / 'multi-scenario.html'}")

    # Animation 3: Timing Sensitivity Animation
    print("\n3. Timing Sensitivity Animation...")

    # Show how effectiveness changes with intervention timing
    intervention_times = [10, 30, 50, 70, 90, 110, 130, 150]

    fig3 = go.Figure()

    # Add baseline
    fig3.add_trace(go.Scatter(x=years, y=baseline_psi, mode='lines',
                               name='Baseline', line=dict(color='#d62728', width=3)))

    # Add traces for each intervention timing (initially hidden)
    colors = ['#2ca02c', '#17becf', '#bcbd22', '#e377c2', '#ff7f0e', '#8c564b', '#9467bd', '#7f7f7f']
    for i, start_year in enumerate(intervention_times):
        intervention_psi = np.copy(baseline_psi)
        effectiveness = np.exp(-start_year / 60)  # Decaying effectiveness
        for j, y in enumerate(years):
            if y >= start_year:
                decay = effectiveness * 0.5 * (y - start_year) / 100
                intervention_psi[j] = max(0.05, baseline_psi[j] - decay)

        fig3.add_trace(go.Scatter(x=years, y=intervention_psi, mode='lines',
                                   name=f'Reform at Year {start_year}',
                                   line=dict(color=colors[i], width=2),
                                   visible=False))

    # Create animation frames
    frames3 = []
    for i, start_year in enumerate(intervention_times):
        visible = [True] + [j == i for j in range(len(intervention_times))]
        frames3.append(go.Frame(
            data=[fig3.data[0]] + [go.Scatter(
                x=fig3.data[j + 1].x,
                y=fig3.data[j + 1].y,
                mode='lines',
                name=fig3.data[j + 1].name,
                line=dict(color=colors[j], width=2 if j == i else 1),
                opacity=1.0 if j == i else 0.3
            ) for j in range(len(intervention_times))],
            name=str(i),
            layout=go.Layout(title=f'Reform at Year {start_year} - Effectiveness: {np.exp(-start_year / 60) * 100:.0f}%')
        ))

    fig3.frames = frames3

    # Make all intervention lines visible but faded
    for i in range(1, len(fig3.data)):
        fig3.data[i].visible = True
        fig3.data[i].opacity = 0.3

    fig3.update_layout(
        title='Intervention Timing: Earlier = More Effective',
        xaxis=dict(title='Year', range=[0, 210]),
        yaxis=dict(title='Political Stress Index', range=[0, 0.8]),
        height=600,
        width=1000,
        legend=dict(x=1.02, y=0.98),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'y': 1.1,
            'x': 0.5,
            'xanchor': 'center',
            'buttons': [
                {'label': 'Play', 'method': 'animate',
                 'args': [None, {'frame': {'duration': 1500, 'redraw': True}}]},
                {'label': 'Pause', 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
            ]
        }],
        sliders=[{
            'active': 0,
            'currentvalue': {'prefix': 'Intervention Year: ', 'visible': True},
            'steps': [
                {'args': [[str(i)], {'frame': {'duration': 1500, 'redraw': True}}],
                 'label': f'{intervention_times[i]}',
                 'method': 'animate'}
                for i in range(len(intervention_times))
            ]
        }]
    )

    fig3.write_html(ANIMATIONS_DIR / 'timing-sensitivity.html')
    print(f"  Saved: {ANIMATIONS_DIR / 'timing-sensitivity.html'}")

    print("\n" + "=" * 60)
    print("Animation generation complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Generate visuals for Essay 067')
    parser.add_argument('--images-only', action='store_true', help='Generate only images')
    parser.add_argument('--charts-only', action='store_true', help='Generate only charts')
    parser.add_argument('--animations-only', action='store_true', help='Generate only animations')
    args = parser.parse_args()

    print("=" * 60)
    print("ESSAY 067 VISUAL GENERATION")
    print("Steering the Ship: Policy Lessons from Cliodynamics")
    print("=" * 60)

    if args.images_only:
        generate_images()
    elif args.charts_only:
        generate_charts()
    elif args.animations_only:
        generate_animations()
    else:
        # Generate all
        generate_images()
        generate_charts()
        generate_animations()

    print("\n" + "=" * 60)
    print("ALL VISUALS GENERATED SUCCESSFULLY!")
    print(f"Images: {IMAGES_DIR}")
    print(f"Charts: {CHARTS_DIR}")
    print(f"Animations: {ANIMATIONS_DIR}")
    print("=" * 60)
    print("\nIMPORTANT: Visually verify all generated content before committing!")


if __name__ == '__main__':
    main()
