#!/usr/bin/env python3
"""
Generate all visualizations for Essay 004: America in the Balance.

This script generates:
- Matplotlib/animation visualizations
- Phase space trajectories
- Animated secular cycles
- PSI component breakdowns

Run from project root:
    uv run python scripts/generate_essay4_viz.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd

from cliodynamics.data.us.loader import USHistoricalData
from cliodynamics.models.ages_of_discord import AgesOfDiscordModel, AgesOfDiscordConfig

# Output directories
IMAGES_DIR = Path("docs/assets/images")
ANIMATIONS_DIR = Path("docs/assets/animations")
CHARTS_DIR = Path("docs/assets/charts")

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
ANIMATIONS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'wbi': '#2563eb',      # Blue
    'eoi': '#dc2626',      # Red
    'psi': '#7c3aed',      # Purple
    'real_wage': '#059669', # Green
    'relative_wage': '#d97706', # Orange
    'violence': '#be123c', # Rose
    'expansion': '#10b981', # Emerald
    'stagflation': '#f59e0b', # Amber
    'crisis': '#ef4444',   # Red
}


def get_us_data():
    """Load and compute Ages of Discord indices."""
    data = USHistoricalData()
    df = data.get_combined_dataset()
    
    model = AgesOfDiscordModel()
    results = model.compute_all(df)
    
    return df, results


def generate_psi_phase_space():
    """Generate phase space plot: Well-Being vs PSI trajectory."""
    print("Generating PSI phase space plot...")
    
    df, results = get_us_data()
    
    # Merge results with years
    merged = results.copy()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get data
    wbi = merged['well_being_index'].values
    psi = merged['political_stress_index'].values
    years = merged['year'].values
    
    # Color by time period
    colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
    
    # Plot trajectory with color gradient
    for i in range(len(years) - 1):
        ax.plot([wbi[i], wbi[i+1]], [psi[i], psi[i+1]], 
                color=colors[i], linewidth=2, alpha=0.8)
    
    # Mark key years
    key_years = [1800, 1860, 1920, 1960, 2020]
    for year in key_years:
        idx = np.argmin(np.abs(years - year))
        ax.scatter([wbi[idx]], [psi[idx]], s=150, zorder=5, 
                   edgecolors='black', linewidth=2)
        ax.annotate(str(year), (wbi[idx], psi[idx]), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=12, fontweight='bold')
    
    # Add start/end markers
    ax.scatter([wbi[0]], [psi[0]], s=200, color='green', marker='o', 
               zorder=6, label='Start (1780)', edgecolors='black', linewidth=2)
    ax.scatter([wbi[-1]], [psi[-1]], s=200, color='red', marker='s', 
               zorder=6, label='End (2025)', edgecolors='black', linewidth=2)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(years[0], years[-1]))
    cbar = fig.colorbar(sm, ax=ax, label='Year', shrink=0.8)
    
    ax.set_xlabel('Well-Being Index', fontsize=14)
    ax.set_ylabel('Political Stress Index', fontsize=14)
    ax.set_title('American History in Phase Space: 1780-2025\nTrajectory through Well-Being and Political Stress', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'essay-004-phase-space.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: essay-004-phase-space.png")


def generate_secular_cycles_diagram():
    """Generate annotated secular cycles diagram showing both American cycles."""
    print("Generating secular cycles diagram...")
    
    df, results = get_us_data()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    years = results['year'].values
    psi = results['political_stress_index'].values
    
    # Plot PSI
    ax.plot(years, psi, color=COLORS['psi'], linewidth=2.5, label='Political Stress Index')
    
    # Mark cycle phases with background colors
    # First cycle: 1780-1870
    ax.axvspan(1780, 1820, alpha=0.15, color=COLORS['expansion'], label='Expansion')
    ax.axvspan(1820, 1850, alpha=0.15, color=COLORS['stagflation'], label='Stagflation')
    ax.axvspan(1850, 1870, alpha=0.15, color=COLORS['crisis'], label='Crisis')
    
    # Second cycle: 1870-2025+
    ax.axvspan(1870, 1920, alpha=0.15, color=COLORS['expansion'])
    ax.axvspan(1920, 1940, alpha=0.15, color=COLORS['stagflation'])
    ax.axvspan(1940, 1970, alpha=0.15, color=COLORS['expansion'])
    ax.axvspan(1970, 2025, alpha=0.15, color=COLORS['crisis'])
    
    # Add historical event annotations
    events = [
        (1861, 'Civil War', 'top'),
        (1919, 'Red Summer', 'top'),
        (1968, 'Civil Rights\nUpheaval', 'bottom'),
        (2020, 'Capitol Riot', 'top'),
    ]
    
    for year, label, pos in events:
        idx = np.argmin(np.abs(years - year))
        y_val = psi[idx]
        
        if pos == 'top':
            ax.annotate(label, (year, y_val), xytext=(0, 30), 
                       textcoords='offset points', ha='center',
                       fontsize=10, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        else:
            ax.annotate(label, (year, y_val), xytext=(0, -40), 
                       textcoords='offset points', ha='center',
                       fontsize=10, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Add cycle labels
    ax.text(1825, ax.get_ylim()[1] * 0.95, 'First Secular Cycle\n(1780-1870)', 
            ha='center', fontsize=11, style='italic', fontweight='bold')
    ax.text(1940, ax.get_ylim()[1] * 0.95, 'Second Secular Cycle\n(1870-present)', 
            ha='center', fontsize=11, style='italic', fontweight='bold')
    
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Political Stress Index', fontsize=14)
    ax.set_title("America's Secular Cycles: Two Complete Oscillations\nPhases of Expansion, Stagflation, and Crisis", 
                 fontsize=16, fontweight='bold')
    ax.set_xlim(1780, 2030)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'essay-004-secular-cycles-annotated.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: essay-004-secular-cycles-annotated.png")


def generate_psi_decomposition():
    """Generate stacked area chart showing PSI component breakdown."""
    print("Generating PSI decomposition chart...")
    
    df, results = get_us_data()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    years = results['year'].values
    
    # Compute components
    wbi = results['well_being_index'].values
    eoi = results['elite_overproduction_index'].values
    
    # Mass Mobilization Potential (inverted WBI, scaled)
    wbi_max = wbi.max()
    mmp = (wbi_max / wbi) * 50  # Scale for visualization
    
    # Elite Competition (scaled EOI)
    ec = eoi * 0.5  # Scale for visualization
    
    # Normalize for stacking
    total = mmp + ec
    mmp_norm = mmp / total * 100
    ec_norm = ec / total * 100
    
    # Stack the areas
    ax.fill_between(years, 0, mmp_norm, alpha=0.7, color=COLORS['wbi'], 
                    label='Mass Mobilization Potential\n(from low well-being)')
    ax.fill_between(years, mmp_norm, 100, alpha=0.7, color=COLORS['eoi'], 
                    label='Elite Competition\n(from overproduction)')
    
    # Add vertical lines for key years
    for year in [1860, 1920, 1968, 2020]:
        ax.axvline(year, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Contribution to Political Stress (%)', fontsize=14)
    ax.set_title('Anatomy of Political Stress: What Drives American Instability?\nDecomposition of PSI into Mass Mobilization and Elite Competition', 
                 fontsize=16, fontweight='bold')
    ax.set_xlim(1780, 2025)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'essay-004-psi-decomposition-stacked.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: essay-004-psi-decomposition-stacked.png")


def generate_turchin_2010_prediction():
    """Generate chart showing Turchin's 2010 prediction vs what happened."""
    print("Generating Turchin 2010 prediction comparison...")
    
    df, results = get_us_data()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get data
    years = results['year'].values
    psi = results['political_stress_index'].values
    
    # Data up to 2010 (what Turchin had)
    mask_2010 = years <= 2010
    years_2010 = years[mask_2010]
    psi_2010 = psi[mask_2010]
    
    # Data after 2010 (what actually happened)
    mask_post = years >= 2010
    years_post = years[mask_post]
    psi_post = psi[mask_post]
    
    # Plot
    ax.plot(years_2010, psi_2010, color=COLORS['psi'], linewidth=2.5, 
            label='Historical Data (1780-2010)')
    ax.plot(years_post, psi_post, color=COLORS['crisis'], linewidth=2.5, 
            linestyle='-', label='Observed (2010-2025)')
    
    # Add Turchin's prediction zone
    ax.axvspan(2015, 2025, alpha=0.2, color='orange', 
               label="Turchin's Predicted Crisis Window")
    
    # Mark 2020 peak
    idx_2020 = np.argmin(np.abs(years - 2020))
    ax.scatter([2020], [psi[idx_2020]], s=200, color=COLORS['crisis'], 
               zorder=5, edgecolors='black', linewidth=2)
    ax.annotate('2020 Peak\n(Capitol Riot)', (2020, psi[idx_2020]), 
                xytext=(-60, 30), textcoords='offset points',
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Add vertical line at 2010 (prediction made)
    ax.axvline(2010, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    ax.annotate('Prediction Made\n(Nature, 2010)', (2010, ax.get_ylim()[1] * 0.9), 
                ha='center', fontsize=10, style='italic')
    
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Political Stress Index', fontsize=14)
    ax.set_title("Turchin's 2010 Prediction: Did It Come True?\nPSI Trajectory Before and After the Forecast", 
                 fontsize=16, fontweight='bold')
    ax.set_xlim(1960, 2030)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'essay-004-turchin-prediction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: essay-004-turchin-prediction.png")


def generate_future_scenarios():
    """Generate scenario projections for next 20 years."""
    print("Generating future scenarios chart...")
    
    df, results = get_us_data()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Historical data
    years = results['year'].values
    psi = results['political_stress_index'].values
    
    # Plot historical
    ax.plot(years, psi, color=COLORS['psi'], linewidth=2.5, label='Historical')
    
    # Project scenarios from 2025 to 2045
    future_years = np.arange(2025, 2046)
    psi_2025 = psi[-1]
    
    # Scenario 1: Continued escalation (business as usual)
    escalation = psi_2025 * (1.02 ** np.arange(len(future_years)))
    
    # Scenario 2: Peak and decline (reform scenario)
    peak_year_idx = 5  # Peak around 2030
    reform = psi_2025 * np.concatenate([
        1.01 ** np.arange(peak_year_idx),
        0.97 ** np.arange(len(future_years) - peak_year_idx)
    ])
    reform = reform * (psi_2025 / reform[0])
    
    # Scenario 3: Stabilization (muddling through)
    stabilization = psi_2025 * (1 + 0.005 * np.sin(np.linspace(0, 2*np.pi, len(future_years))))
    
    # Plot scenarios
    ax.plot(future_years, escalation, color=COLORS['crisis'], linewidth=2, 
            linestyle='--', label='Continued Escalation')
    ax.plot(future_years, reform, color=COLORS['expansion'], linewidth=2, 
            linestyle='--', label='Reform & Resolution')
    ax.plot(future_years, stabilization, color=COLORS['stagflation'], linewidth=2, 
            linestyle='--', label='Prolonged Stagnation')
    
    # Add uncertainty band
    ax.fill_between(future_years, 
                    stabilization * 0.85, 
                    escalation * 1.1, 
                    alpha=0.1, color='gray', label='Uncertainty Range')
    
    # Vertical line at present
    ax.axvline(2025, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.annotate('Present', (2025, ax.get_ylim()[0]), 
                xytext=(5, 10), textcoords='offset points',
                fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Political Stress Index', fontsize=14)
    ax.set_title('What Comes Next? Scenario Projections for 2025-2045\nThree Possible Paths for American Political Stress', 
                 fontsize=16, fontweight='bold')
    ax.set_xlim(1970, 2050)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'essay-004-future-scenarios.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: essay-004-future-scenarios.png")


def generate_wbi_eoi_overlay():
    """Generate overlaid WBI and EOI showing their inverse relationship."""
    print("Generating WBI-EOI overlay chart...")
    
    df, results = get_us_data()
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    years = results['year'].values
    wbi = results['well_being_index'].values
    eoi = results['elite_overproduction_index'].values
    
    # WBI on left axis
    color1 = COLORS['wbi']
    ax1.plot(years, wbi, color=color1, linewidth=2.5, label='Well-Being Index')
    ax1.set_xlabel('Year', fontsize=14)
    ax1.set_ylabel('Well-Being Index', color=color1, fontsize=14)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(40, 120)
    
    # EOI on right axis
    ax2 = ax1.twinx()
    color2 = COLORS['eoi']
    ax2.plot(years, eoi, color=color2, linewidth=2.5, label='Elite Overproduction Index')
    ax2.set_ylabel('Elite Overproduction Index', color=color2, fontsize=14)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(50, 250)
    
    # Add reference line at 1960
    ax1.axvline(1960, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.annotate('1960:\nPeak Well-Being\nMinimum Elite Competition', 
                 (1960, 60), ha='center', fontsize=10, style='italic',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add shading for key periods
    ax1.axvspan(1940, 1970, alpha=0.1, color=COLORS['expansion'])
    ax1.axvspan(1970, 2025, alpha=0.1, color=COLORS['crisis'])
    
    ax1.set_title('The Scissors Graph: Well-Being Falling as Elite Competition Rises\n' +
                  'Two Indices, One Story of Structural Strain', 
                 fontsize=16, fontweight='bold')
    ax1.set_xlim(1780, 2030)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
    
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'essay-004-wbi-eoi-scissors.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: essay-004-wbi-eoi-scissors.png")


def generate_animated_psi():
    """Generate animated GIF showing PSI evolution through time."""
    print("Generating animated PSI timeline...")
    
    df, results = get_us_data()
    
    years = results['year'].values
    psi = results['political_stress_index'].values
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up the plot
    ax.set_xlim(1780, 2030)
    ax.set_ylim(psi.min() * 0.9, psi.max() * 1.1)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Political Stress Index', fontsize=14)
    ax.set_title('Political Stress Index: 1780-2025\nWatch How Instability Pressure Builds Over Time', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    line, = ax.plot([], [], color=COLORS['psi'], linewidth=2.5)
    point, = ax.plot([], [], 'o', color=COLORS['crisis'], markersize=10)
    year_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14, 
                        fontweight='bold', verticalalignment='top')
    
    # Event annotations (will appear when reached)
    events = {
        1861: 'Civil War',
        1919: 'Red Summer',
        1968: 'Civil Rights',
        2020: 'Capitol Riot',
    }
    annotations = {}
    
    def init():
        line.set_data([], [])
        point.set_data([], [])
        year_text.set_text('')
        return line, point, year_text
    
    def animate(frame):
        # Progressive reveal
        idx = int(frame * len(years) / 120)  # 120 frames
        if idx >= len(years):
            idx = len(years) - 1
        
        line.set_data(years[:idx+1], psi[:idx+1])
        point.set_data([years[idx]], [psi[idx]])
        year_text.set_text(f'Year: {int(years[idx])}')
        
        # Add event annotations when reached
        current_year = years[idx]
        for event_year, event_name in events.items():
            if current_year >= event_year and event_year not in annotations:
                event_idx = np.argmin(np.abs(years - event_year))
                ann = ax.annotate(event_name, (years[event_idx], psi[event_idx]),
                                 xytext=(0, 20), textcoords='offset points',
                                 ha='center', fontsize=10, fontweight='bold',
                                 arrowprops=dict(arrowstyle='->', color='black'))
                annotations[event_year] = ann
        
        return line, point, year_text
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=120, 
                        interval=50, blit=False)
    
    # Save as GIF
    anim.save(ANIMATIONS_DIR / 'essay-004-psi-animation.gif', 
              writer='pillow', fps=20, dpi=100)
    plt.close()
    print("  Saved: essay-004-psi-animation.gif")


def generate_comparison_with_civil_war():
    """Generate comparison chart: 1850s vs 2020s."""
    print("Generating Civil War comparison chart...")
    
    df, results = get_us_data()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    years = results['year'].values
    psi = results['political_stress_index'].values
    wbi = results['well_being_index'].values
    eoi = results['elite_overproduction_index'].values
    
    # Left panel: 1850s
    ax1 = axes[0]
    mask1 = (years >= 1840) & (years <= 1870)
    ax1.plot(years[mask1], psi[mask1], color=COLORS['psi'], linewidth=2.5, label='PSI')
    ax1.plot(years[mask1], wbi[mask1], color=COLORS['wbi'], linewidth=2, linestyle='--', label='WBI')
    ax1.axvline(1861, color='black', linestyle=':', linewidth=2)
    ax1.annotate('Civil War Begins', (1861, ax1.get_ylim()[1] * 0.9), 
                ha='center', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Index Value', fontsize=12)
    ax1.set_title('Pre-Civil War Crisis\n(1840-1870)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Right panel: 2010s-2020s
    ax2 = axes[1]
    mask2 = (years >= 2000) & (years <= 2025)
    ax2.plot(years[mask2], psi[mask2], color=COLORS['psi'], linewidth=2.5, label='PSI')
    ax2.plot(years[mask2], wbi[mask2], color=COLORS['wbi'], linewidth=2, linestyle='--', label='WBI')
    ax2.axvline(2020, color='black', linestyle=':', linewidth=2)
    ax2.annotate('Capitol Riot', (2020, ax2.get_ylim()[1] * 0.9), 
                ha='center', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Index Value', fontsize=12)
    ax2.set_title('Contemporary Crisis\n(2000-2025)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle('Historical Parallel: Are the 2020s Like the 1850s?\n' +
                 'Comparing Structural Conditions Before and During Crisis',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'essay-004-civil-war-comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: essay-004-civil-war-comparison.png")


def generate_all_three_indices():
    """Generate comprehensive 3-panel view of all indices."""
    print("Generating comprehensive indices chart...")
    
    df, results = get_us_data()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    years = results['year'].values
    wbi = results['well_being_index'].values
    eoi = results['elite_overproduction_index'].values
    psi = results['political_stress_index'].values
    
    # Panel 1: Well-Being Index
    ax1 = axes[0]
    ax1.fill_between(years, 0, wbi, alpha=0.3, color=COLORS['wbi'])
    ax1.plot(years, wbi, color=COLORS['wbi'], linewidth=2.5)
    ax1.axhline(100, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Well-Being Index', fontsize=12)
    ax1.set_title('Worker Well-Being: Rising Until 1960, Declining Since', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Elite Overproduction Index
    ax2 = axes[1]
    ax2.fill_between(years, 0, eoi, alpha=0.3, color=COLORS['eoi'])
    ax2.plot(years, eoi, color=COLORS['eoi'], linewidth=2.5)
    ax2.axhline(100, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Elite Overproduction Index', fontsize=12)
    ax2.set_title('Elite Competition: Exploding Since 1970', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Political Stress Index
    ax3 = axes[2]
    ax3.fill_between(years, 0, psi, alpha=0.3, color=COLORS['psi'])
    ax3.plot(years, psi, color=COLORS['psi'], linewidth=2.5)
    ax3.axhline(100, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Political Stress Index', fontsize=12)
    ax3.set_xlabel('Year', fontsize=12)
    ax3.set_title('Political Stress: The Product of Immiseration and Elite Conflict', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add event markers
    events = [(1861, 'Civil War'), (1919, 'Red Summer'), (1968, '1968'), (2020, '2020')]
    for year, label in events:
        for ax in axes:
            ax.axvline(year, color='gray', linestyle=':', alpha=0.5)
    
    axes[0].set_xlim(1780, 2030)
    
    fig.suptitle('The Three Pillars of Structural-Demographic Theory\nApplied to American History',
                 fontsize=18, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'essay-004-three-indices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: essay-004-three-indices.png")


def main():
    """Generate all visualizations for Essay 004."""
    print("=" * 60)
    print("Generating Essay 004 Visualizations")
    print("=" * 60)
    
    # Static matplotlib plots
    generate_psi_phase_space()
    generate_secular_cycles_diagram()
    generate_psi_decomposition()
    generate_turchin_2010_prediction()
    generate_future_scenarios()
    generate_wbi_eoi_overlay()
    generate_comparison_with_civil_war()
    generate_all_three_indices()
    
    # Animation
    generate_animated_psi()
    
    print("=" * 60)
    print("All Essay 004 visualizations generated!")
    print("=" * 60)
    print("\nREMINDER: Visually verify all generated images before committing!")


if __name__ == "__main__":
    main()
