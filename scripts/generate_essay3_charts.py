#!/usr/bin/env python3
"""Generate charts for Essay 3: The Mathematics of Societal Collapse.

This script generates visualizations for the SDT model implementation essay.
Uses more stable parameters to avoid numerical instabilities.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import altair as alt
from scipy.integrate import odeint
from pathlib import Path

# Import our SDT implementation
from cliodynamics.models.sdt import SDTModel
from cliodynamics.models.params import SDTParams, SDTState

# Output directory
OUTPUT_DIR = Path('docs/assets/images')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Chart configuration
CHART_WIDTH = 700
CHART_HEIGHT = 400


def configure_chart(chart, title, width=CHART_WIDTH, height=CHART_HEIGHT):
    """Apply standard formatting to an Altair chart."""
    return (
        chart
        .properties(
            width=width,
            height=height,
            title=alt.TitleParams(
                text=title,
                fontSize=18,
                anchor='start'
            )
        )
        .configure_axis(
            labelFontSize=12,
            titleFontSize=14,
            labelLimit=300,
        )
        .configure_legend(
            labelFontSize=11,
            titleFontSize=12,
            labelLimit=200,
        )
    )


def get_stable_params():
    """Return parameters that produce stable dynamics for visualization."""
    return SDTParams(
        r_max=0.015,       # Slower population growth
        K_0=1.0,
        beta=0.5,          # Less sensitive to wages
        mu=0.15,
        alpha=0.003,       # Slower elite formation
        delta_e=0.025,     # Faster elite decay
        gamma=1.5,         # Moderate labor effect
        eta=0.8,           # Lower extraction
        rho=0.15,
        sigma=0.08,
        epsilon=0.03,
        lambda_psi=0.03,   # Slower stress accumulation
        theta_w=0.8,
        theta_e=0.8,
        theta_s=0.8,
        psi_decay=0.03,
        W_0=1.0,
        E_0=0.1,
        S_0=1.0,
    )


def safe_simulate(model, y0, t, max_val=100):
    """Run simulation and cap extreme values for visualization."""
    solution = odeint(model.system, y0, t)
    # Cap values to avoid numerical issues in visualization
    solution = np.clip(solution, -max_val, max_val)
    # Replace NaN/Inf with nearby values
    solution = np.nan_to_num(solution, nan=0.0, posinf=max_val, neginf=-max_val)
    return solution


def generate_population_dynamics_chart():
    """Generate chart showing population dynamics under different wage scenarios."""
    print("Generating population dynamics chart...")

    t = np.linspace(0, 60, 500)

    scenarios = []
    params = get_stable_params()

    # High wages scenario
    model = SDTModel(params)
    y0_high = [0.3, 0.05, 1.3, 1.0, 0.0]
    sol_high = safe_simulate(model, y0_high, t)
    for i, time in enumerate(t[::5]):  # Subsample
        idx = i * 5
        scenarios.append({
            'time': time,
            'population': sol_high[idx, 0],
            'scenario': 'High Initial Wages'
        })

    # Baseline wages scenario
    y0_norm = [0.3, 0.05, 1.0, 1.0, 0.0]
    sol_norm = safe_simulate(model, y0_norm, t)
    for i, time in enumerate(t[::5]):
        idx = i * 5
        scenarios.append({
            'time': time,
            'population': sol_norm[idx, 0],
            'scenario': 'Baseline Wages'
        })

    # Low wages scenario
    y0_low = [0.3, 0.05, 0.7, 1.0, 0.0]
    sol_low = safe_simulate(model, y0_low, t)
    for i, time in enumerate(t[::5]):
        idx = i * 5
        scenarios.append({
            'time': time,
            'population': sol_low[idx, 0],
            'scenario': 'Low Initial Wages'
        })

    df = pd.DataFrame(scenarios)

    chart = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X('time:Q', title='Time (years)'),
        y=alt.Y('population:Q', title='Population (relative to K)',
                scale=alt.Scale(domain=[0.2, 1.1])),
        color=alt.Color('scenario:N',
                       title='Wage Scenario',
                       scale=alt.Scale(domain=['High Initial Wages', 'Baseline Wages', 'Low Initial Wages'],
                                       range=['#2ecc71', '#3498db', '#e74c3c']))
    )

    chart = configure_chart(chart, 'Population Dynamics Under Different Wage Conditions')
    chart.save(str(OUTPUT_DIR / 'chart-population-dynamics.png'), scale_factor=2)
    print(f"  Saved to {OUTPUT_DIR / 'chart-population-dynamics.png'}")


def generate_wage_dynamics_chart():
    """Generate chart showing wage dynamics."""
    print("Generating wage dynamics chart...")

    t = np.linspace(0, 60, 500)

    params = get_stable_params()
    model = SDTModel(params)
    y0 = [0.4, 0.06, 1.1, 1.0, 0.0]
    solution = safe_simulate(model, y0, t)

    data = []
    for i, time in enumerate(t[::5]):
        idx = i * 5
        data.append({'time': time, 'value': solution[idx, 2], 'variable': 'Wages (W)'})
        data.append({'time': time, 'value': solution[idx, 0], 'variable': 'Population (N)'})
        data.append({'time': time, 'value': solution[idx, 1] * 5, 'variable': 'Elites (E x5)'})

    df = pd.DataFrame(data)

    chart = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X('time:Q', title='Time (years)'),
        y=alt.Y('value:Q', title='Value (normalized)',
                scale=alt.Scale(domain=[0, 1.5])),
        color=alt.Color('variable:N', title='Variable',
                       scale=alt.Scale(domain=['Wages (W)', 'Population (N)', 'Elites (E x5)'],
                                       range=['#e74c3c', '#3498db', '#9b59b6']))
    )

    chart = configure_chart(chart, 'Wage Dynamics: Response to Population and Elite Pressure')
    chart.save(str(OUTPUT_DIR / 'chart-wage-dynamics.png'), scale_factor=2)
    print(f"  Saved to {OUTPUT_DIR / 'chart-wage-dynamics.png'}")


def generate_state_dynamics_chart():
    """Generate chart showing state fiscal health."""
    print("Generating state dynamics chart...")

    t = np.linspace(0, 60, 500)

    params = get_stable_params()
    model = SDTModel(params)
    y0 = [0.4, 0.06, 1.1, 1.0, 0.0]
    solution = safe_simulate(model, y0, t)

    data = []
    for i, time in enumerate(t[::5]):
        idx = i * 5
        N, E, W, S, psi = solution[idx]
        data.append({'time': time, 'value': S, 'variable': 'State Health (S)'})
        data.append({'time': time, 'value': E * 5, 'variable': 'Elite Burden (E x5)'})
        data.append({'time': time, 'value': W * N * 0.5, 'variable': 'Revenue Base (W*N/2)'})

    df = pd.DataFrame(data)

    chart = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X('time:Q', title='Time (years)'),
        y=alt.Y('value:Q', title='Value (normalized)',
                scale=alt.Scale(domain=[0, 1.5])),
        color=alt.Color('variable:N', title='Variable',
                       scale=alt.Scale(domain=['State Health (S)', 'Elite Burden (E x5)', 'Revenue Base (W*N/2)'],
                                       range=['#2ecc71', '#e74c3c', '#f39c12']))
    )

    chart = configure_chart(chart, 'State Fiscal Health: Revenue vs Elite Burden')
    chart.save(str(OUTPUT_DIR / 'chart-state-dynamics.png'), scale_factor=2)
    print(f"  Saved to {OUTPUT_DIR / 'chart-state-dynamics.png'}")


def generate_psi_components_chart():
    """Generate chart showing PSI components."""
    print("Generating PSI components chart...")

    t = np.linspace(0, 60, 500)

    params = get_stable_params()
    model = SDTModel(params)
    y0 = [0.4, 0.06, 1.1, 1.0, 0.0]
    solution = safe_simulate(model, y0, t)

    data = []
    for i, time in enumerate(t[::5]):
        idx = i * 5
        N, E, W, S, psi = solution[idx]
        # Calculate stress components (capped for stability)
        wage_stress = min(max(0, params.W_0 / max(W, 0.1) - 1), 5)
        elite_stress = min(max(0, E / params.E_0 - 1), 5)
        state_stress = min(max(0, params.S_0 / max(S, 0.1) - 1), 5)

        data.append({'time': time, 'value': wage_stress, 'component': 'Wage Stress'})
        data.append({'time': time, 'value': elite_stress, 'component': 'Elite Stress'})
        data.append({'time': time, 'value': state_stress, 'component': 'State Stress'})
        data.append({'time': time, 'value': min(psi, 5), 'component': 'Total PSI'})

    df = pd.DataFrame(data)

    chart = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X('time:Q', title='Time (years)'),
        y=alt.Y('value:Q', title='Stress Level', scale=alt.Scale(domain=[0, 3])),
        color=alt.Color('component:N', title='Component',
                       scale=alt.Scale(domain=['Wage Stress', 'Elite Stress', 'State Stress', 'Total PSI'],
                                       range=['#e74c3c', '#9b59b6', '#f39c12', '#2c3e50']))
    )

    chart = configure_chart(chart, 'Political Stress Index: Components Over Time')
    chart.save(str(OUTPUT_DIR / 'chart-psi-components.png'), scale_factor=2)
    print(f"  Saved to {OUTPUT_DIR / 'chart-psi-components.png'}")


def generate_phase_portrait_chart():
    """Generate phase portrait in elite-wage space."""
    print("Generating phase portrait chart...")

    t = np.linspace(0, 80, 600)

    params = get_stable_params()
    model = SDTModel(params)

    all_data = []
    starting_points = [
        ([0.3, 0.04, 1.15, 1.0, 0.0], 'Start 1'),
        ([0.5, 0.08, 0.95, 0.9, 0.1], 'Start 2'),
        ([0.4, 0.06, 1.05, 1.0, 0.05], 'Start 3'),
    ]

    for y0, label in starting_points:
        solution = safe_simulate(model, y0, t)
        for i, time in enumerate(t[::3]):
            idx = i * 3
            all_data.append({
                'elite': solution[idx, 1],
                'wage': solution[idx, 2],
                'time': time,
                'trajectory': label
            })

    df = pd.DataFrame(all_data)

    chart = alt.Chart(df).mark_line(opacity=0.8, strokeWidth=2).encode(
        x=alt.X('wage:Q', title='Wages (W)', scale=alt.Scale(domain=[0.7, 1.2])),
        y=alt.Y('elite:Q', title='Elite Population (E)', scale=alt.Scale(domain=[0.03, 0.15])),
        color=alt.Color('trajectory:N', title='Trajectory',
                       scale=alt.Scale(range=['#3498db', '#e74c3c', '#2ecc71'])),
        order='time:Q'
    )

    chart = configure_chart(chart, 'Phase Portrait: Elite-Wage Space')
    chart.save(str(OUTPUT_DIR / 'chart-phase-portrait-ew.png'), scale_factor=2)
    print(f"  Saved to {OUTPUT_DIR / 'chart-phase-portrait-ew.png'}")


def generate_solver_comparison_chart():
    """Generate chart comparing solver accuracy at different resolutions."""
    print("Generating solver comparison chart...")

    # Use simple logistic growth for a clean comparison
    def logistic(y, t, r=0.1, K=1.0):
        return r * y * (1 - y / K)

    t_fine = np.linspace(0, 50, 1000)
    t_medium = np.linspace(0, 50, 100)
    t_coarse = np.linspace(0, 50, 20)

    y0 = [0.1]

    sol_fine = odeint(logistic, y0, t_fine)
    sol_medium = odeint(logistic, y0, t_medium)
    sol_coarse = odeint(logistic, y0, t_coarse)

    data = []
    for i, time in enumerate(t_fine[::10]):
        idx = i * 10
        data.append({'time': time, 'value': sol_fine[idx, 0], 'method': 'High Resolution (1000 pts)'})

    for i, time in enumerate(t_medium):
        data.append({'time': time, 'value': sol_medium[i, 0], 'method': 'Medium Resolution (100 pts)'})

    for i, time in enumerate(t_coarse):
        data.append({'time': time, 'value': sol_coarse[i, 0], 'method': 'Low Resolution (20 pts)'})

    df = pd.DataFrame(data)

    chart = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X('time:Q', title='Time (years)'),
        y=alt.Y('value:Q', title='Population', scale=alt.Scale(domain=[0, 1.1])),
        color=alt.Color('method:N', title='Resolution',
                       scale=alt.Scale(range=['#2ecc71', '#3498db', '#e74c3c'])),
        strokeDash=alt.StrokeDash('method:N')
    )

    chart = configure_chart(chart, 'Numerical Solver: Effect of Resolution on Accuracy')
    chart.save(str(OUTPUT_DIR / 'chart-solver-comparison.png'), scale_factor=2)
    print(f"  Saved to {OUTPUT_DIR / 'chart-solver-comparison.png'}")


def generate_sensitivity_analysis_chart():
    """Generate sensitivity analysis chart."""
    print("Generating sensitivity analysis chart...")

    t = np.linspace(0, 50, 300)
    y0 = [0.4, 0.06, 1.0, 1.0, 0.0]

    params_base = get_stable_params()
    model_base = SDTModel(params_base)
    sol_base = safe_simulate(model_base, y0, t)
    psi_max_base = np.max(sol_base[:, 4])

    param_names = ['theta_e', 'theta_w', 'lambda_psi', 'eta', 'alpha']
    param_labels = ['Elite Weight', 'Wage Weight', 'Accum. Rate', 'Extraction', 'Mobility']
    variations = []

    for pname, plabel in zip(param_names, param_labels):
        base_val = getattr(params_base, pname)

        # +20%
        param_dict = {
            'r_max': params_base.r_max, 'K_0': params_base.K_0, 'beta': params_base.beta,
            'mu': params_base.mu, 'alpha': params_base.alpha, 'delta_e': params_base.delta_e,
            'gamma': params_base.gamma, 'eta': params_base.eta, 'rho': params_base.rho,
            'sigma': params_base.sigma, 'epsilon': params_base.epsilon,
            'lambda_psi': params_base.lambda_psi, 'theta_w': params_base.theta_w,
            'theta_e': params_base.theta_e, 'theta_s': params_base.theta_s,
            'psi_decay': params_base.psi_decay, 'W_0': params_base.W_0,
            'E_0': params_base.E_0, 'S_0': params_base.S_0
        }
        param_dict[pname] = base_val * 1.2
        params_high = SDTParams(**param_dict)
        model_high = SDTModel(params_high)
        sol_high = safe_simulate(model_high, y0, t)
        psi_max_high = np.max(sol_high[:, 4])

        # -20%
        param_dict[pname] = base_val * 0.8
        params_low = SDTParams(**param_dict)
        model_low = SDTModel(params_low)
        sol_low = safe_simulate(model_low, y0, t)
        psi_max_low = np.max(sol_low[:, 4])

        if psi_max_base > 0.01:
            effect_high = (psi_max_high - psi_max_base) / psi_max_base * 100
            effect_low = (psi_max_low - psi_max_base) / psi_max_base * 100
        else:
            effect_high = 0
            effect_low = 0

        variations.append({'parameter': plabel, 'change': '+20%', 'effect': effect_high})
        variations.append({'parameter': plabel, 'change': '-20%', 'effect': effect_low})

    df = pd.DataFrame(variations)

    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y('parameter:N', title='Parameter', sort=param_labels),
        x=alt.X('effect:Q', title='Change in Max PSI (%)'),
        color=alt.Color('change:N', title='Variation',
                       scale=alt.Scale(domain=['+20%', '-20%'], range=['#3498db', '#e74c3c'])),
        yOffset='change:N'
    )

    chart = configure_chart(chart, 'Sensitivity Analysis: Parameter Effects on Peak Instability', height=300)
    chart.save(str(OUTPUT_DIR / 'chart-sensitivity-analysis.png'), scale_factor=2)
    print(f"  Saved to {OUTPUT_DIR / 'chart-sensitivity-analysis.png'}")


def generate_initial_conditions_chart():
    """Generate chart showing effect of different initial conditions."""
    print("Generating initial conditions chart...")

    t = np.linspace(0, 60, 400)

    params = get_stable_params()
    model = SDTModel(params)

    scenarios = []

    # Favorable start
    y0_good = [0.25, 0.03, 1.2, 1.2, 0.0]
    sol_good = safe_simulate(model, y0_good, t)
    for i, time in enumerate(t[::4]):
        idx = i * 4
        scenarios.append({'time': time, 'psi': sol_good[idx, 4],
                         'condition': 'Favorable Start'})

    # Unfavorable start
    y0_bad = [0.7, 0.12, 0.8, 0.7, 0.3]
    sol_bad = safe_simulate(model, y0_bad, t)
    for i, time in enumerate(t[::4]):
        idx = i * 4
        scenarios.append({'time': time, 'psi': sol_bad[idx, 4],
                         'condition': 'Unfavorable Start'})

    # Neutral start
    y0_mid = [0.45, 0.06, 1.0, 1.0, 0.1]
    sol_mid = safe_simulate(model, y0_mid, t)
    for i, time in enumerate(t[::4]):
        idx = i * 4
        scenarios.append({'time': time, 'psi': sol_mid[idx, 4],
                         'condition': 'Neutral Start'})

    df = pd.DataFrame(scenarios)

    chart = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X('time:Q', title='Time (years)'),
        y=alt.Y('psi:Q', title='Political Stress Index', scale=alt.Scale(domain=[0, 2])),
        color=alt.Color('condition:N', title='Starting Condition',
                       scale=alt.Scale(domain=['Favorable Start', 'Neutral Start', 'Unfavorable Start'],
                                       range=['#2ecc71', '#3498db', '#e74c3c']))
    )

    chart = configure_chart(chart, 'Effect of Initial Conditions on PSI Trajectory')
    chart.save(str(OUTPUT_DIR / 'chart-initial-conditions.png'), scale_factor=2)
    print(f"  Saved to {OUTPUT_DIR / 'chart-initial-conditions.png'}")


def main():
    print("Generating charts for Essay 3...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    generate_population_dynamics_chart()
    generate_wage_dynamics_chart()
    generate_state_dynamics_chart()
    generate_psi_components_chart()
    generate_phase_portrait_chart()
    generate_solver_comparison_chart()
    generate_sensitivity_analysis_chart()
    generate_initial_conditions_chart()

    print()
    print("All charts generated successfully!")


if __name__ == '__main__':
    main()
