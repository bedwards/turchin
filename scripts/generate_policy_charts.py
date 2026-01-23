#!/usr/bin/env python3
"""Generate visualizations for the policy essay.

Creates charts showing:
1. Baseline vs intervention trajectories
2. Policy effectiveness rankings
3. Intervention timing effects (window of opportunity)
4. US PSI projections with interventions
5. Historical counterfactual comparisons
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import altair as alt
import numpy as np
import pandas as pd

from cliodynamics.models import SDTModel, SDTParams
from cliodynamics.policy import (
    CompositeIntervention,
    CounterfactualEngine,
    EliteCap,
    FiscalStimulus,
    InstitutionalReform,
    OutcomeComparison,
    SensitivityAnalysis,
    TaxProgressivity,
    WageBoost,
    WageFloor,
)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "assets" / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_baseline_simulation():
    """Run a baseline simulation representing late Republic/US trajectory."""
    params = SDTParams(
        r_max=0.02,
        K_0=1.0,
        beta=1.0,
        mu=0.2,
        alpha=0.006,  # Faster elite growth for overproduction
        delta_e=0.02,
        gamma=2.0,
        eta=1.0,
        rho=0.2,
        sigma=0.1,
        epsilon=0.06,  # Higher elite burden
        lambda_psi=0.06,
        theta_w=1.0,
        theta_e=1.2,  # Elite contribution slightly higher
        theta_s=0.8,
        psi_decay=0.02,
    )
    
    model = SDTModel(params)
    engine = CounterfactualEngine(model)
    
    initial_conditions = {
        "N": 0.5,
        "E": 0.08,
        "W": 1.0,
        "S": 1.0,
        "psi": 0.0,
    }
    
    baseline = engine.run_baseline(
        initial_conditions=initial_conditions,
        time_span=(0, 200),
        dt=0.5,
    )
    
    return engine, baseline, model


def create_trajectory_comparison_chart(engine, baseline):
    """Create chart comparing baseline vs intervention trajectories."""
    # Run intervention: comprehensive reform at year 50
    reform = CompositeIntervention(
        name="Comprehensive Reform",
        start_time=50,
        interventions=[
            EliteCap(name="Elite cap", start_time=50, max_elite_ratio=1.3),
            WageFloor(name="Wage floor", start_time=50, min_wage_ratio=0.85),
            TaxProgressivity(name="Progressive tax", start_time=50, revenue_boost=0.04),
        ],
    )
    
    intervention_result = engine.run_intervention(baseline, reform)
    
    # Prepare data
    df_base = baseline.df.copy()
    df_base["scenario"] = "Baseline (No Intervention)"
    
    df_int = intervention_result.df.copy()
    df_int["scenario"] = "With Comprehensive Reform"
    
    df_combined = pd.concat([df_base, df_int])
    
    # Create chart for PSI
    psi_chart = alt.Chart(df_combined).mark_line(strokeWidth=2.5).encode(
        x=alt.X("t:Q", title="Time (Years)", scale=alt.Scale(domain=[0, 200])),
        y=alt.Y("psi:Q", title="Political Stress Index (PSI)"),
        color=alt.Color("scenario:N", title="Scenario", 
                       scale=alt.Scale(range=["#e41a1c", "#377eb8"])),
        strokeDash=alt.StrokeDash("scenario:N", legend=None),
    ).properties(
        width=700,
        height=400,
        title="Political Stress Index: Baseline vs. Reform Intervention"
    )
    
    # Add annotation for intervention start
    intervention_line = alt.Chart(pd.DataFrame({"x": [50]})).mark_rule(
        strokeDash=[5, 5], color="gray"
    ).encode(x="x:Q")
    
    annotation = alt.Chart(pd.DataFrame({
        "x": [52], "y": [0.1], "text": ["Reform begins (year 50)"]
    })).mark_text(align="left", fontSize=12, color="gray").encode(
        x="x:Q", y="y:Q", text="text:N"
    )
    
    final_chart = (psi_chart + intervention_line + annotation).configure_axis(
        labelFontSize=12,
        titleFontSize=14,
    ).configure_legend(
        titleFontSize=14,
        labelFontSize=12,
    ).configure_title(
        fontSize=16,
    )
    
    final_chart.save(str(OUTPUT_DIR / "policy_trajectory_comparison.png"), scale_factor=2)
    print(f"Saved: {OUTPUT_DIR / 'policy_trajectory_comparison.png'}")
    
    return intervention_result


def create_policy_effectiveness_chart(engine, baseline):
    """Create bar chart ranking policy effectiveness."""
    # Define interventions to test
    interventions = [
        EliteCap(name="Elite Cap (1.5x)", start_time=50, max_elite_ratio=1.5),
        EliteCap(name="Elite Cap (1.3x)", start_time=50, max_elite_ratio=1.3),
        WageFloor(name="Wage Floor (80%)", start_time=50, min_wage_ratio=0.8),
        WageFloor(name="Wage Floor (85%)", start_time=50, min_wage_ratio=0.85),
        WageBoost(name="Wage Boost (2%/yr)", start_time=50, boost_rate=0.02),
        TaxProgressivity(name="Progressive Taxation", start_time=50, revenue_boost=0.05),
        FiscalStimulus(name="Fiscal Stimulus", start_time=50, wage_boost=0.02, psi_reduction=0.01),
        InstitutionalReform(name="Institutional Reform", start_time=50, legitimacy_boost=0.03),
    ]
    
    # Run all interventions
    results = engine.run_interventions(baseline, interventions)
    
    # Compute effectiveness
    analysis = SensitivityAnalysis(baseline, results)
    df = analysis.to_dataframe()
    
    # Rename columns for clarity
    df = df.rename(columns={
        "intervention": "Policy",
        "psi_peak_reduction": "PSI Reduction",
    })
    
    # Sort by effectiveness
    df = df.sort_values("PSI Reduction", ascending=False)
    
    # Create bar chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("PSI Reduction:Q", title="PSI Peak Reduction (%)", 
               axis=alt.Axis(format=".0%")),
        y=alt.Y("Policy:N", sort="-x", title="Policy Intervention"),
        color=alt.Color("PSI Reduction:Q", 
                       scale=alt.Scale(scheme="blues"),
                       legend=None),
    ).properties(
        width=600,
        height=350,
        title="Policy Effectiveness: Ranking by PSI Peak Reduction"
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14,
    ).configure_title(
        fontSize=16,
    )
    
    chart.save(str(OUTPUT_DIR / "policy_effectiveness_ranking.png"), scale_factor=2)
    print(f"Saved: {OUTPUT_DIR / 'policy_effectiveness_ranking.png'}")
    
    return df


def create_timing_window_chart(engine, baseline):
    """Create chart showing intervention effectiveness by timing."""
    # Test elite cap at different start times
    start_times = list(range(10, 160, 10))
    
    timing_data = []
    for start_time in start_times:
        intervention = EliteCap(
            name=f"Elite Cap at t={start_time}",
            start_time=start_time,
            max_elite_ratio=1.3,
        )
        result = engine.run_intervention(baseline, intervention)
        comparison = OutcomeComparison(baseline, result)
        
        timing_data.append({
            "Start Time": start_time,
            "PSI Reduction": comparison.psi_peak_reduction(),
            "Intervention": "Elite Cap (1.3x)"
        })
    
    # Also test wage floor
    for start_time in start_times:
        intervention = WageFloor(
            name=f"Wage Floor at t={start_time}",
            start_time=start_time,
            min_wage_ratio=0.85,
        )
        result = engine.run_intervention(baseline, intervention)
        comparison = OutcomeComparison(baseline, result)
        
        timing_data.append({
            "Start Time": start_time,
            "PSI Reduction": comparison.psi_peak_reduction(),
            "Intervention": "Wage Floor (85%)"
        })
    
    df = pd.DataFrame(timing_data)
    
    # Create line chart
    chart = alt.Chart(df).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("Start Time:Q", title="Intervention Start Time (Years)"),
        y=alt.Y("PSI Reduction:Q", title="PSI Peak Reduction (%)",
               axis=alt.Axis(format=".0%")),
        color=alt.Color("Intervention:N", 
                       scale=alt.Scale(range=["#377eb8", "#4daf4a"])),
    ).properties(
        width=700,
        height=400,
        title="Window of Opportunity: Earlier Intervention Yields Greater Effect"
    )
    
    # Add vertical line at crisis threshold (approximate peak year)
    baseline_peak_time = baseline.psi_peak_time
    threshold_line = alt.Chart(pd.DataFrame({"x": [baseline_peak_time]})).mark_rule(
        strokeDash=[5, 5], color="red"
    ).encode(x="x:Q")
    
    annotation = alt.Chart(pd.DataFrame({
        "x": [baseline_peak_time + 2], 
        "y": [0.35], 
        "text": ["Crisis Peak"]
    })).mark_text(align="left", fontSize=12, color="red").encode(
        x="x:Q", y="y:Q", text="text:N"
    )
    
    final_chart = (chart + threshold_line + annotation).configure_axis(
        labelFontSize=12,
        titleFontSize=14,
    ).configure_legend(
        titleFontSize=14,
        labelFontSize=12,
    ).configure_title(
        fontSize=16,
    )
    
    final_chart.save(str(OUTPUT_DIR / "policy_timing_window.png"), scale_factor=2)
    print(f"Saved: {OUTPUT_DIR / 'policy_timing_window.png'}")
    
    return df


def create_us_projection_chart():
    """Create chart showing US PSI projections with different intervention scenarios."""
    # Simulate US-like trajectory from 1970 to 2050
    params = SDTParams(
        r_max=0.015,
        K_0=1.0,
        beta=0.8,
        mu=0.25,
        alpha=0.007,
        delta_e=0.018,
        gamma=1.8,
        eta=1.2,
        rho=0.18,
        sigma=0.12,
        epsilon=0.07,
        lambda_psi=0.055,
        theta_w=1.1,
        theta_e=1.3,
        theta_s=0.7,
        psi_decay=0.015,
    )
    
    model = SDTModel(params)
    engine = CounterfactualEngine(model)
    
    # Initial conditions calibrated to ~1970 US
    initial_conditions = {
        "N": 0.7,
        "E": 0.12,
        "W": 1.0,
        "S": 0.95,
        "psi": 0.15,
    }
    
    # Run baseline (1970-2050)
    baseline = engine.run_baseline(
        initial_conditions=initial_conditions,
        time_span=(0, 80),  # 1970-2050
        dt=0.5,
    )
    
    # Scenario 1: No intervention (baseline)
    df_base = baseline.df.copy()
    df_base["Year"] = df_base["t"] + 1970
    df_base["Scenario"] = "No Intervention (Baseline)"
    
    # Scenario 2: Moderate reform starting 2025
    moderate_reform = CompositeIntervention(
        name="Moderate Reform (2025)",
        start_time=55,  # 1970 + 55 = 2025
        interventions=[
            WageFloor(name="Wage protection", start_time=55, min_wage_ratio=0.9),
            TaxProgressivity(name="Tax reform", start_time=55, revenue_boost=0.03),
        ],
    )
    result_moderate = engine.run_intervention(baseline, moderate_reform)
    df_moderate = result_moderate.df.copy()
    df_moderate["Year"] = df_moderate["t"] + 1970
    df_moderate["Scenario"] = "Moderate Reform (2025)"
    
    # Scenario 3: Comprehensive reform starting 2025
    comprehensive_reform = CompositeIntervention(
        name="Comprehensive Reform (2025)",
        start_time=55,
        interventions=[
            EliteCap(name="Elite opportunity management", start_time=55, max_elite_ratio=1.2),
            WageFloor(name="Strong wage protection", start_time=55, min_wage_ratio=0.9),
            WageBoost(name="Productivity sharing", start_time=55, boost_rate=0.015),
            TaxProgressivity(name="Progressive taxation", start_time=55, revenue_boost=0.05),
            InstitutionalReform(name="Democratic renewal", start_time=55, legitimacy_boost=0.02),
        ],
    )
    result_comprehensive = engine.run_intervention(baseline, comprehensive_reform)
    df_comprehensive = result_comprehensive.df.copy()
    df_comprehensive["Year"] = df_comprehensive["t"] + 1970
    df_comprehensive["Scenario"] = "Comprehensive Reform (2025)"
    
    # Combine all scenarios
    df_combined = pd.concat([df_base, df_moderate, df_comprehensive])
    
    # Filter to relevant years
    df_combined = df_combined[df_combined["Year"] >= 1970]
    
    # Create chart
    chart = alt.Chart(df_combined).mark_line(strokeWidth=2.5).encode(
        x=alt.X("Year:Q", title="Year", scale=alt.Scale(domain=[1970, 2050])),
        y=alt.Y("psi:Q", title="Political Stress Index (PSI)"),
        color=alt.Color("Scenario:N", 
                       scale=alt.Scale(range=["#e41a1c", "#ff7f00", "#4daf4a"])),
        strokeDash=alt.StrokeDash("Scenario:N", legend=None),
    ).properties(
        width=700,
        height=400,
        title="US Political Stress: Projected Trajectories Under Different Scenarios"
    )
    
    # Add vertical line at 2025
    intervention_line = alt.Chart(pd.DataFrame({"x": [2025]})).mark_rule(
        strokeDash=[5, 5], color="gray"
    ).encode(x="x:Q")
    
    annotation = alt.Chart(pd.DataFrame({
        "x": [2027], "y": [0.1], "text": ["Reform begins"]
    })).mark_text(align="left", fontSize=12, color="gray").encode(
        x="x:Q", y="y:Q", text="text:N"
    )
    
    # Historical marker
    today_line = alt.Chart(pd.DataFrame({"x": [2026]})).mark_rule(
        color="black", strokeWidth=1
    ).encode(x="x:Q")
    
    today_annotation = alt.Chart(pd.DataFrame({
        "x": [2026], "y": [0.4], "text": ["Today"]
    })).mark_text(align="center", fontSize=11, color="black", dy=-10).encode(
        x="x:Q", y="y:Q", text="text:N"
    )
    
    final_chart = (chart + intervention_line + annotation + today_line + today_annotation).configure_axis(
        labelFontSize=12,
        titleFontSize=14,
    ).configure_legend(
        titleFontSize=14,
        labelFontSize=12,
    ).configure_title(
        fontSize=16,
    )
    
    final_chart.save(str(OUTPUT_DIR / "us_psi_projections.png"), scale_factor=2)
    print(f"Saved: {OUTPUT_DIR / 'us_psi_projections.png'}")


def create_rome_counterfactual_chart():
    """Create chart showing Roman Republic counterfactuals."""
    # Parameters calibrated to late Roman Republic
    params = SDTParams(
        r_max=0.012,
        K_0=1.0,
        beta=0.9,
        mu=0.3,  # High extraction
        alpha=0.008,
        delta_e=0.015,
        gamma=2.2,
        eta=1.4,
        rho=0.15,
        sigma=0.08,
        epsilon=0.08,  # High elite burden
        lambda_psi=0.07,
        theta_w=0.9,
        theta_e=1.4,  # Intra-elite competition dominant
        theta_s=0.8,
        psi_decay=0.012,
    )
    
    model = SDTModel(params)
    engine = CounterfactualEngine(model)
    
    # Initial conditions ~130 BCE
    initial_conditions = {
        "N": 0.75,
        "E": 0.15,  # Already significant elite class
        "W": 0.85,  # Wages already under pressure
        "S": 0.9,
        "psi": 0.2,  # Some existing tension
    }
    
    # Run baseline (130 BCE to 30 BCE, 100 years to civil wars)
    baseline = engine.run_baseline(
        initial_conditions=initial_conditions,
        time_span=(0, 120),
        dt=0.5,
    )
    
    # Scenario 1: Baseline (actual history approximation)
    df_base = baseline.df.copy()
    df_base["Year"] = df_base["t"] - 130  # -130 BCE to -10 BCE
    df_base["Scenario"] = "Actual History (Approx.)"
    
    # Scenario 2: Gracchi reforms succeed (land reform + elite management)
    gracchi_success = CompositeIntervention(
        name="Gracchi Reforms Succeed",
        start_time=0,  # 130 BCE
        interventions=[
            WageFloor(name="Land redistribution", start_time=0, min_wage_ratio=0.9),
            EliteCap(name="Limits on latifundia", start_time=0, max_elite_ratio=1.2),
        ],
    )
    result_gracchi = engine.run_intervention(baseline, gracchi_success)
    df_gracchi = result_gracchi.df.copy()
    df_gracchi["Year"] = df_gracchi["t"] - 130
    df_gracchi["Scenario"] = "Gracchi Reforms Succeed"
    
    # Scenario 3: Earlier comprehensive reform (150 BCE)
    # Need to start earlier, so adjust simulation
    baseline_early = engine.run_baseline(
        initial_conditions={
            "N": 0.65,
            "E": 0.1,
            "W": 0.95,
            "S": 0.95,
            "psi": 0.1,
        },
        time_span=(0, 140),  # 150 BCE to 10 BCE
        dt=0.5,
    )
    early_reform = CompositeIntervention(
        name="Early Reform (150 BCE)",
        start_time=0,
        interventions=[
            WageFloor(name="Early land reform", start_time=0, min_wage_ratio=0.9),
            EliteCap(name="Sumptuary laws enforced", start_time=0, max_elite_ratio=1.15),
            TaxProgressivity(name="Tributary reform", start_time=0, revenue_boost=0.03),
        ],
    )
    result_early = engine.run_intervention(baseline_early, early_reform)
    df_early = result_early.df.copy()
    df_early["Year"] = df_early["t"] - 150
    df_early["Scenario"] = "Early Reform (150 BCE)"
    
    # Combine
    df_combined = pd.concat([df_base, df_gracchi, df_early])
    
    # Create chart
    chart = alt.Chart(df_combined).mark_line(strokeWidth=2.5).encode(
        x=alt.X("Year:Q", title="Year (BCE, negative)", 
               scale=alt.Scale(domain=[-150, -10])),
        y=alt.Y("psi:Q", title="Political Stress Index (PSI)"),
        color=alt.Color("Scenario:N",
                       scale=alt.Scale(range=["#e41a1c", "#377eb8", "#4daf4a"])),
        strokeDash=alt.StrokeDash("Scenario:N", legend=None),
    ).properties(
        width=700,
        height=400,
        title="Roman Republic: What If Reform Had Succeeded?"
    )
    
    # Add markers for key events
    events = pd.DataFrame({
        "Year": [-133, -91, -49],
        "Event": ["Gracchi", "Social War", "Caesar"],
    })
    
    event_rules = alt.Chart(events).mark_rule(
        strokeDash=[3, 3], color="gray"
    ).encode(x="Year:Q")
    
    event_labels = alt.Chart(events).mark_text(
        angle=270, align="left", dy=-5, fontSize=10, color="gray"
    ).encode(
        x="Year:Q",
        y=alt.value(0),
        text="Event:N"
    )
    
    final_chart = (chart + event_rules + event_labels).configure_axis(
        labelFontSize=12,
        titleFontSize=14,
    ).configure_legend(
        titleFontSize=14,
        labelFontSize=12,
        orient="bottom",
    ).configure_title(
        fontSize=16,
    )
    
    final_chart.save(str(OUTPUT_DIR / "rome_counterfactual.png"), scale_factor=2)
    print(f"Saved: {OUTPUT_DIR / 'rome_counterfactual.png'}")


def create_intervention_components_chart(engine, baseline):
    """Create chart showing how individual components contribute to reform success."""
    # Test individual components
    components = [
        ("Elite Cap Alone", EliteCap(name="Elite cap", start_time=50, max_elite_ratio=1.3)),
        ("Wage Floor Alone", WageFloor(name="Wage floor", start_time=50, min_wage_ratio=0.85)),
        ("Tax Reform Alone", TaxProgressivity(name="Tax", start_time=50, revenue_boost=0.05)),
        ("All Three Combined", CompositeIntervention(
            name="Combined",
            start_time=50,
            interventions=[
                EliteCap(name="Elite cap", start_time=50, max_elite_ratio=1.3),
                WageFloor(name="Wage floor", start_time=50, min_wage_ratio=0.85),
                TaxProgressivity(name="Tax", start_time=50, revenue_boost=0.05),
            ],
        )),
    ]
    
    data = []
    for name, intervention in components:
        result = engine.run_intervention(baseline, intervention)
        comparison = OutcomeComparison(baseline, result)
        
        data.append({
            "Component": name,
            "PSI Reduction": comparison.psi_peak_reduction(),
            "Wage Improvement": comparison.wage_improvement(),
            "State Improvement": comparison.state_improvement(),
        })
    
    df = pd.DataFrame(data)
    
    # Reshape for grouped bar chart
    df_melted = df.melt(
        id_vars=["Component"],
        value_vars=["PSI Reduction", "Wage Improvement", "State Improvement"],
        var_name="Metric",
        value_name="Improvement"
    )
    
    chart = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X("Component:N", title="Intervention", sort=list(df["Component"])),
        y=alt.Y("Improvement:Q", title="Improvement (%)",
               axis=alt.Axis(format=".0%")),
        color=alt.Color("Metric:N",
                       scale=alt.Scale(range=["#377eb8", "#4daf4a", "#984ea3"])),
        xOffset="Metric:N",
    ).properties(
        width=600,
        height=400,
        title="Intervention Components: Individual vs. Combined Effects"
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14,
    ).configure_legend(
        titleFontSize=14,
        labelFontSize=12,
    ).configure_title(
        fontSize=16,
    )
    
    chart.save(str(OUTPUT_DIR / "intervention_components.png"), scale_factor=2)
    print(f"Saved: {OUTPUT_DIR / 'intervention_components.png'}")


def main():
    print("Generating policy essay visualizations...")
    print("=" * 60)
    
    # Run baseline simulation
    engine, baseline, model = run_baseline_simulation()
    print(f"Baseline PSI peak: {baseline.psi_peak:.3f} at t={baseline.psi_peak_time:.1f}")
    
    # Generate all charts
    create_trajectory_comparison_chart(engine, baseline)
    create_policy_effectiveness_chart(engine, baseline)
    create_timing_window_chart(engine, baseline)
    create_intervention_components_chart(engine, baseline)
    create_us_projection_chart()
    create_rome_counterfactual_chart()
    
    print("=" * 60)
    print("All charts generated successfully!")


if __name__ == "__main__":
    main()
