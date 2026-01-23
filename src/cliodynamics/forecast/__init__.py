"""Instability forecasting pipeline for cliodynamics models.

This module provides tools for generating probabilistic forecasts of
societal instability using calibrated SDT models, including:

- **Forecaster**: Main class for generating forecasts with uncertainty
- **Scenarios**: Policy interventions and external shock modeling
- **Uncertainty**: Uncertainty quantification and propagation

Example:
    >>> from cliodynamics.forecast import Forecaster, ForecastResult
    >>> from cliodynamics.forecast.scenarios import create_standard_scenarios
    >>> from cliodynamics.models import SDTModel, SDTParams
    >>>
    >>> # Create and calibrate model
    >>> params = SDTParams(r_max=0.02, K_0=1.0)
    >>> model = SDTModel(params)
    >>>
    >>> # Initialize forecaster
    >>> forecaster = Forecaster(model, n_ensemble=100)
    >>>
    >>> # Define current state
    >>> current_state = {
    ...     'N': 0.8,   # Population at 80% of carrying capacity
    ...     'E': 0.15,  # Elite overproduction
    ...     'W': 0.85,  # Depressed wages
    ...     'S': 0.9,   # Fiscal stress
    ...     'psi': 0.3  # Rising instability
    ... }
    >>>
    >>> # Generate baseline forecast
    >>> forecast = forecaster.predict(
    ...     current_state=current_state,
    ...     horizon_years=20,
    ...     confidence_level=0.90
    ... )
    >>>
    >>> print(f"Peak probability: {forecast.peak_probability:.1%}")
    >>> print(f"Final PSI: {forecast.mean['psi'].iloc[-1]:.3f}")
    >>>
    >>> # Scenario comparison
    >>> scenarios = create_standard_scenarios()
    >>> forecast_with_scenarios = forecaster.predict(
    ...     current_state=current_state,
    ...     horizon_years=20,
    ...     scenarios=['baseline', 'wealth_pump_off', 'reform_package'],
    ...     scenario_modifiers={
    ...         'wealth_pump_off': {'mu': 0.05},
    ...         'reform_package': {'mu': 0.08, 'alpha': 0.003}
    ...     }
    ... )

References:
    Turchin, P. (2016). Ages of Discord. Beresta Books.
    Turchin, P. & Nefedov, S. (2009). Secular Cycles. Princeton.
"""

from cliodynamics.forecast.forecaster import Forecaster, ForecastResult
from cliodynamics.forecast.scenarios import (
    Scenario,
    ScenarioManager,
    create_shock_scenario,
    create_standard_scenarios,
)
from cliodynamics.forecast.uncertainty import (
    UncertaintyEstimate,
    UncertaintyQuantifier,
    combine_uncertainties,
    compute_ensemble_statistics,
    generate_initial_condition_samples,
    generate_parameter_samples,
)

__all__ = [
    # Core forecasting
    "Forecaster",
    "ForecastResult",
    # Scenarios
    "Scenario",
    "ScenarioManager",
    "create_standard_scenarios",
    "create_shock_scenario",
    # Uncertainty
    "UncertaintyQuantifier",
    "UncertaintyEstimate",
    "combine_uncertainties",
    "compute_ensemble_statistics",
    "generate_parameter_samples",
    "generate_initial_condition_samples",
]
