"""Analysis tools for cliodynamics simulations.

This module provides tools for analyzing simulation results:
- Sensitivity analysis (Sobol indices, correlation-based)
- Parameter importance ranking
- Variance decomposition

Example:
    >>> from cliodynamics.analysis import SensitivityAnalyzer
    >>> from cliodynamics.models import SDTModel
    >>>
    >>> analyzer = SensitivityAnalyzer(
    ...     model=SDTModel(),
    ...     parameter_bounds={
    ...         'r_max': (0.01, 0.03),
    ...         'alpha': (0.003, 0.007),
    ...     }
    ... )
    >>> results = analyzer.sobol_analysis(
    ...     initial_conditions={...},
    ...     time_span=(0, 100),
    ...     target_variable='psi'
    ... )
    >>> print(results.summary())
"""

from cliodynamics.analysis.sensitivity import SensitivityAnalyzer, SensitivityResults

__all__ = ["SensitivityAnalyzer", "SensitivityResults"]
