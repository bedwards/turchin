"""Sensitivity analysis for cliodynamics models.

This module provides tools for understanding which model parameters have
the greatest influence on simulation outputs, using:

1. **Sobol Sensitivity Indices**: Variance-based global sensitivity analysis
   - First-order indices (S1): Individual parameter effects
   - Total-order indices (ST): Including interactions

2. **Parameter Importance Ranking**: Correlation-based ranking
   - Pearson and Spearman correlations
   - Partial rank correlation coefficients (PRCC)

3. **Variance Decomposition**: How much of output variance is explained
   by each parameter

Example:
    >>> from cliodynamics.analysis.sensitivity import SensitivityAnalyzer
    >>> from cliodynamics.models import SDTModel
    >>>
    >>> analyzer = SensitivityAnalyzer(
    ...     model=SDTModel(),
    ...     parameter_bounds={
    ...         'r_max': (0.01, 0.03),
    ...         'alpha': (0.003, 0.007),
    ...         'gamma': (1.0, 3.0),
    ...     }
    ... )
    >>> results = analyzer.sobol_analysis(
    ...     initial_conditions={'N': 0.5, 'E': 0.05, 'W': 1.0, 'S': 1.0, 'psi': 0.0},
    ...     time_span=(0, 100),
    ...     target_variable='psi'
    ... )
    >>> print(results.ranking)  # Parameters ranked by importance

References:
    Saltelli, A. et al. (2008). Global Sensitivity Analysis: The Primer. Wiley.
    Sobol, I.M. (2001). Global sensitivity indices for nonlinear
        mathematical models and their Monte Carlo estimates.
        Mathematics and Computers in Simulation, 55(1-3), 271-280.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from cliodynamics.models import SDTModel


@dataclass
class SensitivityResults:
    """Container for sensitivity analysis results.

    Attributes:
        first_order: First-order Sobol indices (S1) for each parameter.
        total_order: Total-order Sobol indices (ST) for each parameter.
        confidence_intervals: Dict mapping param -> (S1_CI, ST_CI).
        parameter_names: List of parameter names analyzed.
        target_variable: Variable that was analyzed.
        target_time: Time point analyzed.
        n_samples: Number of samples used.
        method: Method used ('sobol', 'correlation', 'prcc').
        metadata: Additional information.
    """

    first_order: dict[str, float]
    total_order: dict[str, float]
    confidence_intervals: dict[str, tuple[tuple[float, float], tuple[float, float]]]
    parameter_names: list[str]
    target_variable: str
    target_time: float
    n_samples: int
    method: str = "sobol"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ranking(self) -> list[tuple[str, float, float]]:
        """Get parameters ranked by total-order sensitivity.

        Returns:
            List of (param_name, S1, ST) tuples sorted by ST descending.
        """
        items = []
        for name in self.parameter_names:
            s1 = self.first_order.get(name, 0.0)
            st = self.total_order.get(name, 0.0)
            items.append((name, s1, st))
        return sorted(items, key=lambda x: abs(x[2]), reverse=True)

    @property
    def interaction_strength(self) -> dict[str, float]:
        """Compute interaction strength for each parameter.

        Interaction strength = ST - S1. Higher values mean the parameter
        has more influence through interactions with other parameters.

        Returns:
            Dict mapping parameter names to interaction strength.
        """
        return {
            name: self.total_order.get(name, 0.0) - self.first_order.get(name, 0.0)
            for name in self.parameter_names
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for easy inspection.

        Returns:
            DataFrame with columns: parameter, S1, ST, S1_lower, S1_upper,
            ST_lower, ST_upper, interaction.
        """
        data = []
        for name in self.parameter_names:
            s1 = self.first_order.get(name, 0.0)
            st = self.total_order.get(name, 0.0)

            if name in self.confidence_intervals:
                s1_ci, st_ci = self.confidence_intervals[name]
                s1_lower, s1_upper = s1_ci
                st_lower, st_upper = st_ci
            else:
                s1_lower = s1_upper = s1
                st_lower = st_upper = st

            data.append(
                {
                    "parameter": name,
                    "S1": s1,
                    "ST": st,
                    "S1_lower": s1_lower,
                    "S1_upper": s1_upper,
                    "ST_lower": st_lower,
                    "ST_upper": st_upper,
                    "interaction": st - s1,
                }
            )

        return pd.DataFrame(data)

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns:
            Formatted summary string.
        """
        lines = [
            f"Sensitivity Analysis Summary ({self.method.upper()})",
            "=" * 50,
            f"Target: {self.target_variable} at t={self.target_time}",
            f"Samples: {self.n_samples}",
            "",
            "Parameter Rankings (by Total-Order Index ST):",
            "-" * 50,
            f"{'Parameter':<15} {'S1':<10} {'ST':<10} {'Interaction':<12}",
            "-" * 50,
        ]

        for name, s1, st in self.ranking:
            interaction = st - s1
            lines.append(f"{name:<15} {s1:<10.4f} {st:<10.4f} {interaction:<12.4f}")

        # Interpretation
        lines.extend(
            [
                "",
                "Interpretation:",
                "  S1 (First-order): Individual effect of parameter",
                "  ST (Total-order): Effect including all interactions",
                "  ST-S1: Strength of interactions with other parameters",
            ]
        )

        # Highlight most important
        if self.ranking:
            top_param = self.ranking[0][0]
            lines.append(f"\nMost influential parameter: {top_param}")

        return "\n".join(lines)


class SensitivityAnalyzer:
    """Analyzer for global sensitivity of SDT models.

    Provides multiple methods for sensitivity analysis:
    - Sobol indices (variance-based, global)
    - Correlation-based (simpler, local)
    - PRCC (partial rank correlation)

    Attributes:
        model: SDT model instance.
        parameter_bounds: Dict mapping param names to (low, high) bounds.
        n_samples: Number of samples for analysis.
        seed: Random seed for reproducibility.

    Example:
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
    """

    def __init__(
        self,
        model: "SDTModel",
        parameter_bounds: dict[str, tuple[float, float]],
        n_samples: int = 1024,
        seed: int | None = None,
    ) -> None:
        """Initialize the sensitivity analyzer.

        Args:
            model: SDT model instance (provides structure).
            parameter_bounds: Dict mapping parameter names to (low, high) bounds.
                Parameters not specified use their default values.
            n_samples: Number of base samples (actual simulations = n_samples * (2k + 2)
                where k is number of parameters for Sobol analysis).
            seed: Random seed.
        """
        self.model = model
        self.parameter_bounds = parameter_bounds
        self.n_samples = n_samples
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.parameter_names = list(parameter_bounds.keys())
        self.n_params = len(self.parameter_names)

    def _generate_sobol_samples(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Generate Sobol quasi-random samples for sensitivity analysis.

        Uses Saltelli's sampling scheme which creates two independent sample
        matrices A and B, plus k matrices where column i of B is replaced
        with column i of A.

        Returns:
            Tuple of (A matrix, B matrix) each of shape (n_samples, n_params).
        """
        # For simplicity, use pseudo-random sampling
        # (True Sobol sequences would be better but require additional dependencies)
        n = self.n_samples
        k = self.n_params

        # Generate two independent sample matrices in [0, 1]
        A = self._rng.random((n, k))
        B = self._rng.random((n, k))

        # Scale to parameter bounds
        for i, name in enumerate(self.parameter_names):
            low, high = self.parameter_bounds[name]
            A[:, i] = low + (high - low) * A[:, i]
            B[:, i] = low + (high - low) * B[:, i]

        return A, B

    def _run_simulations(
        self,
        parameter_matrix: NDArray[np.float64],
        initial_conditions: dict[str, float],
        time_span: tuple[float, float],
        target_variable: str,
        target_time: float,
        dt: float = 1.0,
    ) -> NDArray[np.float64]:
        """Run simulations for each row in parameter matrix.

        Args:
            parameter_matrix: Array of shape (n, k) with parameter values.
            initial_conditions: Initial state dict.
            time_span: Simulation time span.
            target_variable: Variable to extract.
            target_time: Time at which to evaluate.
            dt: Time step.

        Returns:
            Array of output values, one per simulation.
        """
        from dataclasses import fields

        from cliodynamics.models import SDTModel, SDTParams
        from cliodynamics.simulation import Simulator

        n_sims = parameter_matrix.shape[0]
        outputs = np.zeros(n_sims)

        # Get base parameters
        base_params = {
            f.name: getattr(self.model.params, f.name)
            for f in fields(self.model.params)
        }

        for i in range(n_sims):
            # Update parameters
            params_dict = base_params.copy()
            for j, name in enumerate(self.parameter_names):
                params_dict[name] = float(parameter_matrix[i, j])

            try:
                params = SDTParams(**params_dict)
                model = SDTModel(params)
                sim = Simulator(model)

                result = sim.run(
                    initial_conditions=initial_conditions,
                    time_span=time_span,
                    dt=dt,
                    method="RK45",
                )

                # Extract target value
                time_idx = np.argmin(np.abs(result.df["t"].values - target_time))
                outputs[i] = result.df[target_variable].values[time_idx]

            except Exception:
                outputs[i] = np.nan

        return outputs

    def sobol_analysis(
        self,
        initial_conditions: dict[str, float],
        time_span: tuple[float, float],
        target_variable: str = "psi",
        target_time: float | None = None,
        dt: float = 1.0,
        n_bootstrap: int = 100,
    ) -> SensitivityResults:
        """Compute Sobol sensitivity indices.

        Uses Saltelli's method to estimate first-order (S1) and total-order (ST)
        sensitivity indices for each parameter.

        Args:
            initial_conditions: Initial state dict.
            time_span: Simulation time span.
            target_variable: Variable to analyze.
            target_time: Time at which to evaluate (default: end of simulation).
            dt: Time step.
            n_bootstrap: Number of bootstrap samples for confidence intervals.

        Returns:
            SensitivityResults with Sobol indices.

        Note:
            This runs approximately n_samples * (2 * n_params + 2) simulations,
            which can be computationally expensive for large n_samples.
        """
        if target_time is None:
            target_time = time_span[1]

        k = self.n_params
        n = self.n_samples

        # Generate sample matrices
        A, B = self._generate_sobol_samples()

        # Run simulations for A and B
        f_A = self._run_simulations(
            A, initial_conditions, time_span, target_variable, target_time, dt
        )
        f_B = self._run_simulations(
            B, initial_conditions, time_span, target_variable, target_time, dt
        )

        # Run simulations for AB_i matrices (column i of A replaced with B)
        f_AB = np.zeros((n, k))
        for i in range(k):
            AB_i = A.copy()
            AB_i[:, i] = B[:, i]
            f_AB[:, i] = self._run_simulations(
                AB_i, initial_conditions, time_span, target_variable, target_time, dt
            )

        # Run simulations for BA_i matrices (column i of B replaced with A)
        f_BA = np.zeros((n, k))
        for i in range(k):
            BA_i = B.copy()
            BA_i[:, i] = A[:, i]
            f_BA[:, i] = self._run_simulations(
                BA_i, initial_conditions, time_span, target_variable, target_time, dt
            )

        # Remove NaN values
        valid_A = ~np.isnan(f_A)
        valid_B = ~np.isnan(f_B)
        valid_all = valid_A & valid_B
        for i in range(k):
            valid_all &= ~np.isnan(f_AB[:, i]) & ~np.isnan(f_BA[:, i])

        f_A = f_A[valid_all]
        f_B = f_B[valid_all]
        f_AB = f_AB[valid_all, :]
        f_BA = f_BA[valid_all, :]
        n_valid = len(f_A)

        if n_valid < 10:
            raise ValueError(
                f"Too few valid simulations ({n_valid}). Check model stability."
            )

        # Compute variance
        f_all = np.concatenate([f_A, f_B])
        V = np.var(f_all)

        if V < 1e-10:
            # No variance - all outputs are the same
            first_order = {name: 0.0 for name in self.parameter_names}
            total_order = {name: 0.0 for name in self.parameter_names}
            confidence_intervals = {
                name: ((0.0, 0.0), (0.0, 0.0)) for name in self.parameter_names
            }
        else:
            # Compute Sobol indices using Jansen estimator
            first_order = {}
            total_order = {}

            for i, name in enumerate(self.parameter_names):
                # First-order index (S1) - Jansen estimator
                # S1_i = V[E[Y|Xi]] / V[Y]
                # Estimated as: mean(f_B * (f_AB_i - f_A)) / V
                s1 = np.mean(f_B * (f_AB[:, i] - f_A)) / V
                first_order[name] = float(np.clip(s1, 0, 1))

                # Total-order index (ST) - Jansen estimator
                # ST_i = E[(f_A - f_AB_i)^2] / (2V)
                st = np.mean((f_A - f_AB[:, i]) ** 2) / (2 * V)
                total_order[name] = float(np.clip(st, 0, 1))

            # Bootstrap confidence intervals
            confidence_intervals = self._bootstrap_confidence_intervals(
                f_A, f_B, f_AB, f_BA, V, n_bootstrap
            )

        total_sims = n * (2 * k + 2)

        return SensitivityResults(
            first_order=first_order,
            total_order=total_order,
            confidence_intervals=confidence_intervals,
            parameter_names=self.parameter_names,
            target_variable=target_variable,
            target_time=target_time,
            n_samples=n_valid,
            method="sobol",
            metadata={
                "total_simulations": total_sims,
                "valid_simulations": n_valid,
                "variance": float(V),
            },
        )

    def _bootstrap_confidence_intervals(
        self,
        f_A: NDArray[np.float64],
        f_B: NDArray[np.float64],
        f_AB: NDArray[np.float64],
        f_BA: NDArray[np.float64],
        V: float,
        n_bootstrap: int,
    ) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
        """Compute bootstrap confidence intervals for Sobol indices.

        Args:
            f_A, f_B: Output arrays from A and B samples.
            f_AB, f_BA: Output arrays from mixed samples.
            V: Total variance.
            n_bootstrap: Number of bootstrap samples.

        Returns:
            Dict mapping param names to ((S1_low, S1_high), (ST_low, ST_high)).
        """
        n = len(f_A)

        s1_samples = {name: [] for name in self.parameter_names}
        st_samples = {name: [] for name in self.parameter_names}

        for _ in range(n_bootstrap):
            # Bootstrap resample
            indices = self._rng.choice(n, size=n, replace=True)
            f_A_b = f_A[indices]
            f_B_b = f_B[indices]
            f_AB_b = f_AB[indices, :]
            f_all_b = np.concatenate([f_A_b, f_B_b])
            V_b = np.var(f_all_b)

            if V_b < 1e-10:
                continue

            for i, name in enumerate(self.parameter_names):
                s1_b = np.mean(f_B_b * (f_AB_b[:, i] - f_A_b)) / V_b
                st_b = np.mean((f_A_b - f_AB_b[:, i]) ** 2) / (2 * V_b)
                s1_samples[name].append(float(np.clip(s1_b, -0.5, 1.5)))
                st_samples[name].append(float(np.clip(st_b, -0.5, 1.5)))

        # Compute 95% CI
        confidence_intervals = {}
        for name in self.parameter_names:
            if s1_samples[name]:
                s1_arr = np.array(s1_samples[name])
                st_arr = np.array(st_samples[name])
                s1_ci = (
                    float(np.percentile(s1_arr, 2.5)),
                    float(np.percentile(s1_arr, 97.5)),
                )
                st_ci = (
                    float(np.percentile(st_arr, 2.5)),
                    float(np.percentile(st_arr, 97.5)),
                )
            else:
                s1_ci = (0.0, 0.0)
                st_ci = (0.0, 0.0)
            confidence_intervals[name] = (s1_ci, st_ci)

        return confidence_intervals

    def correlation_analysis(
        self,
        initial_conditions: dict[str, float],
        time_span: tuple[float, float],
        target_variable: str = "psi",
        target_time: float | None = None,
        dt: float = 1.0,
        method: str = "pearson",
    ) -> SensitivityResults:
        """Compute correlation-based sensitivity measures.

        Simpler and faster than Sobol analysis, but captures only linear
        relationships (Pearson) or monotonic relationships (Spearman).

        Args:
            initial_conditions: Initial state dict.
            time_span: Simulation time span.
            target_variable: Variable to analyze.
            target_time: Time at which to evaluate.
            dt: Time step.
            method: 'pearson' or 'spearman'.

        Returns:
            SensitivityResults with correlation coefficients as indices.
        """
        if target_time is None:
            target_time = time_span[1]

        # Generate random samples
        samples = np.zeros((self.n_samples, self.n_params))
        for i, name in enumerate(self.parameter_names):
            low, high = self.parameter_bounds[name]
            samples[:, i] = self._rng.uniform(low, high, self.n_samples)

        # Run simulations
        outputs = self._run_simulations(
            samples, initial_conditions, time_span, target_variable, target_time, dt
        )

        # Remove NaN values
        valid = ~np.isnan(outputs)
        samples = samples[valid]
        outputs = outputs[valid]

        # Compute correlations
        first_order = {}
        total_order = {}

        for i, name in enumerate(self.parameter_names):
            param_values = samples[:, i]

            if method == "spearman":
                # Rank correlation
                from scipy.stats import spearmanr

                corr, _ = spearmanr(param_values, outputs)
            else:
                # Pearson correlation
                corr = np.corrcoef(param_values, outputs)[0, 1]

            # Use squared correlation as sensitivity index
            first_order[name] = float(corr**2)
            total_order[name] = float(abs(corr))  # Absolute correlation

        return SensitivityResults(
            first_order=first_order,
            total_order=total_order,
            confidence_intervals={},
            parameter_names=self.parameter_names,
            target_variable=target_variable,
            target_time=target_time,
            n_samples=len(outputs),
            method=f"correlation_{method}",
            metadata={"method": method},
        )

    def prcc_analysis(
        self,
        initial_conditions: dict[str, float],
        time_span: tuple[float, float],
        target_variable: str = "psi",
        target_time: float | None = None,
        dt: float = 1.0,
    ) -> SensitivityResults:
        """Compute Partial Rank Correlation Coefficients (PRCC).

        PRCC measures the relationship between each parameter and the output
        while controlling for all other parameters. Useful for models with
        correlated inputs.

        Args:
            initial_conditions: Initial state dict.
            time_span: Simulation time span.
            target_variable: Variable to analyze.
            target_time: Time at which to evaluate.
            dt: Time step.

        Returns:
            SensitivityResults with PRCC values as indices.
        """
        if target_time is None:
            target_time = time_span[1]

        # Generate Latin Hypercube samples for better space coverage
        samples = self._latin_hypercube_samples(self.n_samples)

        # Scale to parameter bounds
        for i, name in enumerate(self.parameter_names):
            low, high = self.parameter_bounds[name]
            samples[:, i] = low + (high - low) * samples[:, i]

        # Run simulations
        outputs = self._run_simulations(
            samples, initial_conditions, time_span, target_variable, target_time, dt
        )

        # Remove NaN values
        valid = ~np.isnan(outputs)
        samples = samples[valid]
        outputs = outputs[valid]

        if len(outputs) < self.n_params + 2:
            raise ValueError("Not enough valid simulations for PRCC analysis")

        # Compute PRCC using rank transformation
        from scipy.stats import rankdata

        # Rank transform all data
        ranked_samples = np.zeros_like(samples)
        for i in range(self.n_params):
            ranked_samples[:, i] = rankdata(samples[:, i])
        ranked_outputs = rankdata(outputs)

        # Compute partial correlations
        first_order = {}
        total_order = {}

        for i, name in enumerate(self.parameter_names):
            # Get the parameter column and output
            x = ranked_samples[:, i]
            y = ranked_outputs

            # Get all other parameters
            other_indices = [j for j in range(self.n_params) if j != i]
            Z = ranked_samples[:, other_indices]

            # Regress x on Z and y on Z
            if len(other_indices) > 0:
                # Add constant term
                Z_with_const = np.column_stack([np.ones(len(x)), Z])

                # Residuals of x regressed on Z
                try:
                    beta_x = np.linalg.lstsq(Z_with_const, x, rcond=None)[0]
                    resid_x = x - Z_with_const @ beta_x

                    # Residuals of y regressed on Z
                    beta_y = np.linalg.lstsq(Z_with_const, y, rcond=None)[0]
                    resid_y = y - Z_with_const @ beta_y

                    # Correlation of residuals is the partial correlation
                    prcc = np.corrcoef(resid_x, resid_y)[0, 1]
                except Exception:
                    prcc = 0.0
            else:
                prcc = np.corrcoef(x, y)[0, 1]

            first_order[name] = float(prcc**2)
            total_order[name] = float(abs(prcc))

        return SensitivityResults(
            first_order=first_order,
            total_order=total_order,
            confidence_intervals={},
            parameter_names=self.parameter_names,
            target_variable=target_variable,
            target_time=target_time,
            n_samples=len(outputs),
            method="prcc",
            metadata={},
        )

    def _latin_hypercube_samples(self, n: int) -> NDArray[np.float64]:
        """Generate Latin Hypercube samples.

        Provides better coverage of parameter space than random sampling.

        Args:
            n: Number of samples.

        Returns:
            Array of shape (n, n_params) with values in [0, 1].
        """
        k = self.n_params
        samples = np.zeros((n, k))

        for i in range(k):
            # Create n evenly spaced intervals
            intervals = np.arange(n) / n
            # Random point within each interval
            samples[:, i] = intervals + self._rng.uniform(0, 1 / n, n)
            # Shuffle to break correlations between parameters
            self._rng.shuffle(samples[:, i])

        return samples

    def variance_decomposition(
        self,
        initial_conditions: dict[str, float],
        time_span: tuple[float, float],
        target_variable: str = "psi",
        target_time: float | None = None,
        dt: float = 1.0,
    ) -> pd.DataFrame:
        """Decompose output variance by parameter contributions.

        Uses ANOVA-like decomposition to estimate how much of total output
        variance is attributable to each parameter.

        Args:
            initial_conditions: Initial state dict.
            time_span: Simulation time span.
            target_variable: Variable to analyze.
            target_time: Time at which to evaluate.
            dt: Time step.

        Returns:
            DataFrame with columns: parameter, variance_contribution,
            percentage, cumulative_percentage.
        """
        # Run Sobol analysis to get sensitivity indices
        results = self.sobol_analysis(
            initial_conditions=initial_conditions,
            time_span=time_span,
            target_variable=target_variable,
            target_time=target_time,
            dt=dt,
            n_bootstrap=0,  # Skip bootstrap for speed
        )

        total_variance = results.metadata.get("variance", 1.0)

        # First-order indices give individual variance contributions
        data = []
        for name in self.parameter_names:
            s1 = results.first_order.get(name, 0.0)
            var_contrib = s1 * total_variance
            data.append(
                {
                    "parameter": name,
                    "variance_contribution": var_contrib,
                    "percentage": s1 * 100,
                }
            )

        df = pd.DataFrame(data)
        df = df.sort_values("percentage", ascending=False)
        df["cumulative_percentage"] = df["percentage"].cumsum()

        return df


__all__ = ["SensitivityAnalyzer", "SensitivityResults"]
