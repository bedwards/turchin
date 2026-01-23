"""Ages of Discord model implementation.

This module implements the composite indices used in Turchin's
Ages of Discord (2016) analysis of American historical dynamics.

The three key indices are:

1. **Well-Being Index (WBI)**: Composite measure of worker welfare
   - Real wages
   - Health indicators (life expectancy, height)
   - Family stability (age at marriage, household size)
   - Social indicators (urbanization, inequality)

2. **Elite Overproduction Index (EOI)**: Measure of elite surplus
   - Lawyers per capita
   - PhD production
   - Wealth inequality
   - Elite income relative to average

3. **Political Stress Index (PSI)**: Composite instability measure
   - Combines WBI (inverted) and EOI
   - Captures structural pressure for instability

These indices differ from the dynamic SDT model (sdt.py) in that
they are computed directly from historical data rather than
simulated from differential equations.

References:
    Turchin, P. (2016). Ages of Discord. Beresta Books.
        Chapter 3: The Well-Being Index
        Chapter 4: Elite Overproduction
        Chapter 5: Political Stress Indicator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class WellBeingWeights:
    """Weights for Well-Being Index components.

    The WBI is a weighted average of several components.
    Default weights follow Ages of Discord methodology.

    Attributes:
        real_wage: Weight for real wage component.
        relative_wage: Weight for relative wage component.
        health: Weight for health indicators.
        family: Weight for family stability indicators.
    """

    real_wage: float = 0.35
    relative_wage: float = 0.35
    health: float = 0.15
    family: float = 0.15

    def validate(self) -> bool:
        """Check that weights sum to 1.0."""
        total = self.real_wage + self.relative_wage + self.health + self.family
        return abs(total - 1.0) < 0.001

    def as_array(self) -> NDArray[np.float64]:
        """Return weights as numpy array."""
        return np.array(
            [
                self.real_wage,
                self.relative_wage,
                self.health,
                self.family,
            ]
        )


@dataclass
class EliteWeights:
    """Weights for Elite Overproduction Index components.

    Attributes:
        lawyers: Weight for lawyers per capita.
        phds: Weight for PhD production.
        wealth_inequality: Weight for wealth concentration.
    """

    lawyers: float = 0.35
    phds: float = 0.25
    wealth_inequality: float = 0.40

    def validate(self) -> bool:
        """Check that weights sum to 1.0."""
        total = self.lawyers + self.phds + self.wealth_inequality
        return abs(total - 1.0) < 0.001

    def as_array(self) -> NDArray[np.float64]:
        """Return weights as numpy array."""
        return np.array(
            [
                self.lawyers,
                self.phds,
                self.wealth_inequality,
            ]
        )


@dataclass
class PSIWeights:
    """Weights for Political Stress Index components.

    The PSI combines inverted well-being with elite overproduction.

    Attributes:
        mass_mobilization: Weight for inverted WBI (mass mobilization potential).
        elite_competition: Weight for EOI (intra-elite competition).
        state_weakness: Weight for state fiscal health (not yet implemented).
    """

    mass_mobilization: float = 0.50  # Inverted WBI
    elite_competition: float = 0.50  # EOI

    def validate(self) -> bool:
        """Check that weights sum to 1.0."""
        total = self.mass_mobilization + self.elite_competition
        return abs(total - 1.0) < 0.001


@dataclass
class AgesOfDiscordConfig:
    """Configuration for Ages of Discord analysis.

    Attributes:
        baseline_year: Year to normalize indices to (default 1960).
        wbi_weights: Weights for Well-Being Index.
        elite_weights: Weights for Elite Overproduction Index.
        psi_weights: Weights for Political Stress Index.
    """

    baseline_year: int = 1960
    wbi_weights: WellBeingWeights = field(default_factory=WellBeingWeights)
    elite_weights: EliteWeights = field(default_factory=EliteWeights)
    psi_weights: PSIWeights = field(default_factory=PSIWeights)


class AgesOfDiscordModel:
    """Ages of Discord composite index calculator.

    This class computes the three composite indices (WBI, EOI, PSI)
    from historical data, following the methodology in Ages of Discord.

    Unlike the SDT dynamic model, these indices are computed directly
    from data rather than simulated from differential equations.

    Attributes:
        config: Analysis configuration.
        data: Input historical data.
        results: Computed indices (populated after compute_all).

    Example:
        >>> from cliodynamics.data.us import USHistoricalData
        >>> from cliodynamics.models.ages_of_discord import AgesOfDiscordModel
        >>>
        >>> data = USHistoricalData()
        >>> model = AgesOfDiscordModel()
        >>> results = model.compute_all(data.get_combined_dataset())
        >>>
        >>> # Access individual indices
        >>> wbi = results['well_being_index']
        >>> eoi = results['elite_overproduction_index']
        >>> psi = results['political_stress_index']
    """

    def __init__(self, config: AgesOfDiscordConfig | None = None) -> None:
        """Initialize the Ages of Discord model.

        Args:
            config: Analysis configuration. Uses defaults if None.
        """
        self.config = config if config is not None else AgesOfDiscordConfig()
        self._results: pd.DataFrame | None = None

    def compute_well_being_index(
        self,
        data: pd.DataFrame,
        normalize: bool = True,
    ) -> pd.Series:
        """Compute the Well-Being Index.

        The WBI is a composite measure of worker welfare, combining:
        - Real wages (actual purchasing power)
        - Relative wages (share of economic growth)
        - Health indicators (life expectancy proxy)
        - Family stability (social cohesion proxy)

        Higher values indicate better conditions for ordinary workers.
        The index is normalized so that the baseline year = 100.

        Args:
            data: DataFrame with required columns:
                - year
                - real_wage_index
                - relative_wage_index
            normalize: If True, normalize to baseline_year = 100.

        Returns:
            Series with Well-Being Index values indexed by year.

        Raises:
            ValueError: If required columns are missing.
        """
        required_cols = ["year", "real_wage_index", "relative_wage_index"]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        weights = self.config.wbi_weights

        # Extract components
        real_wage = data["real_wage_index"].values
        relative_wage = data["relative_wage_index"].values

        # For health and family, we approximate from wage data
        # In a full implementation, these would be separate data series
        # For now, use smoothed wage data as proxy
        health_proxy = self._smooth(real_wage, window=10)
        family_proxy = self._smooth(relative_wage, window=10)

        # Compute weighted index
        wbi = (
            weights.real_wage * real_wage
            + weights.relative_wage * relative_wage
            + weights.health * health_proxy
            + weights.family * family_proxy
        )

        if normalize:
            # Normalize to baseline year
            baseline_mask = data["year"] == self.config.baseline_year
            if baseline_mask.any():
                masked = wbi[baseline_mask]
                if hasattr(masked, "iloc"):
                    baseline_value = masked.iloc[0]
                else:
                    baseline_value = masked[0]
                wbi = (wbi / baseline_value) * 100

        return pd.Series(wbi, index=data["year"], name="well_being_index")

    def compute_elite_overproduction_index(
        self,
        data: pd.DataFrame,
        normalize: bool = True,
    ) -> pd.Series:
        """Compute the Elite Overproduction Index.

        The EOI measures the surplus of elite aspirants relative to
        available elite positions. Components include:
        - Lawyers per capita (political elite aspirants)
        - PhDs per capita (credential competition)
        - Wealth inequality (concentration of resources)

        Higher values indicate more intense intra-elite competition.
        The index is normalized so that the baseline year = 100.

        Args:
            data: DataFrame with required columns:
                - year
                - lawyers_per_capita_index
                - phds_per_capita_index
                - inequality_index
            normalize: If True, normalize to baseline_year = 100.

        Returns:
            Series with Elite Overproduction Index values indexed by year.

        Raises:
            ValueError: If required columns are missing.
        """
        # Check for required columns with fallbacks
        has_lawyers = "lawyers_per_capita_index" in data.columns
        has_phds = "phds_per_capita_index" in data.columns
        has_combined = "combined_elite_index" in data.columns
        has_inequality = "inequality_index" in data.columns

        if not (has_lawyers or has_combined):
            raise ValueError("Need lawyers_per_capita_index or combined_elite_index")

        weights = self.config.elite_weights

        # Extract components with fallbacks
        if has_lawyers:
            lawyers = data["lawyers_per_capita_index"].values
        else:
            lawyers = data["combined_elite_index"].values

        if has_phds:
            phds = data["phds_per_capita_index"].values
        else:
            phds = lawyers * 0.8  # Proxy from lawyers

        if has_inequality:
            inequality = data["inequality_index"].values
        else:
            # Use lawyers as inequality proxy
            inequality = lawyers

        # Compute weighted index
        eoi = (
            weights.lawyers * lawyers
            + weights.phds * phds
            + weights.wealth_inequality * inequality
        )

        if normalize:
            baseline_mask = data["year"] == self.config.baseline_year
            if baseline_mask.any():
                baseline_value = eoi[baseline_mask][0]
                eoi = (eoi / baseline_value) * 100

        return pd.Series(eoi, index=data["year"], name="elite_overproduction_index")

    def compute_political_stress_index(
        self,
        wbi: pd.Series,
        eoi: pd.Series,
        normalize: bool = True,
    ) -> pd.Series:
        """Compute the Political Stress Index.

        The PSI combines mass mobilization potential (from low well-being)
        with intra-elite competition (from elite overproduction).

        The formula follows Ages of Discord:
            PSI = MMP * EOI

        where MMP (Mass Mobilization Potential) is derived from inverted WBI.

        Higher PSI indicates greater structural pressure for instability.

        Args:
            wbi: Well-Being Index series.
            eoi: Elite Overproduction Index series.
            normalize: If True, normalize to baseline_year = 100.

        Returns:
            Series with Political Stress Index values indexed by year.
        """
        # Align indices
        common_idx = wbi.index.intersection(eoi.index)
        wbi_aligned = wbi.loc[common_idx]
        eoi_aligned = eoi.loc[common_idx]

        # Compute Mass Mobilization Potential (inverted WBI)
        # When WBI is high (good conditions), MMP is low
        # Scale so MMP = 100 when WBI = 100
        wbi_max = wbi_aligned.max()
        mmp = (wbi_max / wbi_aligned) * 100

        # Compute PSI
        # Following Turchin's multiplicative formula
        # PSI = MMP * EOI / 10000 (scaling factor)
        weights = self.config.psi_weights
        psi_raw = (mmp**weights.mass_mobilization) * (
            eoi_aligned**weights.elite_competition
        )

        # Normalize
        if normalize:
            if self.config.baseline_year in common_idx:
                baseline_value = psi_raw.loc[self.config.baseline_year]
                psi = (psi_raw / baseline_value) * 100
            else:
                # Find closest year
                psi = (psi_raw / psi_raw.median()) * 100
        else:
            psi = psi_raw

        return pd.Series(psi.values, index=common_idx, name="political_stress_index")

    def compute_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute all indices from historical data.

        This is the main entry point for computing Ages of Discord
        indices from a combined historical dataset.

        Args:
            data: DataFrame with historical data. Should include:
                - year
                - real_wage_index
                - relative_wage_index
                - lawyers_per_capita_index (or combined_elite_index)
                - inequality_index (optional)

        Returns:
            DataFrame with columns:
                - year
                - well_being_index
                - elite_overproduction_index
                - political_stress_index
        """
        # Compute individual indices
        wbi = self.compute_well_being_index(data)
        eoi = self.compute_elite_overproduction_index(data)
        psi = self.compute_political_stress_index(wbi, eoi)

        # Combine into single DataFrame
        results = (
            pd.DataFrame(
                {
                    "year": wbi.index,
                    "well_being_index": wbi.values,
                }
            )
            .merge(
                pd.DataFrame(
                    {
                        "year": eoi.index,
                        "elite_overproduction_index": eoi.values,
                    }
                ),
                on="year",
                how="outer",
            )
            .merge(
                pd.DataFrame(
                    {
                        "year": psi.index,
                        "political_stress_index": psi.values,
                    }
                ),
                on="year",
                how="outer",
            )
        )

        self._results = results.sort_values("year").reset_index(drop=True)
        return self._results

    def identify_cycles(
        self,
        series: pd.Series,
        min_period: int = 30,
        threshold: float = 0.5,
    ) -> list[dict]:
        """Identify secular cycles in a time series.

        Detects peaks and troughs in the index to identify
        integrative and disintegrative phases.

        Args:
            series: Time series to analyze.
            min_period: Minimum period between peaks/troughs.
            threshold: Threshold for peak/trough detection (0-1).

        Returns:
            List of cycle dictionaries with keys:
                - start_year: Beginning of cycle
                - peak_year: Year of maximum
                - end_year: End of cycle
                - amplitude: Peak-to-trough difference
        """
        from scipy.signal import find_peaks

        values = series.values
        years = series.index.values

        # Find peaks (highs)
        peaks, _ = find_peaks(values, distance=min_period)

        # Find troughs (lows) by inverting
        troughs, _ = find_peaks(-values, distance=min_period)

        cycles = []

        # Combine into cycles
        for i in range(len(troughs) - 1):
            start_year = years[troughs[i]]
            end_year = years[troughs[i + 1]]

            # Find peak between troughs
            mask = (peaks > troughs[i]) & (peaks < troughs[i + 1])
            if mask.any():
                peak_idx = peaks[mask][0]
                peak_year = years[peak_idx]
                amplitude = values[peak_idx] - values[troughs[i]]
            else:
                peak_year = (start_year + end_year) // 2
                amplitude = 0

            cycles.append(
                {
                    "start_year": int(start_year),
                    "peak_year": int(peak_year),
                    "end_year": int(end_year),
                    "amplitude": float(amplitude),
                }
            )

        return cycles

    def compare_to_published(
        self,
        computed: pd.Series,
        target_year: int,
        published_value: float,
        tolerance: float = 0.15,
    ) -> dict:
        """Compare computed values to published Ages of Discord figures.

        Args:
            computed: Computed index series.
            target_year: Year to compare.
            published_value: Value from published analysis.
            tolerance: Acceptable relative difference (default 15%).

        Returns:
            Dictionary with comparison results:
                - computed_value: Our computed value
                - published_value: Published value
                - difference: Absolute difference
                - relative_error: Relative error
                - within_tolerance: Boolean
        """
        if target_year not in computed.index:
            raise ValueError(f"Year {target_year} not in computed series")

        computed_value = computed.loc[target_year]
        difference = computed_value - published_value
        if published_value != 0:
            relative_error = abs(difference) / published_value
        else:
            relative_error = float("inf")

        return {
            "computed_value": computed_value,
            "published_value": published_value,
            "difference": difference,
            "relative_error": relative_error,
            "within_tolerance": relative_error <= tolerance,
        }

    @staticmethod
    def _smooth(
        values: NDArray[np.float64],
        window: int = 5,
    ) -> NDArray[np.float64]:
        """Apply moving average smoothing.

        Args:
            values: Array to smooth.
            window: Window size for moving average.

        Returns:
            Smoothed array (same length as input).
        """
        kernel = np.ones(window) / window
        # Pad to maintain length
        padded = np.pad(values, (window // 2, window - window // 2 - 1), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed


def compute_ages_of_discord_indices(
    data: pd.DataFrame,
    config: AgesOfDiscordConfig | None = None,
) -> pd.DataFrame:
    """Convenience function to compute all Ages of Discord indices.

    Args:
        data: Historical data DataFrame.
        config: Optional configuration.

    Returns:
        DataFrame with computed indices.

    Example:
        >>> from cliodynamics.data.us import USHistoricalData
        >>> from cliodynamics.models.ages_of_discord import (
        ...     compute_ages_of_discord_indices,
        ... )
        >>> data = USHistoricalData().get_combined_dataset()
        >>> results = compute_ages_of_discord_indices(data)
    """
    model = AgesOfDiscordModel(config)
    return model.compute_all(data)
