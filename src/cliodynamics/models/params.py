"""Parameter dataclasses for Structural-Demographic Theory models.

This module defines the parameters used in SDT differential equations,
based on formulations from:
- Turchin, P. (2016). Ages of Discord, Chapter 2
- Turchin & Nefedov (2009). Secular Cycles, Mathematical Appendix
"""

from dataclasses import dataclass


@dataclass
class SDTParams:
    """Parameters for the Structural-Demographic Theory model.

    The SDT model captures the dynamics of population, elites, wages,
    state health, and political instability through coupled ODEs.
    Parameter values should be calibrated to historical data for specific
    societies/time periods.

    Attributes:
        r_max: Maximum population growth rate (per year).
            Represents intrinsic growth rate when resources are abundant.
            Typical range: 0.01-0.03 for pre-industrial societies.
            Source: Ages of Discord, Ch. 2

        K_0: Base carrying capacity (population units).
            Maximum sustainable population under optimal conditions.
            Scales with available resources/technology.
            Source: Secular Cycles, Mathematical Appendix

        beta: Wage sensitivity of population growth.
            How strongly real wages affect birth/death rates.
            Higher values mean population responds more to economic conditions.
            Typical range: 0.5-2.0
            Source: Ages of Discord, Eq. 2.1

        mu: Elite extraction rate (fraction of output).
            Proportion of economic surplus captured by elites.
            Higher values increase elite growth but depress wages.
            Typical range: 0.1-0.3
            Source: Secular Cycles, Ch. 2

        alpha: Elite upward mobility rate (per year).
            Rate at which commoners enter the elite class.
            Depends on wealth accumulation opportunities.
            Typical range: 0.001-0.01
            Source: Ages of Discord, Eq. 2.3

        delta_e: Elite death/downward mobility rate (per year).
            Combined rate of elite mortality and status loss.
            Typical range: 0.01-0.05
            Source: Ages of Discord, Eq. 2.3

        gamma: Labor supply effect on wages.
            Elasticity of wages with respect to labor supply (N/K).
            Higher values mean wages fall faster with overpopulation.
            Typical range: 1.0-3.0
            Source: Secular Cycles, Mathematical Appendix

        eta: Elite competition effect on wages.
            How elite extraction affects worker compensation.
            Higher values mean more elite predation on wages.
            Typical range: 0.5-2.0
            Source: Ages of Discord, Eq. 2.4

        rho: State revenue coefficient.
            Efficiency of state tax collection.
            Typical range: 0.1-0.3
            Source: Secular Cycles, Ch. 3

        sigma: State expenditure rate.
            Base rate of state spending (military, administration).
            Typical range: 0.05-0.2
            Source: Secular Cycles, Ch. 3

        epsilon: Elite burden on state.
            How elite population strains state resources
            (patronage, competition for offices).
            Typical range: 0.01-0.1
            Source: Ages of Discord, Eq. 2.5

        lambda_psi: Instability growth rate.
            Rate at which political stress accumulates.
            Typical range: 0.01-0.1
            Source: Ages of Discord, Eq. 2.6

        theta_w: Wage contribution to instability.
            Weight of popular immiseration (low wages) in PSI.
            Higher values mean economic hardship drives more unrest.
            Typical range: 0.5-2.0
            Source: Ages of Discord, Ch. 2

        theta_e: Elite contribution to instability.
            Weight of elite overproduction in PSI.
            Captures intra-elite competition effects.
            Typical range: 0.5-2.0
            Source: Ages of Discord, Ch. 2

        theta_s: State weakness contribution to instability.
            Weight of fiscal crisis in PSI.
            Typical range: 0.5-2.0
            Source: Ages of Discord, Ch. 2

        psi_decay: Natural decay rate of instability (per year).
            Rate at which tensions dissipate absent driving forces.
            Typical range: 0.01-0.05
            Source: Ages of Discord, Eq. 2.6

        W_0: Baseline/reference wage level.
            Normalization constant for wage dynamics.
            Set to 1.0 for relative wage interpretation.
            Source: Secular Cycles, Mathematical Appendix

        E_0: Baseline/reference elite population.
            Normalization constant for elite dynamics.
            Typically set relative to total population.
            Source: Ages of Discord, Ch. 2

        S_0: Baseline/reference state fiscal health.
            Normalization constant for state dynamics.
            Set to 1.0 for relative interpretation.
            Source: Secular Cycles, Ch. 3
    """

    # Population dynamics parameters
    r_max: float = 0.02
    K_0: float = 1.0
    beta: float = 1.0

    # Elite dynamics parameters
    mu: float = 0.2
    alpha: float = 0.005
    delta_e: float = 0.02

    # Wage dynamics parameters
    gamma: float = 2.0
    eta: float = 1.0

    # State dynamics parameters
    rho: float = 0.2
    sigma: float = 0.1
    epsilon: float = 0.05

    # Instability dynamics parameters
    lambda_psi: float = 0.05
    theta_w: float = 1.0
    theta_e: float = 1.0
    theta_s: float = 1.0
    psi_decay: float = 0.02

    # Reference values for normalization
    W_0: float = 1.0
    E_0: float = 0.1
    S_0: float = 1.0

    def validate(self) -> list[str]:
        """Validate parameter values are physically reasonable.

        Returns:
            List of validation error messages (empty if all valid).
        """
        errors = []

        # Growth rates should be positive
        if self.r_max <= 0:
            errors.append("r_max must be positive")
        if self.K_0 <= 0:
            errors.append("K_0 must be positive")

        # Rates should be non-negative
        for rate_name in [
            "alpha",
            "delta_e",
            "rho",
            "sigma",
            "epsilon",
            "lambda_psi",
            "psi_decay",
        ]:
            if getattr(self, rate_name) < 0:
                errors.append(f"{rate_name} must be non-negative")

        # Elasticities should be positive
        for param_name in ["beta", "gamma", "eta", "theta_w", "theta_e", "theta_s"]:
            if getattr(self, param_name) <= 0:
                errors.append(f"{param_name} must be positive")

        # Extraction rate should be between 0 and 1
        if not 0 <= self.mu <= 1:
            errors.append("mu (extraction rate) must be between 0 and 1")

        # Reference values should be positive
        for ref_name in ["W_0", "E_0", "S_0"]:
            if getattr(self, ref_name) <= 0:
                errors.append(f"{ref_name} must be positive")

        return errors

    def is_valid(self) -> bool:
        """Check if all parameters are valid.

        Returns:
            True if all parameters pass validation.
        """
        return len(self.validate()) == 0


@dataclass
class SDTState:
    """State variables for the SDT model at a point in time.

    Attributes:
        N: Population size (normalized to carrying capacity).
        E: Elite population (absolute or normalized).
        W: Real wages / worker well-being (normalized to W_0).
        S: State fiscal health (normalized to S_0).
        psi: Political Stress Index (dimensionless, 0 = stable).
    """

    N: float = 0.5  # Start at half carrying capacity
    E: float = 0.05  # Small initial elite fraction
    W: float = 1.0  # Wages at baseline
    S: float = 1.0  # State health at baseline
    psi: float = 0.0  # No initial instability

    def to_array(self) -> list[float]:
        """Convert state to array for ODE solver.

        Returns:
            List of state variables [N, E, W, S, psi].
        """
        return [self.N, self.E, self.W, self.S, self.psi]

    @classmethod
    def from_array(cls, y: list[float]) -> "SDTState":
        """Create state from ODE solver array.

        Args:
            y: Array of state variables [N, E, W, S, psi].

        Returns:
            SDTState instance.
        """
        return cls(N=y[0], E=y[1], W=y[2], S=y[3], psi=y[4])
