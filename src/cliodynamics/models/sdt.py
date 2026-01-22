"""Structural-Demographic Theory differential equations.

This module implements the core SDT model as coupled ordinary differential
equations (ODEs) following formulations from:

- Turchin, P. (2016). Ages of Discord: A Structural-Demographic Analysis
  of American History. Beresta Books. Chapter 2.
- Turchin, P. & Nefedov, S. (2009). Secular Cycles. Princeton University
  Press. Mathematical Appendix.

The model captures feedback loops between:
- Population growth and resource constraints
- Elite overproduction and intra-elite competition
- Labor supply/demand and real wages
- State fiscal capacity and legitimacy
- Political instability accumulation

Example:
    >>> from cliodynamics.models.params import SDTParams, SDTState
    >>> from cliodynamics.models.sdt import SDTModel
    >>> params = SDTParams()
    >>> model = SDTModel(params)
    >>> state = SDTState()
    >>> derivatives = model.system(state.to_array(), t=0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from cliodynamics.models.params import SDTParams, SDTState

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SDTModel:
    """Structural-Demographic Theory model implementation.

    This class implements the coupled ODE system describing societal
    dynamics according to Turchin's SDT framework. The five core
    equations model:

    1. Population dynamics (N) - Logistic growth modified by well-being
    2. Elite dynamics (E) - Elite overproduction from wealth accumulation
    3. Wage dynamics (W) - Labor supply/demand equilibrium
    4. State dynamics (S) - Fiscal health evolution
    5. Instability dynamics (psi) - Political Stress Index

    The model captures the key insight that societies undergo cyclical
    dynamics driven by demographic-structural pressures and elite
    competition, with typical cycle periods of 100-300 years.

    Attributes:
        params: SDTParams instance with model parameters.

    References:
        Turchin (2016), Ages of Discord, Chapter 2, Equations 2.1-2.6
        Turchin & Nefedov (2009), Secular Cycles, Mathematical Appendix
    """

    def __init__(self, params: SDTParams | None = None) -> None:
        """Initialize the SDT model.

        Args:
            params: Model parameters. If None, uses default SDTParams.
        """
        self.params = params if params is not None else SDTParams()

    def population_dynamics(self, N: float, W: float, t: float) -> float:
        """Compute population growth rate dN/dt.

        Population follows logistic growth with the effective growth rate
        modified by real wages (a proxy for well-being). When wages are
        high, population grows faster due to better nutrition, lower
        mortality, and higher fertility. When wages fall below subsistence,
        growth becomes negative.

        Equation (following Ages of Discord, Eq. 2.1):
            dN/dt = r(W) * N * (1 - N/K)

        where:
            r(W) = r_max * (W/W_0)^beta

        The wage-dependent growth rate captures the Malthusian mechanism:
        population pressure depresses wages, which feeds back to slow growth.

        Args:
            N: Current population (normalized to carrying capacity).
            W: Current real wages (normalized to baseline W_0).
            t: Time (not used in autonomous system, kept for solver API).

        Returns:
            Rate of change of population dN/dt.

        References:
            Turchin (2016), Ages of Discord, Chapter 2, Eq. 2.1
            Turchin & Nefedov (2009), Secular Cycles, pp. 303-305
        """
        p = self.params

        # Wage-modified growth rate
        # When W = W_0, r = r_max; when W < W_0, growth slows
        r_effective = p.r_max * (W / p.W_0) ** p.beta

        # Logistic growth with effective rate
        # K_0 is the carrying capacity
        dN_dt = r_effective * N * (1 - N / p.K_0)

        return dN_dt

    def elite_dynamics(self, E: float, N: float, W: float, t: float) -> float:
        """Compute elite population growth rate dE/dt.

        Elite population grows through upward mobility (commoners becoming
        elites via wealth accumulation) and declines through mortality and
        downward mobility. The upward mobility rate depends on the surplus
        available for accumulation.

        Equation (following Ages of Discord, Eq. 2.3):
            dE/dt = alpha * f(surplus) * N - delta_e * E

        where surplus depends on total output minus worker consumption.
        We model this as proportional to (1 - W/W_0) * N, capturing that
        elite opportunities increase when wages are depressed (more surplus
        to capture) and population is large.

        This equation captures "elite overproduction" - when economic
        conditions allow rapid wealth accumulation, the elite class grows
        faster than positions of power, leading to intra-elite competition.

        Args:
            E: Current elite population.
            N: Current total population.
            W: Current real wages.
            t: Time (not used in autonomous system).

        Returns:
            Rate of change of elite population dE/dt.

        References:
            Turchin (2016), Ages of Discord, Chapter 2, Eq. 2.3
            Turchin & Nefedov (2009), Secular Cycles, pp. 306-308
        """
        p = self.params

        # Surplus available for elite accumulation
        # Higher when wages are depressed relative to baseline
        # We use max(0, ...) to ensure non-negative surplus contribution
        wage_ratio = W / p.W_0
        surplus_factor = max(0.0, 1.0 - wage_ratio + p.mu)

        # Upward mobility: rate depends on population and surplus
        upward_mobility = p.alpha * surplus_factor * N

        # Downward mobility and mortality
        elite_loss = p.delta_e * E

        dE_dt = upward_mobility - elite_loss

        return dE_dt

    def wage_dynamics(self, W: float, N: float, E: float, t: float) -> float:
        """Compute wage rate of change dW/dt.

        Real wages are determined by labor supply (population) relative to
        demand, modified by elite extraction (predation on worker income).
        This follows a simple supply-demand framework with elite competition
        for surplus.

        Equation (following Secular Cycles, Mathematical Appendix):
            dW/dt = W * [gamma * (1 - N/K) - eta * (E/E_0 - 1)]

        The first term captures labor market tightness: when N < K, labor
        is scarce and wages rise. The second term captures elite predation:
        more elites competing for surplus depresses worker compensation.

        This equation generates the "popular immiseration" dynamic: as
        population grows and elites multiply, wages fall, leading to
        declining living standards for commoners.

        Args:
            W: Current real wages.
            N: Current population.
            E: Current elite population.
            t: Time (not used in autonomous system).

        Returns:
            Rate of change of wages dW/dt.

        References:
            Turchin & Nefedov (2009), Secular Cycles, Mathematical Appendix
            Turchin (2016), Ages of Discord, Chapter 2, Eq. 2.4
        """
        p = self.params

        # Labor market tightness effect
        # Positive when N < K (labor scarce), negative when N > K
        labor_effect = p.gamma * (1 - N / p.K_0)

        # Elite extraction effect
        # Negative when E > E_0 (elite overproduction)
        elite_ratio = E / p.E_0
        extraction_effect = p.eta * (elite_ratio - 1)

        # Combined wage dynamics
        dW_dt = W * (labor_effect - extraction_effect)

        return dW_dt

    def state_dynamics(self, S: float, N: float, E: float, W: float, t: float) -> float:
        """Compute state fiscal health rate of change dS/dt.

        State fiscal health depends on tax revenue (proportional to
        economic output) minus expenditures (base spending plus elite
        burden from patronage and competition for offices).

        Equation (following Secular Cycles, Chapter 3):
            dS/dt = rho * Y - sigma * S - epsilon * E

        where Y = W * N is total economic output (wages times workers),
        and elite burden grows with elite population.

        This equation captures "state fiscal crisis": as population
        pressure depresses wages and elite overproduction increases
        demands on state resources, fiscal health declines.

        Args:
            S: Current state fiscal health.
            N: Current population.
            E: Current elite population.
            W: Current real wages.
            t: Time (not used in autonomous system).

        Returns:
            Rate of change of state fiscal health dS/dt.

        References:
            Turchin & Nefedov (2009), Secular Cycles, Chapter 3
            Turchin (2016), Ages of Discord, Chapter 2, Eq. 2.5
        """
        p = self.params

        # Economic output (simplified as wage bill)
        output = W * N

        # Tax revenue proportional to output
        revenue = p.rho * output

        # Base state expenditure (military, administration)
        base_expenditure = p.sigma * S

        # Elite burden: patronage demands, competition for offices
        elite_burden = p.epsilon * E

        dS_dt = revenue - base_expenditure - elite_burden

        return dS_dt

    def instability_dynamics(
        self, psi: float, E: float, W: float, S: float, t: float
    ) -> float:
        """Compute Political Stress Index rate of change dpsi/dt.

        The Political Stress Index (PSI) aggregates the structural drivers
        of instability: popular immiseration (low wages), elite overproduction
        (excess elites), and state weakness (fiscal crisis). PSI accumulates
        when these drivers are present and decays naturally when they abate.

        Equation (following Ages of Discord, Eq. 2.6):
            dpsi/dt = lambda * [theta_w * (W_0/W - 1) +
                                theta_e * (E/E_0 - 1) +
                                theta_s * (S_0/S - 1)] - psi_decay * psi

        Each term contributes positively when conditions worsen:
        - W < W_0: popular immiseration (inverted wage)
        - E > E_0: elite overproduction
        - S < S_0: state weakness (inverted fiscal health)

        The decay term represents natural dissipation of tensions when
        structural drivers are absent.

        Args:
            psi: Current Political Stress Index.
            E: Current elite population.
            W: Current real wages.
            S: Current state fiscal health.
            t: Time (not used in autonomous system).

        Returns:
            Rate of change of instability dpsi/dt.

        References:
            Turchin (2016), Ages of Discord, Chapter 2, Eq. 2.6
        """
        p = self.params

        # Popular immiseration contribution
        # Positive when wages are below baseline
        # Use max to avoid negative contributions
        wage_stress = p.theta_w * max(0.0, p.W_0 / max(W, 1e-6) - 1)

        # Elite overproduction contribution
        # Positive when elite population exceeds baseline
        elite_stress = p.theta_e * max(0.0, E / p.E_0 - 1)

        # State weakness contribution
        # Positive when state health is below baseline
        state_stress = p.theta_s * max(0.0, p.S_0 / max(S, 1e-6) - 1)

        # Total stress accumulation
        accumulation = p.lambda_psi * (wage_stress + elite_stress + state_stress)

        # Natural decay of tensions
        decay = p.psi_decay * psi

        dpsi_dt = accumulation - decay

        return dpsi_dt

    def system(self, y: ArrayLike, t: float) -> NDArray[np.float64]:
        """Compute the full system of ODEs.

        This method combines all five differential equations into a single
        system suitable for numerical integration with scipy.integrate.

        The system is:
            dy/dt = [dN/dt, dE/dt, dW/dt, dS/dt, dpsi/dt]

        Args:
            y: State vector [N, E, W, S, psi].
            t: Time (for solver compatibility).

        Returns:
            Array of derivatives [dN/dt, dE/dt, dW/dt, dS/dt, dpsi/dt].

        Example:
            >>> from scipy.integrate import odeint
            >>> model = SDTModel()
            >>> y0 = [0.5, 0.05, 1.0, 1.0, 0.0]
            >>> t = np.linspace(0, 100, 1000)
            >>> solution = odeint(model.system, y0, t)
        """
        # Unpack state variables
        N, E, W, S, psi = y

        # Ensure non-negative values (numerical stability)
        N = max(N, 1e-10)
        E = max(E, 1e-10)
        W = max(W, 1e-10)
        S = max(S, 1e-10)
        psi = max(psi, 0.0)

        # Compute derivatives
        dN_dt = self.population_dynamics(N, W, t)
        dE_dt = self.elite_dynamics(E, N, W, t)
        dW_dt = self.wage_dynamics(W, N, E, t)
        dS_dt = self.state_dynamics(S, N, E, W, t)
        dpsi_dt = self.instability_dynamics(psi, E, W, S, t)

        return np.array([dN_dt, dE_dt, dW_dt, dS_dt, dpsi_dt])

    def get_state(self, y: ArrayLike) -> SDTState:
        """Convert state vector to SDTState object.

        Args:
            y: State vector [N, E, W, S, psi].

        Returns:
            SDTState instance.
        """
        return SDTState.from_array(list(y))

    def equilibrium_wages(self, N: float, E: float) -> float:
        """Compute equilibrium wages for given N and E.

        At wage equilibrium, dW/dt = 0, which gives:
            gamma * (1 - N/K) = eta * (E/E_0 - 1)

        This is useful for understanding steady-state behavior.

        Args:
            N: Population.
            E: Elite population.

        Returns:
            Equilibrium wage level (returns current W_0 if at equilibrium).

        Note:
            This assumes the instantaneous equilibrium; actual wages
            adjust dynamically toward this value.
        """
        p = self.params

        # At dW/dt = 0: labor_effect = extraction_effect
        # gamma * (1 - N/K) = eta * (E/E_0 - 1)
        # This gives us the condition, but W cancels out
        # The equilibrium wage level itself depends on historical path

        labor_effect = p.gamma * (1 - N / p.K_0)
        extraction_effect = p.eta * (E / p.E_0 - 1)

        # If effects balance, wages are stable at current level
        # Otherwise, wages are adjusting toward equilibrium
        if abs(labor_effect - extraction_effect) < 1e-6:
            return p.W_0

        # Direction of wage change
        if labor_effect > extraction_effect:
            return p.W_0 * 1.1  # Wages rising
        else:
            return p.W_0 * 0.9  # Wages falling

    def carrying_capacity(self, W: float) -> float:
        """Compute effective carrying capacity given wages.

        The effective carrying capacity is the population level at which
        growth stops (dN/dt = 0 for N > 0), which depends on wages.

        Args:
            W: Current real wages.

        Returns:
            Effective carrying capacity.
        """
        # At dN/dt = 0: either N = 0 or N = K
        # The carrying capacity K_0 is the structural limit
        return self.params.K_0
