"""Tests for Structural-Demographic Theory models.

Tests cover:
- SDTParams validation and defaults
- SDTState conversions
- SDTModel equation implementations
- System integration and sanity checks
"""

import numpy as np
from numpy.testing import assert_allclose

from cliodynamics.models import SDTModel, SDTParams, SDTState


class TestSDTParams:
    """Tests for SDTParams dataclass."""

    def test_default_params_valid(self):
        """Default parameters should pass validation."""
        params = SDTParams()
        assert params.is_valid()
        assert len(params.validate()) == 0

    def test_default_values(self):
        """Check default parameter values are reasonable."""
        params = SDTParams()

        # Growth rates
        assert params.r_max == 0.02
        assert params.K_0 == 1.0

        # Elite parameters
        assert params.alpha == 0.005
        assert params.delta_e == 0.02

        # Reference values
        assert params.W_0 == 1.0
        assert params.E_0 == 0.1
        assert params.S_0 == 1.0

    def test_custom_params(self):
        """Custom parameters should be stored correctly."""
        params = SDTParams(r_max=0.03, K_0=2.0, mu=0.25)
        assert params.r_max == 0.03
        assert params.K_0 == 2.0
        assert params.mu == 0.25

    def test_invalid_negative_r_max(self):
        """Negative r_max should fail validation."""
        params = SDTParams(r_max=-0.01)
        assert not params.is_valid()
        errors = params.validate()
        assert any("r_max" in e for e in errors)

    def test_invalid_zero_carrying_capacity(self):
        """Zero carrying capacity should fail validation."""
        params = SDTParams(K_0=0)
        assert not params.is_valid()
        errors = params.validate()
        assert any("K_0" in e for e in errors)

    def test_invalid_mu_range(self):
        """Extraction rate outside [0,1] should fail validation."""
        params = SDTParams(mu=1.5)
        assert not params.is_valid()
        errors = params.validate()
        assert any("mu" in e for e in errors)

        params = SDTParams(mu=-0.1)
        assert not params.is_valid()

    def test_invalid_negative_rate(self):
        """Negative rates should fail validation."""
        params = SDTParams(alpha=-0.01)
        errors = params.validate()
        assert any("alpha" in e for e in errors)

    def test_invalid_zero_elasticity(self):
        """Zero elasticity parameters should fail validation."""
        params = SDTParams(beta=0)
        errors = params.validate()
        assert any("beta" in e for e in errors)


class TestSDTState:
    """Tests for SDTState dataclass."""

    def test_default_state(self):
        """Default state should have reasonable initial values."""
        state = SDTState()
        assert state.N == 0.5
        assert state.E == 0.05
        assert state.W == 1.0
        assert state.S == 1.0
        assert state.psi == 0.0

    def test_to_array(self):
        """State should convert to array correctly."""
        state = SDTState(N=0.6, E=0.08, W=0.9, S=0.8, psi=0.1)
        arr = state.to_array()
        assert arr == [0.6, 0.08, 0.9, 0.8, 0.1]

    def test_from_array(self):
        """State should be reconstructed from array."""
        arr = [0.7, 0.1, 0.85, 0.75, 0.2]
        state = SDTState.from_array(arr)
        assert state.N == 0.7
        assert state.E == 0.1
        assert state.W == 0.85
        assert state.S == 0.75
        assert state.psi == 0.2

    def test_roundtrip(self):
        """Array conversion should round-trip correctly."""
        original = SDTState(N=0.55, E=0.06, W=0.95, S=0.85, psi=0.05)
        reconstructed = SDTState.from_array(original.to_array())
        assert original.N == reconstructed.N
        assert original.E == reconstructed.E
        assert original.W == reconstructed.W
        assert original.S == reconstructed.S
        assert original.psi == reconstructed.psi


class TestSDTModelPopulation:
    """Tests for population dynamics equation."""

    def test_logistic_growth_at_low_population(self):
        """Population should grow when N << K."""
        model = SDTModel()
        # Low population, normal wages -> positive growth
        dN_dt = model.population_dynamics(N=0.1, W=1.0, t=0)
        assert dN_dt > 0

    def test_logistic_saturation(self):
        """Population growth should slow near carrying capacity."""
        model = SDTModel()
        # Near K -> growth approaches zero
        dN_dt = model.population_dynamics(N=0.99, W=1.0, t=0)
        assert dN_dt > 0
        assert dN_dt < 0.01  # Very slow growth

    def test_at_carrying_capacity(self):
        """Population should be stable at K."""
        model = SDTModel()
        dN_dt = model.population_dynamics(N=1.0, W=1.0, t=0)
        assert_allclose(dN_dt, 0.0, atol=1e-10)

    def test_wage_effect_on_growth(self):
        """Higher wages should increase population growth."""
        model = SDTModel()
        dN_low = model.population_dynamics(N=0.5, W=0.5, t=0)
        dN_high = model.population_dynamics(N=0.5, W=1.5, t=0)
        assert dN_high > dN_low

    def test_low_wages_reduce_growth(self):
        """Very low wages should dramatically reduce growth."""
        model = SDTModel()
        dN_normal = model.population_dynamics(N=0.5, W=1.0, t=0)
        dN_low = model.population_dynamics(N=0.5, W=0.1, t=0)
        assert dN_low < dN_normal / 2


class TestSDTModelElite:
    """Tests for elite dynamics equation."""

    def test_elite_growth_with_surplus(self):
        """Elites should grow when surplus is available."""
        model = SDTModel()
        # Low wages = high surplus for elite accumulation
        dE_dt = model.elite_dynamics(E=0.05, N=0.5, W=0.5, t=0)
        assert dE_dt > 0

    def test_elite_decay_without_surplus(self):
        """Elite population should decline when surplus is minimal."""
        model = SDTModel()
        # Very high wages = low surplus, elite decay dominates
        # With high wages and small population, surplus factor approaches 0
        params = SDTParams(mu=0.0)  # No extraction
        model = SDTModel(params)
        dE_dt = model.elite_dynamics(E=0.1, N=0.1, W=2.0, t=0)
        # Should be negative due to delta_e * E decay
        assert dE_dt < 0

    def test_upward_mobility_scales_with_population(self):
        """Larger population should support faster elite growth."""
        model = SDTModel()
        dE_small = model.elite_dynamics(E=0.05, N=0.3, W=0.8, t=0)
        dE_large = model.elite_dynamics(E=0.05, N=0.7, W=0.8, t=0)
        assert dE_large > dE_small


class TestSDTModelWages:
    """Tests for wage dynamics equation."""

    def test_wages_rise_with_labor_scarcity(self):
        """Wages should rise when population is low."""
        model = SDTModel()
        # Low N = tight labor market
        dW_dt = model.wage_dynamics(W=1.0, N=0.3, E=0.1, t=0)
        assert dW_dt > 0

    def test_wages_fall_with_overpopulation(self):
        """Wages should fall when population exceeds capacity."""
        model = SDTModel()
        # High N = excess labor supply
        dW_dt = model.wage_dynamics(W=1.0, N=1.2, E=0.1, t=0)
        assert dW_dt < 0

    def test_elite_extraction_depresses_wages(self):
        """More elites should reduce wages."""
        model = SDTModel()
        dW_few = model.wage_dynamics(W=1.0, N=0.5, E=0.05, t=0)
        dW_many = model.wage_dynamics(W=1.0, N=0.5, E=0.2, t=0)
        assert dW_many < dW_few

    def test_wage_equilibrium(self):
        """Wages should stabilize when effects balance."""
        params = SDTParams(gamma=2.0, eta=1.0, K_0=1.0, E_0=0.1)
        model = SDTModel(params)
        # Find N where labor effect = extraction effect
        # At N=0.5, E=0.1: labor_effect = 2*(1-0.5) = 1.0
        # extraction_effect = 1.0*(0.1/0.1 - 1) = 0
        # So dW > 0
        dW_dt = model.wage_dynamics(W=1.0, N=0.5, E=0.1, t=0)
        assert dW_dt > 0  # Wages rising toward equilibrium


class TestSDTModelState:
    """Tests for state fiscal health dynamics."""

    def test_state_growth_with_prosperity(self):
        """State should grow with high output and few elites."""
        model = SDTModel()
        # High wages, decent population, few elites
        dS_dt = model.state_dynamics(S=1.0, N=0.5, E=0.05, W=1.2, t=0)
        assert dS_dt > 0

    def test_state_decline_with_crisis(self):
        """State should decline under fiscal pressure."""
        model = SDTModel()
        # Low wages (low revenue), many elites (high burden)
        dS_dt = model.state_dynamics(S=0.5, N=0.5, E=0.3, W=0.3, t=0)
        assert dS_dt < 0

    def test_elite_burden_effect(self):
        """More elites should strain state resources."""
        model = SDTModel()
        dS_few = model.state_dynamics(S=1.0, N=0.5, E=0.05, W=1.0, t=0)
        dS_many = model.state_dynamics(S=1.0, N=0.5, E=0.3, W=1.0, t=0)
        assert dS_few > dS_many


class TestSDTModelInstability:
    """Tests for Political Stress Index dynamics."""

    def test_psi_rises_with_immiseration(self):
        """Instability should rise with low wages."""
        model = SDTModel()
        # Low wages = popular immiseration
        dpsi_dt = model.instability_dynamics(psi=0.1, E=0.1, W=0.5, S=1.0, t=0)
        assert dpsi_dt > 0

    def test_psi_rises_with_elite_overproduction(self):
        """Instability should rise with elite overproduction."""
        model = SDTModel()
        # High E = intra-elite competition
        dpsi_dt = model.instability_dynamics(psi=0.1, E=0.3, W=1.0, S=1.0, t=0)
        assert dpsi_dt > 0

    def test_psi_rises_with_state_weakness(self):
        """Instability should rise with weak state."""
        model = SDTModel()
        # Low S = fiscal crisis
        dpsi_dt = model.instability_dynamics(psi=0.1, E=0.1, W=1.0, S=0.3, t=0)
        assert dpsi_dt > 0

    def test_psi_decays_in_stability(self):
        """Instability should decay when conditions are good."""
        model = SDTModel()
        # High wages, few elites, strong state -> decay
        # With baseline conditions, only decay term remains
        dpsi_dt = model.instability_dynamics(psi=0.5, E=0.05, W=1.5, S=1.5, t=0)
        # All stress terms should be zero (conditions better than baseline)
        # Only decay term: -psi_decay * psi
        assert dpsi_dt < 0

    def test_psi_zero_at_baseline_conditions(self):
        """PSI should be near-stable at baseline with zero initial stress."""
        model = SDTModel()
        # At baseline: W=W_0, E=E_0, S=S_0, psi=0
        dpsi_dt = model.instability_dynamics(psi=0.0, E=0.1, W=1.0, S=1.0, t=0)
        # No stress accumulation (all at baseline), no decay (psi=0)
        assert_allclose(dpsi_dt, 0.0, atol=1e-10)


class TestSDTModelSystem:
    """Tests for combined ODE system."""

    def test_system_returns_correct_shape(self):
        """System should return 5-element array."""
        model = SDTModel()
        y = [0.5, 0.1, 1.0, 1.0, 0.0]
        dy = model.system(y, t=0)
        assert len(dy) == 5
        assert isinstance(dy, np.ndarray)

    def test_system_matches_individual_equations(self):
        """System output should match individual equation calls."""
        model = SDTModel()
        N, E, W, S, psi = 0.5, 0.08, 0.9, 0.85, 0.1
        y = [N, E, W, S, psi]

        dy = model.system(y, t=0)

        assert_allclose(dy[0], model.population_dynamics(N, W, t=0))
        assert_allclose(dy[1], model.elite_dynamics(E, N, W, t=0))
        assert_allclose(dy[2], model.wage_dynamics(W, N, E, t=0))
        assert_allclose(dy[3], model.state_dynamics(S, N, E, W, t=0))
        assert_allclose(dy[4], model.instability_dynamics(psi, E, W, S, t=0))

    def test_system_handles_near_zero_values(self):
        """System should handle near-zero values without errors."""
        model = SDTModel()
        y = [1e-6, 1e-6, 1e-6, 1e-6, 0.0]
        dy = model.system(y, t=0)
        assert np.all(np.isfinite(dy))

    def test_get_state_conversion(self):
        """get_state should correctly convert array to SDTState."""
        model = SDTModel()
        y = [0.6, 0.1, 0.8, 0.9, 0.15]
        state = model.get_state(y)
        assert state.N == 0.6
        assert state.E == 0.1
        assert state.W == 0.8
        assert state.S == 0.9
        assert state.psi == 0.15


class TestSDTModelIntegration:
    """Integration tests for model behavior over time."""

    def test_odeint_compatible(self):
        """Model should integrate with scipy.integrate.odeint."""
        from scipy.integrate import odeint

        model = SDTModel()
        y0 = [0.5, 0.05, 1.0, 1.0, 0.0]
        t = np.linspace(0, 100, 100)

        solution = odeint(model.system, y0, t)

        assert solution.shape == (100, 5)
        assert np.all(np.isfinite(solution))

    def test_population_converges(self):
        """Population should converge toward carrying capacity."""
        from scipy.integrate import odeint

        params = SDTParams(r_max=0.05, alpha=0.0, eta=0.0)  # Simplify dynamics
        model = SDTModel(params)
        y0 = [0.2, 0.05, 1.0, 1.0, 0.0]
        t = np.linspace(0, 500, 1000)

        solution = odeint(model.system, y0, t)

        # Population should approach K_0 = 1.0
        final_N = solution[-1, 0]
        assert 0.8 < final_N < 1.2

    def test_instability_cycle(self):
        """Model should exhibit instability dynamics."""
        from scipy.integrate import odeint

        model = SDTModel()
        # Start with slight perturbation
        y0 = [0.6, 0.12, 0.9, 0.9, 0.0]
        t = np.linspace(0, 200, 2000)

        solution = odeint(model.system, y0, t)

        # PSI should rise at some point due to stressed conditions
        psi_max = np.max(solution[:, 4])
        assert psi_max > 0.01  # Some instability develops

    def test_variables_remain_positive_short_term(self):
        """State variables should remain positive over short integrations.

        Note: The default parameters are not calibrated for stability.
        This test uses parameters that produce more stable dynamics
        to verify the numerical integration works correctly.
        """
        from scipy.integrate import odeint

        # Use parameters that produce more stable dynamics
        # Lower gamma reduces wage sensitivity, higher eta adds damping
        params = SDTParams(
            r_max=0.02,
            gamma=0.5,  # Reduced wage sensitivity to labor market
            eta=0.5,  # Moderate elite extraction
            lambda_psi=0.02,  # Slower instability accumulation
        )
        model = SDTModel(params)

        # Start near equilibrium
        y0 = [0.8, 0.1, 1.0, 1.0, 0.0]
        t = np.linspace(0, 20, 200)

        solution = odeint(model.system, y0, t)

        # All variables should stay positive (or non-negative for psi)
        assert np.all(solution[:, :4] > 0)  # N, E, W, S positive
        assert np.all(solution[:, 4] >= 0)  # psi non-negative


class TestSDTModelEquilibrium:
    """Tests for equilibrium analysis methods."""

    def test_carrying_capacity(self):
        """Carrying capacity method should return K_0."""
        params = SDTParams(K_0=2.0)
        model = SDTModel(params)
        assert model.carrying_capacity(W=1.0) == 2.0

    def test_equilibrium_wages_at_balance(self):
        """Equilibrium wages should be identified when effects balance."""
        model = SDTModel()
        # This is a simplified check - the method indicates direction
        w_eq = model.equilibrium_wages(N=0.5, E=0.1)
        assert w_eq > 0  # Should return positive value
