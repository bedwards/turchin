"""Tests for policy interventions.

Tests cover:
- Intervention activation logic
- State modification by each intervention type
- Derivative modification by each intervention type
- Composite interventions
"""

import numpy as np
import pytest

from cliodynamics.models.params import SDTParams
from cliodynamics.policy.interventions import (
    CompositeIntervention,
    EliteCap,
    ElitePurge,
    FiscalAusterity,
    FiscalStimulus,
    FrontierExpansion,
    InstitutionalReform,
    MigrationControl,
    TaxProgressivity,
    WageBoost,
    WageFloor,
    create_elite_management_policy,
    create_welfare_policy,
)


class TestInterventionActivation:
    """Tests for intervention is_active() logic."""

    def test_before_start(self):
        """Intervention inactive before start_time."""
        intervention = EliteCap(name="test", start_time=50)
        assert not intervention.is_active(0)
        assert not intervention.is_active(49)

    def test_at_start(self):
        """Intervention active at start_time."""
        intervention = EliteCap(name="test", start_time=50)
        assert intervention.is_active(50)

    def test_after_start(self):
        """Intervention active after start_time."""
        intervention = EliteCap(name="test", start_time=50)
        assert intervention.is_active(100)
        assert intervention.is_active(1000)

    def test_with_end_time(self):
        """Intervention inactive after end_time."""
        intervention = EliteCap(name="test", start_time=50, end_time=100)
        assert intervention.is_active(50)
        assert intervention.is_active(99)
        assert not intervention.is_active(100)
        assert not intervention.is_active(200)


class TestEliteCap:
    """Tests for EliteCap intervention."""

    @pytest.fixture
    def params(self):
        return SDTParams(E_0=0.1)

    @pytest.fixture
    def intervention(self):
        return EliteCap(
            name="Elite cap",
            start_time=0,
            max_elite_ratio=1.5,
        )

    def test_cap_not_exceeded(self, intervention, params):
        """State unchanged when elite below cap."""
        state = np.array([0.5, 0.1, 1.0, 1.0, 0.1])  # E = 0.1 = E_0
        modified = intervention.modify_state(state, t=10, params=params)
        assert modified[1] == 0.1

    def test_cap_exceeded(self, intervention, params):
        """Elite capped when above maximum."""
        state = np.array([0.5, 0.2, 1.0, 1.0, 0.1])  # E = 0.2 = 2*E_0 > 1.5*E_0
        modified = intervention.modify_state(state, t=10, params=params)
        assert modified[1] == pytest.approx(0.15)  # 1.5 * E_0

    def test_derivative_capped(self, intervention, params):
        """Positive dE/dt zeroed when at or above cap."""
        # Use value clearly above cap (0.16 > 0.15) to avoid floating-point issues
        state = np.array([0.5, 0.16, 1.0, 1.0, 0.1])  # Above cap
        derivatives = np.array([0.01, 0.01, 0.0, 0.0, 0.01])  # dE/dt positive
        modified = intervention.modify_derivatives(
            derivatives, state, t=10, params=params
        )
        assert modified[1] == 0.0  # dE/dt zeroed

    def test_derivative_not_capped_below_cap(self, intervention, params):
        """dE/dt unchanged when below cap."""
        state = np.array([0.5, 0.1, 1.0, 1.0, 0.1])  # Below cap
        derivatives = np.array([0.01, 0.01, 0.0, 0.0, 0.01])
        modified = intervention.modify_derivatives(
            derivatives, state, t=10, params=params
        )
        assert modified[1] == 0.01  # Unchanged

    def test_inactive_when_before_start(self, params):
        """Intervention has no effect before start_time."""
        intervention = EliteCap(name="test", start_time=50, max_elite_ratio=1.5)
        state = np.array([0.5, 0.2, 1.0, 1.0, 0.1])  # Above cap
        modified = intervention.modify_state(state, t=10, params=params)
        assert modified[1] == 0.2  # Unchanged


class TestElitePurge:
    """Tests for ElitePurge intervention."""

    @pytest.fixture
    def params(self):
        return SDTParams(E_0=0.1)

    def test_one_time_reduction(self, params):
        """Elite reduced once at start_time."""
        intervention = ElitePurge(
            name="Purge",
            start_time=50,
            reduction_fraction=0.2,
        )
        state = np.array([0.5, 0.1, 1.0, 1.0, 0.1])

        # First application
        modified = intervention.modify_state(state, t=50, params=params)
        assert modified[1] == pytest.approx(0.08)

        # Second application - should not reduce again
        modified2 = intervention.modify_state(modified, t=51, params=params)
        assert modified2[1] == pytest.approx(0.08)

    def test_invalid_fraction(self):
        """Reject invalid reduction_fraction values."""
        with pytest.raises(ValueError):
            ElitePurge(name="test", start_time=0, reduction_fraction=1.5)
        with pytest.raises(ValueError):
            ElitePurge(name="test", start_time=0, reduction_fraction=-0.1)


class TestWageFloor:
    """Tests for WageFloor intervention."""

    @pytest.fixture
    def params(self):
        return SDTParams(W_0=1.0)

    @pytest.fixture
    def intervention(self):
        return WageFloor(
            name="Wage floor",
            start_time=0,
            min_wage_ratio=0.8,
        )

    def test_floor_not_reached(self, intervention, params):
        """Wages unchanged when above floor."""
        state = np.array([0.5, 0.1, 1.0, 1.0, 0.1])  # W = 1.0 > 0.8
        modified = intervention.modify_state(state, t=10, params=params)
        assert modified[2] == 1.0

    def test_floor_enforced(self, intervention, params):
        """Wages raised to floor when below."""
        state = np.array([0.5, 0.1, 0.5, 1.0, 0.1])  # W = 0.5 < 0.8
        modified = intervention.modify_state(state, t=10, params=params)
        assert modified[2] == pytest.approx(0.8)

    def test_derivative_floor(self, intervention, params):
        """Negative dW/dt zeroed at or below floor."""
        # Use value clearly below floor to avoid floating-point issues
        state = np.array([0.5, 0.1, 0.79, 1.0, 0.1])  # Below floor
        derivatives = np.array([0.01, 0.01, -0.05, 0.0, 0.01])  # dW/dt negative
        modified = intervention.modify_derivatives(
            derivatives, state, t=10, params=params
        )
        assert modified[2] == 0.0


class TestWageBoost:
    """Tests for WageBoost intervention."""

    @pytest.fixture
    def params(self):
        return SDTParams(W_0=1.0)

    def test_boost_rate_applied(self, params):
        """Wage growth boosted by specified rate."""
        intervention = WageBoost(
            name="Boost",
            start_time=0,
            boost_rate=0.02,
        )
        state = np.array([0.5, 0.1, 1.0, 1.0, 0.1])
        derivatives = np.array([0.01, 0.01, 0.0, 0.0, 0.01])

        modified = intervention.modify_derivatives(
            derivatives, state, t=10, params=params
        )
        assert modified[2] == pytest.approx(0.02)  # 0.02 * 1.0


class TestTaxProgressivity:
    """Tests for TaxProgressivity intervention."""

    @pytest.fixture
    def params(self):
        return SDTParams(rho=0.2)

    def test_revenue_boost(self, params):
        """State revenue increased."""
        intervention = TaxProgressivity(
            name="Tax",
            start_time=0,
            revenue_boost=0.05,
            elite_drain=0.01,
        )
        state = np.array([0.5, 0.1, 1.0, 1.0, 0.1])  # N=0.5, E=0.1, W=1.0
        derivatives = np.array([0.01, 0.01, 0.0, 0.0, 0.01])

        modified = intervention.modify_derivatives(
            derivatives, state, t=10, params=params
        )

        # Revenue boost: 0.05 * W * N = 0.05 * 1.0 * 0.5 = 0.025
        assert modified[3] == pytest.approx(0.025)
        # Elite drain: 0.01 * E = 0.01 * 0.1 = 0.001
        assert modified[1] == pytest.approx(0.01 - 0.001)


class TestFiscalAusterity:
    """Tests for FiscalAusterity intervention."""

    @pytest.fixture
    def params(self):
        return SDTParams(sigma=0.1)

    def test_spending_reduction(self, params):
        """State spending reduced."""
        intervention = FiscalAusterity(
            name="Austerity",
            start_time=0,
            spending_reduction=0.2,
        )
        state = np.array([0.5, 0.1, 1.0, 1.0, 0.1])  # S = 1.0
        derivatives = np.array([0.01, 0.01, 0.0, 0.0, 0.01])

        modified = intervention.modify_derivatives(
            derivatives, state, t=10, params=params
        )

        # Expenditure saved: 0.2 * 0.1 * 1.0 = 0.02
        assert modified[3] == pytest.approx(0.02)


class TestFiscalStimulus:
    """Tests for FiscalStimulus intervention."""

    @pytest.fixture
    def params(self):
        return SDTParams()

    def test_stimulus_effects(self, params):
        """Stimulus boosts wages, reduces PSI, spends state resources."""
        intervention = FiscalStimulus(
            name="Stimulus",
            start_time=0,
            wage_boost=0.02,
            psi_reduction=0.01,
            spending_rate=0.1,
        )
        state = np.array([0.5, 0.1, 1.0, 1.0, 0.5])  # S=1.0, psi=0.5
        derivatives = np.array([0.01, 0.01, 0.0, 0.0, 0.01])

        modified = intervention.modify_derivatives(
            derivatives, state, t=10, params=params
        )

        # Wage boost: 0.02 * W = 0.02
        assert modified[2] == pytest.approx(0.02)
        # PSI reduction: -0.01 * psi = -0.005
        assert modified[4] == pytest.approx(0.01 - 0.005)
        # Spending: -0.1 * S = -0.1
        assert modified[3] == pytest.approx(-0.1)


class TestMigrationControl:
    """Tests for MigrationControl intervention."""

    @pytest.fixture
    def params(self):
        return SDTParams()

    def test_population_reduction(self, params):
        """Negative effect reduces population growth."""
        intervention = MigrationControl(
            name="Emigration",
            start_time=0,
            population_effect=-0.01,
        )
        state = np.array([0.5, 0.1, 1.0, 1.0, 0.1])
        derivatives = np.array([0.02, 0.01, 0.0, 0.0, 0.01])

        modified = intervention.modify_derivatives(
            derivatives, state, t=10, params=params
        )

        # Population effect: -0.01 * N = -0.005
        assert modified[0] == pytest.approx(0.02 - 0.005)


class TestFrontierExpansion:
    """Tests for FrontierExpansion intervention."""

    @pytest.fixture
    def params(self):
        return SDTParams(K_0=1.0, r_max=0.02)

    def test_expansion_effects(self, params):
        """Expansion boosts capacity and reduces elite competition."""
        intervention = FrontierExpansion(
            name="Expansion",
            start_time=0,
            carrying_capacity_boost=0.1,
            elite_opportunity_boost=0.01,
        )
        # N/K > 0.7 to trigger capacity relief
        state = np.array([0.8, 0.1, 1.0, 1.0, 0.1])
        derivatives = np.array([0.01, 0.01, 0.0, 0.0, 0.01])

        modified = intervention.modify_derivatives(
            derivatives, state, t=10, params=params
        )

        # Capacity relief: 0.1 * r_max * N = 0.1 * 0.02 * 0.8 = 0.0016
        assert modified[0] > 0.01
        # Elite opportunity: -0.01 * E = -0.001
        assert modified[1] == pytest.approx(0.01 - 0.001)
        # Wage boost: 0.01 * W = 0.01
        assert modified[2] == pytest.approx(0.01)


class TestInstitutionalReform:
    """Tests for InstitutionalReform intervention."""

    @pytest.fixture
    def params(self):
        return SDTParams(rho=0.2, epsilon=0.05)

    def test_reform_effects(self, params):
        """Reform reduces instability and improves state efficiency."""
        intervention = InstitutionalReform(
            name="Reform",
            start_time=0,
            legitimacy_boost=0.02,
            efficiency_boost=0.02,
            elite_restraint=0.01,
        )
        state = np.array([0.5, 0.1, 1.0, 1.0, 0.1])  # N=0.5, E=0.1, W=1.0
        derivatives = np.array([0.01, 0.01, 0.0, 0.0, 0.05])  # dpsi/dt = 0.05

        modified = intervention.modify_derivatives(
            derivatives, state, t=10, params=params
        )

        # Legitimacy: dpsi *= (1 - 0.02)
        assert modified[4] == pytest.approx(0.05 * 0.98)
        # Efficiency: 0.02 * rho * W * N = 0.02 * 0.2 * 1.0 * 0.5 = 0.002
        # Elite restraint: 0.01 * epsilon * E = 0.01 * 0.05 * 0.1 = 0.00005
        assert modified[3] == pytest.approx(0.002 + 0.00005)


class TestCompositeIntervention:
    """Tests for CompositeIntervention."""

    @pytest.fixture
    def params(self):
        return SDTParams(W_0=1.0, E_0=0.1)

    def test_composite_applies_all(self, params):
        """Composite applies all constituent interventions."""
        composite = CompositeIntervention(
            name="Package",
            start_time=0,
            interventions=[
                WageFloor(name="Floor", start_time=0, min_wage_ratio=0.8),
                EliteCap(name="Cap", start_time=0, max_elite_ratio=1.5),
            ],
        )

        # State with low wage and high elite
        state = np.array([0.5, 0.2, 0.5, 1.0, 0.1])
        modified = composite.modify_state(state, t=10, params=params)

        # Wage floor enforced
        assert modified[2] == pytest.approx(0.8)
        # Elite cap enforced
        assert modified[1] == pytest.approx(0.15)


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_elite_management_policy(self):
        """Factory creates correct elite management policy."""
        policy = create_elite_management_policy(
            start_time=50,
            cap_ratio=1.5,
            purge_fraction=0.2,
        )
        assert isinstance(policy, CompositeIntervention)
        assert len(policy.interventions) == 2

    def test_create_welfare_policy(self):
        """Factory creates correct welfare policy."""
        policy = create_welfare_policy(
            start_time=50,
            wage_floor=0.8,
            wage_boost=0.01,
        )
        assert isinstance(policy, CompositeIntervention)
        assert len(policy.interventions) == 2
