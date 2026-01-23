"""Tests for the scenarios module."""

from __future__ import annotations

import pytest

from cliodynamics.forecast.scenarios import (
    Scenario,
    ScenarioManager,
    create_shock_scenario,
    create_standard_scenarios,
)


class TestScenario:
    """Tests for the Scenario class."""

    def test_scenario_creation(self) -> None:
        """Test basic Scenario creation."""
        scenario = Scenario(
            name="test_scenario",
            description="A test scenario",
            param_changes={"mu": 0.05},
        )

        assert scenario.name == "test_scenario"
        assert scenario.description == "A test scenario"
        assert scenario.param_changes == {"mu": 0.05}

    def test_scenario_defaults(self) -> None:
        """Test Scenario default values."""
        scenario = Scenario(name="minimal")

        assert scenario.description == ""
        assert scenario.param_changes == {}
        assert scenario.state_changes == {}
        assert scenario.start_year == 0.0
        assert scenario.end_year is None
        assert scenario.ramp_years == 0.0
        assert scenario.probability == 1.0

    def test_get_param_value_no_change(self) -> None:
        """Test get_param_value when param is not in changes."""
        scenario = Scenario(name="test", param_changes={"mu": 0.05})

        # Parameter not in changes should return baseline
        result = scenario.get_param_value("r_max", baseline_value=0.02, t=5)
        assert result == 0.02

    def test_get_param_value_instant_change(self) -> None:
        """Test get_param_value with instant change (no ramping)."""
        scenario = Scenario(
            name="test",
            param_changes={"mu": 0.05},
            start_year=5,
            ramp_years=0,
        )

        # Before start
        assert scenario.get_param_value("mu", baseline_value=0.2, t=3) == 0.2

        # At and after start
        assert scenario.get_param_value("mu", baseline_value=0.2, t=5) == 0.05
        assert scenario.get_param_value("mu", baseline_value=0.2, t=10) == 0.05

    def test_get_param_value_with_ramping(self) -> None:
        """Test get_param_value with gradual ramping."""
        scenario = Scenario(
            name="test",
            param_changes={"mu": 0.0},
            start_year=0,
            ramp_years=10,
        )

        baseline = 0.2

        # At start (t=0)
        result = scenario.get_param_value("mu", baseline_value=baseline, t=0)
        assert abs(result - baseline) < 0.01

        # Halfway through ramp (t=5)
        result = scenario.get_param_value("mu", baseline_value=baseline, t=5)
        assert abs(result - 0.1) < 0.01  # Halfway between 0.2 and 0.0

        # After ramp complete (t=10)
        result = scenario.get_param_value("mu", baseline_value=baseline, t=10)
        assert abs(result - 0.0) < 0.01

    def test_get_param_value_with_end_year(self) -> None:
        """Test get_param_value with temporary change."""
        scenario = Scenario(
            name="test",
            param_changes={"mu": 0.05},
            start_year=5,
            end_year=15,
        )

        baseline = 0.2

        # Before start
        assert scenario.get_param_value("mu", baseline_value=baseline, t=3) == baseline

        # During active period
        assert scenario.get_param_value("mu", baseline_value=baseline, t=10) == 0.05

        # After end
        assert scenario.get_param_value("mu", baseline_value=baseline, t=20) == baseline

    def test_get_state_shock(self) -> None:
        """Test state shock application."""
        scenario = Scenario(
            name="shock",
            state_changes={"W": -0.1, "S": -0.2},
            start_year=5,
        )

        # Before shock
        assert scenario.get_state_shock("W", t=3) == 0.0

        # At shock time
        assert scenario.get_state_shock("W", t=5, dt=1.0) == -0.1
        assert scenario.get_state_shock("S", t=5, dt=1.0) == -0.2

        # After shock (one-time impulse)
        assert scenario.get_state_shock("W", t=7) == 0.0

    def test_is_active(self) -> None:
        """Test is_active method."""
        scenario = Scenario(
            name="test",
            start_year=5,
            end_year=15,
        )

        assert not scenario.is_active(t=3)
        assert scenario.is_active(t=5)
        assert scenario.is_active(t=10)
        assert not scenario.is_active(t=20)

    def test_is_active_no_end(self) -> None:
        """Test is_active with no end year (permanent change)."""
        scenario = Scenario(
            name="permanent",
            start_year=5,
            end_year=None,
        )

        assert not scenario.is_active(t=3)
        assert scenario.is_active(t=5)
        assert scenario.is_active(t=100)


class TestScenarioManager:
    """Tests for the ScenarioManager class."""

    def test_manager_creation(self) -> None:
        """Test ScenarioManager can be created."""
        manager = ScenarioManager()
        assert len(manager.list_scenarios()) == 0

    def test_add_scenario(self) -> None:
        """Test adding scenarios."""
        manager = ScenarioManager()
        scenario = Scenario(name="test")

        manager.add_scenario(scenario)

        assert "test" in manager.list_scenarios()
        assert manager.get_scenario("test") is scenario

    def test_add_duplicate_raises(self) -> None:
        """Test that adding duplicate scenario raises error."""
        manager = ScenarioManager()
        manager.add_scenario(Scenario(name="test"))

        with pytest.raises(ValueError, match="already exists"):
            manager.add_scenario(Scenario(name="test"))

    def test_get_nonexistent_raises(self) -> None:
        """Test that getting nonexistent scenario raises error."""
        manager = ScenarioManager()

        with pytest.raises(KeyError, match="not found"):
            manager.get_scenario("nonexistent")

    def test_remove_scenario(self) -> None:
        """Test removing scenarios."""
        manager = ScenarioManager()
        manager.add_scenario(Scenario(name="test"))

        manager.remove_scenario("test")

        assert "test" not in manager.list_scenarios()

    def test_remove_nonexistent_raises(self) -> None:
        """Test that removing nonexistent scenario raises error."""
        manager = ScenarioManager()

        with pytest.raises(KeyError, match="not found"):
            manager.remove_scenario("nonexistent")

    def test_list_scenarios(self) -> None:
        """Test listing scenarios."""
        manager = ScenarioManager()
        manager.add_scenario(Scenario(name="a"))
        manager.add_scenario(Scenario(name="b"))
        manager.add_scenario(Scenario(name="c"))

        names = manager.list_scenarios()

        assert len(names) == 3
        assert set(names) == {"a", "b", "c"}

    def test_items(self) -> None:
        """Test iterating over scenarios."""
        manager = ScenarioManager()
        manager.add_scenario(Scenario(name="a", description="A"))
        manager.add_scenario(Scenario(name="b", description="B"))

        items = manager.items()

        assert len(items) == 2
        names = [name for name, _ in items]
        assert set(names) == {"a", "b"}

    def test_get_param_modifiers(self) -> None:
        """Test getting parameter modifiers."""
        manager = ScenarioManager()
        manager.add_scenario(
            Scenario(
                name="test",
                param_changes={"mu": 0.05, "alpha": 0.003},
            )
        )

        modifiers = manager.get_param_modifiers("test")

        assert modifiers["mu"] == 0.05
        assert modifiers["alpha"] == 0.003

    def test_scenarios_property(self) -> None:
        """Test scenarios property returns copy."""
        manager = ScenarioManager()
        manager.add_scenario(Scenario(name="test"))

        scenarios = manager.scenarios

        # Should be a copy
        assert scenarios is not manager._scenarios
        assert "test" in scenarios


class TestStandardScenarios:
    """Tests for create_standard_scenarios function."""

    def test_standard_scenarios_created(self) -> None:
        """Test that standard scenarios are created."""
        manager = create_standard_scenarios()

        expected = [
            "baseline",
            "wealth_pump_off",
            "elite_reduction",
            "economic_shock",
            "reform_package",
            "state_strengthening",
        ]

        for name in expected:
            assert name in manager.list_scenarios()

    def test_baseline_has_no_changes(self) -> None:
        """Test that baseline scenario has no param changes."""
        manager = create_standard_scenarios()
        baseline = manager.get_scenario("baseline")

        assert baseline.param_changes == {}

    def test_wealth_pump_off_reduces_extraction(self) -> None:
        """Test that wealth_pump_off reduces extraction rate."""
        manager = create_standard_scenarios()
        scenario = manager.get_scenario("wealth_pump_off")

        # mu should be reduced from typical 0.2
        assert "mu" in scenario.param_changes
        assert scenario.param_changes["mu"] < 0.2

    def test_economic_shock_has_state_changes(self) -> None:
        """Test that economic_shock has state shocks."""
        manager = create_standard_scenarios()
        scenario = manager.get_scenario("economic_shock")

        assert len(scenario.state_changes) > 0
        assert "W" in scenario.state_changes or "S" in scenario.state_changes


class TestCreateShockScenario:
    """Tests for create_shock_scenario function."""

    def test_economic_shock(self) -> None:
        """Test creating economic shock scenario."""
        scenario = create_shock_scenario(
            name="recession",
            description="Economic recession",
            shock_type="economic",
            magnitude=0.2,
            start_year=5,
        )

        assert scenario.name == "recession"
        assert "W" in scenario.state_changes
        assert "S" in scenario.state_changes
        assert scenario.start_year == 5

    def test_demographic_shock(self) -> None:
        """Test creating demographic shock scenario."""
        scenario = create_shock_scenario(
            name="pandemic",
            description="Pandemic",
            shock_type="demographic",
            magnitude=0.1,
        )

        assert "N" in scenario.state_changes
        assert scenario.state_changes["N"] == -0.1

    def test_political_shock(self) -> None:
        """Test creating political shock scenario."""
        scenario = create_shock_scenario(
            name="coup",
            description="Political crisis",
            shock_type="political",
            magnitude=0.3,
        )

        assert "psi" in scenario.state_changes
        assert scenario.state_changes["psi"] > 0  # Increases instability

    def test_invalid_shock_type_raises(self) -> None:
        """Test that invalid shock type raises error."""
        with pytest.raises(ValueError, match="Unknown shock_type"):
            create_shock_scenario(
                name="test",
                description="Test",
                shock_type="invalid",
            )
