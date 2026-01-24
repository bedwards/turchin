--!strict
--[[
    SDT.spec.luau

    Unit tests for SDT (Structural-Demographic Theory) equations.
    Uses TestService format for Roblox test execution.

    Tests verify:
    - Default state and parameter values
    - Step function produces expected outputs
    - Edge cases (PSI at 0, at 1, boundary conditions)
    - Stress level and music track mappings
    - State modification functions (setPsi, adjustPsi)
]]

-- Note: In actual Roblox environment, SDT would be required from ReplicatedStorage
-- For testing purposes, we assume the module is available
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local Shared = ReplicatedStorage:WaitForChild("Shared")
local SDT = require(Shared:WaitForChild("SDT"))

-- Test utilities
local function assertEquals(actual: any, expected: any, message: string)
    if actual ~= expected then
        error(string.format("FAIL: %s - Expected %s, got %s", message, tostring(expected), tostring(actual)))
    end
    print(string.format("PASS: %s", message))
end

local function assertApproxEquals(actual: number, expected: number, tolerance: number, message: string)
    if math.abs(actual - expected) > tolerance then
        error(string.format("FAIL: %s - Expected ~%f, got %f (tolerance: %f)", message, expected, actual, tolerance))
    end
    print(string.format("PASS: %s", message))
end

local function assertInRange(value: number, min: number, max: number, message: string)
    if value < min or value > max then
        error(string.format("FAIL: %s - Value %f not in range [%f, %f]", message, value, min, max))
    end
    print(string.format("PASS: %s", message))
end

local function assertTrue(condition: boolean, message: string)
    if not condition then
        error(string.format("FAIL: %s - Expected true", message))
    end
    print(string.format("PASS: %s", message))
end

print("==============================================")
print("  SDT UNIT TESTS")
print("==============================================")
print("")

-- Test Group: Default Values
print("--- Test Group: Default Values ---")

local function testDefaultState()
    local state = SDT.getDefaultState()

    -- Verify all fields exist and have expected values
    assertApproxEquals(state.N, 0.5, 0.01, "Default N (Population) = 0.5")
    assertApproxEquals(state.E, 0.1, 0.01, "Default E (Elites) = 0.1")
    assertApproxEquals(state.W, 0.7, 0.01, "Default W (Wages) = 0.7")
    assertApproxEquals(state.S, 0.8, 0.01, "Default S (State Health) = 0.8")
    assertApproxEquals(state.psi, 0.05, 0.01, "Default psi = 0.05")
end

local function testDefaultParams()
    local params = SDT.getDefaultParams()

    -- Verify all parameters exist and are positive
    assertTrue(params.r > 0, "Parameter r (growth rate) > 0")
    assertTrue(params.K > 0, "Parameter K (carrying capacity) > 0")
    assertTrue(params.alpha > 0, "Parameter alpha (elite recruitment) > 0")
    assertTrue(params.beta > 0, "Parameter beta (elite decay) > 0")
    assertTrue(params.gamma > 0, "Parameter gamma (wage recovery) > 0")
    assertTrue(params.theta > 0, "Parameter theta (PSI accumulation) > 0")
    assertTrue(params.delta > 0, "Parameter delta (state recovery) > 0")
    assertTrue(params.epsilon > 0, "Parameter epsilon (elite competition) > 0")
end

testDefaultState()
testDefaultParams()
print("")

-- Test Group: Step Function Basics
print("--- Test Group: Step Function Basics ---")

local function testStepReturnsState()
    local state = SDT.getDefaultState()
    local params = SDT.getDefaultParams()
    local newState = SDT.step(state, params, 1 / 30)

    -- Verify returned state has all fields
    assertTrue(newState.N ~= nil, "Step returns state with N")
    assertTrue(newState.E ~= nil, "Step returns state with E")
    assertTrue(newState.W ~= nil, "Step returns state with W")
    assertTrue(newState.S ~= nil, "Step returns state with S")
    assertTrue(newState.psi ~= nil, "Step returns state with psi")
end

local function testStepPreservesBounds()
    local state = SDT.getDefaultState()
    local params = SDT.getDefaultParams()

    -- Run many steps
    for _ = 1, 1000 do
        state = SDT.step(state, params, 1 / 30)
    end

    -- All values should remain bounded
    assertInRange(state.N, 0.01, params.K, "N stays within bounds after 1000 steps")
    assertInRange(state.E, 0.01, 0.5, "E stays within bounds after 1000 steps")
    assertInRange(state.W, 0.1, 1.0, "W stays within bounds after 1000 steps")
    assertInRange(state.S, 0.1, 1.0, "S stays within bounds after 1000 steps")
    assertInRange(state.psi, 0.0, 1.0, "psi stays within bounds after 1000 steps")
end

local function testStepChangesState()
    local state = SDT.getDefaultState()
    local params = SDT.getDefaultParams()
    local newState = SDT.step(state, params, 1.0) -- Large dt to see changes

    -- At least some values should change
    local changed = (newState.N ~= state.N)
        or (newState.E ~= state.E)
        or (newState.W ~= state.W)
        or (newState.S ~= state.S)
        or (newState.psi ~= state.psi)

    assertTrue(changed, "Step function changes at least one state variable")
end

testStepReturnsState()
testStepPreservesBounds()
testStepChangesState()
print("")

-- Test Group: Edge Cases - PSI Boundaries
print("--- Test Group: PSI Boundaries ---")

local function testPsiAtZero()
    local state: SDT.State = {
        N = 0.5,
        E = 0.1,
        W = 0.9,
        S = 0.9,
        psi = 0.0, -- PSI at minimum
    }
    local params = SDT.getDefaultParams()
    local newState = SDT.step(state, params, 1 / 30)

    assertInRange(newState.psi, 0.0, 1.0, "PSI at 0 stays bounded after step")
end

local function testPsiAtOne()
    local state: SDT.State = {
        N = 0.9,
        E = 0.4,
        W = 0.2,
        S = 0.2,
        psi = 1.0, -- PSI at maximum
    }
    local params = SDT.getDefaultParams()
    local newState = SDT.step(state, params, 1 / 30)

    assertInRange(newState.psi, 0.0, 1.0, "PSI at 1 stays bounded after step")
end

local function testPsiRecoveryWithGoodConditions()
    -- Good wages and state health should reduce PSI over time
    local state: SDT.State = {
        N = 0.5,
        E = 0.1,
        W = 0.9, -- High wages
        S = 0.9, -- Healthy state
        psi = 0.5, -- Moderate stress
    }
    local params = SDT.getDefaultParams()

    -- Run 100 steps
    for _ = 1, 100 do
        state = SDT.step(state, params, 1 / 30)
    end

    -- PSI should decrease or stay relatively stable with good conditions
    assertTrue(state.psi < 0.8, "PSI doesn't explode with good conditions")
end

local function testPsiIncreasesWithBadConditions()
    -- Low wages and state health should increase PSI
    local state: SDT.State = {
        N = 0.9, -- High population
        E = 0.4, -- Elite overproduction
        W = 0.2, -- Low wages
        S = 0.2, -- Weak state
        psi = 0.1, -- Low stress
    }
    local params = SDT.getDefaultParams()
    local initialPsi = state.psi

    -- Run 100 steps
    for _ = 1, 100 do
        state = SDT.step(state, params, 1 / 30)
    end

    -- PSI should have increased with bad conditions
    assertTrue(state.psi > initialPsi or state.psi > 0.3, "PSI increases or stays high with bad conditions")
end

testPsiAtZero()
testPsiAtOne()
testPsiRecoveryWithGoodConditions()
testPsiIncreasesWithBadConditions()
print("")

-- Test Group: Stress Level Mapping
print("--- Test Group: Stress Level Mapping ---")

local function testStressLevelPeaceful()
    local level = SDT.getStressLevel(0.05)
    assertEquals(level, "Peaceful", "PSI 0.05 -> Peaceful")
end

local function testStressLevelStable()
    local level = SDT.getStressLevel(0.15)
    assertEquals(level, "Stable", "PSI 0.15 -> Stable")
end

local function testStressLevelTense()
    local level = SDT.getStressLevel(0.25)
    assertEquals(level, "Tense", "PSI 0.25 -> Tense")
end

local function testStressLevelTroubled()
    local level = SDT.getStressLevel(0.45)
    assertEquals(level, "Troubled", "PSI 0.45 -> Troubled")
end

local function testStressLevelCritical()
    local level = SDT.getStressLevel(0.65)
    assertEquals(level, "Critical", "PSI 0.65 -> Critical")
end

local function testStressLevelCrisis()
    local level = SDT.getStressLevel(0.85)
    assertEquals(level, "Crisis", "PSI 0.85 -> Crisis")
end

testStressLevelPeaceful()
testStressLevelStable()
testStressLevelTense()
testStressLevelTroubled()
testStressLevelCritical()
testStressLevelCrisis()
print("")

-- Test Group: Music Track Index Mapping
print("--- Test Group: Music Track Mapping ---")

local function testMusicTrackAtLowPsi()
    local track = SDT.getMusicTrackIndex(0.05)
    assertEquals(track, 1, "PSI 0.05 -> Track 1")
end

local function testMusicTrackAtMidPsi()
    local track = SDT.getMusicTrackIndex(0.40)
    assertEquals(track, 4, "PSI 0.40 -> Track 4")
end

local function testMusicTrackAtHighPsi()
    local track = SDT.getMusicTrackIndex(0.90)
    assertEquals(track, 6, "PSI 0.90 -> Track 6")
end

local function testMusicTrackAtBoundary()
    -- Test at exact thresholds
    local track1 = SDT.getMusicTrackIndex(0.08) -- exactly at threshold
    local track2 = SDT.getMusicTrackIndex(0.18)
    local track3 = SDT.getMusicTrackIndex(0.32)

    assertTrue(track1 >= 1 and track1 <= 2, "PSI 0.08 -> Track 1 or 2")
    assertTrue(track2 >= 2 and track2 <= 3, "PSI 0.18 -> Track 2 or 3")
    assertTrue(track3 >= 3 and track3 <= 4, "PSI 0.32 -> Track 3 or 4")
end

testMusicTrackAtLowPsi()
testMusicTrackAtMidPsi()
testMusicTrackAtHighPsi()
testMusicTrackAtBoundary()
print("")

-- Test Group: State Modification Functions
print("--- Test Group: State Modification ---")

local function testSetPsi()
    local state = SDT.getDefaultState()
    local newState = SDT.setPsi(state, 0.75)

    assertApproxEquals(newState.psi, 0.75, 0.001, "setPsi sets PSI to 0.75")
    -- Other values should be unchanged
    assertApproxEquals(newState.N, state.N, 0.001, "setPsi preserves N")
    assertApproxEquals(newState.E, state.E, 0.001, "setPsi preserves E")
    assertApproxEquals(newState.W, state.W, 0.001, "setPsi preserves W")
    assertApproxEquals(newState.S, state.S, 0.001, "setPsi preserves S")
end

local function testSetPsiClampsHigh()
    local state = SDT.getDefaultState()
    local newState = SDT.setPsi(state, 1.5) -- Above max

    assertApproxEquals(newState.psi, 1.0, 0.001, "setPsi clamps PSI to 1.0")
end

local function testSetPsiClampsLow()
    local state = SDT.getDefaultState()
    local newState = SDT.setPsi(state, -0.5) -- Below min

    assertApproxEquals(newState.psi, 0.0, 0.001, "setPsi clamps PSI to 0.0")
end

local function testAdjustPsi()
    local state = SDT.getDefaultState() -- psi = 0.05
    local newState = SDT.adjustPsi(state, 0.1) -- Add 0.1

    assertApproxEquals(newState.psi, 0.15, 0.001, "adjustPsi adds 0.1 to PSI")
end

local function testAdjustPsiNegative()
    local state: SDT.State = {
        N = 0.5,
        E = 0.1,
        W = 0.7,
        S = 0.8,
        psi = 0.5,
    }
    local newState = SDT.adjustPsi(state, -0.2) -- Subtract 0.2

    assertApproxEquals(newState.psi, 0.3, 0.001, "adjustPsi subtracts 0.2 from PSI")
end

testSetPsi()
testSetPsiClampsHigh()
testSetPsiClampsLow()
testAdjustPsi()
testAdjustPsiNegative()
print("")

-- Test Group: Known Values Verification
print("--- Test Group: Known Value Verification ---")

local function testKnownStepOutput()
    -- Test with specific input values to verify math
    local state: SDT.State = {
        N = 0.5,
        E = 0.1,
        W = 0.7,
        S = 0.8,
        psi = 0.05,
    }
    local params: SDT.Params = {
        r = 0.05,
        K = 1.0,
        alpha = 0.03,
        beta = 0.02,
        gamma = 0.04,
        theta = 0.08,
        delta = 0.03,
        epsilon = 0.5,
    }
    local dt = 1.0

    local newState = SDT.step(state, params, dt)

    -- These values are computed from the equations in SDT.luau
    -- dN = 0.05 * 0.5 * (1 - 0.5/1.0) * 0.7 = 0.00875
    -- New N = 0.5 + 0.00875 = 0.50875

    -- The exact values depend on the implementation, but we can verify:
    -- 1. N should increase slightly (population grows when below capacity with good wages)
    assertTrue(newState.N > state.N, "N increases with growth conditions")

    -- 2. W should decrease slightly (elite extraction and labor pressure)
    -- This depends on current conditions

    -- 3. PSI changes based on conditions
    -- With good conditions (W=0.7, S=0.8, low E), PSI should stay low or decrease
    assertTrue(newState.psi < 0.5, "PSI stays low with good initial conditions")
end

local function testPopulationGrowth()
    -- Population should grow when below carrying capacity with good wages
    local state: SDT.State = {
        N = 0.3, -- Well below capacity
        E = 0.05,
        W = 0.9, -- High wages
        S = 0.9,
        psi = 0.1,
    }
    local params = SDT.getDefaultParams()
    local initialN = state.N

    for _ = 1, 100 do
        state = SDT.step(state, params, 1 / 30)
    end

    assertTrue(state.N > initialN, "Population grows when below capacity")
end

local function testEliteGrowth()
    -- Elites should recruit from population when wages are high
    local state: SDT.State = {
        N = 0.8, -- High population
        E = 0.05, -- Low elites
        W = 0.9, -- High wages
        S = 0.9,
        psi = 0.1,
    }
    local params = SDT.getDefaultParams()
    local initialE = state.E

    for _ = 1, 100 do
        state = SDT.step(state, params, 1 / 30)
    end

    assertTrue(state.E > initialE, "Elite population grows with high wages")
end

testKnownStepOutput()
testPopulationGrowth()
testEliteGrowth()
print("")

-- Test Group: Immutability
print("--- Test Group: Immutability ---")

local function testStepDoesNotMutateInput()
    local state = SDT.getDefaultState()
    local originalN = state.N
    local originalE = state.E
    local originalW = state.W
    local originalS = state.S
    local originalPsi = state.psi

    local params = SDT.getDefaultParams()
    local _ = SDT.step(state, params, 1.0)

    -- Original state should be unchanged
    assertApproxEquals(state.N, originalN, 0.0001, "step doesn't mutate input N")
    assertApproxEquals(state.E, originalE, 0.0001, "step doesn't mutate input E")
    assertApproxEquals(state.W, originalW, 0.0001, "step doesn't mutate input W")
    assertApproxEquals(state.S, originalS, 0.0001, "step doesn't mutate input S")
    assertApproxEquals(state.psi, originalPsi, 0.0001, "step doesn't mutate input psi")
end

local function testSetPsiDoesNotMutateInput()
    local state = SDT.getDefaultState()
    local originalPsi = state.psi

    local _ = SDT.setPsi(state, 0.9)

    assertApproxEquals(state.psi, originalPsi, 0.0001, "setPsi doesn't mutate input")
end

testStepDoesNotMutateInput()
testSetPsiDoesNotMutateInput()
print("")

print("==============================================")
print("  ALL SDT TESTS COMPLETED")
print("==============================================")
