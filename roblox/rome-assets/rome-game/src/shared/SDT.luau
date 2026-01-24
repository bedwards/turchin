--!strict
--[[
    SDT.luau

    Simplified Structural-Demographic Theory (SDT) equations for game simulation.
    Based on Peter Turchin's cliodynamics research.

    Core variables:
    - N: Population
    - E: Elite population
    - W: Real wages / well-being
    - S: State fiscal health
    - psi: Political Stress Index (PSI)

    The key output is PSI, which drives the music system:
    - Low PSI (< 0.2): Peaceful, prosperous times
    - Medium PSI (0.2 - 0.5): Growing tensions
    - High PSI (> 0.5): Crisis and instability
]]

local SDT = {}

-- Type definitions for SDT state and parameters
export type State = {
    N: number, -- Population (normalized 0-1)
    E: number, -- Elite population (normalized 0-1)
    W: number, -- Real wages / well-being (normalized 0-1)
    S: number, -- State fiscal health (normalized 0-1)
    psi: number, -- Political Stress Index (normalized 0-1)
}

export type Params = {
    r: number, -- Population growth rate
    K: number, -- Carrying capacity
    alpha: number, -- Elite recruitment rate
    beta: number, -- Elite decay rate
    gamma: number, -- Wage recovery rate
    theta: number, -- PSI accumulation rate
    delta: number, -- State recovery rate
    epsilon: number, -- Elite competition effect
}

--[[
    Default parameters calibrated for interesting gameplay dynamics.
    These produce secular cycles on ~15 minute timescales.
]]
function SDT.getDefaultParams(): Params
    return {
        r = 0.05, -- Population growth rate
        K = 1.0, -- Carrying capacity
        alpha = 0.03, -- Elite recruitment
        beta = 0.02, -- Elite decay
        gamma = 0.04, -- Wage recovery
        theta = 0.01, -- PSI accumulation (key for music transitions) - reduced 8x for slower gameplay
        delta = 0.03, -- State recovery
        epsilon = 0.5, -- Elite competition effect
    }
end

--[[
    Default initial state representing a stable society.
]]
function SDT.getDefaultState(): State
    return {
        N = 0.5, -- 50% of carrying capacity
        E = 0.1, -- 10% elite population
        W = 0.7, -- Good wages
        S = 0.8, -- Healthy state
        psi = 0.05, -- Low stress
    }
end

--[[
    Perform one simulation step using simplified SDT equations.

    The equations capture key feedback loops:
    1. Population grows logistically toward carrying capacity
    2. Elites recruit from population when wages are high
    3. Wages decrease with population pressure and elite competition
    4. State health decreases with elite competition
    5. PSI increases with low wages, elite surplus, and weak state

    @param state: Current system state
    @param params: Model parameters
    @param dt: Time step (in simulation units)
    @return: New state after time step
]]
function SDT.step(state: State, params: Params, dt: number): State
    local N = state.N
    local E = state.E
    local W = state.W
    local S = state.S
    local psi = state.psi

    -- Population dynamics (logistic growth with carrying capacity)
    -- dN/dt = r * N * (1 - N/K) * W
    local dN = params.r * N * (1 - N / params.K) * W

    -- Elite dynamics (recruitment proportional to wealth, decay from competition)
    -- dE/dt = alpha * N * W - beta * E * (E/N)
    local eliteRatio = if N > 0.01 then E / N else 0
    local dE = params.alpha * N * W - params.beta * E * eliteRatio

    -- Wage dynamics (recover toward equilibrium, decrease with labor supply and elite extraction)
    -- dW/dt = gamma * (1 - W) - epsilon * E - (N - 0.5)
    local laborPressure = math.max(0, N - 0.5)
    local dW = params.gamma * (1 - W) - params.epsilon * E - laborPressure * 0.1

    -- State dynamics (recover naturally, decrease with elite competition)
    -- dS/dt = delta * (1 - S) - epsilon * E^2
    local dS = params.delta * (1 - S) - params.epsilon * E * E

    -- PSI dynamics (accumulate from immiseration and elite competition)
    -- dpsi/dt = theta * ((1 - W) + E + (1 - S)) - decay
    local immiseration = math.max(0, 1 - W)
    local eliteSurplus = math.max(0, E - 0.1)
    local stateDyfunction = math.max(0, 1 - S)
    local dPsi = params.theta * (immiseration * 0.4 + eliteSurplus * 0.3 + stateDyfunction * 0.3)
    -- PSI naturally decays when conditions improve
    dPsi = dPsi - params.theta * 0.5 * psi * W * S

    -- Apply Euler integration with clamping
    local newState: State = {
        N = math.clamp(N + dN * dt, 0.01, params.K),
        E = math.clamp(E + dE * dt, 0.01, 0.5),
        W = math.clamp(W + dW * dt, 0.1, 1.0),
        S = math.clamp(S + dS * dt, 0.1, 1.0),
        psi = math.clamp(psi + dPsi * dt, 0.0, 1.0),
    }

    return newState
end

--[[
    Get a human-readable description of the current stress level.
]]
function SDT.getStressLevel(psi: number): string
    if psi < 0.1 then
        return "Peaceful"
    elseif psi < 0.2 then
        return "Stable"
    elseif psi < 0.35 then
        return "Tense"
    elseif psi < 0.55 then
        return "Troubled"
    elseif psi < 0.75 then
        return "Critical"
    else
        return "Crisis"
    end
end

--[[
    Get the music track index based on PSI value.
    Returns 1-6 corresponding to the track thresholds.
]]
function SDT.getMusicTrackIndex(psi: number): number
    local thresholds = { 0.08, 0.18, 0.32, 0.52, 0.76, 1.00 }
    for i, threshold in ipairs(thresholds) do
        if psi <= threshold then
            return i
        end
    end
    return 6
end

--[[
    Directly set PSI value (for player control interaction).
    Returns a new state with the modified PSI.
]]
function SDT.setPsi(state: State, newPsi: number): State
    return {
        N = state.N,
        E = state.E,
        W = state.W,
        S = state.S,
        psi = math.clamp(newPsi, 0.0, 1.0),
    }
end

--[[
    Adjust PSI by a delta amount (for gradual control changes).
    Returns a new state with the modified PSI.
]]
function SDT.adjustPsi(state: State, deltaPsi: number): State
    return SDT.setPsi(state, state.psi + deltaPsi)
end

return SDT
