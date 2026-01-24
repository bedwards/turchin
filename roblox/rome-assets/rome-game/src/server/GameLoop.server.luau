--!strict
--[[
    GameLoop.server.luau

    Main game loop for the SDT simulation.

    Responsibilities:
    - Run SDT simulation each frame
    - Broadcast state to clients via RemoteEvents
    - Handle player control inputs from 4 control stations:
      1. PSI Control - Direct PSI adjustment
      2. Senate Seats - Elite recruitment rate (alpha)
      3. Grain Dole - Wage recovery rate (gamma)
      4. Legion Funding - State recovery rate (delta)
]]

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService = game:GetService("RunService")

-- Wait for shared modules
local Shared = ReplicatedStorage:WaitForChild("Shared")
local SDT = require(Shared:WaitForChild("SDT"))

-- Create RemoteEvents folder
local RemoteEvents = Instance.new("Folder")
RemoteEvents.Name = "RemoteEvents"
RemoteEvents.Parent = ReplicatedStorage

-- Create RemoteEvent for PSI updates
local PsiUpdateEvent = Instance.new("RemoteEvent")
PsiUpdateEvent.Name = "PsiUpdate"
PsiUpdateEvent.Parent = RemoteEvents

-- Create RemoteEvent for full state updates (less frequent)
local StateUpdateEvent = Instance.new("RemoteEvent")
StateUpdateEvent.Name = "StateUpdate"
StateUpdateEvent.Parent = RemoteEvents

-- Create RemoteEvent for control inputs from clients
local ControlInputEvent = Instance.new("RemoteEvent")
ControlInputEvent.Name = "ControlInput"
ControlInputEvent.Parent = RemoteEvents

-- Create RemoteEvent for parameter updates (for HUD display)
local ParamsUpdateEvent = Instance.new("RemoteEvent")
ParamsUpdateEvent.Name = "ParamsUpdate"
ParamsUpdateEvent.Parent = RemoteEvents

-- Simulation state
local gameState = SDT.getDefaultState()
local gameParams = SDT.getDefaultParams()

-- Timing
local SIMULATION_DT = 1 / 30 -- Simulation runs at 30 Hz
local PSI_BROADCAST_DT = 0.1 -- Broadcast PSI 10 times per second
local STATE_BROADCAST_DT = 1.0 -- Broadcast full state once per second

local simulationAccumulator = 0
local psiBroadcastAccumulator = 0
local stateBroadcastAccumulator = 0

-- Control state from Controls.server.luau (via BindableEvent)
local controlState: { [string]: number } = {
    decrease_psi = 0,
    increase_psi = 0,
    decrease_alpha = 0,
    increase_alpha = 0,
    decrease_gamma = 0,
    increase_gamma = 0,
    decrease_delta = 0,
    increase_delta = 0,
}

-- Adjustment rates per player on control plate (per second)
local PSI_ADJUSTMENT_RATE = 0.05 -- PSI change per second
local PARAM_ADJUSTMENT_RATE = 0.01 -- Parameter change per second

-- Parameter bounds
local PARAM_BOUNDS = {
    alpha = { min = 0.01, max = 0.10 }, -- Elite recruitment: 0.01 to 0.10
    gamma = { min = 0.01, max = 0.10 }, -- Wage recovery: 0.01 to 0.10
    delta = { min = 0.01, max = 0.10 }, -- State recovery: 0.01 to 0.10
}

print("==============================================")
print("  SDT GAME LOOP INITIALIZED")
print("==============================================")
print("")
print("Initial State:")
print(string.format("  Population (N): %.2f", gameState.N))
print(string.format("  Elites (E): %.2f", gameState.E))
print(string.format("  Wages (W): %.2f", gameState.W))
print(string.format("  State Health (S): %.2f", gameState.S))
print(string.format("  Political Stress (psi): %.2f", gameState.psi))
print("")
print("Initial Parameters:")
print(string.format("  Elite Recruitment (alpha): %.3f", gameParams.alpha))
print(string.format("  Wage Recovery (gamma): %.3f", gameParams.gamma))
print(string.format("  State Recovery (delta): %.3f", gameParams.delta))
print("")

-- Wait for ControlBindable from Controls.server.luau
task.spawn(function()
    local controlBindable = ReplicatedStorage:WaitForChild("ControlBindable", 30) :: BindableEvent?
    if controlBindable then
        controlBindable.Event:Connect(function(state: { [string]: number })
            controlState = state
        end)
        print("[GameLoop] Connected to control inputs from Controls.server.luau")
    else
        warn("[GameLoop] ControlBindable not found - controls may not work")
    end
end)

--[[
    Clamp a parameter value within its bounds.
]]
local function clampParam(paramName: string, value: number): number
    local bounds = PARAM_BOUNDS[paramName]
    if bounds then
        return math.clamp(value, bounds.min, bounds.max)
    end
    return value
end

--[[
    Apply active control inputs to PSI and parameters.
]]
local function applyControlInputs(dt: number)
    local paramsChanged = false

    -- PSI direct control
    local netPsiPlayers = (controlState.increase_psi or 0) - (controlState.decrease_psi or 0)
    if netPsiPlayers ~= 0 then
        local totalAdjustment = netPsiPlayers * PSI_ADJUSTMENT_RATE * dt
        gameState = SDT.adjustPsi(gameState, totalAdjustment)
    end

    -- Alpha (Elite Recruitment) control
    local netAlphaPlayers = (controlState.increase_alpha or 0) - (controlState.decrease_alpha or 0)
    if netAlphaPlayers ~= 0 then
        local alphaAdjustment = netAlphaPlayers * PARAM_ADJUSTMENT_RATE * dt
        gameParams.alpha = clampParam("alpha", gameParams.alpha + alphaAdjustment)
        paramsChanged = true
    end

    -- Gamma (Wage Recovery) control
    local netGammaPlayers = (controlState.increase_gamma or 0) - (controlState.decrease_gamma or 0)
    if netGammaPlayers ~= 0 then
        local gammaAdjustment = netGammaPlayers * PARAM_ADJUSTMENT_RATE * dt
        gameParams.gamma = clampParam("gamma", gameParams.gamma + gammaAdjustment)
        paramsChanged = true
    end

    -- Delta (State Recovery) control
    local netDeltaPlayers = (controlState.increase_delta or 0) - (controlState.decrease_delta or 0)
    if netDeltaPlayers ~= 0 then
        local deltaAdjustment = netDeltaPlayers * PARAM_ADJUSTMENT_RATE * dt
        gameParams.delta = clampParam("delta", gameParams.delta + deltaAdjustment)
        paramsChanged = true
    end

    return paramsChanged
end

--[[
    Broadcast parameter values to clients.
]]
local function broadcastParams()
    ParamsUpdateEvent:FireAllClients({
        alpha = gameParams.alpha,
        gamma = gameParams.gamma,
        delta = gameParams.delta,
    })
end

--[[
    Main game loop - runs every frame.
]]
RunService.Heartbeat:Connect(function(dt: number)
    -- Accumulate time
    simulationAccumulator = simulationAccumulator + dt
    psiBroadcastAccumulator = psiBroadcastAccumulator + dt
    stateBroadcastAccumulator = stateBroadcastAccumulator + dt

    -- Run simulation at fixed timestep
    local paramsChanged = false
    while simulationAccumulator >= SIMULATION_DT do
        -- Apply player control inputs
        if applyControlInputs(SIMULATION_DT) then
            paramsChanged = true
        end

        -- Step the SDT simulation
        gameState = SDT.step(gameState, gameParams, SIMULATION_DT)

        simulationAccumulator = simulationAccumulator - SIMULATION_DT
    end

    -- Broadcast params if changed
    if paramsChanged then
        broadcastParams()
    end

    -- Broadcast PSI to clients (frequent, for smooth music transitions)
    if psiBroadcastAccumulator >= PSI_BROADCAST_DT then
        PsiUpdateEvent:FireAllClients(gameState.psi)
        psiBroadcastAccumulator = 0
    end

    -- Broadcast full state to clients (less frequent, for HUD updates)
    if stateBroadcastAccumulator >= STATE_BROADCAST_DT then
        StateUpdateEvent:FireAllClients({
            N = gameState.N,
            E = gameState.E,
            W = gameState.W,
            S = gameState.S,
            psi = gameState.psi,
        })
        stateBroadcastAccumulator = 0
    end
end)

print("Game loop started! SDT simulation running at 30 Hz.")
print("PSI broadcasts: 10/sec, Full state broadcasts: 1/sec")
print("")
print("Control Stations:")
print("  - PSI Control: Direct stress adjustment")
print("  - Senate Seats: Elite recruitment rate (alpha)")
print("  - Grain Dole: Wage recovery rate (gamma)")
print("  - Legion Funding: State recovery rate (delta)")
