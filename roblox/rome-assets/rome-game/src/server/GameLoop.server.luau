--!strict
--[[
    GameLoop.server.luau

    Main game loop for the SDT simulation.

    Responsibilities:
    - Run SDT simulation each frame
    - Broadcast state to clients via RemoteEvents
    - Handle player control inputs
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

-- Simulation state
local gameState = SDT.getDefaultState()
local gameParams = SDT.getDefaultParams()

-- Timing
local SIMULATION_DT = 1/30     -- Simulation runs at 30 Hz
local PSI_BROADCAST_DT = 0.1   -- Broadcast PSI 10 times per second
local STATE_BROADCAST_DT = 1.0 -- Broadcast full state once per second

local simulationAccumulator = 0
local psiBroadcastAccumulator = 0
local stateBroadcastAccumulator = 0

-- Control state from Controls.server.luau (via BindableEvent)
local controlState = {
    decreasePsi = 0,
    increasePsi = 0,
}

-- PSI adjustment rate per player on control plate
local PSI_ADJUSTMENT_RATE = 0.05  -- per second per player

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

-- Wait for ControlBindable from Controls.server.luau
task.spawn(function()
    local controlBindable = ReplicatedStorage:WaitForChild("ControlBindable", 30) :: BindableEvent?
    if controlBindable then
        controlBindable.Event:Connect(function(state: {decreasePsi: number, increasePsi: number})
            controlState = state
        end)
        print("[GameLoop] Connected to control inputs from Controls.server.luau")
    else
        warn("[GameLoop] ControlBindable not found - controls may not work")
    end
end)

--[[
    Apply active control inputs to PSI.
]]
local function applyControlInputs(dt: number)
    -- Calculate net adjustment from all players on plates
    local netPlayers = controlState.increasePsi - controlState.decreasePsi
    local totalAdjustment = netPlayers * PSI_ADJUSTMENT_RATE * dt

    if totalAdjustment ~= 0 then
        gameState = SDT.adjustPsi(gameState, totalAdjustment)
    end
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
    while simulationAccumulator >= SIMULATION_DT do
        -- Apply player control inputs
        applyControlInputs(SIMULATION_DT)

        -- Step the SDT simulation
        gameState = SDT.step(gameState, gameParams, SIMULATION_DT)

        simulationAccumulator = simulationAccumulator - SIMULATION_DT
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
