--!strict
--[[
    SpawnSetup.server.luau

    Handles player spawning at the Forum center with correct ground elevation.
    Uses PlayerAdded/CharacterAdded events to ensure players always spawn
    at ground level, even if terrain generates after they join.
]]

local Players = game:GetService("Players")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local Shared = ReplicatedStorage:WaitForChild("Shared")
local TerrainUtils = require(Shared:WaitForChild("TerrainUtils"))

-- Spawn at Forum center
local SPAWN_X = 0
local SPAWN_Z = 0
local SPAWN_HEIGHT_OFFSET = 3 -- Headroom above ground

-- Create a basic spawn location immediately (will be at origin height)
-- This ensures there's always a spawn point, even before terrain loads
local spawn = Instance.new("SpawnLocation")
spawn.Name = "ForumSpawn"
spawn.Anchored = true
spawn.CanCollide = false
spawn.Transparency = 1
spawn.Size = Vector3.new(6, 1, 6)
spawn.Position = Vector3.new(SPAWN_X, 10, SPAWN_Z) -- Temporary height
spawn.Parent = workspace

print("[SpawnSetup] Created initial spawn location")

--[[
    Teleport a character to the correct ground level at spawn.
    Called whenever a character is added to ensure correct positioning.
]]
local function teleportToGround(character: Model)
    -- Wait for the HumanoidRootPart to exist
    local rootPart = character:WaitForChild("HumanoidRootPart", 5) :: BasePart?
    if not rootPart then
        warn("[SpawnSetup] No HumanoidRootPart found for character")
        return
    end

    -- Small delay to let physics settle
    task.wait(0.1)

    -- Get current ground level (raycast or fallback)
    local groundY = TerrainUtils.getGroundY(SPAWN_X, SPAWN_Z)
    local targetY = groundY + SPAWN_HEIGHT_OFFSET

    -- Check if player is significantly below ground or at wrong height
    local currentY = rootPart.Position.Y
    if currentY < groundY or math.abs(currentY - targetY) > 5 then
        -- Teleport to correct position
        local newPosition = Vector3.new(SPAWN_X, targetY, SPAWN_Z)
        rootPart.CFrame = CFrame.new(newPosition)
        print("[SpawnSetup] Teleported character to ground level Y =", targetY)
    end

    -- Update spawn location for future spawns
    spawn.Position = Vector3.new(SPAWN_X, groundY + 0.5, SPAWN_Z)
end

--[[
    Handle a new player joining the game.
    Sets up CharacterAdded listener for ground-level spawning.
]]
local function onPlayerAdded(player: Player)
    -- Handle current character if exists
    if player.Character then
        task.spawn(function()
            teleportToGround(player.Character :: Model)
        end)
    end

    -- Handle future character spawns
    player.CharacterAdded:Connect(function(character)
        task.spawn(function()
            teleportToGround(character)
        end)
    end)
end

-- Connect to all current and future players
for _, player in Players:GetPlayers() do
    task.spawn(function()
        onPlayerAdded(player)
    end)
end
Players.PlayerAdded:Connect(onPlayerAdded)

print("[SpawnSetup] Player spawn handlers initialized")
