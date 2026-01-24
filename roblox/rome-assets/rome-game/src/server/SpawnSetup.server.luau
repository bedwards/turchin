--!strict
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local Shared = ReplicatedStorage:WaitForChild("Shared")
local TerrainUtils = require(Shared:WaitForChild("TerrainUtils"))

-- Spawn at Forum center
local SPAWN_X = 0
local SPAWN_Z = 0

-- Wait for terrain to generate
task.wait(2)

local groundY = TerrainUtils.getGroundY(SPAWN_X, SPAWN_Z)
print("Setting spawn at ground level Y =", groundY)

local spawn = Instance.new("SpawnLocation")
spawn.Name = "ForumSpawn"
spawn.Anchored = true
spawn.CanCollide = false
spawn.Transparency = 1
spawn.Size = Vector3.new(6, 1, 6)
spawn.Position = Vector3.new(SPAWN_X, groundY + 0.5, SPAWN_Z)
spawn.Parent = workspace

print("Spawn location created at", spawn.Position)
