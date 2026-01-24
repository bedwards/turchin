--!strict
--[[
    GenerateTerrain.server.luau

    Server script that generates the terrain when the game starts.
    This runs once on server initialization to create the procedural
    terrain for Ancient Rome.

    The terrain includes:
    - Dramatic undulating valleys using layered Perlin noise
    - Flat zones for the Forum, Colosseum, temples, markets, and residential areas
    - The Tiber River with water containment
    - Material variation based on slope and elevation
]]

local ReplicatedStorage = game:GetService("ReplicatedStorage")

-- Wait for shared modules to be available
local Shared = ReplicatedStorage:WaitForChild("Shared")
local TerrainGenerator = require(Shared:WaitForChild("TerrainGenerator"))

-- Configuration for Ancient Rome terrain
local ROME_CONFIG: TerrainGenerator.TerrainConfig = {
    seed = 753, -- 753 BC - legendary founding of Rome
    mapSize = 512,
    resolution = 4,
    baseHeight = 20,
    maxHeight = 100,
    waterLevel = 15,
}

print("==============================================")
print("  ANCIENT ROME TERRAIN GENERATOR")
print("  Cliodynamics Simulation Game")
print("==============================================")
print("")

-- Generate the terrain
print("Initiating terrain generation...")
print(string.format("Using seed %d (year of Rome's founding)", ROME_CONFIG.seed))
print("")

local startTime = os.clock()

-- Run generation
TerrainGenerator.generate(ROME_CONFIG)

local elapsed = os.clock() - startTime

print("")
print("==============================================")
print(string.format("  TERRAIN GENERATION COMPLETE"))
print(string.format("  Total time: %.2f seconds", elapsed))
print("==============================================")

-- Log flat zone locations for reference
print("")
print("Building Zone Locations:")
print("------------------------")

for _, zone in ipairs(TerrainGenerator.getFlatZones()) do
    print(string.format(
        "  %s: Center (%.0f, %.0f), Size (%.0f x %.0f), Elevation %.0f",
        zone.name,
        zone.centerX,
        zone.centerZ,
        zone.radiusX * 2,
        zone.radiusZ * 2,
        zone.elevation
    ))
end

print("")
print("Terrain ready for building placement!")
print("Use TerrainUtils.getGroundY(x, z) to get elevation at any point.")
