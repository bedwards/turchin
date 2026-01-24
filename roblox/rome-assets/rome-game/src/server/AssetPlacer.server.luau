--!strict
--[[
    AssetPlacer.server.luau

    Places Roman building 3D models in the environment using mesh asset IDs.

    Features:
    - Loads 3D meshes from Roblox asset library
    - Places buildings at appropriate flat zones
    - Creates placeholder Parts if mesh loading fails
    - Uses TerrainUtils for proper ground placement

    Buildings Placed:
    - Colosseum at Colosseum zone
    - Pantheon at Forum
    - Temple at Temple District
    - Columns in rows at Forum
    - Market stalls at Market zone
    - Aqueduct chain across terrain
]]

local ReplicatedStorage = game:GetService("ReplicatedStorage")

-- Wait for terrain to generate before placing assets
task.wait(6)

-- Wait for shared modules
local Shared = ReplicatedStorage:WaitForChild("Shared")
local TerrainUtils = require(Shared:WaitForChild("TerrainUtils"))
local AssetConfig = require(Shared:WaitForChild("AssetConfig"))
local TerrainGenerator = require(Shared:WaitForChild("TerrainGenerator"))

print("==============================================")
print("  ASSET PLACER INITIALIZING")
print("==============================================")
print("")

-- Create a folder to hold all placed assets
local assetsFolder = Instance.new("Folder")
assetsFolder.Name = "RomanBuildings"
assetsFolder.Parent = workspace

-- Get flat zones from terrain generator
local flatZones = TerrainGenerator.getFlatZones()

--[[
    Find a flat zone by name.
]]
local function findZone(name: string): { centerX: number, centerZ: number, radiusX: number, radiusZ: number, elevation: number }?
    for _, zone in ipairs(flatZones) do
        if zone.name == name then
            return zone
        end
    end
    return nil
end

--[[
    Create a MeshPart with the specified mesh ID.
    Returns the created part or nil if creation fails.
]]
local function createMeshPart(name: string, meshId: string, scale: Vector3, color: Color3): MeshPart
    local meshPart = Instance.new("MeshPart")
    meshPart.Name = name
    meshPart.Size = scale
    meshPart.MeshId = meshId
    meshPart.Color = color
    meshPart.Material = Enum.Material.Limestone
    meshPart.Anchored = true
    meshPart.CanCollide = true
    meshPart.CastShadow = true

    return meshPart
end

--[[
    Create a placeholder Part when mesh loading fails.
]]
local function createPlaceholder(name: string, size: Vector3, color: Color3): Part
    local part = Instance.new("Part")
    part.Name = name .. "_Placeholder"
    part.Size = size
    part.Color = color
    part.Material = Enum.Material.Limestone
    part.Anchored = true
    part.CanCollide = true
    part.CastShadow = true

    -- Add a label to indicate placeholder
    local billboard = Instance.new("BillboardGui")
    billboard.Size = UDim2.new(0, 100, 0, 30)
    billboard.StudsOffset = Vector3.new(0, size.Y / 2 + 5, 0)
    billboard.AlwaysOnTop = false

    local label = Instance.new("TextLabel")
    label.Size = UDim2.new(1, 0, 1, 0)
    label.BackgroundTransparency = 1
    label.Text = name
    label.TextColor3 = Color3.new(1, 1, 1)
    label.TextStrokeTransparency = 0.5
    label.Font = Enum.Font.GothamBold
    label.TextScaled = true
    label.Parent = billboard

    billboard.Parent = part

    return part
end

--[[
    Place a building at the specified position using TerrainUtils.
]]
local function placeBuilding(
    name: string,
    meshId: string,
    x: number,
    z: number,
    scale: Vector3,
    color: Color3,
    rotation: number?
): BasePart
    local rot = rotation or 0
    local groundY = TerrainUtils.getGroundY(x, z)

    local building: BasePart

    -- Try to create mesh part
    local success, result = pcall(function()
        return createMeshPart(name, meshId, scale, color)
    end)

    if success and result then
        building = result
    else
        warn(string.format("[AssetPlacer] Mesh failed for %s, using placeholder", name))
        building = createPlaceholder(name, scale, color)
    end

    -- Position the building
    local yOffset = scale.Y / 2
    building.CFrame = CFrame.new(x, groundY + yOffset, z) * CFrame.Angles(0, math.rad(rot), 0)
    building.Parent = assetsFolder

    print(string.format("[AssetPlacer] Placed %s at (%.0f, %.1f, %.0f)", name, x, groundY, z))

    return building
end

--[[
    Place the Colosseum at its designated zone.
]]
local function placeColosseum()
    local zone = findZone("Colosseum")
    if not zone then
        warn("[AssetPlacer] Colosseum zone not found")
        return
    end

    local config = AssetConfig.Placements.Colosseum
    placeBuilding(
        "Colosseum",
        AssetConfig.Meshes.Colosseum,
        zone.centerX,
        zone.centerZ,
        config.scale,
        AssetConfig.Colors.Limestone,
        config.rotation
    )
end

--[[
    Place the Pantheon at the Forum.
]]
local function placePantheon()
    local zone = findZone("Forum")
    if not zone then
        warn("[AssetPlacer] Forum zone not found")
        return
    end

    -- Place Pantheon offset from center to not overlap with control plates
    local config = AssetConfig.Placements.Pantheon
    placeBuilding(
        "Pantheon",
        AssetConfig.Meshes.Pantheon,
        zone.centerX - 30,
        zone.centerZ + 20,
        config.scale,
        AssetConfig.Colors.Marble,
        config.rotation
    )
end

--[[
    Place the Temple at Temple District.
]]
local function placeTemple()
    local zone = findZone("Temple District")
    if not zone then
        warn("[AssetPlacer] Temple District zone not found")
        return
    end

    -- Place temple offset from control plates
    local config = AssetConfig.Placements.RomanTemple
    placeBuilding(
        "RomanTemple",
        AssetConfig.Meshes.RomanTemple,
        zone.centerX + 20,
        zone.centerZ - 15,
        config.scale,
        AssetConfig.Colors.Marble,
        config.rotation
    )
end

--[[
    Place a Triumphal Arch near the Forum entrance.
]]
local function placeTriumphalArch()
    local zone = findZone("Forum")
    if not zone then
        return
    end

    local config = AssetConfig.Placements.TriumphalArch
    placeBuilding(
        "TriumphalArch",
        AssetConfig.Meshes.TriumphalArch,
        zone.centerX + 40,
        zone.centerZ - 25,
        config.scale,
        AssetConfig.Colors.Limestone,
        config.rotation
    )
end

--[[
    Place rows of columns in the Forum area.
]]
local function placeColumns()
    local zone = findZone("Forum")
    if not zone then
        return
    end

    local config = AssetConfig.Placements.DoricColumn
    local columnSpacing = 8
    local rowOffset = 15

    -- Two rows of columns along the Forum
    for row = 1, 2 do
        local rowZ = zone.centerZ + (row == 1 and -rowOffset or rowOffset)

        for i = 1, 5 do
            local colX = zone.centerX + 15 + (i - 1) * columnSpacing
            placeBuilding(
                "DoricColumn_" .. row .. "_" .. i,
                AssetConfig.Meshes.DoricColumn,
                colX,
                rowZ,
                config.scale,
                AssetConfig.Colors.Marble,
                0
            )
        end
    end
end

--[[
    Place market stalls in the Market zone.
]]
local function placeMarketStalls()
    local zone = findZone("Market")
    if not zone then
        warn("[AssetPlacer] Market zone not found")
        return
    end

    local config = AssetConfig.Placements.MarketStall
    local stallPositions = {
        { x = zone.centerX - 15, z = zone.centerZ - 20, rot = 0 },
        { x = zone.centerX + 15, z = zone.centerZ - 20, rot = 0 },
        { x = zone.centerX - 15, z = zone.centerZ + 10, rot = 180 },
        { x = zone.centerX + 15, z = zone.centerZ + 10, rot = 180 },
        { x = zone.centerX - 25, z = zone.centerZ - 5, rot = 90 },
        { x = zone.centerX + 25, z = zone.centerZ - 5, rot = 270 },
    }

    for i, pos in ipairs(stallPositions) do
        placeBuilding(
            "MarketStall_" .. i,
            AssetConfig.Meshes.MarketStall,
            pos.x,
            pos.z,
            config.scale,
            AssetConfig.Colors.Wood,
            pos.rot
        )
    end
end

--[[
    Place aqueduct segments across the terrain.
]]
local function placeAqueduct()
    local config = AssetConfig.Placements.RomanAqueduct
    local segmentSpacing = 25
    local startX = -150
    local startZ = 160 -- North of the main areas, near river

    -- Create 5 aqueduct segments in a line
    for i = 1, 5 do
        local x = startX + (i - 1) * segmentSpacing
        local z = startZ + math.sin(i * 0.3) * 10 -- Slight curve

        placeBuilding(
            "Aqueduct_" .. i,
            AssetConfig.Meshes.RomanAqueduct,
            x,
            z,
            config.scale,
            AssetConfig.Colors.Limestone,
            75 -- Angle across terrain
        )
    end
end

-- Place all buildings
print("Placing Roman buildings...")
print("")

placeColosseum()
placePantheon()
placeTemple()
placeTriumphalArch()
placeColumns()
placeMarketStalls()
placeAqueduct()

print("")
print("==============================================")
print("  ASSET PLACEMENT COMPLETE")
print("==============================================")
print("")
print("Placed buildings:")
print("  - Colosseum")
print("  - Pantheon")
print("  - Roman Temple")
print("  - Triumphal Arch")
print("  - 10 Doric Columns")
print("  - 6 Market Stalls")
print("  - 5 Aqueduct Segments")
