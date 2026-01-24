--!strict
--[[
    ProceduralBuildings.server.luau

    Generates procedural Part-based buildings for residential areas.

    Features:
    - Insulae (Roman apartment blocks) - stacked colored Parts with doorways
    - Small shops - open-front structures
    - Colorful, enterable buildings (no blocked doorways)
    - Placed in residential zones using TerrainUtils

    Building Types:
    - Insulae: 2-4 story apartment blocks with windows and doors
    - Shops: Single-story open-front structures with awnings
]]

local ReplicatedStorage = game:GetService("ReplicatedStorage")

-- Wait for terrain to generate
task.wait(7)

-- Wait for shared modules
local Shared = ReplicatedStorage:WaitForChild("Shared")
local TerrainUtils = require(Shared:WaitForChild("TerrainUtils"))
local TerrainGenerator = require(Shared:WaitForChild("TerrainGenerator"))

print("==============================================")
print("  PROCEDURAL BUILDINGS GENERATOR")
print("==============================================")
print("")

-- Create folder for procedural buildings
local buildingsFolder = Instance.new("Folder")
buildingsFolder.Name = "ProceduralBuildings"
buildingsFolder.Parent = workspace

-- Building colors (warm Mediterranean palette)
local BUILDING_COLORS = {
    Color3.fromRGB(240, 220, 180), -- Cream
    Color3.fromRGB(255, 200, 150), -- Peach
    Color3.fromRGB(230, 180, 140), -- Terracotta light
    Color3.fromRGB(200, 170, 130), -- Tan
    Color3.fromRGB(220, 200, 160), -- Beige
    Color3.fromRGB(250, 230, 200), -- Off-white
}

local ROOF_COLORS = {
    Color3.fromRGB(180, 80, 60), -- Red tile
    Color3.fromRGB(160, 70, 50), -- Dark red tile
    Color3.fromRGB(170, 90, 70), -- Rust
}

local DOOR_COLOR = Color3.fromRGB(100, 70, 40) -- Dark wood
local WINDOW_COLOR = Color3.fromRGB(60, 80, 100) -- Dark blue (shuttered)
local AWNING_COLORS = {
    Color3.fromRGB(180, 60, 40), -- Red
    Color3.fromRGB(60, 100, 140), -- Blue
    Color3.fromRGB(80, 120, 60), -- Green
    Color3.fromRGB(180, 140, 60), -- Gold
}

-- Random seed for consistency
local rng = Random.new(753) -- Rome founding year seed

-- Get flat zones
local flatZones = TerrainGenerator.getFlatZones()

--[[
    Find zones by partial name match.
]]
local function findZonesByName(pattern: string): { any }
    local matches = {}
    for _, zone in ipairs(flatZones) do
        if string.find(zone.name:lower(), pattern:lower()) then
            table.insert(matches, zone)
        end
    end
    return matches
end

--[[
    Pick a random color from a list.
]]
local function randomColor(colors: { Color3 }): Color3
    return colors[rng:NextInteger(1, #colors)]
end

--[[
    Create a basic Part with standard properties.
]]
local function createPart(name: string, size: Vector3, color: Color3, material: Enum.Material?): Part
    local part = Instance.new("Part")
    part.Name = name
    part.Size = size
    part.Color = color
    part.Material = material or Enum.Material.Brick
    part.Anchored = true
    part.CanCollide = true
    part.CastShadow = true
    part.TopSurface = Enum.SurfaceType.Smooth
    part.BottomSurface = Enum.SurfaceType.Smooth
    return part
end

--[[
    Create an Insula (Roman apartment block).

    Structure:
    - 2-4 stories high
    - Main wall sections with window/door cutouts
    - Flat or sloped roof
    - Ground floor doorway for entry
]]
local function createInsula(x: number, z: number, rotation: number): Model
    local model = Instance.new("Model")
    model.Name = "Insula"

    local stories = rng:NextInteger(2, 4)
    local width = rng:NextNumber(12, 18)
    local depth = rng:NextNumber(10, 14)
    local storyHeight = 4
    local wallThickness = 0.5

    local buildingColor = randomColor(BUILDING_COLORS)
    local roofColor = randomColor(ROOF_COLORS)

    local groundY = TerrainUtils.getGroundY(x, z)
    local totalHeight = stories * storyHeight

    -- Create main walls
    -- Front wall (with door)
    local frontWall = createPart("FrontWall", Vector3.new(width, totalHeight, wallThickness), buildingColor)
    frontWall.CFrame = CFrame.new(0, totalHeight / 2, -depth / 2)
    frontWall.Parent = model

    -- Back wall
    local backWall = createPart("BackWall", Vector3.new(width, totalHeight, wallThickness), buildingColor)
    backWall.CFrame = CFrame.new(0, totalHeight / 2, depth / 2)
    backWall.Parent = model

    -- Left wall
    local leftWall = createPart("LeftWall", Vector3.new(wallThickness, totalHeight, depth), buildingColor)
    leftWall.CFrame = CFrame.new(-width / 2, totalHeight / 2, 0)
    leftWall.Parent = model

    -- Right wall
    local rightWall = createPart("RightWall", Vector3.new(wallThickness, totalHeight, depth), buildingColor)
    rightWall.CFrame = CFrame.new(width / 2, totalHeight / 2, 0)
    rightWall.Parent = model

    -- Floor/ceiling for each story
    for story = 0, stories do
        local floorY = story * storyHeight
        local floor = createPart("Floor_" .. story, Vector3.new(width - wallThickness, 0.3, depth - wallThickness), buildingColor)
        floor.CFrame = CFrame.new(0, floorY + 0.15, 0)
        floor.Parent = model
    end

    -- Roof
    local roof = createPart("Roof", Vector3.new(width + 1, 0.5, depth + 1), roofColor, Enum.Material.Slate)
    roof.CFrame = CFrame.new(0, totalHeight + 0.25, 0)
    roof.Parent = model

    -- Door (opening in front wall)
    local doorWidth = 2.5
    local doorHeight = 3
    local door = createPart("Door", Vector3.new(doorWidth, doorHeight, wallThickness + 0.1), DOOR_COLOR, Enum.Material.Wood)
    door.CFrame = CFrame.new(0, doorHeight / 2, -depth / 2 - 0.05)
    door.Transparency = 0.3 -- Semi-transparent for "open" feel
    door.CanCollide = false -- Walkthrough
    door.Parent = model

    -- Windows for each upper story
    local windowSize = Vector3.new(1.5, 2, 0.2)
    for story = 2, stories do
        local windowY = (story - 1) * storyHeight + storyHeight / 2

        -- Front windows
        for wx = -1, 1, 2 do
            local windowX = wx * (width / 4)
            local window = createPart("Window_Front_" .. story .. "_" .. wx, windowSize, WINDOW_COLOR, Enum.Material.Glass)
            window.CFrame = CFrame.new(windowX, windowY, -depth / 2 - 0.1)
            window.Parent = model
        end
    end

    -- Position the entire model
    local primaryPart = frontWall
    model.PrimaryPart = primaryPart
    model:SetPrimaryPartCFrame(
        CFrame.new(x, groundY + totalHeight / 2, z) * CFrame.Angles(0, math.rad(rotation), 0)
    )

    return model
end

--[[
    Create a small shop structure.

    Structure:
    - Single story with open front
    - Counter at front
    - Colorful awning
]]
local function createShop(x: number, z: number, rotation: number): Model
    local model = Instance.new("Model")
    model.Name = "Shop"

    local width = rng:NextNumber(6, 10)
    local depth = rng:NextNumber(5, 8)
    local height = 3.5
    local wallThickness = 0.4

    local buildingColor = randomColor(BUILDING_COLORS)
    local awningColor = randomColor(AWNING_COLORS)

    local groundY = TerrainUtils.getGroundY(x, z)

    -- Back wall
    local backWall = createPart("BackWall", Vector3.new(width, height, wallThickness), buildingColor)
    backWall.CFrame = CFrame.new(0, height / 2, depth / 2)
    backWall.Parent = model

    -- Left wall
    local leftWall = createPart("LeftWall", Vector3.new(wallThickness, height, depth), buildingColor)
    leftWall.CFrame = CFrame.new(-width / 2, height / 2, 0)
    leftWall.Parent = model

    -- Right wall
    local rightWall = createPart("RightWall", Vector3.new(wallThickness, height, depth), buildingColor)
    rightWall.CFrame = CFrame.new(width / 2, height / 2, 0)
    rightWall.Parent = model

    -- Floor
    local floor = createPart("Floor", Vector3.new(width, 0.3, depth), buildingColor)
    floor.CFrame = CFrame.new(0, 0.15, 0)
    floor.Parent = model

    -- Roof
    local roof = createPart("Roof", Vector3.new(width + 0.5, 0.4, depth + 0.5), randomColor(ROOF_COLORS), Enum.Material.Slate)
    roof.CFrame = CFrame.new(0, height + 0.2, 0)
    roof.Parent = model

    -- Counter at front
    local counter = createPart("Counter", Vector3.new(width - 0.5, 1, 0.6), Color3.fromRGB(120, 80, 50), Enum.Material.Wood)
    counter.CFrame = CFrame.new(0, 0.5, -depth / 2 + 0.3)
    counter.Parent = model

    -- Awning over front
    local awning = createPart("Awning", Vector3.new(width + 1, 0.2, 2), awningColor, Enum.Material.Fabric)
    awning.CFrame = CFrame.new(0, height - 0.3, -depth / 2 - 0.8) * CFrame.Angles(math.rad(15), 0, 0)
    awning.Parent = model

    -- Position the entire model
    model.PrimaryPart = backWall
    model:SetPrimaryPartCFrame(
        CFrame.new(x, groundY + height / 2, z) * CFrame.Angles(0, math.rad(rotation), 0)
    )

    return model
end

--[[
    Generate buildings in a residential zone.
]]
local function populateResidentialZone(zone: any)
    print(string.format("[ProceduralBuildings] Populating %s", zone.name))

    local centerX = zone.centerX
    local centerZ = zone.centerZ
    local radiusX = zone.radiusX * 0.7 -- Stay within zone
    local radiusZ = zone.radiusZ * 0.7

    local buildingCount = 0

    -- Grid of buildings with some randomness
    local gridSpacing = 20
    local positions: { { x: number, z: number, rot: number } } = {}

    -- Generate positions in a grid pattern
    for gx = -1, 1 do
        for gz = -1, 1 do
            if gx == 0 and gz == 0 then
                continue -- Leave center open
            end

            local x = centerX + gx * gridSpacing + rng:NextNumber(-3, 3)
            local z = centerZ + gz * gridSpacing + rng:NextNumber(-3, 3)

            -- Check if within zone bounds
            local dx = math.abs(x - centerX) / radiusX
            local dz = math.abs(z - centerZ) / radiusZ
            if dx * dx + dz * dz < 1 then
                local rotation = rng:NextInteger(0, 3) * 90
                table.insert(positions, { x = x, z = z, rot = rotation })
            end
        end
    end

    -- Create buildings at positions
    for _, pos in ipairs(positions) do
        local building: Model

        -- 70% insulae, 30% shops
        if rng:NextNumber() < 0.7 then
            building = createInsula(pos.x, pos.z, pos.rot)
        else
            building = createShop(pos.x, pos.z, pos.rot)
        end

        building.Parent = buildingsFolder
        buildingCount = buildingCount + 1
    end

    print(string.format("  Created %d buildings in %s", buildingCount, zone.name))
end

-- Find and populate residential zones
local residentialZones = findZonesByName("Residential")

print(string.format("Found %d residential zones", #residentialZones))
print("")

for _, zone in ipairs(residentialZones) do
    populateResidentialZone(zone)
end

-- Also add some buildings near Palatine Hill (wealthy area)
local palatineZone = nil
for _, zone in ipairs(flatZones) do
    if zone.name == "Palatine Hill" then
        palatineZone = zone
        break
    end
end

if palatineZone then
    print("Adding villas to Palatine Hill...")

    -- Add a few larger insulae (wealthy apartments)
    for i = 1, 3 do
        local angle = (i - 1) * 120
        local dist = 20
        local x = palatineZone.centerX + math.cos(math.rad(angle)) * dist
        local z = palatineZone.centerZ + math.sin(math.rad(angle)) * dist

        local villa = createInsula(x, z, angle + 180)
        villa.Name = "Villa_" .. i
        villa.Parent = buildingsFolder
    end

    print("  Created 3 villas on Palatine Hill")
end

print("")
print("==============================================")
print("  PROCEDURAL BUILDINGS COMPLETE")
print("==============================================")
