--!strict
--[[
    ProceduralBuildings.server.luau

    Generates procedural Part-based buildings for residential areas.

    Features:
    - Insulae (Roman apartment blocks) - stacked colored Parts with doorways
    - Small shops - open-front structures
    - Temples - open-air columned structures
    - Bathhouses - large structures with pools
    - Courtyard houses - domus style with central courtyard
    - Warehouses - large open storage structures
    - Colorful, enterable buildings (no blocked doorways)
    - Placed in residential zones using TerrainUtils

    Building Types:
    - Insulae: 2-4 story apartment blocks with windows and doors
    - Shops: Single-story open-front structures with awnings
    - Temples: Open-air columned structures with raised platforms
    - Bathhouses: Large structures with central pools and columns
    - Courtyard Houses: Domus-style with central courtyards and fountains
    - Warehouses: Large open structures with wide archways
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

-- Building colors (expanded Mediterranean palette with bold accents)
local BUILDING_COLORS = {
    Color3.fromRGB(240, 220, 180), -- Cream
    Color3.fromRGB(255, 200, 150), -- Peach
    Color3.fromRGB(230, 180, 140), -- Terracotta light
    Color3.fromRGB(200, 170, 130), -- Tan
    Color3.fromRGB(220, 200, 160), -- Beige
    Color3.fromRGB(250, 230, 200), -- Off-white
    Color3.fromRGB(180, 200, 220), -- Sky blue
    Color3.fromRGB(160, 190, 180), -- Seafoam green
    Color3.fromRGB(200, 180, 160), -- Warm stone
}

-- Accent colors for special buildings (temples, bathhouses)
local ACCENT_COLORS = {
    Color3.fromRGB(70, 100, 140),  -- Deep blue
    Color3.fromRGB(60, 110, 90),   -- Forest green
    Color3.fromRGB(200, 160, 80),  -- Gold
    Color3.fromRGB(150, 70, 80),   -- Deep red
    Color3.fromRGB(100, 140, 160), -- Teal
}

-- Column colors (marble and stone)
local COLUMN_COLORS = {
    Color3.fromRGB(245, 245, 240), -- White marble
    Color3.fromRGB(230, 220, 200), -- Cream marble
    Color3.fromRGB(200, 190, 175), -- Aged stone
    Color3.fromRGB(180, 170, 155), -- Weathered limestone
}

-- Water color for pools
local WATER_COLOR = Color3.fromRGB(80, 140, 160)

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
    Create an open-air temple.

    Structure:
    - Raised platform base
    - Ring of columns (no walls)
    - Completely walkable
    - Optional central altar
]]
local function createTemple(x: number, z: number, rotation: number): Model
    local model = Instance.new("Model")
    model.Name = "Temple"

    local width = rng:NextNumber(16, 24)
    local depth = rng:NextNumber(20, 28)
    local platformHeight = 2
    local columnHeight = 8
    local columnRadius = 0.8

    local columnColor = randomColor(COLUMN_COLORS)
    local platformColor = randomColor(BUILDING_COLORS)
    local accentColor = randomColor(ACCENT_COLORS)

    local groundY = TerrainUtils.getGroundY(x, z)

    -- Raised platform base
    local platform = createPart("Platform", Vector3.new(width, platformHeight, depth), platformColor)
    platform.CFrame = CFrame.new(0, platformHeight / 2, 0)
    platform.Parent = model

    -- Steps at front (3 wide steps)
    local stepWidth = width * 0.6
    local stepDepth = 1.5
    for i = 1, 3 do
        local stepY = (i - 1) * (platformHeight / 3)
        local stepZ = -depth / 2 - stepDepth * (4 - i)
        local step = createPart("Step_" .. i, Vector3.new(stepWidth, platformHeight / 3, stepDepth), platformColor)
        step.CFrame = CFrame.new(0, stepY + (platformHeight / 6), stepZ)
        step.Parent = model
    end

    -- Columns around perimeter
    local columnSpacing = 4
    local columnsX = math.floor(width / columnSpacing)
    local columnsZ = math.floor(depth / columnSpacing)

    -- Front and back columns
    for i = 0, columnsX do
        local colX = -width / 2 + i * (width / columnsX) + 1
        for _, colZ in ipairs({ -depth / 2 + 1.5, depth / 2 - 1.5 }) do
            local column = Instance.new("Part")
            column.Name = "Column"
            column.Shape = Enum.PartType.Cylinder
            column.Size = Vector3.new(columnHeight, columnRadius * 2, columnRadius * 2)
            column.Color = columnColor
            column.Material = Enum.Material.Marble
            column.Anchored = true
            column.CFrame = CFrame.new(colX, platformHeight + columnHeight / 2, colZ) * CFrame.Angles(0, 0, math.rad(90))
            column.Parent = model
        end
    end

    -- Side columns (excluding corners already placed)
    for i = 1, columnsZ - 1 do
        local colZ = -depth / 2 + i * (depth / columnsZ) + 1.5
        for _, colX in ipairs({ -width / 2 + 1, width / 2 - 1 }) do
            local column = Instance.new("Part")
            column.Name = "Column"
            column.Shape = Enum.PartType.Cylinder
            column.Size = Vector3.new(columnHeight, columnRadius * 2, columnRadius * 2)
            column.Color = columnColor
            column.Material = Enum.Material.Marble
            column.Anchored = true
            column.CFrame = CFrame.new(colX, platformHeight + columnHeight / 2, colZ) * CFrame.Angles(0, 0, math.rad(90))
            column.Parent = model
        end
    end

    -- Roof/entablature on top of columns
    local roof = createPart("Roof", Vector3.new(width + 1, 1, depth + 1), accentColor, Enum.Material.Slate)
    roof.CFrame = CFrame.new(0, platformHeight + columnHeight + 0.5, 0)
    roof.Parent = model

    -- Central altar (optional decoration)
    local altar = createPart("Altar", Vector3.new(3, 1.5, 3), columnColor, Enum.Material.Marble)
    altar.CFrame = CFrame.new(0, platformHeight + 0.75, 2)
    altar.Parent = model

    -- Position the entire model
    model.PrimaryPart = platform
    model:SetPrimaryPartCFrame(
        CFrame.new(x, groundY + platformHeight / 2, z) * CFrame.Angles(0, math.rad(rotation), 0)
    )

    return model
end

--[[
    Create a bathhouse structure.

    Structure:
    - Large rectangular with open central pool
    - Columns around perimeter
    - Multiple entry archways
]]
local function createBathhouse(x: number, z: number, rotation: number): Model
    local model = Instance.new("Model")
    model.Name = "Bathhouse"

    local width = rng:NextNumber(24, 32)
    local depth = rng:NextNumber(20, 28)
    local wallHeight = 6
    local wallThickness = 0.6
    local poolDepth = 1.5

    local buildingColor = randomColor(BUILDING_COLORS)
    local columnColor = randomColor(COLUMN_COLORS)
    local accentColor = randomColor(ACCENT_COLORS)

    local groundY = TerrainUtils.getGroundY(x, z)

    -- Floor
    local floor = createPart("Floor", Vector3.new(width, 0.4, depth), buildingColor, Enum.Material.Marble)
    floor.CFrame = CFrame.new(0, 0.2, 0)
    floor.Parent = model

    -- Central pool (sunken)
    local poolWidth = width * 0.5
    local poolLength = depth * 0.5
    local pool = createPart("Pool", Vector3.new(poolWidth, poolDepth, poolLength), WATER_COLOR, Enum.Material.Glass)
    pool.CFrame = CFrame.new(0, -poolDepth / 2, 0)
    pool.Transparency = 0.3
    pool.Parent = model

    -- Pool edge/rim
    local rimWidth = 0.8
    for _, side in ipairs({
        { Vector3.new(poolWidth + rimWidth * 2, 0.5, rimWidth), CFrame.new(0, 0.25, -poolLength / 2 - rimWidth / 2) },
        { Vector3.new(poolWidth + rimWidth * 2, 0.5, rimWidth), CFrame.new(0, 0.25, poolLength / 2 + rimWidth / 2) },
        { Vector3.new(rimWidth, 0.5, poolLength), CFrame.new(-poolWidth / 2 - rimWidth / 2, 0.25, 0) },
        { Vector3.new(rimWidth, 0.5, poolLength), CFrame.new(poolWidth / 2 + rimWidth / 2, 0.25, 0) },
    }) do
        local rim = createPart("PoolRim", side[1], columnColor, Enum.Material.Marble)
        rim.CFrame = side[2]
        rim.Parent = model
    end

    -- Columns around perimeter
    local columnHeight = wallHeight - 1
    local columnRadius = 0.6
    local columnSpacing = 5

    -- Place columns along all four sides
    local colPositions = {}
    -- Front and back
    for i = 0, math.floor(width / columnSpacing) do
        local colX = -width / 2 + 2 + i * ((width - 4) / math.floor(width / columnSpacing))
        table.insert(colPositions, { colX, -depth / 2 + 1.5 })
        table.insert(colPositions, { colX, depth / 2 - 1.5 })
    end
    -- Sides
    for i = 1, math.floor(depth / columnSpacing) - 1 do
        local colZ = -depth / 2 + 2 + i * ((depth - 4) / math.floor(depth / columnSpacing))
        table.insert(colPositions, { -width / 2 + 1.5, colZ })
        table.insert(colPositions, { width / 2 - 1.5, colZ })
    end

    for _, pos in ipairs(colPositions) do
        local column = Instance.new("Part")
        column.Name = "Column"
        column.Shape = Enum.PartType.Cylinder
        column.Size = Vector3.new(columnHeight, columnRadius * 2, columnRadius * 2)
        column.Color = columnColor
        column.Material = Enum.Material.Marble
        column.Anchored = true
        column.CFrame = CFrame.new(pos[1], columnHeight / 2, pos[2]) * CFrame.Angles(0, 0, math.rad(90))
        column.Parent = model
    end

    -- Low walls between columns (waist height, allows visibility)
    local lowWallHeight = 1.5
    -- Front wall with archway opening
    local archWidth = 6
    local frontWallLeft = createPart("FrontWallLeft", Vector3.new((width - archWidth) / 2 - 1, lowWallHeight, wallThickness), buildingColor)
    frontWallLeft.CFrame = CFrame.new(-(width + archWidth) / 4, lowWallHeight / 2, -depth / 2)
    frontWallLeft.Parent = model

    local frontWallRight = createPart("FrontWallRight", Vector3.new((width - archWidth) / 2 - 1, lowWallHeight, wallThickness), buildingColor)
    frontWallRight.CFrame = CFrame.new((width + archWidth) / 4, lowWallHeight / 2, -depth / 2)
    frontWallRight.Parent = model

    -- Side walls
    local sideWallLeft = createPart("SideWallLeft", Vector3.new(wallThickness, lowWallHeight, depth - 2), buildingColor)
    sideWallLeft.CFrame = CFrame.new(-width / 2, lowWallHeight / 2, 0)
    sideWallLeft.Parent = model

    local sideWallRight = createPart("SideWallRight", Vector3.new(wallThickness, lowWallHeight, depth - 2), buildingColor)
    sideWallRight.CFrame = CFrame.new(width / 2, lowWallHeight / 2, 0)
    sideWallRight.Parent = model

    -- Back wall
    local backWall = createPart("BackWall", Vector3.new(width - 2, lowWallHeight, wallThickness), buildingColor)
    backWall.CFrame = CFrame.new(0, lowWallHeight / 2, depth / 2)
    backWall.Parent = model

    -- Roof (supported by columns)
    local roof = createPart("Roof", Vector3.new(width + 2, 0.6, depth + 2), accentColor, Enum.Material.Slate)
    roof.CFrame = CFrame.new(0, columnHeight + 0.3, 0)
    roof.Parent = model

    -- Position the entire model
    model.PrimaryPart = floor
    model:SetPrimaryPartCFrame(
        CFrame.new(x, groundY + 0.2, z) * CFrame.Angles(0, math.rad(rotation), 0)
    )

    return model
end

--[[
    Create a courtyard house (domus style).

    Structure:
    - Walls with large archway entrance
    - Open center courtyard
    - Optional fountain in center
]]
local function createCourtyardHouse(x: number, z: number, rotation: number): Model
    local model = Instance.new("Model")
    model.Name = "CourtyardHouse"

    local width = rng:NextNumber(18, 26)
    local depth = rng:NextNumber(16, 22)
    local wallHeight = 5
    local wallThickness = 0.5
    local courtyardRatio = 0.5

    local buildingColor = randomColor(BUILDING_COLORS)
    local roofColor = randomColor(ROOF_COLORS)

    local groundY = TerrainUtils.getGroundY(x, z)

    local courtyardWidth = width * courtyardRatio
    local courtyardDepth = depth * courtyardRatio

    -- Floor
    local floor = createPart("Floor", Vector3.new(width, 0.3, depth), buildingColor)
    floor.CFrame = CFrame.new(0, 0.15, 0)
    floor.Parent = model

    -- Outer walls
    -- Front wall with large archway
    local archWidth = 4
    local archHeight = 3.5
    local frontWallLeft = createPart("FrontWallLeft", Vector3.new((width - archWidth) / 2, wallHeight, wallThickness), buildingColor)
    frontWallLeft.CFrame = CFrame.new(-(width + archWidth) / 4, wallHeight / 2, -depth / 2)
    frontWallLeft.Parent = model

    local frontWallRight = createPart("FrontWallRight", Vector3.new((width - archWidth) / 2, wallHeight, wallThickness), buildingColor)
    frontWallRight.CFrame = CFrame.new((width + archWidth) / 4, wallHeight / 2, -depth / 2)
    frontWallRight.Parent = model

    -- Arch top piece
    local archTop = createPart("ArchTop", Vector3.new(archWidth, wallHeight - archHeight, wallThickness), buildingColor)
    archTop.CFrame = CFrame.new(0, archHeight + (wallHeight - archHeight) / 2, -depth / 2)
    archTop.Parent = model

    -- Back wall
    local backWall = createPart("BackWall", Vector3.new(width, wallHeight, wallThickness), buildingColor)
    backWall.CFrame = CFrame.new(0, wallHeight / 2, depth / 2)
    backWall.Parent = model

    -- Side walls
    local leftWall = createPart("LeftWall", Vector3.new(wallThickness, wallHeight, depth), buildingColor)
    leftWall.CFrame = CFrame.new(-width / 2, wallHeight / 2, 0)
    leftWall.Parent = model

    local rightWall = createPart("RightWall", Vector3.new(wallThickness, wallHeight, depth), buildingColor)
    rightWall.CFrame = CFrame.new(width / 2, wallHeight / 2, 0)
    rightWall.Parent = model

    -- Roofed sections around courtyard (just the roof parts)
    local roofThickness = 0.4
    local coveredWidth = (width - courtyardWidth) / 2

    -- Front covered section
    local frontRoof = createPart("FrontRoof", Vector3.new(width - wallThickness * 2, roofThickness, coveredWidth), roofColor, Enum.Material.Slate)
    frontRoof.CFrame = CFrame.new(0, wallHeight - roofThickness / 2, -depth / 2 + coveredWidth / 2 + wallThickness)
    frontRoof.Parent = model

    -- Back covered section
    local backRoof = createPart("BackRoof", Vector3.new(width - wallThickness * 2, roofThickness, coveredWidth), roofColor, Enum.Material.Slate)
    backRoof.CFrame = CFrame.new(0, wallHeight - roofThickness / 2, depth / 2 - coveredWidth / 2 - wallThickness)
    backRoof.Parent = model

    -- Side covered sections
    local leftRoof = createPart("LeftRoof", Vector3.new(coveredWidth, roofThickness, courtyardDepth), roofColor, Enum.Material.Slate)
    leftRoof.CFrame = CFrame.new(-width / 2 + coveredWidth / 2 + wallThickness, wallHeight - roofThickness / 2, 0)
    leftRoof.Parent = model

    local rightRoof = createPart("RightRoof", Vector3.new(coveredWidth, roofThickness, courtyardDepth), roofColor, Enum.Material.Slate)
    rightRoof.CFrame = CFrame.new(width / 2 - coveredWidth / 2 - wallThickness, wallHeight - roofThickness / 2, 0)
    rightRoof.Parent = model

    -- Fountain in center (decorative)
    local fountainBase = createPart("FountainBase", Vector3.new(3, 0.6, 3), Color3.fromRGB(180, 170, 160), Enum.Material.Marble)
    fountainBase.CFrame = CFrame.new(0, 0.45, 0)
    fountainBase.Parent = model

    local fountainBowl = Instance.new("Part")
    fountainBowl.Name = "FountainBowl"
    fountainBowl.Shape = Enum.PartType.Cylinder
    fountainBowl.Size = Vector3.new(0.8, 2.5, 2.5)
    fountainBowl.Color = Color3.fromRGB(180, 170, 160)
    fountainBowl.Material = Enum.Material.Marble
    fountainBowl.Anchored = true
    fountainBowl.CFrame = CFrame.new(0, 1, 0) * CFrame.Angles(0, 0, math.rad(90))
    fountainBowl.Parent = model

    -- Position the entire model
    model.PrimaryPart = floor
    model:SetPrimaryPartCFrame(
        CFrame.new(x, groundY + 0.15, z) * CFrame.Angles(0, math.rad(rotation), 0)
    )

    return model
end

--[[
    Create a warehouse (horreum).

    Structure:
    - Large open rectangular structure
    - Wide entrance archways on multiple sides
    - Simple utilitarian design
]]
local function createWarehouse(x: number, z: number, rotation: number): Model
    local model = Instance.new("Model")
    model.Name = "Warehouse"

    local width = rng:NextNumber(20, 30)
    local depth = rng:NextNumber(25, 35)
    local wallHeight = 7
    local wallThickness = 0.6

    local buildingColor = randomColor(BUILDING_COLORS)
    local roofColor = randomColor(ROOF_COLORS)

    local groundY = TerrainUtils.getGroundY(x, z)

    -- Floor
    local floor = createPart("Floor", Vector3.new(width, 0.4, depth), buildingColor)
    floor.CFrame = CFrame.new(0, 0.2, 0)
    floor.Parent = model

    -- Walls with large archway openings
    local archWidth = 5
    local archHeight = 5

    -- Front wall - two large archways
    local frontArchSpacing = width / 3
    local frontWallSections = {
        { Vector3.new(frontArchSpacing - archWidth / 2, wallHeight, wallThickness), CFrame.new(-frontArchSpacing, wallHeight / 2, -depth / 2) },
        { Vector3.new(archWidth, wallHeight - archHeight, wallThickness), CFrame.new(-frontArchSpacing / 2, archHeight + (wallHeight - archHeight) / 2, -depth / 2) },
        { Vector3.new(frontArchSpacing - archWidth, wallHeight, wallThickness), CFrame.new(0, wallHeight / 2, -depth / 2) },
        { Vector3.new(archWidth, wallHeight - archHeight, wallThickness), CFrame.new(frontArchSpacing / 2, archHeight + (wallHeight - archHeight) / 2, -depth / 2) },
        { Vector3.new(frontArchSpacing - archWidth / 2, wallHeight, wallThickness), CFrame.new(frontArchSpacing, wallHeight / 2, -depth / 2) },
    }

    for i, section in ipairs(frontWallSections) do
        local wall = createPart("FrontWall_" .. i, section[1], buildingColor)
        wall.CFrame = section[2]
        wall.Parent = model
    end

    -- Back wall (solid)
    local backWall = createPart("BackWall", Vector3.new(width, wallHeight, wallThickness), buildingColor)
    backWall.CFrame = CFrame.new(0, wallHeight / 2, depth / 2)
    backWall.Parent = model

    -- Side walls with single archway each
    for _, side in ipairs({ -1, 1 }) do
        local sideX = side * width / 2
        local sideName = side == -1 and "Left" or "Right"

        -- Wall sections around archway
        local wallFront = createPart(sideName .. "WallFront", Vector3.new(wallThickness, wallHeight, (depth - archWidth) / 2), buildingColor)
        wallFront.CFrame = CFrame.new(sideX, wallHeight / 2, -depth / 4 - archWidth / 4)
        wallFront.Parent = model

        local wallBack = createPart(sideName .. "WallBack", Vector3.new(wallThickness, wallHeight, (depth - archWidth) / 2), buildingColor)
        wallBack.CFrame = CFrame.new(sideX, wallHeight / 2, depth / 4 + archWidth / 4)
        wallBack.Parent = model

        local archTop = createPart(sideName .. "ArchTop", Vector3.new(wallThickness, wallHeight - archHeight, archWidth), buildingColor)
        archTop.CFrame = CFrame.new(sideX, archHeight + (wallHeight - archHeight) / 2, 0)
        archTop.Parent = model
    end

    -- Simple pitched roof
    local roofPitch = 1.5
    local roofLeft = createPart("RoofLeft", Vector3.new(width / 2 + 1, 0.5, depth + 2), roofColor, Enum.Material.Slate)
    roofLeft.CFrame = CFrame.new(-width / 4, wallHeight + roofPitch / 2, 0) * CFrame.Angles(0, 0, math.rad(10))
    roofLeft.Parent = model

    local roofRight = createPart("RoofRight", Vector3.new(width / 2 + 1, 0.5, depth + 2), roofColor, Enum.Material.Slate)
    roofRight.CFrame = CFrame.new(width / 4, wallHeight + roofPitch / 2, 0) * CFrame.Angles(0, 0, math.rad(-10))
    roofRight.Parent = model

    -- Internal support columns
    local columnColor = randomColor(COLUMN_COLORS)
    local columnSpacingZ = depth / 4
    for i = 1, 3 do
        local colZ = -depth / 2 + i * columnSpacingZ
        for _, colX in ipairs({ -width / 4, width / 4 }) do
            local column = Instance.new("Part")
            column.Name = "SupportColumn"
            column.Shape = Enum.PartType.Cylinder
            column.Size = Vector3.new(wallHeight - 1, 1, 1)
            column.Color = columnColor
            column.Material = Enum.Material.Concrete
            column.Anchored = true
            column.CFrame = CFrame.new(colX, (wallHeight - 1) / 2, colZ) * CFrame.Angles(0, 0, math.rad(90))
            column.Parent = model
        end
    end

    -- Position the entire model
    model.PrimaryPart = floor
    model:SetPrimaryPartCFrame(
        CFrame.new(x, groundY + 0.2, z) * CFrame.Angles(0, math.rad(rotation), 0)
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
    Now includes more building types and higher density.
]]
local function populateResidentialZone(zone: any)
    print(string.format("[ProceduralBuildings] Populating %s", zone.name))

    local centerX = zone.centerX
    local centerZ = zone.centerZ
    local radiusX = zone.radiusX * 0.85 -- Expand to fill more of zone
    local radiusZ = zone.radiusZ * 0.85

    local buildingCount = 0

    -- Denser grid of buildings with some randomness
    local gridSpacing = 15 -- Reduced from 20 for more density
    local positions: { { x: number, z: number, rot: number } } = {}

    -- Generate positions in a larger grid pattern
    for gx = -2, 2 do
        for gz = -2, 2 do
            -- Skip center for plaza/open space
            if math.abs(gx) <= 0.5 and math.abs(gz) <= 0.5 then
                continue
            end

            local x = centerX + gx * gridSpacing + rng:NextNumber(-4, 4)
            local z = centerZ + gz * gridSpacing + rng:NextNumber(-4, 4)

            -- Check if within zone bounds
            local dx = math.abs(x - centerX) / radiusX
            local dz = math.abs(z - centerZ) / radiusZ
            if dx * dx + dz * dz < 1 then
                local rotation = rng:NextInteger(0, 3) * 90
                table.insert(positions, { x = x, z = z, rot = rotation })
            end
        end
    end

    -- Create buildings at positions with diverse types
    for _, pos in ipairs(positions) do
        local building: Model
        local roll = rng:NextNumber()

        -- Distribution: 40% insulae, 25% shops, 15% courtyard houses, 10% warehouses, 10% temples/bathhouses
        if roll < 0.40 then
            building = createInsula(pos.x, pos.z, pos.rot)
        elseif roll < 0.65 then
            building = createShop(pos.x, pos.z, pos.rot)
        elseif roll < 0.80 then
            building = createCourtyardHouse(pos.x, pos.z, pos.rot)
        elseif roll < 0.90 then
            building = createWarehouse(pos.x, pos.z, pos.rot)
        elseif roll < 0.95 then
            building = createTemple(pos.x, pos.z, pos.rot)
        else
            building = createBathhouse(pos.x, pos.z, pos.rot)
        end

        building.Parent = buildingsFolder
        buildingCount = buildingCount + 1
    end

    print(string.format("  Created %d buildings in %s", buildingCount, zone.name))
    return buildingCount
end

-- Find and populate residential zones
local residentialZones = findZonesByName("Residential")

print(string.format("Found %d residential zones", #residentialZones))
print("")

local totalBuildings = 0
for _, zone in ipairs(residentialZones) do
    totalBuildings = totalBuildings + populateResidentialZone(zone)
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
    print("Adding villas and temples to Palatine Hill...")

    -- Add wealthy courtyard houses (villas)
    for i = 1, 4 do
        local angle = (i - 1) * 90
        local dist = 25
        local x = palatineZone.centerX + math.cos(math.rad(angle)) * dist
        local z = palatineZone.centerZ + math.sin(math.rad(angle)) * dist

        local villa = createCourtyardHouse(x, z, angle + 180)
        villa.Name = "Villa_" .. i
        villa.Parent = buildingsFolder
        totalBuildings = totalBuildings + 1
    end

    -- Add a temple at the center
    local temple = createTemple(palatineZone.centerX, palatineZone.centerZ, 0)
    temple.Name = "PalatineTemple"
    temple.Parent = buildingsFolder
    totalBuildings = totalBuildings + 1

    print("  Created 4 villas and 1 temple on Palatine Hill")
end

-- Add buildings to Forum area if it exists
local forumZone = nil
for _, zone in ipairs(flatZones) do
    if zone.name == "Forum" or string.find(zone.name:lower(), "forum") then
        forumZone = zone
        break
    end
end

if forumZone then
    print("Adding public buildings to Forum...")

    -- Add bathhouse
    local bathX = forumZone.centerX + 30
    local bathZ = forumZone.centerZ
    local bathhouse = createBathhouse(bathX, bathZ, 90)
    bathhouse.Name = "ForumBathhouse"
    bathhouse.Parent = buildingsFolder
    totalBuildings = totalBuildings + 1

    -- Add temple
    local templeX = forumZone.centerX - 30
    local templeZ = forumZone.centerZ
    local forumTemple = createTemple(templeX, templeZ, 270)
    forumTemple.Name = "ForumTemple"
    forumTemple.Parent = buildingsFolder
    totalBuildings = totalBuildings + 1

    -- Add warehouses near edges
    for i = 1, 2 do
        local warehouseX = forumZone.centerX + (i == 1 and 40 or -40)
        local warehouseZ = forumZone.centerZ + 35
        local warehouse = createWarehouse(warehouseX, warehouseZ, 0)
        warehouse.Name = "ForumWarehouse_" .. i
        warehouse.Parent = buildingsFolder
        totalBuildings = totalBuildings + 1
    end

    print("  Created 4 public buildings near Forum")
end

print("")
print("==============================================")
print(string.format("  PROCEDURAL BUILDINGS COMPLETE: %d total", totalBuildings))
print("==============================================")
