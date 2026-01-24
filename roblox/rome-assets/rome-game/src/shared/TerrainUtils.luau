--!strict
--[[
    TerrainUtils.luau

    Utility functions for working with terrain.
    CRITICAL API for placing objects at correct ground elevation.

    Key functions:
    - getGroundY(x, z): Raycast from sky to find terrain surface
    - placeOnGround(model, x, z): Position model at correct elevation
]]

local TerrainUtils = {}

-- Import TerrainGenerator for fallback height calculation
local TerrainGenerator = require(script.Parent.TerrainGenerator)

-- Constants
local SKY_HEIGHT = 500 -- Height to raycast from
local RAYCAST_DISTANCE = 1000 -- Maximum raycast distance

-- Raycast parameters for terrain detection
local raycastParams: RaycastParams = RaycastParams.new()
raycastParams.FilterType = Enum.RaycastFilterType.Include
raycastParams.FilterDescendantsInstances = {workspace.Terrain}
raycastParams.IgnoreWater = true

--[[
    Get the ground Y position at the given X, Z coordinates.
    Uses raycast from sky to find terrain surface.

    @param x: number - The X coordinate
    @param z: number - The Z coordinate
    @return number - The Y position of the ground surface

    Example:
        local groundY = TerrainUtils.getGroundY(100, 50)
        print("Ground at (100, 50) is at Y =", groundY)
]]
function TerrainUtils.getGroundY(x: number, z: number): number
    local origin = Vector3.new(x, SKY_HEIGHT, z)
    local direction = Vector3.new(0, -RAYCAST_DISTANCE, 0)

    local result = workspace:Raycast(origin, direction, raycastParams)

    if result then
        return result.Position.Y
    end

    -- Fallback to TerrainGenerator calculation if raycast fails
    -- This can happen if terrain hasn't been generated yet
    return TerrainGenerator.getHeightAt(x, z)
end

--[[
    Get detailed ground information at the given coordinates.

    @param x: number - The X coordinate
    @param z: number - The Z coordinate
    @return table - Contains Y position, normal vector, and material

    Example:
        local info = TerrainUtils.getGroundInfo(100, 50)
        print("Ground Y:", info.y)
        print("Surface normal:", info.normal)
        print("Material:", info.material)
]]
function TerrainUtils.getGroundInfo(x: number, z: number): {y: number, normal: Vector3, material: Enum.Material?}
    local origin = Vector3.new(x, SKY_HEIGHT, z)
    local direction = Vector3.new(0, -RAYCAST_DISTANCE, 0)

    local result = workspace:Raycast(origin, direction, raycastParams)

    if result then
        return {
            y = result.Position.Y,
            normal = result.Normal,
            material = result.Material,
        }
    end

    -- Fallback values
    return {
        y = TerrainGenerator.getHeightAt(x, z),
        normal = Vector3.new(0, 1, 0),
        material = nil,
    }
end

--[[
    Place a Model at the correct ground elevation.
    The model's PrimaryPart will be positioned at ground level.

    @param model: Model - The model to place (must have PrimaryPart set)
    @param x: number - The X coordinate
    @param z: number - The Z coordinate
    @param yOffset: number? - Optional offset above ground (default 0)

    Example:
        local building = game.ReplicatedStorage.Buildings.Temple:Clone()
        TerrainUtils.placeOnGround(building, 50, -80)
        building.Parent = workspace
]]
function TerrainUtils.placeOnGround(model: Model, x: number, z: number, yOffset: number?)
    local offset = yOffset or 0
    local groundY = TerrainUtils.getGroundY(x, z)

    if model.PrimaryPart then
        -- Get the model's bounding box to determine base offset
        local _, size = model:GetBoundingBox()
        local baseOffset = size.Y / 2

        model:SetPrimaryPartCFrame(
            CFrame.new(x, groundY + baseOffset + offset, z)
        )
    else
        warn("[TerrainUtils] Model has no PrimaryPart set:", model.Name)
        -- Try to position anyway using MoveTo
        model:MoveTo(Vector3.new(x, groundY + offset, z))
    end
end

--[[
    Place a BasePart at the correct ground elevation.

    @param part: BasePart - The part to place
    @param x: number - The X coordinate
    @param z: number - The Z coordinate
    @param yOffset: number? - Optional offset above ground (default 0)

    Example:
        local marker = Instance.new("Part")
        marker.Size = Vector3.new(2, 2, 2)
        TerrainUtils.placePartOnGround(marker, 100, 100)
        marker.Parent = workspace
]]
function TerrainUtils.placePartOnGround(part: BasePart, x: number, z: number, yOffset: number?)
    local offset = yOffset or 0
    local groundY = TerrainUtils.getGroundY(x, z)

    -- Offset by half the part height so bottom sits on ground
    local baseOffset = part.Size.Y / 2

    part.Position = Vector3.new(x, groundY + baseOffset + offset, z)
end

--[[
    Align a model to the terrain slope at its position.
    Useful for placing objects that should follow terrain contour.

    @param model: Model - The model to align (must have PrimaryPart set)

    Example:
        local road = game.ReplicatedStorage.Roads.RoadSegment:Clone()
        TerrainUtils.placeOnGround(road, 0, 0)
        TerrainUtils.alignToSlope(road)
]]
function TerrainUtils.alignToSlope(model: Model)
    if not model.PrimaryPart then
        warn("[TerrainUtils] Model has no PrimaryPart set:", model.Name)
        return
    end

    local pos = model.PrimaryPart.Position
    local info = TerrainUtils.getGroundInfo(pos.X, pos.Z)

    -- Calculate rotation to align with surface normal
    local up = info.normal
    local look = model.PrimaryPart.CFrame.LookVector

    -- Project look vector onto plane perpendicular to surface normal
    local right = look:Cross(up).Unit
    local alignedLook = up:Cross(right).Unit

    local cf = CFrame.fromMatrix(pos, right, up, -alignedLook)
    model:SetPrimaryPartCFrame(cf)
end

--[[
    Check if a position is on solid ground (not water).

    @param x: number - The X coordinate
    @param z: number - The Z coordinate
    @return boolean - True if position is on solid ground

    Example:
        if TerrainUtils.isOnSolidGround(100, 50) then
            print("Safe to place building here")
        end
]]
function TerrainUtils.isOnSolidGround(x: number, z: number): boolean
    local info = TerrainUtils.getGroundInfo(x, z)
    return info.material ~= Enum.Material.Water and info.material ~= nil
end

--[[
    Get the slope angle at a position in degrees.

    @param x: number - The X coordinate
    @param z: number - The Z coordinate
    @return number - Slope angle in degrees (0 = flat, 90 = vertical)

    Example:
        local slope = TerrainUtils.getSlopeAngle(100, 50)
        if slope > 30 then
            print("Too steep for building!")
        end
]]
function TerrainUtils.getSlopeAngle(x: number, z: number): number
    local info = TerrainUtils.getGroundInfo(x, z)
    local upVector = Vector3.new(0, 1, 0)
    local angle = math.acos(info.normal:Dot(upVector))
    return math.deg(angle)
end

--[[
    Find a flat area near the given position.
    Searches in expanding circles until a flat spot is found.

    @param x: number - Starting X coordinate
    @param z: number - Starting Z coordinate
    @param maxSlope: number? - Maximum acceptable slope in degrees (default 15)
    @param searchRadius: number? - Maximum search distance (default 50)
    @return Vector3? - Position of flat area, or nil if none found

    Example:
        local flatSpot = TerrainUtils.findFlatArea(100, 50)
        if flatSpot then
            TerrainUtils.placeOnGround(building, flatSpot.X, flatSpot.Z)
        end
]]
function TerrainUtils.findFlatArea(x: number, z: number, maxSlope: number?, searchRadius: number?): Vector3?
    local slopeThreshold = maxSlope or 15
    local radius = searchRadius or 50
    local step = 5

    -- Check original position first
    if TerrainUtils.getSlopeAngle(x, z) <= slopeThreshold then
        return Vector3.new(x, TerrainUtils.getGroundY(x, z), z)
    end

    -- Search in expanding circles
    for r = step, radius, step do
        for angle = 0, 360, 30 do
            local rad = math.rad(angle)
            local checkX = x + math.cos(rad) * r
            local checkZ = z + math.sin(rad) * r

            if TerrainUtils.getSlopeAngle(checkX, checkZ) <= slopeThreshold then
                return Vector3.new(checkX, TerrainUtils.getGroundY(checkX, checkZ), checkZ)
            end
        end
    end

    return nil
end

--[[
    Sample ground heights along a line between two points.
    Useful for placing roads or checking terrain profiles.

    @param startX: number - Starting X coordinate
    @param startZ: number - Starting Z coordinate
    @param endX: number - Ending X coordinate
    @param endZ: number - Ending Z coordinate
    @param samples: number? - Number of samples (default 10)
    @return {Vector3} - Array of ground positions along the line

    Example:
        local profile = TerrainUtils.getTerrainProfile(0, 0, 100, 100)
        for _, point in profile do
            print(point)
        end
]]
function TerrainUtils.getTerrainProfile(
    startX: number,
    startZ: number,
    endX: number,
    endZ: number,
    samples: number?
): {Vector3}
    local numSamples = samples or 10
    local points: {Vector3} = {}

    for i = 0, numSamples - 1 do
        local t = i / (numSamples - 1)
        local x = startX + (endX - startX) * t
        local z = startZ + (endZ - startZ) * t
        local y = TerrainUtils.getGroundY(x, z)

        table.insert(points, Vector3.new(x, y, z))
    end

    return points
end

return TerrainUtils
