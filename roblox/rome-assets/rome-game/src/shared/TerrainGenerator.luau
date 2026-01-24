--!strict
--[[
    TerrainGenerator.luau

    Procedural terrain generation for Ancient Rome using layered Perlin noise.
    Creates dramatic undulating valleys with flat zones for buildings.

    Features:
    - Seeded generation for reproducibility
    - Multiple noise octaves (large valleys, medium hills, small detail)
    - Flat zones carved out for Forum, Colosseum, temples, markets
    - Material assignment based on slope and zone type
    - Tiber River with water containment
]]

local TerrainGenerator = {}

-- Types
export type FlatZone = {
    name: string,
    centerX: number,
    centerZ: number,
    radiusX: number,
    radiusZ: number,
    elevation: number,
    material: Enum.Material,
}

export type TerrainConfig = {
    seed: number,
    mapSize: number,
    resolution: number,
    baseHeight: number,
    maxHeight: number,
    waterLevel: number,
}

-- Default configuration
local DEFAULT_CONFIG: TerrainConfig = {
    seed = 42,
    mapSize = 512,
    resolution = 4, -- studs per terrain cell
    baseHeight = 20,
    maxHeight = 100,
    waterLevel = 15,
}

-- Flat zones for building placement (relative to map center)
local FLAT_ZONES: {FlatZone} = {
    {
        name = "Forum",
        centerX = 0,
        centerZ = 0,
        radiusX = 60,
        radiusZ = 40,
        elevation = 25,
        material = Enum.Material.Cobblestone,
    },
    {
        name = "Colosseum",
        centerX = 120,
        centerZ = 50,
        radiusX = 50,
        radiusZ = 50,
        elevation = 22,
        material = Enum.Material.Cobblestone,
    },
    {
        name = "Temple District",
        centerX = -80,
        centerZ = 60,
        radiusX = 40,
        radiusZ = 35,
        elevation = 35,
        material = Enum.Material.Cobblestone,
    },
    {
        name = "Market",
        centerX = 40,
        centerZ = -80,
        radiusX = 45,
        radiusZ = 30,
        elevation = 24,
        material = Enum.Material.Cobblestone,
    },
    {
        name = "Residential North",
        centerX = -100,
        centerZ = -60,
        radiusX = 50,
        radiusZ = 40,
        elevation = 30,
        material = Enum.Material.Ground,
    },
    {
        name = "Residential South",
        centerX = 80,
        centerZ = -100,
        radiusX = 45,
        radiusZ = 35,
        elevation = 28,
        material = Enum.Material.Ground,
    },
    {
        name = "Palatine Hill",
        centerX = -40,
        centerZ = -40,
        radiusX = 35,
        radiusZ = 35,
        elevation = 45,
        material = Enum.Material.Grass,
    },
}

-- Tiber River path (series of control points)
local TIBER_RIVER_PATH: {{x: number, z: number, width: number}} = {
    {x = -256, z = 100, width = 30},
    {x = -180, z = 80, width = 35},
    {x = -100, z = 120, width = 40},
    {x = -20, z = 90, width = 45},
    {x = 60, z = 130, width = 40},
    {x = 140, z = 100, width = 35},
    {x = 200, z = 140, width = 30},
    {x = 256, z = 120, width = 25},
}

-- Store config for access
local currentConfig: TerrainConfig = DEFAULT_CONFIG
local heightCache: {[string]: number} = {}

--[[
    Seeded noise function using math.noise with offset based on seed
]]
local function seededNoise(x: number, z: number, seed: number, scale: number): number
    local seedOffset = seed * 1000
    return math.noise(x / scale + seedOffset, z / scale + seedOffset)
end

--[[
    Multi-octave noise for natural terrain variation
    Combines large-scale valleys with medium hills and small detail
]]
local function getLayeredNoise(x: number, z: number, seed: number): number
    -- Large valleys and mountains (low frequency, high amplitude)
    local large = seededNoise(x, z, seed, 200) * 0.5

    -- Medium hills (medium frequency, medium amplitude)
    local medium = seededNoise(x, z, seed + 1, 80) * 0.3

    -- Small detail (high frequency, low amplitude)
    local small = seededNoise(x, z, seed + 2, 30) * 0.15

    -- Micro detail for texture
    local micro = seededNoise(x, z, seed + 3, 10) * 0.05

    return large + medium + small + micro
end

--[[
    Check if a point is within a flat zone
    Returns the zone if found, nil otherwise
]]
local function getZoneAtPoint(x: number, z: number): FlatZone?
    for _, zone in ipairs(FLAT_ZONES) do
        local dx = (x - zone.centerX) / zone.radiusX
        local dz = (z - zone.centerZ) / zone.radiusZ
        local distance = dx * dx + dz * dz

        if distance < 1 then
            return zone
        end
    end
    return nil
end

--[[
    Get blend factor for smooth transition at zone edges
    Returns 0 at zone center, 1 at zone edge
]]
local function getZoneBlendFactor(x: number, z: number, zone: FlatZone): number
    local dx = (x - zone.centerX) / zone.radiusX
    local dz = (z - zone.centerZ) / zone.radiusZ
    local distance = math.sqrt(dx * dx + dz * dz)

    -- Smooth blend in outer 30% of zone
    local blendStart = 0.7
    if distance < blendStart then
        return 0
    end
    return (distance - blendStart) / (1 - blendStart)
end

--[[
    Check if point is in Tiber River
    Returns distance to river center (0 = in river, >1 = outside)
]]
local function getDistanceToRiver(x: number, z: number): number
    local minDistance = math.huge

    -- Check distance to each river segment
    for i = 1, #TIBER_RIVER_PATH - 1 do
        local p1 = TIBER_RIVER_PATH[i]
        local p2 = TIBER_RIVER_PATH[i + 1]

        -- Project point onto line segment
        local dx = p2.x - p1.x
        local dz = p2.z - p1.z
        local length = math.sqrt(dx * dx + dz * dz)

        if length > 0 then
            local t = math.clamp(((x - p1.x) * dx + (z - p1.z) * dz) / (length * length), 0, 1)

            local projX = p1.x + t * dx
            local projZ = p1.z + t * dz
            local projWidth = p1.width + t * (p2.width - p1.width)

            local distToLine = math.sqrt((x - projX) ^ 2 + (z - projZ) ^ 2)
            local normalizedDist = distToLine / projWidth

            minDistance = math.min(minDistance, normalizedDist)
        end
    end

    return minDistance
end

--[[
    Calculate terrain height at a point
    Combines noise with flat zones and river carving
]]
function TerrainGenerator.getHeightAt(x: number, z: number): number
    local config = currentConfig

    -- Check cache first
    local cacheKey = string.format("%d,%d", math.floor(x), math.floor(z))
    if heightCache[cacheKey] then
        return heightCache[cacheKey]
    end

    -- Base noise height
    local noise = getLayeredNoise(x, z, config.seed)
    local height = config.baseHeight + (noise + 0.5) * (config.maxHeight - config.baseHeight)

    -- Check for flat zones
    local zone = getZoneAtPoint(x, z)
    if zone then
        local blend = getZoneBlendFactor(x, z, zone)
        height = zone.elevation * (1 - blend) + height * blend
    end

    -- Carve river valley
    local riverDist = getDistanceToRiver(x, z)
    if riverDist < 2 then
        local riverDepth = config.waterLevel - 5
        local riverBlend = math.clamp(riverDist, 0, 1)

        -- Smooth valley sides
        if riverDist < 1 then
            -- In river bed
            height = riverDepth
        else
            -- Valley slopes
            local valleyBlend = (riverDist - 1) / 1
            height = riverDepth + (height - riverDepth) * valleyBlend
        end
    end

    heightCache[cacheKey] = height
    return height
end

--[[
    Determine terrain material based on height, slope, and zone
]]
function TerrainGenerator.getMaterialAt(x: number, z: number, height: number): Enum.Material
    local config = currentConfig

    -- Check for flat zone material
    local zone = getZoneAtPoint(x, z)
    if zone then
        local blend = getZoneBlendFactor(x, z, zone)
        if blend < 0.5 then
            return zone.material
        end
    end

    -- Check for river/water
    local riverDist = getDistanceToRiver(x, z)
    if riverDist < 1.2 then
        return Enum.Material.Sand
    end

    -- Calculate slope by checking neighbor heights
    local dx = 2
    local h1 = TerrainGenerator.getHeightAt(x - dx, z)
    local h2 = TerrainGenerator.getHeightAt(x + dx, z)
    local h3 = TerrainGenerator.getHeightAt(x, z - dx)
    local h4 = TerrainGenerator.getHeightAt(x, z + dx)

    local slopeX = math.abs(h2 - h1) / (dx * 2)
    local slopeZ = math.abs(h4 - h3) / (dx * 2)
    local slope = math.max(slopeX, slopeZ)

    -- Steep slopes get rock
    if slope > 0.8 then
        return Enum.Material.Rock
    elseif slope > 0.5 then
        return Enum.Material.Slate
    end

    -- Height-based materials
    if height < config.waterLevel + 5 then
        return Enum.Material.Sand
    elseif height > config.maxHeight * 0.7 then
        return Enum.Material.Rock
    else
        return Enum.Material.Grass
    end
end

--[[
    Generate the terrain in workspace
]]
function TerrainGenerator.generate(config: TerrainConfig?)
    currentConfig = config or DEFAULT_CONFIG
    heightCache = {} -- Clear cache for new generation

    local terrain = workspace.Terrain
    local halfSize = currentConfig.mapSize / 2
    local resolution = currentConfig.resolution

    print("[TerrainGenerator] Starting terrain generation...")
    print(string.format("  Seed: %d", currentConfig.seed))
    print(string.format("  Map size: %d studs", currentConfig.mapSize))
    print(string.format("  Resolution: %d studs/cell", resolution))

    -- Clear existing terrain
    terrain:Clear()

    local startTime = os.clock()
    local cellCount = 0

    -- Generate terrain cells
    for x = -halfSize, halfSize, resolution do
        for z = -halfSize, halfSize, resolution do
            local height = TerrainGenerator.getHeightAt(x, z)
            local material = TerrainGenerator.getMaterialAt(x, z, height)

            -- Fill from bottom to height
            local region = Region3.new(
                Vector3.new(x - resolution/2, 0, z - resolution/2),
                Vector3.new(x + resolution/2, height, z + resolution/2)
            )

            terrain:FillBlock(
                CFrame.new(x, height/2, z),
                Vector3.new(resolution, height, resolution),
                material
            )

            cellCount += 1
        end

        -- Yield periodically to prevent timeout
        if cellCount % 1000 == 0 then
            task.wait()
        end
    end

    print(string.format("[TerrainGenerator] Terrain generated in %.2f seconds", os.clock() - startTime))
    print(string.format("  Total cells: %d", cellCount))

    -- Generate water for Tiber River
    TerrainGenerator.generateWater()

    print("[TerrainGenerator] Generation complete!")
end

--[[
    Generate water for the Tiber River using terrain water material
]]
function TerrainGenerator.generateWater()
    local config = currentConfig
    local terrain = workspace.Terrain
    local halfSize = config.mapSize / 2

    print("[TerrainGenerator] Generating Tiber River water...")

    -- Fill water in river areas
    for x = -halfSize, halfSize, config.resolution do
        for z = -halfSize, halfSize, config.resolution do
            local riverDist = getDistanceToRiver(x, z)

            if riverDist < 1 then
                -- Fill with water up to water level
                local waterHeight = config.waterLevel
                local riverBed = config.waterLevel - 5

                terrain:FillBlock(
                    CFrame.new(x, (riverBed + waterHeight) / 2, z),
                    Vector3.new(config.resolution, waterHeight - riverBed, config.resolution),
                    Enum.Material.Water
                )
            end
        end
    end

    -- Create invisible containment walls
    TerrainGenerator.createWaterContainment()
end

--[[
    Create invisible walls to contain water (swimming pool strategy)
]]
function TerrainGenerator.createWaterContainment()
    local config = currentConfig
    local halfSize = config.mapSize / 2

    -- Create container folder
    local container = Instance.new("Folder")
    container.Name = "RiverContainment"
    container.Parent = workspace

    -- Create invisible walls at map edges
    local wallHeight = config.waterLevel + 20
    local wallThickness = 5

    local walls = {
        {name = "North", pos = Vector3.new(0, wallHeight/2, halfSize), size = Vector3.new(config.mapSize, wallHeight, wallThickness)},
        {name = "South", pos = Vector3.new(0, wallHeight/2, -halfSize), size = Vector3.new(config.mapSize, wallHeight, wallThickness)},
        {name = "East", pos = Vector3.new(halfSize, wallHeight/2, 0), size = Vector3.new(wallThickness, wallHeight, config.mapSize)},
        {name = "West", pos = Vector3.new(-halfSize, wallHeight/2, 0), size = Vector3.new(wallThickness, wallHeight, config.mapSize)},
    }

    for _, wallData in ipairs(walls) do
        local wall = Instance.new("Part")
        wall.Name = wallData.name .. "Wall"
        wall.Position = wallData.pos
        wall.Size = wallData.size
        wall.Anchored = true
        wall.Transparency = 1 -- Invisible
        wall.CanCollide = true
        wall.Parent = container
    end

    print("[TerrainGenerator] Water containment walls created")
end

--[[
    Get flat zones configuration (for external use)
]]
function TerrainGenerator.getFlatZones(): {FlatZone}
    return FLAT_ZONES
end

--[[
    Get current configuration
]]
function TerrainGenerator.getConfig(): TerrainConfig
    return currentConfig
end

--[[
    Set seed for reproducible generation
]]
function TerrainGenerator.setSeed(seed: number)
    currentConfig.seed = seed
    heightCache = {}
end

return TerrainGenerator
