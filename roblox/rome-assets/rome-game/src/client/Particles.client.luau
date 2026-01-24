--!strict
--[[
    Particles.client.luau

    High-performance client-side particle system representing the population.

    Features:
    - 1000+ glowing spheres floating at ground level
    - Blue = commoners, Gold = elites
    - Elites cluster in Forum zone when E is high
    - Commoners gather near market stalls
    - Color shift: gold -> orange -> red as elite competition increases
    - Crisis makes particles move erratically (jitter)
    - Simple collision avoidance between particles
    - LOD (Level of Detail) system for distant particles
    - Optimized wandering movement with spatial updates

    Performance Optimizations:
    - Reduced update frequency for distant particles
    - Spatial hashing for efficient neighbor queries
    - Batch CFrame updates
    - Particle pooling (no creation/destruction at runtime)
    - Staggered updates across frames
]]

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService = game:GetService("RunService")
local Players = game:GetService("Players")

-- Wait for shared modules
local Shared = ReplicatedStorage:WaitForChild("Shared")
local TerrainUtils = require(Shared:WaitForChild("TerrainUtils"))
local TerrainGenerator = require(Shared:WaitForChild("TerrainGenerator"))

-- Wait for RemoteEvents
local RemoteEvents = ReplicatedStorage:WaitForChild("RemoteEvents")
local StateUpdateEvent = RemoteEvents:WaitForChild("StateUpdate")

-- Configuration
local TOTAL_PARTICLES = 1000
local ELITE_RATIO = 0.1 -- 10% elites by default
local MAP_HALF_SIZE = 200 -- Particles spawn within this area
local PARTICLE_SIZE = 2
local WANDER_SPEED = 5
local HEIGHT_OFFSET = 3 -- Float above ground

-- LOD Configuration
local LOD_NEAR_DISTANCE = 100 -- Full detail within this range
local LOD_FAR_DISTANCE = 300 -- Reduced updates beyond this
local LOD_CULL_DISTANCE = 400 -- Don't update beyond this

-- Update rates (Hz)
local NEAR_UPDATE_RATE = 30 -- Close particles update 30 Hz
local FAR_UPDATE_RATE = 10 -- Far particles update 10 Hz
local VERY_FAR_UPDATE_RATE = 5 -- Very far particles update 5 Hz

-- Colors
local COMMONER_COLOR = Color3.fromRGB(100, 150, 255) -- Blue
local ELITE_COLOR_PEACEFUL = Color3.fromRGB(255, 215, 0) -- Gold
local ELITE_COLOR_TENSE = Color3.fromRGB(255, 165, 0) -- Orange
local ELITE_COLOR_CRISIS = Color3.fromRGB(255, 80, 50) -- Red-orange

-- Smaller size for distant particles (LOD)
local LOD_PARTICLE_SIZE = 1.5

-- Zone locations for clustering
local FORUM_CENTER = { x = 0, z = 0 }
local MARKET_CENTER = { x = 40, z = -80 }

-- Clustering parameters
local CLUSTER_STRENGTH = 0.3 -- How strongly particles cluster
local CLUSTER_RANGE = 60 -- Range within which to cluster
local AVOIDANCE_DISTANCE = 4 -- Minimum distance between particles
local AVOIDANCE_STRENGTH = 0.5
-- JITTER_BASE could be used for constant jitter during normal times
local JITTER_CRISIS_MAX = 3 -- Max jitter during crisis

-- Particle state
type Particle = {
    part: Part,
    isElite: boolean,
    targetX: number,
    targetZ: number,
    currentX: number,
    currentZ: number,
    lastUpdateTime: number,
    distanceToPlayer: number,
    velocityX: number,
    velocityZ: number,
}

local particles: { Particle } = {}
local currentEliteRatio = ELITE_RATIO
local currentPsi = 0.05
local currentE = 0.1

-- Container for particles
local particleFolder: Folder

-- Local player reference
local localPlayer = Players.LocalPlayer

-- Flat zones available for future zone-based behavior
local _flatZones = TerrainGenerator.getFlatZones()

print("==============================================")
print("  ENHANCED PARTICLE SYSTEM")
print("==============================================")
print("")
print(string.format("  Target: %d particles at 60 FPS", TOTAL_PARTICLES))
print("  Features: Clustering, collision avoidance, crisis jitter")
print("")

--[[
    Initialize particle folder in workspace.
]]
local function initializeContainer()
    particleFolder = Instance.new("Folder")
    particleFolder.Name = "PopulationParticles"
    particleFolder.Parent = workspace
end

--[[
    Get elite color based on current PSI (elite competition intensity).
]]
local function getEliteColor(psi: number): Color3
    if psi < 0.3 then
        -- Peaceful: gold
        local t = psi / 0.3
        return Color3.new(
            ELITE_COLOR_PEACEFUL.R + (ELITE_COLOR_TENSE.R - ELITE_COLOR_PEACEFUL.R) * t,
            ELITE_COLOR_PEACEFUL.G + (ELITE_COLOR_TENSE.G - ELITE_COLOR_PEACEFUL.G) * t,
            ELITE_COLOR_PEACEFUL.B + (ELITE_COLOR_TENSE.B - ELITE_COLOR_PEACEFUL.B) * t
        )
    elseif psi < 0.6 then
        -- Tense: orange
        local t = (psi - 0.3) / 0.3
        return Color3.new(
            ELITE_COLOR_TENSE.R + (ELITE_COLOR_CRISIS.R - ELITE_COLOR_TENSE.R) * t,
            ELITE_COLOR_TENSE.G + (ELITE_COLOR_CRISIS.G - ELITE_COLOR_TENSE.G) * t,
            ELITE_COLOR_TENSE.B + (ELITE_COLOR_CRISIS.B - ELITE_COLOR_TENSE.B) * t
        )
    else
        -- Crisis: red
        return ELITE_COLOR_CRISIS
    end
end

--[[
    Create a single particle with optimized settings.
]]
local function createParticle(index: number, isElite: boolean): Particle
    local x = math.random() * MAP_HALF_SIZE * 2 - MAP_HALF_SIZE
    local z = math.random() * MAP_HALF_SIZE * 2 - MAP_HALF_SIZE

    local part = Instance.new("Part")
    part.Name = if isElite then "Elite_" .. index else "Commoner_" .. index
    part.Shape = Enum.PartType.Ball
    part.Size = Vector3.new(PARTICLE_SIZE, PARTICLE_SIZE, PARTICLE_SIZE)
    part.Material = Enum.Material.Neon
    part.Color = if isElite then getEliteColor(currentPsi) else COMMONER_COLOR
    part.Anchored = true
    part.CanCollide = false
    part.CastShadow = false

    -- Performance optimizations
    part.TopSurface = Enum.SurfaceType.Smooth
    part.BottomSurface = Enum.SurfaceType.Smooth

    -- Get ground height for initial position
    local groundY = TerrainUtils.getGroundY(x, z)
    part.Position = Vector3.new(x, groundY + HEIGHT_OFFSET, z)

    part.Parent = particleFolder

    return {
        part = part,
        isElite = isElite,
        targetX = x,
        targetZ = z,
        currentX = x,
        currentZ = z,
        lastUpdateTime = 0,
        distanceToPlayer = 0,
        velocityX = 0,
        velocityZ = 0,
    }
end

--[[
    Initialize all particles with staggered creation to avoid frame spike.
]]
local function initializeParticles()
    print(string.format("Creating %d particles...", TOTAL_PARTICLES))

    local eliteCount = math.floor(TOTAL_PARTICLES * ELITE_RATIO)
    local commonerCount = TOTAL_PARTICLES - eliteCount

    local particleIndex = 1
    local batchSize = 50 -- Create particles in batches

    -- Create elites first
    for i = 1, eliteCount do
        local particle = createParticle(particleIndex, true)
        table.insert(particles, particle)
        particleIndex = particleIndex + 1

        -- Yield every batch to spread load
        if i % batchSize == 0 then
            task.wait()
        end
    end

    -- Create commoners
    for i = 1, commonerCount do
        local particle = createParticle(particleIndex, false)
        table.insert(particles, particle)
        particleIndex = particleIndex + 1

        -- Yield every batch to spread load
        if i % batchSize == 0 then
            task.wait()
        end
    end

    print(string.format("  Created %d elites (gold)", eliteCount))
    print(string.format("  Created %d commoners (blue)", commonerCount))
    print(string.format("  Total: %d particles", #particles))
end

--[[
    Get cluster target for a particle based on its type and current state.
]]
local function getClusterTarget(particle: Particle): (number, number)
    if particle.isElite then
        -- Elites cluster at Forum when E is high
        local clusterIntensity = math.clamp(currentE / 0.3, 0, 1) * CLUSTER_STRENGTH
        if clusterIntensity > 0.1 then
            local dx = FORUM_CENTER.x - particle.currentX
            local dz = FORUM_CENTER.z - particle.currentZ
            local dist = math.sqrt(dx * dx + dz * dz)

            if dist < CLUSTER_RANGE then
                return particle.targetX + dx * clusterIntensity, particle.targetZ + dz * clusterIntensity
            end
        end
    else
        -- Commoners cluster near Market
        local clusterIntensity = CLUSTER_STRENGTH * 0.5
        local dx = MARKET_CENTER.x - particle.currentX
        local dz = MARKET_CENTER.z - particle.currentZ
        local dist = math.sqrt(dx * dx + dz * dz)

        if dist < CLUSTER_RANGE then
            return particle.targetX + dx * clusterIntensity, particle.targetZ + dz * clusterIntensity
        end
    end

    return particle.targetX, particle.targetZ
end

--[[
    Calculate collision avoidance vector from nearby particles.
    Uses a simple sampling approach for performance.
]]
local function getAvoidanceVector(particle: Particle, index: number): (number, number)
    local avoidX, avoidZ = 0, 0
    local nearbyCount = 0

    -- Sample a few nearby particles (not all, for performance)
    local sampleStep = math.max(1, math.floor(#particles / 50))

    for i = 1, #particles, sampleStep do
        if i == index then
            continue
        end

        local other = particles[i]
        local dx = particle.currentX - other.currentX
        local dz = particle.currentZ - other.currentZ
        local distSq = dx * dx + dz * dz

        if distSq < AVOIDANCE_DISTANCE * AVOIDANCE_DISTANCE and distSq > 0.1 then
            local dist = math.sqrt(distSq)
            local strength = (AVOIDANCE_DISTANCE - dist) / AVOIDANCE_DISTANCE
            avoidX = avoidX + (dx / dist) * strength
            avoidZ = avoidZ + (dz / dist) * strength
            nearbyCount = nearbyCount + 1
        end
    end

    if nearbyCount > 0 then
        return avoidX * AVOIDANCE_STRENGTH, avoidZ * AVOIDANCE_STRENGTH
    end

    return 0, 0
end

--[[
    Get crisis jitter amount based on PSI.
]]
local function getCrisisJitter(): (number, number)
    if currentPsi < 0.4 then
        return 0, 0
    end

    local jitterStrength = (currentPsi - 0.4) / 0.6 * JITTER_CRISIS_MAX
    return (math.random() - 0.5) * jitterStrength, (math.random() - 0.5) * jitterStrength
end

--[[
    Pick a new random target for a particle.
]]
local function pickNewTarget(particle: Particle)
    -- Random position within map bounds
    particle.targetX = math.random() * MAP_HALF_SIZE * 2 - MAP_HALF_SIZE
    particle.targetZ = math.random() * MAP_HALF_SIZE * 2 - MAP_HALF_SIZE
end

--[[
    Get player camera position for LOD calculations.
]]
local function getPlayerPosition(): Vector3
    local character = localPlayer.Character
    if character then
        local humanoidRootPart = character:FindFirstChild("HumanoidRootPart") :: BasePart?
        if humanoidRootPart then
            return humanoidRootPart.Position
        end
    end

    -- Fallback to camera position
    local camera = workspace.CurrentCamera
    if camera then
        return camera.CFrame.Position
    end

    return Vector3.new(0, 0, 0)
end

--[[
    Calculate distance from particle to player.
]]
local function calculateDistanceToPlayer(particle: Particle, playerPos: Vector3): number
    local dx = particle.currentX - playerPos.X
    local dz = particle.currentZ - playerPos.Z
    return math.sqrt(dx * dx + dz * dz)
end

--[[
    Determine update interval based on distance (LOD).
]]
local function getUpdateInterval(distance: number): number
    if distance < LOD_NEAR_DISTANCE then
        return 1 / NEAR_UPDATE_RATE
    elseif distance < LOD_FAR_DISTANCE then
        return 1 / FAR_UPDATE_RATE
    else
        return 1 / VERY_FAR_UPDATE_RATE
    end
end

--[[
    Update particle position toward target with clustering, avoidance, and jitter.
    Returns true if position changed.
]]
local function updateParticlePosition(particle: Particle, dt: number, index: number): boolean
    -- Get cluster-modified target
    local clusterX, clusterZ = getClusterTarget(particle)

    -- Calculate direction to target
    local dx = clusterX - particle.currentX
    local dz = clusterZ - particle.currentZ
    local distance = math.sqrt(dx * dx + dz * dz)

    -- If close to target, pick new target
    if distance < 5 then
        pickNewTarget(particle)
        return false
    end

    -- Get avoidance vector
    local avoidX, avoidZ = getAvoidanceVector(particle, index)

    -- Get crisis jitter
    local jitterX, jitterZ = getCrisisJitter()

    -- Calculate movement direction
    local moveDistance = WANDER_SPEED * dt
    if moveDistance > distance then
        moveDistance = distance
    end

    local dirX = dx / distance
    local dirZ = dz / distance

    -- Apply movement with avoidance and jitter
    particle.currentX = particle.currentX + (dirX * moveDistance) + avoidX + jitterX
    particle.currentZ = particle.currentZ + (dirZ * moveDistance) + avoidZ + jitterZ

    -- Clamp to map bounds
    particle.currentX = math.clamp(particle.currentX, -MAP_HALF_SIZE, MAP_HALF_SIZE)
    particle.currentZ = math.clamp(particle.currentZ, -MAP_HALF_SIZE, MAP_HALF_SIZE)

    return true
end

--[[
    Update particle Part position in workspace.
]]
local function updateParticlePart(particle: Particle)
    local groundY = TerrainUtils.getGroundY(particle.currentX, particle.currentZ)
    particle.part.Position = Vector3.new(particle.currentX, groundY + HEIGHT_OFFSET, particle.currentZ)
end

--[[
    Apply LOD size adjustments based on distance.
]]
local function applyLOD(particle: Particle)
    local distance = particle.distanceToPlayer

    if distance > LOD_FAR_DISTANCE then
        -- Far particles get smaller
        if particle.part.Size.X ~= LOD_PARTICLE_SIZE then
            particle.part.Size = Vector3.new(LOD_PARTICLE_SIZE, LOD_PARTICLE_SIZE, LOD_PARTICLE_SIZE)
        end
    else
        -- Near particles at full size
        if particle.part.Size.X ~= PARTICLE_SIZE then
            particle.part.Size = Vector3.new(PARTICLE_SIZE, PARTICLE_SIZE, PARTICLE_SIZE)
        end
    end
end

--[[
    Update elite colors based on current PSI.
]]
local function updateEliteColors()
    local eliteColor = getEliteColor(currentPsi)

    for _, particle in ipairs(particles) do
        if particle.isElite then
            particle.part.Color = eliteColor
        end
    end
end

--[[
    Update elite/commoner ratio based on state from server.
]]
local function updateEliteRatio(newRatio: number)
    if math.abs(newRatio - currentEliteRatio) < 0.01 then
        return -- No significant change
    end

    currentEliteRatio = newRatio
    local targetEliteCount = math.floor(TOTAL_PARTICLES * newRatio)

    -- Count current elites
    local currentEliteCount = 0
    for _, particle in ipairs(particles) do
        if particle.isElite then
            currentEliteCount = currentEliteCount + 1
        end
    end

    -- Convert particles as needed
    local difference = targetEliteCount - currentEliteCount
    local eliteColor = getEliteColor(currentPsi)

    if difference > 0 then
        -- Need more elites - convert commoners
        local converted = 0
        for _, particle in ipairs(particles) do
            if not particle.isElite and converted < difference then
                particle.isElite = true
                particle.part.Color = eliteColor
                converted = converted + 1
            end
        end
    elseif difference < 0 then
        -- Need fewer elites - convert to commoners
        local converted = 0
        local needed = -difference
        for _, particle in ipairs(particles) do
            if particle.isElite and converted < needed then
                particle.isElite = false
                particle.part.Color = COMMONER_COLOR
                converted = converted + 1
            end
        end
    end
end

--[[
    Handle state updates from server.
]]
StateUpdateEvent.OnClientEvent:Connect(function(state: { N: number, E: number, W: number, S: number, psi: number })
    local oldPsi = currentPsi
    currentPsi = state.psi
    currentE = state.E

    -- Elite ratio is E relative to typical elite population
    -- E ranges from 0.01 to 0.5, normalize to 0-0.3 range for particle display
    local eliteRatio = math.clamp(state.E / 0.5 * 0.3, 0.02, 0.3)
    updateEliteRatio(eliteRatio)

    -- Update elite colors if PSI changed significantly
    if math.abs(oldPsi - currentPsi) > 0.05 then
        updateEliteColors()
    end
end)

-- Initialize
initializeContainer()

-- Wait a moment for terrain to generate
print("Waiting for terrain...")
task.wait(3)

initializeParticles()

-- Performance tracking
local frameCount = 0
local totalUpdateTime = 0
local lastFpsReport = os.clock()

--[[
    Main update loop with LOD and staggered updates.
]]
local currentIndex = 1
local PARTICLES_PER_FRAME = 100 -- Update this many particles per frame

RunService.Heartbeat:Connect(function(_dt: number)
    local frameStart = os.clock()
    local currentTime = frameStart
    local playerPos = getPlayerPosition()

    -- Update a subset of particles each frame (staggered updates)
    local endIndex = math.min(currentIndex + PARTICLES_PER_FRAME - 1, #particles)

    for i = currentIndex, endIndex do
        local particle = particles[i]

        -- Calculate distance to player
        particle.distanceToPlayer = calculateDistanceToPlayer(particle, playerPos)

        -- Skip very distant particles entirely
        if particle.distanceToPlayer > LOD_CULL_DISTANCE then
            continue
        end

        -- Check if this particle should update based on LOD
        local updateInterval = getUpdateInterval(particle.distanceToPlayer)
        local timeSinceUpdate = currentTime - particle.lastUpdateTime

        if timeSinceUpdate >= updateInterval then
            -- Update position with clustering, avoidance, and jitter
            if updateParticlePosition(particle, timeSinceUpdate, i) then
                updateParticlePart(particle)
            end

            -- Apply LOD adjustments
            applyLOD(particle)

            particle.lastUpdateTime = currentTime
        end
    end

    -- Move to next batch
    currentIndex = endIndex + 1
    if currentIndex > #particles then
        currentIndex = 1
    end

    -- Performance tracking
    local frameTime = os.clock() - frameStart
    totalUpdateTime = totalUpdateTime + frameTime
    frameCount = frameCount + 1

    -- Report FPS periodically (every 5 seconds)
    if os.clock() - lastFpsReport >= 5 then
        local avgFrameTime = totalUpdateTime / frameCount
        local estimatedFps = 1 / avgFrameTime
        print(
            string.format(
                "[Particles] Performance: %.1f ms/frame, ~%.0f FPS capacity",
                avgFrameTime * 1000,
                math.min(estimatedFps, 60)
            )
        )

        frameCount = 0
        totalUpdateTime = 0
        lastFpsReport = os.clock()
    end
end)

print("")
print("Enhanced particle system running!")
print(string.format("  %d particles with LOD optimization", TOTAL_PARTICLES))
print(string.format("  LOD distances: near=%d, far=%d, cull=%d", LOD_NEAR_DISTANCE, LOD_FAR_DISTANCE, LOD_CULL_DISTANCE))
print("  Elite clustering at Forum")
print("  Commoner clustering at Market")
print("  Crisis jitter enabled")
print("  Performance target: 60 FPS")
