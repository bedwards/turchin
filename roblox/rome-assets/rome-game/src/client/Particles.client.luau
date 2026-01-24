--!strict
--[[
    Particles.client.luau

    Client-side particle system representing the population.

    Features:
    - 500 glowing spheres floating at ground level
    - Blue = commoners, Gold = elites
    - Basic wandering movement
    - Performance optimized for 60fps
]]

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService = game:GetService("RunService")

-- Wait for shared modules
local Shared = ReplicatedStorage:WaitForChild("Shared")
local TerrainUtils = require(Shared:WaitForChild("TerrainUtils"))

-- Wait for RemoteEvents
local RemoteEvents = ReplicatedStorage:WaitForChild("RemoteEvents")
local StateUpdateEvent = RemoteEvents:WaitForChild("StateUpdate")

-- Configuration
local TOTAL_PARTICLES = 500
local ELITE_RATIO = 0.1  -- 10% elites by default
local MAP_HALF_SIZE = 200 -- Particles spawn within this area
local PARTICLE_SIZE = 2
local WANDER_SPEED = 5
local HEIGHT_OFFSET = 3  -- Float above ground

-- Colors
local COMMONER_COLOR = Color3.fromRGB(100, 150, 255) -- Blue
local ELITE_COLOR = Color3.fromRGB(255, 215, 0)      -- Gold

-- Particle state
type Particle = {
    part: Part,
    isElite: boolean,
    targetX: number,
    targetZ: number,
    currentX: number,
    currentZ: number,
}

local particles: {Particle} = {}
local currentEliteRatio = ELITE_RATIO

-- Container for particles
local particleFolder: Folder

print("==============================================")
print("  PARTICLE SYSTEM INITIALIZING")
print("==============================================")
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
    Create a single particle.
]]
local function createParticle(index: number, isElite: boolean): Particle
    local x = math.random() * MAP_HALF_SIZE * 2 - MAP_HALF_SIZE
    local z = math.random() * MAP_HALF_SIZE * 2 - MAP_HALF_SIZE

    local part = Instance.new("Part")
    part.Name = if isElite then "Elite_" .. index else "Commoner_" .. index
    part.Shape = Enum.PartType.Ball
    part.Size = Vector3.new(PARTICLE_SIZE, PARTICLE_SIZE, PARTICLE_SIZE)
    part.Material = Enum.Material.Neon
    part.Color = if isElite then ELITE_COLOR else COMMONER_COLOR
    part.Anchored = true
    part.CanCollide = false
    part.CastShadow = false

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
    }
end

--[[
    Initialize all particles.
]]
local function initializeParticles()
    print(string.format("Creating %d particles...", TOTAL_PARTICLES))

    local eliteCount = math.floor(TOTAL_PARTICLES * ELITE_RATIO)
    local commonerCount = TOTAL_PARTICLES - eliteCount

    -- Create elites first
    for i = 1, eliteCount do
        local particle = createParticle(i, true)
        table.insert(particles, particle)
    end

    -- Create commoners
    for i = 1, commonerCount do
        local particle = createParticle(i, false)
        table.insert(particles, particle)
    end

    print(string.format("  Created %d elites (gold)", eliteCount))
    print(string.format("  Created %d commoners (blue)", commonerCount))
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
    Update particle position toward target.
]]
local function updateParticle(particle: Particle, dt: number)
    -- Calculate direction to target
    local dx = particle.targetX - particle.currentX
    local dz = particle.targetZ - particle.currentZ
    local distance = math.sqrt(dx * dx + dz * dz)

    -- If close to target, pick new target
    if distance < 5 then
        pickNewTarget(particle)
        return
    end

    -- Move toward target
    local moveDistance = WANDER_SPEED * dt
    if moveDistance > distance then
        moveDistance = distance
    end

    local dirX = dx / distance
    local dirZ = dz / distance

    particle.currentX = particle.currentX + dirX * moveDistance
    particle.currentZ = particle.currentZ + dirZ * moveDistance

    -- Update part position
    local groundY = TerrainUtils.getGroundY(particle.currentX, particle.currentZ)
    particle.part.Position = Vector3.new(
        particle.currentX,
        groundY + HEIGHT_OFFSET,
        particle.currentZ
    )
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

    if difference > 0 then
        -- Need more elites - convert commoners
        local converted = 0
        for _, particle in ipairs(particles) do
            if not particle.isElite and converted < difference then
                particle.isElite = true
                particle.part.Color = ELITE_COLOR
                particle.part.Name = "Elite_converted"
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
                particle.part.Name = "Commoner_converted"
                converted = converted + 1
            end
        end
    end
end

--[[
    Handle state updates from server.
]]
StateUpdateEvent.OnClientEvent:Connect(function(state: {N: number, E: number, W: number, S: number, psi: number})
    -- Elite ratio is E relative to typical elite population
    -- E ranges from 0.01 to 0.5, normalize to 0-0.3 range for particle display
    local eliteRatio = math.clamp(state.E / 0.5 * 0.3, 0.02, 0.3)
    updateEliteRatio(eliteRatio)
end)

-- Initialize
initializeContainer()

-- Wait a moment for terrain to generate
print("Waiting for terrain...")
task.wait(3)

initializeParticles()

-- Main update loop
local updateAccumulator = 0
local UPDATE_INTERVAL = 1/30 -- 30 Hz update for particles

RunService.Heartbeat:Connect(function(dt: number)
    updateAccumulator = updateAccumulator + dt

    if updateAccumulator >= UPDATE_INTERVAL then
        -- Update all particles
        for _, particle in ipairs(particles) do
            updateParticle(particle, updateAccumulator)
        end

        updateAccumulator = 0
    end
end)

print("")
print("Particle system running!")
print(string.format("  %d particles wandering the terrain", TOTAL_PARTICLES))
print("  Performance target: 60 FPS")
