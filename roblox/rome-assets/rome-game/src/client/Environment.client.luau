--!strict
--[[
    Environment.client.luau

    Environmental visual effects that respond to game state.

    Features:
    - Building condition visuals: darken/brighten based on State Health (S)
    - Market "busyness" indicator based on Wages (W)
    - Ambient particle dust in sunbeams
    - Visual feedback reinforcing the cliodynamics simulation

    This creates an immersive Rome where the health of the state
    is reflected in the environment itself.
]]

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local Lighting = game:GetService("Lighting")
local TweenService = game:GetService("TweenService")
local RunService = game:GetService("RunService")

-- Wait for RemoteEvents
local RemoteEvents = ReplicatedStorage:WaitForChild("RemoteEvents")
local StateUpdateEvent = RemoteEvents:WaitForChild("StateUpdate")

print("==============================================")
print("  ENVIRONMENT SYSTEM INITIALIZING")
print("==============================================")
print("")

-- Configuration
local TWEEN_DURATION = 2.0
local DUST_PARTICLE_COUNT = 50
local SUNBEAM_UPDATE_RATE = 0.5 -- How often to update dust positions

-- Current state
local currentStateHealth: number = 0.8
local currentWages: number = 0.7
-- currentPsi available for future PSI-based environmental effects

-- Tween info
local TWEEN_INFO = TweenInfo.new(TWEEN_DURATION, Enum.EasingStyle.Quad, Enum.EasingDirection.InOut)

-- Track building parts for darkening
local buildingParts: { [BasePart]: Color3 } = {} -- Part -> original color
local marketStalls: { BasePart } = {}

-- Dust particles container
local dustFolder: Folder?
local dustParticles: { Part } = {}

--[[
    Count entries in a dictionary table.
]]
local function countDictionary(dict: { [any]: any }): number
    local count = 0
    for _ in pairs(dict) do
        count = count + 1
    end
    return count
end

--[[
    Find and catalog all building parts for visual effects.
]]
local function catalogBuildings()
    -- Wait for buildings to be placed
    task.wait(8)

    -- Find Roman buildings folder
    local romanBuildings = workspace:FindFirstChild("RomanBuildings")
    if romanBuildings then
        for _, child in ipairs(romanBuildings:GetDescendants()) do
            if child:IsA("BasePart") then
                buildingParts[child] = child.Color
            end
        end
    end

    -- Find procedural buildings folder
    local proceduralBuildings = workspace:FindFirstChild("ProceduralBuildings")
    if proceduralBuildings then
        for _, child in ipairs(proceduralBuildings:GetDescendants()) do
            if child:IsA("BasePart") then
                buildingParts[child] = child.Color
            end
        end
    end

    print(string.format("[Environment] Cataloged %d building parts", countDictionary(buildingParts)))

    -- Find market stalls specifically
    -- Check RomanBuildings for direct children or descendants named MarketStall*
    if romanBuildings then
        for _, child in ipairs(romanBuildings:GetChildren()) do
            if child.Name:find("MarketStall") then
                -- If it's a BasePart (from AssetPlacer), add it directly
                if child:IsA("BasePart") then
                    table.insert(marketStalls, child)
                else
                    -- If it's a Model, add its descendant parts
                    for _, part in ipairs(child:GetDescendants()) do
                        if part:IsA("BasePart") then
                            table.insert(marketStalls, part)
                        end
                    end
                end
            end
        end
    end

    print(string.format("[Environment] Found %d market stall parts", #marketStalls))
end

--[[
    Darken or brighten a color by a factor.
    Factor > 1 = brighter, Factor < 1 = darker
]]
local function adjustBrightness(color: Color3, factor: number): Color3
    return Color3.new(
        math.clamp(color.R * factor, 0, 1),
        math.clamp(color.G * factor, 0, 1),
        math.clamp(color.B * factor, 0, 1)
    )
end

--[[
    Update building colors based on State Health.
    Low health = darker, grayer buildings (decay)
    High health = bright, clean buildings (prosperity)
]]
local function updateBuildingCondition(stateHealth: number)
    -- Map state health (0.1 to 1.0) to brightness factor (0.5 to 1.0)
    local normalizedHealth = math.clamp((stateHealth - 0.1) / 0.9, 0, 1)
    local brightnessFactor = 0.5 + normalizedHealth * 0.5

    -- Also desaturate during low health
    local saturationFactor = 0.6 + normalizedHealth * 0.4

    for part, originalColor in pairs(buildingParts) do
        if part and part.Parent then
            -- Adjust brightness
            local adjustedColor = adjustBrightness(originalColor, brightnessFactor)

            -- Apply desaturation by mixing with gray
            local gray = (adjustedColor.R + adjustedColor.G + adjustedColor.B) / 3
            local grayColor = Color3.new(gray, gray, gray)

            local finalColor = Color3.new(
                adjustedColor.R * saturationFactor + grayColor.R * (1 - saturationFactor),
                adjustedColor.G * saturationFactor + grayColor.G * (1 - saturationFactor),
                adjustedColor.B * saturationFactor + grayColor.B * (1 - saturationFactor)
            )

            -- Tween to new color
            local tween = TweenService:Create(part, TWEEN_INFO, {
                Color = finalColor,
            })
            tween:Play()
        end
    end
end

--[[
    Update market "busyness" visual based on wages.
    Higher wages = more active market (brighter stalls)
    Lower wages = quieter market (dimmer stalls)
]]
local function updateMarketBusyness(wages: number)
    if #marketStalls == 0 then
        return
    end

    -- Map wages (0.1 to 1.0) to pulsing intensity
    local normalizedWages = math.clamp((wages - 0.1) / 0.9, 0, 1)

    -- Brightness factor for market stalls
    local brightness = 0.7 + normalizedWages * 0.5 -- 0.7 to 1.2

    for _, stall in ipairs(marketStalls) do
        if stall and stall.Parent then
            -- Add a slight color shift toward warm tones when busy
            local warmth = normalizedWages * 0.1
            local baseColor = buildingParts[stall] or stall.Color

            local adjustedColor = Color3.new(
                math.clamp(baseColor.R * brightness + warmth, 0, 1),
                math.clamp(baseColor.G * brightness, 0, 1),
                math.clamp(baseColor.B * brightness - warmth * 0.5, 0, 1)
            )

            local tween = TweenService:Create(stall, TWEEN_INFO, {
                Color = adjustedColor,
            })
            tween:Play()
        end
    end
end

--[[
    Create ambient dust particles floating in sunbeams.
]]
local function createDustParticles()
    dustFolder = Instance.new("Folder")
    dustFolder.Name = "AmbientDust"
    dustFolder.Parent = workspace

    -- Create dust particles
    for i = 1, DUST_PARTICLE_COUNT do
        local dust = Instance.new("Part")
        dust.Name = "Dust_" .. i
        dust.Shape = Enum.PartType.Ball
        dust.Size = Vector3.new(0.3, 0.3, 0.3)
        dust.Material = Enum.Material.Neon
        dust.Color = Color3.fromRGB(255, 250, 220) -- Warm white
        dust.Transparency = 0.6
        dust.Anchored = true
        dust.CanCollide = false
        dust.CastShadow = false

        -- Random starting position (will be updated)
        dust.Position = Vector3.new(
            math.random(-50, 50),
            math.random(5, 30),
            math.random(-50, 50)
        )

        dust.Parent = dustFolder
        table.insert(dustParticles, dust)
    end

    print(string.format("[Environment] Created %d ambient dust particles", DUST_PARTICLE_COUNT))
end

--[[
    Update dust particle positions to float in sunbeams.
]]
local function updateDustParticles(_dt: number)
    -- Get sun direction from lighting
    local sunDirection = Vector3.new(
        math.sin(math.rad(Lighting.ClockTime * 15)),
        0.5,
        math.cos(math.rad(Lighting.ClockTime * 15))
    ).Unit

    for i, dust in ipairs(dustParticles) do
        if dust and dust.Parent then
            local pos = dust.Position

            -- Gentle floating motion
            local time = os.clock() + i * 0.5
            local floatX = math.sin(time * 0.3) * 0.02
            local floatY = math.sin(time * 0.5) * 0.01
            local floatZ = math.cos(time * 0.4) * 0.02

            -- Slowly drift in sunbeam direction
            local drift = sunDirection * 0.01

            local newPos = Vector3.new(
                pos.X + floatX + drift.X,
                pos.Y + floatY,
                pos.Z + floatZ + drift.Z
            )

            -- Reset if out of bounds
            if newPos.Y < 3 or newPos.Y > 40 or math.abs(newPos.X) > 100 or math.abs(newPos.Z) > 100 then
                newPos = Vector3.new(
                    math.random(-50, 50),
                    math.random(10, 35),
                    math.random(-50, 50)
                )
            end

            dust.Position = newPos

            -- Adjust transparency based on height (brighter in sunbeams)
            local heightFactor = math.clamp((pos.Y - 5) / 30, 0, 1)
            dust.Transparency = 0.4 + (1 - heightFactor) * 0.4
        end
    end
end

--[[
    Handle state updates from server.
]]
StateUpdateEvent.OnClientEvent:Connect(function(state: { N: number, E: number, W: number, S: number, psi: number })
    local oldStateHealth = currentStateHealth
    local oldWages = currentWages

    currentStateHealth = state.S
    currentWages = state.W
    -- state.psi available for future PSI-based effects

    -- Update building condition if state health changed significantly
    if math.abs(oldStateHealth - currentStateHealth) > 0.05 then
        updateBuildingCondition(currentStateHealth)
    end

    -- Update market busyness if wages changed significantly
    if math.abs(oldWages - currentWages) > 0.05 then
        updateMarketBusyness(currentWages)
    end
end)

-- Initialize
task.spawn(function()
    catalogBuildings()

    -- Initial update
    updateBuildingCondition(currentStateHealth)
    updateMarketBusyness(currentWages)
end)

-- Create ambient dust
createDustParticles()

-- Dust particle update loop
local dustAccumulator = 0
RunService.Heartbeat:Connect(function(dt: number)
    dustAccumulator = dustAccumulator + dt

    if dustAccumulator >= SUNBEAM_UPDATE_RATE then
        updateDustParticles(dustAccumulator)
        dustAccumulator = 0
    end
end)

print("")
print("Environment system running!")
print("  Building condition responds to State Health (S)")
print("  Market busyness responds to Wages (W)")
print("  Ambient dust particles floating in sunbeams")
print(string.format("  Initial state: S=%.2f, W=%.2f", currentStateHealth, currentWages))
