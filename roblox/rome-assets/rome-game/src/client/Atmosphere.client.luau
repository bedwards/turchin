--!strict
--[[
    Atmosphere.client.luau

    Dynamic atmospheric system that responds to game state.

    Features:
    - Sky color transitions with PSI: golden (peace) -> orange (tension) -> red (crisis)
    - Fog density driven by PSI (ominous during crisis) and terrain height
    - Low-lying areas (river valleys) have denser fog
    - Lighting brightness and ambient adjustments based on conditions
    - Smooth transitions using TweenService

    This creates visual feedback that reinforces the cliodynamics simulation,
    allowing players to literally see the health of Rome through the atmosphere.
]]

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local Lighting = game:GetService("Lighting")
local TweenService = game:GetService("TweenService")
local Players = game:GetService("Players")
local RunService = game:GetService("RunService")

-- Wait for RemoteEvents
local RemoteEvents = ReplicatedStorage:WaitForChild("RemoteEvents")
local StateUpdateEvent = RemoteEvents:WaitForChild("StateUpdate")
local PsiUpdateEvent = RemoteEvents:WaitForChild("PsiUpdate")

print("==============================================")
print("  ATMOSPHERE SYSTEM INITIALIZING")
print("==============================================")
print("")

-- Configuration
local TWEEN_DURATION = 2.0 -- Smooth transition duration
local BASE_CLOCK_TIME = 14 -- 2 PM, pleasant afternoon light
local TERRAIN_FOG_UPDATE_INTERVAL = 0.5 -- How often to check terrain height (seconds)

-- Terrain height thresholds for fog calculation
-- Based on TerrainGenerator: baseHeight=20, maxHeight=100, waterLevel=15
local TERRAIN_LOW_HEIGHT = 25 -- River valleys and low areas (near water level)
local TERRAIN_HIGH_HEIGHT = 60 -- Hilltops and elevated areas
local TERRAIN_HEIGHT_RANGE = TERRAIN_HIGH_HEIGHT - TERRAIN_LOW_HEIGHT

-- Fog density configuration
-- PSI is the primary driver, terrain provides local variation
local FOG_MIN_DENSITY = 0.15 -- Clear day, high ground, low PSI
local FOG_MAX_DENSITY = 0.85 -- Crisis fog, low ground, high PSI
local FOG_PSI_WEIGHT = 0.7 -- PSI contributes 70% to fog intensity
local FOG_TERRAIN_WEIGHT = 0.3 -- Terrain contributes 30% to fog intensity

-- Color presets for different stress levels
local SKY_COLORS = {
    peaceful = Color3.fromRGB(255, 220, 160), -- Golden warm
    stable = Color3.fromRGB(255, 200, 140), -- Soft gold
    tense = Color3.fromRGB(255, 180, 100), -- Orange gold
    troubled = Color3.fromRGB(255, 140, 80), -- Deep orange
    critical = Color3.fromRGB(255, 100, 60), -- Red orange
    crisis = Color3.fromRGB(200, 60, 60), -- Dark red
}

local AMBIENT_COLORS = {
    peaceful = Color3.fromRGB(180, 170, 160),
    stable = Color3.fromRGB(170, 160, 150),
    tense = Color3.fromRGB(160, 140, 130),
    troubled = Color3.fromRGB(140, 110, 100),
    critical = Color3.fromRGB(120, 80, 70),
    crisis = Color3.fromRGB(100, 60, 60),
}

-- Tween info for smooth transitions
local TWEEN_INFO = TweenInfo.new(TWEEN_DURATION, Enum.EasingStyle.Quad, Enum.EasingDirection.InOut)
local TERRAIN_TWEEN_INFO = TweenInfo.new(1.0, Enum.EasingStyle.Quad, Enum.EasingDirection.InOut)

-- Current state
local currentPsi: number = 0.05
local currentStateHealth: number = 0.8
local currentTerrainFactor: number = 0.5 -- 0 = high ground (less fog), 1 = low ground (more fog)
local atmosphere: Atmosphere?
local localPlayer = Players.LocalPlayer
local lastTerrainUpdate: number = 0

--[[
    Create or get the Atmosphere instance in Lighting.
]]
local function getOrCreateAtmosphere(): Atmosphere
    local existing = Lighting:FindFirstChildOfClass("Atmosphere")
    if existing then
        return existing
    end

    local atm = Instance.new("Atmosphere")
    atm.Name = "RomeAtmosphere"
    atm.Density = 0.3
    atm.Offset = 0
    atm.Color = Color3.fromRGB(199, 170, 107) -- Default golden haze
    atm.Decay = Color3.fromRGB(92, 60, 13)
    atm.Glare = 0.3
    atm.Haze = 2
    atm.Parent = Lighting

    return atm
end

--[[
    Initialize lighting settings for Rome atmosphere.
]]
local function initializeLighting()
    -- Set base lighting properties
    Lighting.ClockTime = BASE_CLOCK_TIME
    Lighting.GeographicLatitude = 41.9 -- Rome's latitude
    Lighting.Brightness = 2

    -- Warm outdoor ambient for Roman afternoon
    Lighting.OutdoorAmbient = Color3.fromRGB(180, 170, 160)
    Lighting.Ambient = Color3.fromRGB(80, 75, 70)

    -- Color correction for warmer tones
    local colorCorrection = Lighting:FindFirstChildOfClass("ColorCorrectionEffect")
    if not colorCorrection then
        colorCorrection = Instance.new("ColorCorrectionEffect")
        colorCorrection.Name = "RomeColorCorrection"
        colorCorrection.Parent = Lighting
    end
    colorCorrection.Brightness = 0
    colorCorrection.Contrast = 0.1
    colorCorrection.Saturation = 0.1
    colorCorrection.TintColor = Color3.fromRGB(255, 250, 240) -- Slight warm tint

    print("  Lighting configured for Roman afternoon")
end

--[[
    Interpolate between two Color3 values.
]]
local function lerpColor(c1: Color3, c2: Color3, t: number): Color3
    return Color3.new(c1.R + (c2.R - c1.R) * t, c1.G + (c2.G - c1.G) * t, c1.B + (c2.B - c1.B) * t)
end

--[[
    Calculate atmospheric color based on PSI with smooth interpolation.
]]
local function calculateAtmosphereColor(psi: number): Color3
    -- Map PSI to color gradient
    if psi < 0.1 then
        return lerpColor(SKY_COLORS.peaceful, SKY_COLORS.stable, psi / 0.1)
    elseif psi < 0.2 then
        return lerpColor(SKY_COLORS.stable, SKY_COLORS.tense, (psi - 0.1) / 0.1)
    elseif psi < 0.35 then
        return lerpColor(SKY_COLORS.tense, SKY_COLORS.troubled, (psi - 0.2) / 0.15)
    elseif psi < 0.55 then
        return lerpColor(SKY_COLORS.troubled, SKY_COLORS.critical, (psi - 0.35) / 0.2)
    elseif psi < 0.75 then
        return lerpColor(SKY_COLORS.critical, SKY_COLORS.crisis, (psi - 0.55) / 0.2)
    else
        return SKY_COLORS.crisis
    end
end

--[[
    Calculate ambient color based on PSI.
]]
local function calculateAmbientColor(psi: number): Color3
    if psi < 0.1 then
        return lerpColor(AMBIENT_COLORS.peaceful, AMBIENT_COLORS.stable, psi / 0.1)
    elseif psi < 0.2 then
        return lerpColor(AMBIENT_COLORS.stable, AMBIENT_COLORS.tense, (psi - 0.1) / 0.1)
    elseif psi < 0.35 then
        return lerpColor(AMBIENT_COLORS.tense, AMBIENT_COLORS.troubled, (psi - 0.2) / 0.15)
    elseif psi < 0.55 then
        return lerpColor(AMBIENT_COLORS.troubled, AMBIENT_COLORS.critical, (psi - 0.35) / 0.2)
    elseif psi < 0.75 then
        return lerpColor(AMBIENT_COLORS.critical, AMBIENT_COLORS.crisis, (psi - 0.55) / 0.2)
    else
        return AMBIENT_COLORS.crisis
    end
end

--[[
    Calculate terrain factor based on player's current height.
    Returns a value from 0 (high ground, less fog) to 1 (low ground, more fog).
]]
local function calculateTerrainFactor(): number
    if not localPlayer then
        return 0.5 -- Default mid-value if no player
    end

    local character = localPlayer.Character
    if not character then
        return 0.5
    end

    local rootPart = character:FindFirstChild("HumanoidRootPart") :: BasePart?
    if not rootPart then
        return 0.5
    end

    local playerY = rootPart.Position.Y

    -- Normalize height to terrain factor
    -- Low areas (near water/valleys) -> 1.0 (more fog)
    -- High areas (hilltops) -> 0.0 (less fog)
    local normalizedHeight = (playerY - TERRAIN_LOW_HEIGHT) / TERRAIN_HEIGHT_RANGE
    local terrainFactor = 1 - math.clamp(normalizedHeight, 0, 1)

    return terrainFactor
end

--[[
    Calculate fog density based on PSI and terrain height.
    PSI is the primary driver (70%), terrain provides local variation (30%).
]]
local function calculateFogDensity(psi: number, terrainFactor: number): number
    -- PSI contribution: Higher PSI = more fog (ominous atmosphere during crisis)
    -- PSI ranges from ~0.05 (peaceful) to ~1.0 (crisis)
    local psiContribution = math.clamp(psi, 0, 1)

    -- Terrain contribution: Lower terrain = more fog (valley mist effect)
    -- terrainFactor is 0 (high ground) to 1 (low ground)
    local terrainContribution = terrainFactor

    -- Weighted combination
    local combinedFactor = (psiContribution * FOG_PSI_WEIGHT) + (terrainContribution * FOG_TERRAIN_WEIGHT)

    -- Map to density range
    local density = FOG_MIN_DENSITY + (FOG_MAX_DENSITY - FOG_MIN_DENSITY) * combinedFactor

    return density
end

--[[
    Calculate lighting brightness based on conditions.
]]
local function calculateBrightness(psi: number, stateHealth: number): number
    -- Base brightness reduced slightly during crisis
    local psiFactor = 1 - psi * 0.3 -- 0% to 30% reduction
    local healthFactor = 0.8 + stateHealth * 0.2 -- 80% to 100%

    local brightness = 2 * psiFactor * healthFactor
    return math.clamp(brightness, 1.2, 2.5)
end

--[[
    Update atmosphere based on current game state and terrain.
]]
local function updateAtmosphere(psi: number, stateHealth: number, terrainFactor: number)
    if not atmosphere then
        return
    end

    -- Calculate target values
    local targetSkyColor = calculateAtmosphereColor(psi)
    local targetAmbientColor = calculateAmbientColor(psi)
    local targetFogDensity = calculateFogDensity(psi, terrainFactor)
    local targetBrightness = calculateBrightness(psi, stateHealth)

    -- Calculate haze based on PSI (primary) and terrain (secondary)
    -- More haze during crisis and in low areas
    local targetHaze = 1 + psi * 4 + terrainFactor * 2

    -- Tween atmosphere properties
    local atmTween = TweenService:Create(atmosphere, TWEEN_INFO, {
        Color = targetSkyColor,
        Density = targetFogDensity,
        Haze = targetHaze,
    })
    atmTween:Play()

    -- Tween lighting properties
    local lightingTween = TweenService:Create(Lighting, TWEEN_INFO, {
        OutdoorAmbient = targetAmbientColor,
        Brightness = targetBrightness,
    })
    lightingTween:Play()

    -- Update color correction for warmth/coldness
    local colorCorrection = Lighting:FindFirstChildOfClass("ColorCorrectionEffect")
    if colorCorrection then
        local warmth = 1 - psi -- High PSI = cooler (less warm)
        local tintColor = lerpColor(
            Color3.fromRGB(200, 180, 180), -- Cool tint for crisis
            Color3.fromRGB(255, 250, 240), -- Warm tint for peace
            warmth
        )
        local ccTween = TweenService:Create(colorCorrection, TWEEN_INFO, {
            TintColor = tintColor,
            Saturation = 0.1 - psi * 0.2, -- Desaturate during crisis
        })
        ccTween:Play()
    end
end

--[[
    Update fog based on terrain height only (for smooth local transitions).
    Uses faster tween for responsive terrain-based fog changes.
]]
local function updateTerrainFog(terrainFactor: number)
    if not atmosphere then
        return
    end

    local targetFogDensity = calculateFogDensity(currentPsi, terrainFactor)
    local targetHaze = 1 + currentPsi * 4 + terrainFactor * 2

    local atmTween = TweenService:Create(atmosphere, TERRAIN_TWEEN_INFO, {
        Density = targetFogDensity,
        Haze = targetHaze,
    })
    atmTween:Play()
end

--[[
    Handle PSI updates from server.
]]
PsiUpdateEvent.OnClientEvent:Connect(function(psi: number)
    currentPsi = psi
    updateAtmosphere(currentPsi, currentStateHealth, currentTerrainFactor)
end)

--[[
    Handle full state updates from server.
]]
StateUpdateEvent.OnClientEvent:Connect(function(state: { N: number, E: number, W: number, S: number, psi: number })
    currentPsi = state.psi
    currentStateHealth = state.S
    updateAtmosphere(currentPsi, currentStateHealth, currentTerrainFactor)
end)

--[[
    Continuous terrain height monitoring for local fog adjustments.
    Runs on heartbeat but throttled to TERRAIN_FOG_UPDATE_INTERVAL.
]]
RunService.Heartbeat:Connect(function()
    local currentTime = os.clock()
    if currentTime - lastTerrainUpdate < TERRAIN_FOG_UPDATE_INTERVAL then
        return
    end
    lastTerrainUpdate = currentTime

    local newTerrainFactor = calculateTerrainFactor()

    -- Only update if terrain factor changed significantly (avoid unnecessary tweens)
    if math.abs(newTerrainFactor - currentTerrainFactor) > 0.05 then
        currentTerrainFactor = newTerrainFactor
        updateTerrainFog(currentTerrainFactor)
    end
end)

-- Initialize
initializeLighting()
atmosphere = getOrCreateAtmosphere()

-- Set initial atmosphere
currentTerrainFactor = calculateTerrainFactor()
updateAtmosphere(currentPsi, currentStateHealth, currentTerrainFactor)

print("")
print("Atmosphere system running!")
print("  Sky color responds to Political Stress (PSI)")
print("  Fog density responds to PSI (primary) and terrain height (secondary)")
print("  Low-lying areas experience denser fog")
print(string.format("  Current state: PSI=%.2f, S=%.2f, Terrain=%.2f", currentPsi, currentStateHealth, currentTerrainFactor))
