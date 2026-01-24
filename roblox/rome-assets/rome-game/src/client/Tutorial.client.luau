--!strict
--[[
    Tutorial.client.luau

    3D tutorial hints system for guiding new players.

    Features:
    - Glowing path particles leading to first control
    - Floating BillboardGui text above controls
    - "Stand on green to increase" hint text
    - Hints fade after player uses each control once
    - Stores hint state locally (per session)

    Tutorial Flow:
    1. On first play, show glowing path to Forum (PSI control)
    2. Show floating hint text above controls
    3. Each control's hint fades after first use
    4. Path particles fade after player reaches Forum
]]

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local TweenService = game:GetService("TweenService")
local RunService = game:GetService("RunService")
local Players = game:GetService("Players")

-- Wait for shared modules
local Shared = ReplicatedStorage:WaitForChild("Shared")
local AssetConfig = require(Shared:WaitForChild("AssetConfig"))
local TerrainUtils = require(Shared:WaitForChild("TerrainUtils"))

-- Wait for RemoteEvents
local RemoteEvents = ReplicatedStorage:WaitForChild("RemoteEvents")
local ControlInputEvent = RemoteEvents:WaitForChild("ControlInput")

print("==============================================")
print("  TUTORIAL SYSTEM INITIALIZING")
print("==============================================")
print("")

-- Configuration
local HINT_FADE_DURATION = 2.0
local PATH_PARTICLE_COUNT = 20
-- PATH_PARTICLE_SPACING could be used for variable-spaced particles

-- Tutorial state (per session, not persisted)
local tutorialState = {
    pathShown = true,
    controlsUsed = {
        decrease_psi = false,
        increase_psi = false,
        decrease_alpha = false,
        increase_alpha = false,
        decrease_gamma = false,
        increase_gamma = false,
        decrease_delta = false,
        increase_delta = false,
    },
    hintsVisible = true,
}

-- Control station locations (must match Controls.server.luau)
local STATION_LOCATIONS = {
    forum = { x = 0, z = 0, name = "POLITICAL STRESS" },
    senateForum = { x = 30, z = -25, name = "SENATE SEATS" },
    market = { x = 40, z = -80, name = "GRAIN DOLE" },
    templeDistrict = { x = -80, z = 60, name = "TAX COLLECTORS" },
}

-- Tween info
local FADE_TWEEN_INFO = TweenInfo.new(HINT_FADE_DURATION, Enum.EasingStyle.Quad, Enum.EasingDirection.Out)

-- Local player
local localPlayer = Players.LocalPlayer

-- Containers
local tutorialFolder: Folder?
local pathParticles: { Part } = {}
local hintBillboards: { BillboardGui } = {}

--[[
    Create a glowing path particle.
]]
local function createPathParticle(x: number, z: number, index: number): Part
    local groundY = TerrainUtils.getGroundY(x, z)

    local particle = Instance.new("Part")
    particle.Name = "PathParticle_" .. index
    particle.Shape = Enum.PartType.Ball
    particle.Size = Vector3.new(1.5, 1.5, 1.5)
    particle.Material = Enum.Material.Neon
    particle.Color = AssetConfig.Tutorial.PathParticleColor
    particle.Anchored = true
    particle.CanCollide = false
    particle.CastShadow = false
    particle.Position = Vector3.new(x, groundY + 2, z)

    -- Add pulsing light
    local light = Instance.new("PointLight")
    light.Name = "ParticleLight"
    light.Color = AssetConfig.Tutorial.PathParticleColor
    light.Brightness = 1
    light.Range = 10
    light.Parent = particle

    return particle
end

--[[
    Create path particles from spawn to Forum.
]]
local function createPathToForum()
    if not tutorialFolder then
        return
    end

    -- Calculate path from spawn (assumed 0, 0, -50) to Forum (0, 0)
    local startX, startZ = 0, -50
    local endX, endZ = STATION_LOCATIONS.forum.x, STATION_LOCATIONS.forum.z

    for i = 1, PATH_PARTICLE_COUNT do
        local t = (i - 1) / (PATH_PARTICLE_COUNT - 1)
        local x = startX + (endX - startX) * t
        local z = startZ + (endZ - startZ) * t

        local particle = createPathParticle(x, z, i)
        particle.Parent = tutorialFolder

        table.insert(pathParticles, particle)
    end

    print(string.format("[Tutorial] Created %d path particles to Forum", PATH_PARTICLE_COUNT))
end

--[[
    Animate path particles with pulsing glow.
]]
local function animatePathParticles(_dt: number)
    local time = os.clock()

    for i, particle in ipairs(pathParticles) do
        if particle and particle.Parent then
            -- Wave effect along path
            local phase = time * 3 + i * 0.3
            local scale = 0.8 + math.sin(phase) * 0.3

            particle.Size = Vector3.new(1.5 * scale, 1.5 * scale, 1.5 * scale)

            local light = particle:FindFirstChild("ParticleLight") :: PointLight?
            if light then
                light.Brightness = 0.8 + math.sin(phase) * 0.4
            end

            -- Bobbing motion
            local groundY = TerrainUtils.getGroundY(particle.Position.X, particle.Position.Z)
            particle.Position = Vector3.new(
                particle.Position.X,
                groundY + 2 + math.sin(phase) * 0.5,
                particle.Position.Z
            )
        end
    end
end

--[[
    Fade out path particles.
]]
local function fadePathParticles()
    for _, particle in ipairs(pathParticles) do
        if particle and particle.Parent then
            local tween = TweenService:Create(particle, FADE_TWEEN_INFO, {
                Transparency = 1,
            })
            tween:Play()

            local light = particle:FindFirstChild("ParticleLight") :: PointLight?
            if light then
                local lightTween = TweenService:Create(light, FADE_TWEEN_INFO, {
                    Brightness = 0,
                })
                lightTween:Play()
            end
        end
    end

    tutorialState.pathShown = false
    print("[Tutorial] Path particles fading")
end

--[[
    Create a floating hint billboard above a control (subtle, smaller).
]]
local function createHintBillboard(x: number, z: number, stationName: string): BillboardGui
    local groundY = TerrainUtils.getGroundY(x, z)

    -- Create anchor part
    local anchor = Instance.new("Part")
    anchor.Name = stationName .. "_HintAnchor"
    anchor.Size = Vector3.new(1, 1, 1)
    anchor.Position = Vector3.new(x, groundY + 8, z)
    anchor.Anchored = true
    anchor.Transparency = 1
    anchor.CanCollide = false
    anchor.Parent = tutorialFolder

    -- Create billboard (smaller, not always on top)
    local billboard = Instance.new("BillboardGui")
    billboard.Name = stationName .. "_Hint"
    billboard.Size = UDim2.new(0, 120, 0, 35)
    billboard.StudsOffset = Vector3.new(0, 0, 0)
    billboard.Adornee = anchor
    billboard.AlwaysOnTop = false
    billboard.Parent = anchor

    -- Background frame (more transparent)
    local frame = Instance.new("Frame")
    frame.Name = "Background"
    frame.Size = UDim2.new(1, 0, 1, 0)
    frame.BackgroundColor3 = Color3.fromRGB(30, 30, 40)
    frame.BackgroundTransparency = 0.6
    frame.BorderSizePixel = 0
    frame.Parent = billboard

    local corner = Instance.new("UICorner")
    corner.CornerRadius = UDim.new(0, 4)
    corner.Parent = frame

    -- Hint text (simplified, smaller)
    local hintLabel = Instance.new("TextLabel")
    hintLabel.Name = "HintText"
    hintLabel.Size = UDim2.new(1, -6, 1, -6)
    hintLabel.Position = UDim2.new(0, 3, 0, 3)
    hintLabel.BackgroundTransparency = 1
    hintLabel.Text = "Step on plates"
    hintLabel.TextColor3 = AssetConfig.Tutorial.HintColor
    hintLabel.TextTransparency = 0.2
    hintLabel.TextSize = 10
    hintLabel.Font = Enum.Font.Gotham
    hintLabel.TextXAlignment = Enum.TextXAlignment.Center
    hintLabel.Parent = frame

    -- Removed instruction text for cleaner appearance

    -- Subtle border
    local stroke = Instance.new("UIStroke")
    stroke.Color = AssetConfig.Tutorial.GlowColor
    stroke.Thickness = 1
    stroke.Transparency = 0.6
    stroke.Parent = frame

    table.insert(hintBillboards, billboard)

    return billboard
end

--[[
    Create hint billboards for all control stations.
]]
local function createAllHints()
    if not tutorialFolder then
        return
    end

    for _, station in pairs(STATION_LOCATIONS) do
        createHintBillboard(station.x, station.z, station.name)
    end

    print(string.format("[Tutorial] Created %d hint billboards", #hintBillboards))
end

--[[
    Fade out all hint billboards.
]]
local function fadeAllHints()
    for _, billboard in ipairs(hintBillboards) do
        if billboard and billboard.Parent then
            local frame = billboard:FindFirstChild("Background") :: Frame?
            if frame then
                local tween = TweenService:Create(frame, FADE_TWEEN_INFO, {
                    BackgroundTransparency = 1,
                })
                tween:Play()

                for _, child in ipairs(frame:GetChildren()) do
                    if child:IsA("TextLabel") then
                        local textTween = TweenService:Create(child, FADE_TWEEN_INFO, {
                            TextTransparency = 1,
                        })
                        textTween:Play()
                    elseif child:IsA("UIStroke") then
                        local strokeTween = TweenService:Create(child, FADE_TWEEN_INFO, {
                            Transparency = 1,
                        })
                        strokeTween:Play()
                    end
                end
            end
        end
    end

    tutorialState.hintsVisible = false
    print("[Tutorial] All hints fading")
end

--[[
    Check if all controls have been used.
]]
local function allControlsUsed(): boolean
    for _, used in pairs(tutorialState.controlsUsed) do
        if not used then
            return false
        end
    end
    return true
end

--[[
    Check if player is near Forum (to fade path).
]]
local function checkPlayerNearForum()
    local character = localPlayer.Character
    if not character then
        return
    end

    local humanoidRootPart = character:FindFirstChild("HumanoidRootPart") :: BasePart?
    if not humanoidRootPart then
        return
    end

    local pos = humanoidRootPart.Position
    local forumX = STATION_LOCATIONS.forum.x
    local forumZ = STATION_LOCATIONS.forum.z

    local distance = math.sqrt((pos.X - forumX) ^ 2 + (pos.Z - forumZ) ^ 2)

    if distance < 20 and tutorialState.pathShown then
        fadePathParticles()
    end
end

--[[
    Handle control input events.
]]
ControlInputEvent.OnClientEvent:Connect(function(controlType: string, isActivating: boolean)
    if isActivating then
        -- Mark this control as used
        tutorialState.controlsUsed[controlType] = true

        print(string.format("[Tutorial] Control used: %s", controlType))

        -- Check if all controls used - fade all hints
        if allControlsUsed() and tutorialState.hintsVisible then
            fadeAllHints()
        end
    end
end)

-- Initialize tutorial folder
tutorialFolder = Instance.new("Folder")
tutorialFolder.Name = "Tutorial"
tutorialFolder.Parent = workspace

-- Wait for terrain and controls
task.wait(10)

-- Create tutorial elements
createPathToForum()
createAllHints()

-- Animation loop for path particles
RunService.Heartbeat:Connect(function(dt: number)
    if tutorialState.pathShown then
        animatePathParticles(dt)
        checkPlayerNearForum()
    end
end)

print("")
print("Tutorial system running!")
print("  Glowing path leads to Forum")
print("  Hint billboards above each control")
print("  Hints fade after controls are used")
