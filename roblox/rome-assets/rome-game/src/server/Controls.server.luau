--!strict
--[[
    Controls.server.luau

    Creates pressure plate controls for player interaction with SDT parameters.
    Themed with Roman administrative systems for immersive gameplay.

    Features:
    - 4 control stations placed at key Roman locations
    - Each station has paired red/green plates for bidirectional control
    - Gradual parameter adjustment while standing on plate
    - Visual and audio feedback
    - Roman-themed names with descriptive floating text

    Control Stations (Roman Themed):
    1. POLITICAL STRESS (Forum) - Direct PSI adjustment for testing
       "Control public unrest through spectacles and speeches"

    2. SENATE SEATS (Near Forum) - Controls elite recruitment rate (alpha)
       "Expand or restrict access to the Senatorial class"

    3. GRAIN DOLE (Market) - Controls wage recovery rate (gamma)
       "The bread that feeds Rome's hungry masses"

    4. TAX COLLECTORS (Temple District) - Controls state income/recovery (delta)
       "Fund the legions or let the treasury empty"
]]

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local Players = game:GetService("Players")

-- Wait for terrain to generate before placing controls
task.wait(5)

-- Wait for shared modules
local Shared = ReplicatedStorage:WaitForChild("Shared")
local TerrainUtils = require(Shared:WaitForChild("TerrainUtils"))

-- Wait for RemoteEvents
local RemoteEvents = ReplicatedStorage:WaitForChild("RemoteEvents")
local ControlInputEvent = RemoteEvents:WaitForChild("ControlInput")

-- Control plate configuration
local PLATE_SIZE = Vector3.new(8, 1, 8)
local PLATE_SPACING = 12 -- Distance between red/green plates at each station

-- Station locations (from TerrainGenerator flat zones)
local STATION_LOCATIONS = {
    forum = { x = 0, z = 0 },
    senateForum = { x = 30, z = -25 }, -- Near Forum, offset
    market = { x = 40, z = -80 },
    templeDistrict = { x = -80, z = 60 },
}

-- Track which players are on which plates
local playersOnPlates: { [Player]: string } = {}

-- Control types for different parameters:
-- "decrease_psi", "increase_psi"
-- "decrease_alpha", "increase_alpha"
-- "decrease_gamma", "increase_gamma"
-- "decrease_delta", "increase_delta"

--[[
    Create a control plate at the specified position.
]]
local function createControlPlate(name: string, x: number, z: number, color: Color3, controlType: string): Part
    local groundY = TerrainUtils.getGroundY(x, z)

    local plate = Instance.new("Part")
    plate.Name = name
    plate.Size = PLATE_SIZE
    plate.Position = Vector3.new(x, groundY + PLATE_SIZE.Y / 2, z)
    plate.Anchored = true
    plate.Material = Enum.Material.Neon
    plate.Color = color
    plate.Transparency = 0.3
    plate.CanCollide = true

    -- Add a subtle glow effect via PointLight
    local light = Instance.new("PointLight")
    light.Name = "PlateLight"
    light.Color = color
    light.Brightness = 0.5
    light.Range = 15
    light.Parent = plate

    -- Add activation sound
    local sound = Instance.new("Sound")
    sound.Name = "ActivationSound"
    sound.SoundId = "rbxassetid://6042053626" -- Generic UI click sound
    sound.Volume = 0.5
    sound.Parent = plate

    -- Store control type as attribute for detection
    plate:SetAttribute("ControlType", controlType)

    plate.Parent = workspace

    print(string.format("[Controls] Created %s plate at (%.0f, %.1f, %.0f)", name, x, groundY, z))

    return plate
end

--[[
    Create a label above a plate (subtle, smaller).
]]
local function createPlateLabel(plate: Part, text: string)
    local billboard = Instance.new("BillboardGui")
    billboard.Name = "PlateLabel"
    billboard.Size = UDim2.new(0, 80, 0, 20)
    billboard.StudsOffset = Vector3.new(0, 2, 0)
    billboard.Adornee = plate
    billboard.AlwaysOnTop = false

    local label = Instance.new("TextLabel")
    label.Name = "Label"
    label.Size = UDim2.new(1, 0, 1, 0)
    label.BackgroundTransparency = 1
    label.Text = text
    label.TextColor3 = Color3.new(1, 1, 1)
    label.TextTransparency = 0.3
    label.TextStrokeTransparency = 0.7
    label.TextScaled = true
    label.Font = Enum.Font.Gotham
    label.Parent = billboard

    billboard.Parent = plate
end

--[[
    Create a station title above the control area (subtle, no subtitle).
]]
local function createStationTitle(x: number, z: number, title: string, _subtitle: string)
    local groundY = TerrainUtils.getGroundY(x, z)

    -- Create an invisible anchor part
    local anchor = Instance.new("Part")
    anchor.Name = title .. "_Title"
    anchor.Size = Vector3.new(1, 1, 1)
    anchor.Position = Vector3.new(x, groundY + 5, z)
    anchor.Anchored = true
    anchor.Transparency = 1
    anchor.CanCollide = false
    anchor.Parent = workspace

    local billboard = Instance.new("BillboardGui")
    billboard.Name = "StationTitle"
    billboard.Size = UDim2.new(0, 100, 0, 25)
    billboard.StudsOffset = Vector3.new(0, 0, 0)
    billboard.Adornee = anchor
    billboard.AlwaysOnTop = false

    local titleLabel = Instance.new("TextLabel")
    titleLabel.Name = "Title"
    titleLabel.Size = UDim2.new(1, 0, 1, 0)
    titleLabel.Position = UDim2.new(0, 0, 0, 0)
    titleLabel.BackgroundTransparency = 1
    titleLabel.Text = title
    titleLabel.TextColor3 = Color3.fromRGB(255, 220, 150)
    titleLabel.TextTransparency = 0.3
    titleLabel.TextStrokeTransparency = 0.6
    titleLabel.TextScaled = true
    titleLabel.Font = Enum.Font.Gotham
    titleLabel.Parent = billboard

    -- Removed subtitle for cleaner appearance

    billboard.Parent = anchor
end

--[[
    Set plate active state with visual feedback.
]]
local function setPlateActive(plate: Part, active: boolean)
    local light = plate:FindFirstChild("PlateLight") :: PointLight?

    if active then
        plate.Transparency = 0.1
        if light then
            light.Brightness = 2
        end
    else
        plate.Transparency = 0.3
        if light then
            light.Brightness = 0.5
        end
    end
end

--[[
    Play activation sound on plate.
]]
local function playActivationSound(plate: Part)
    local sound = plate:FindFirstChild("ActivationSound") :: Sound?
    if sound then
        sound:Play()
    end
end

--[[
    Check if a character is standing on a plate.
]]
local function isCharacterOnPlate(character: Model, plate: Part): boolean
    local humanoidRootPart = character:FindFirstChild("HumanoidRootPart") :: BasePart?
    if not humanoidRootPart then
        return false
    end

    local charPos = humanoidRootPart.Position
    local platePos = plate.Position
    local plateSize = plate.Size

    -- Check if character is within plate bounds (with some tolerance)
    local dx = math.abs(charPos.X - platePos.X)
    local dz = math.abs(charPos.Z - platePos.Z)
    local dy = charPos.Y - platePos.Y

    -- Within horizontal bounds and above plate (standing on it)
    return dx < plateSize.X / 2 + 2 and dz < plateSize.Z / 2 + 2 and dy > 0 and dy < 5
end

-- Store all plates for detection
local allPlates: { Part } = {}

--[[
    Handle player entering plate area.
]]
local function onPlayerEnterPlate(player: Player, plate: Part, controlType: string)
    if playersOnPlates[player] == controlType then
        return -- Already on this plate
    end

    playersOnPlates[player] = controlType
    setPlateActive(plate, true)
    playActivationSound(plate)

    -- Notify client
    ControlInputEvent:FireClient(player, controlType, true)

    print(string.format("[Controls] %s stepped on %s plate", player.Name, controlType))
end

--[[
    Handle player leaving plate area.
]]
local function onPlayerLeavePlate(player: Player, plate: Part, controlType: string)
    if playersOnPlates[player] ~= controlType then
        return -- Not on this plate
    end

    playersOnPlates[player] = nil
    setPlateActive(plate, false)

    print(string.format("[Controls] %s left %s plate", player.Name, controlType))
end

--[[
    Create a control station with paired red/green plates.
]]
local function createControlStation(
    stationName: string,
    centerX: number,
    centerZ: number,
    decreaseType: string,
    increaseType: string,
    decreaseLabel: string,
    increaseLabel: string,
    stationTitle: string,
    stationSubtitle: string
)
    print(string.format("\n[Controls] Creating %s station at (%.0f, %.0f)", stationName, centerX, centerZ))

    -- Create decrease (red) plate
    local decreasePlate = createControlPlate(
        stationName .. "_Decrease",
        centerX - PLATE_SPACING / 2,
        centerZ,
        Color3.fromRGB(255, 100, 100), -- Red
        decreaseType
    )
    createPlateLabel(decreasePlate, decreaseLabel)
    table.insert(allPlates, decreasePlate)

    -- Create increase (green) plate
    local increasePlate = createControlPlate(
        stationName .. "_Increase",
        centerX + PLATE_SPACING / 2,
        centerZ,
        Color3.fromRGB(100, 255, 100), -- Green
        increaseType
    )
    createPlateLabel(increasePlate, increaseLabel)
    table.insert(allPlates, increasePlate)

    -- Create station title
    createStationTitle(centerX, centerZ, stationTitle, stationSubtitle)

    return decreasePlate, increasePlate
end

-- Create control stations
print("==============================================")
print("  CREATING CONTROL STATIONS")
print("==============================================")

-- Station 1: PSI Control at Forum
createControlStation(
    "PSI",
    STATION_LOCATIONS.forum.x,
    STATION_LOCATIONS.forum.z,
    "decrease_psi",
    "increase_psi",
    "CALM THE MOB",
    "ROUSE THE MOB",
    "POLITICAL STRESS",
    "Control public unrest through spectacles and speeches"
)

-- Station 2: Senate Seats (Elite Recruitment - alpha)
createControlStation(
    "Senate",
    STATION_LOCATIONS.senateForum.x,
    STATION_LOCATIONS.senateForum.z,
    "decrease_alpha",
    "increase_alpha",
    "FEWER SEATS",
    "MORE SEATS",
    "SENATE SEATS",
    "Expand or restrict access to the Senatorial class"
)

-- Station 3: Grain Dole (Wage Recovery - gamma)
createControlStation(
    "GrainDole",
    STATION_LOCATIONS.market.x,
    STATION_LOCATIONS.market.z,
    "decrease_gamma",
    "increase_gamma",
    "REDUCE DOLE",
    "INCREASE DOLE",
    "GRAIN DOLE",
    "The bread that feeds Rome's hungry masses"
)

-- Station 4: Tax Collectors (State Recovery - delta)
createControlStation(
    "TaxCollectors",
    STATION_LOCATIONS.templeDistrict.x,
    STATION_LOCATIONS.templeDistrict.z,
    "decrease_delta",
    "increase_delta",
    "REDUCE TAXES",
    "RAISE TAXES",
    "TAX COLLECTORS",
    "Fund the legions or let the treasury empty"
)

-- Polling loop to detect players on plates
-- (More reliable than Touched events for this use case)
task.spawn(function()
    while true do
        for _, player in ipairs(Players:GetPlayers()) do
            local character = player.Character
            if character then
                local foundPlate: Part? = nil
                local foundControlType: string? = nil

                -- Check all plates
                for _, plate in ipairs(allPlates) do
                    if isCharacterOnPlate(character, plate) then
                        foundPlate = plate
                        foundControlType = plate:GetAttribute("ControlType") :: string?
                        break
                    end
                end

                -- Handle plate transitions
                local currentPlateType = playersOnPlates[player]

                if foundControlType and currentPlateType ~= foundControlType then
                    -- Leaving current plate (if any)
                    if currentPlateType then
                        for _, plate in ipairs(allPlates) do
                            if plate:GetAttribute("ControlType") == currentPlateType then
                                onPlayerLeavePlate(player, plate, currentPlateType)
                                break
                            end
                        end
                    end
                    -- Entering new plate
                    if foundPlate and foundControlType then
                        onPlayerEnterPlate(player, foundPlate, foundControlType)
                    end
                elseif not foundControlType and currentPlateType then
                    -- Left all plates
                    for _, plate in ipairs(allPlates) do
                        if plate:GetAttribute("ControlType") == currentPlateType then
                            onPlayerLeavePlate(player, plate, currentPlateType)
                            break
                        end
                    end
                end
            end
        end

        task.wait(0.1) -- Check 10 times per second
    end
end)

-- Handle players leaving the game
Players.PlayerRemoving:Connect(function(player: Player)
    playersOnPlates[player] = nil
end)

-- Server-side control tracking for GameLoop
-- We need to communicate with GameLoop about which controls are active
-- Since both are server scripts, we can use a BindableEvent

local ControlBindable = Instance.new("BindableEvent")
ControlBindable.Name = "ControlBindable"
ControlBindable.Parent = ReplicatedStorage

-- Track active controls and notify GameLoop
task.spawn(function()
    while true do
        -- Count players on each plate type
        local controlCounts: { [string]: number } = {
            decrease_psi = 0,
            increase_psi = 0,
            decrease_alpha = 0,
            increase_alpha = 0,
            decrease_gamma = 0,
            increase_gamma = 0,
            decrease_delta = 0,
            increase_delta = 0,
        }

        for _, controlType in pairs(playersOnPlates) do
            controlCounts[controlType] = (controlCounts[controlType] or 0) + 1
        end

        -- Fire control state to any listeners
        ControlBindable:Fire(controlCounts)

        task.wait(0.1)
    end
end)

print("")
print("==============================================")
print("  CONTROL STATIONS READY")
print("==============================================")
print("")
print("4 Roman Control Stations:")
print("  1. Political Stress (Forum) - Control public unrest")
print("  2. Senate Seats (Near Forum) - Elite recruitment rate")
print("  3. Grain Dole (Market) - Worker wage recovery")
print("  4. Tax Collectors (Temple) - State fiscal recovery")
print("")
print("Stand on plates to adjust parameters (~0.05/second)")
print("Green = Increase, Red = Decrease")
