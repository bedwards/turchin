--!strict
--[[
    Controls.server.luau

    Creates pressure plate controls for player interaction with PSI.

    Features:
    - Two plates in the Forum: Red (decrease PSI) and Green (increase PSI)
    - Gradual PSI adjustment while standing on plate (~0.05/sec)
    - Visual feedback: plates glow brighter when active
    - Audio feedback: tone on activation
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
local PLATE_SPACING = 15 -- Distance between plates
local FORUM_CENTER_X = 0
local FORUM_CENTER_Z = 0

-- Track which players are on which plates
local playersOnPlates: {[Player]: string} = {}

--[[
    Create a control plate at the specified position.
]]
local function createControlPlate(
    name: string,
    x: number,
    z: number,
    color: Color3,
    controlType: string
): Part
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

    print(string.format("[Controls] Created %s plate at (%.0f, %.0f, %.0f)",
        name, x, groundY, z))

    return plate
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
    return dx < plateSize.X / 2 + 2 and
           dz < plateSize.Z / 2 + 2 and
           dy > 0 and dy < 5
end

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

    -- Notify game loop
    ControlInputEvent:FireClient(player, controlType, true)

    -- For server-side processing, fire to server (actually we need to track server-side)
    -- The GameLoop already listens on OnServerEvent, but we need client to fire
    -- So we'll handle this differently - track on server

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

-- Create the control plates in the Forum
print("==============================================")
print("  CREATING CONTROL PLATES")
print("==============================================")
print("")

local decreasePlate = createControlPlate(
    "DecreasePsiPlate",
    FORUM_CENTER_X - PLATE_SPACING,
    FORUM_CENTER_Z,
    Color3.fromRGB(255, 100, 100), -- Red
    "decrease_psi"
)

local increasePlate = createControlPlate(
    "IncreasePsiPlate",
    FORUM_CENTER_X + PLATE_SPACING,
    FORUM_CENTER_Z,
    Color3.fromRGB(100, 255, 100), -- Green
    "increase_psi"
)

-- Create labels above plates
local function createPlateLabel(plate: Part, text: string)
    local billboard = Instance.new("BillboardGui")
    billboard.Name = "PlateLabel"
    billboard.Size = UDim2.new(0, 200, 0, 50)
    billboard.StudsOffset = Vector3.new(0, 4, 0)
    billboard.Adornee = plate
    billboard.AlwaysOnTop = true

    local label = Instance.new("TextLabel")
    label.Name = "Label"
    label.Size = UDim2.new(1, 0, 1, 0)
    label.BackgroundTransparency = 1
    label.Text = text
    label.TextColor3 = Color3.new(1, 1, 1)
    label.TextStrokeTransparency = 0.5
    label.TextScaled = true
    label.Font = Enum.Font.GothamBold
    label.Parent = billboard

    billboard.Parent = plate
end

createPlateLabel(decreasePlate, "DECREASE STRESS")
createPlateLabel(increasePlate, "INCREASE STRESS")

-- Polling loop to detect players on plates
-- (More reliable than Touched events for this use case)
task.spawn(function()
    while true do
        for _, player in ipairs(Players:GetPlayers()) do
            local character = player.Character
            if character then
                -- Check decrease plate
                local onDecrease = isCharacterOnPlate(character, decreasePlate)
                if onDecrease and playersOnPlates[player] ~= "decrease_psi" then
                    -- Leaving increase plate if on it
                    if playersOnPlates[player] == "increase_psi" then
                        onPlayerLeavePlate(player, increasePlate, "increase_psi")
                    end
                    onPlayerEnterPlate(player, decreasePlate, "decrease_psi")
                elseif not onDecrease and playersOnPlates[player] == "decrease_psi" then
                    onPlayerLeavePlate(player, decreasePlate, "decrease_psi")
                end

                -- Check increase plate
                local onIncrease = isCharacterOnPlate(character, increasePlate)
                if onIncrease and playersOnPlates[player] ~= "increase_psi" then
                    -- Leaving decrease plate if on it
                    if playersOnPlates[player] == "decrease_psi" then
                        onPlayerLeavePlate(player, decreasePlate, "decrease_psi")
                    end
                    onPlayerEnterPlate(player, increasePlate, "increase_psi")
                elseif not onIncrease and playersOnPlates[player] == "increase_psi" then
                    onPlayerLeavePlate(player, increasePlate, "increase_psi")
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
        local decreaseCount = 0
        local increaseCount = 0

        for _, controlType in pairs(playersOnPlates) do
            if controlType == "decrease_psi" then
                decreaseCount = decreaseCount + 1
            elseif controlType == "increase_psi" then
                increaseCount = increaseCount + 1
            end
        end

        -- Fire control state to any listeners
        ControlBindable:Fire({
            decreasePsi = decreaseCount,
            increasePsi = increaseCount,
        })

        task.wait(0.1)
    end
end)

print("")
print("Control plates ready!")
print("  Red plate (-15, Y, 0): Decrease Political Stress")
print("  Green plate (+15, Y, 0): Increase Political Stress")
print("Stand on a plate to adjust PSI by ~0.05/second")
