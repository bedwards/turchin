--!strict
--[[
    HUD.client.luau

    Minimal HUD displaying Political Stress Index percentage.

    Features:
    - Single text label at top-center
    - Clean, unobtrusive design
    - Updates from server state broadcasts
]]

local Players = game:GetService("Players")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

-- Wait for RemoteEvents
local RemoteEvents = ReplicatedStorage:WaitForChild("RemoteEvents")
local PsiUpdateEvent = RemoteEvents:WaitForChild("PsiUpdate")
local StateUpdateEvent = RemoteEvents:WaitForChild("StateUpdate")

-- Wait for shared modules
local Shared = ReplicatedStorage:WaitForChild("Shared")
local SDT = require(Shared:WaitForChild("SDT"))
local MusicManager = require(Shared:WaitForChild("MusicManager"))

print("==============================================")
print("  HUD INITIALIZING")
print("==============================================")
print("")

-- Get the local player's PlayerGui
local player = Players.LocalPlayer
local playerGui = player:WaitForChild("PlayerGui")

-- Create ScreenGui
local screenGui = Instance.new("ScreenGui")
screenGui.Name = "StressHUD"
screenGui.ResetOnSpawn = false
screenGui.IgnoreGuiInset = true
screenGui.Parent = playerGui

-- Create main container frame (top-center)
local container = Instance.new("Frame")
container.Name = "Container"
container.Size = UDim2.new(0, 300, 0, 80)
container.Position = UDim2.new(0.5, -150, 0, 20)
container.BackgroundColor3 = Color3.fromRGB(20, 20, 30)
container.BackgroundTransparency = 0.3
container.BorderSizePixel = 0
container.Parent = screenGui

-- Add corner rounding
local corner = Instance.new("UICorner")
corner.CornerRadius = UDim.new(0, 12)
corner.Parent = container

-- Create title label
local titleLabel = Instance.new("TextLabel")
titleLabel.Name = "Title"
titleLabel.Size = UDim2.new(1, 0, 0, 24)
titleLabel.Position = UDim2.new(0, 0, 0, 8)
titleLabel.BackgroundTransparency = 1
titleLabel.Text = "POLITICAL STRESS"
titleLabel.TextColor3 = Color3.fromRGB(180, 180, 200)
titleLabel.TextSize = 14
titleLabel.Font = Enum.Font.GothamBold
titleLabel.Parent = container

-- Create percentage label
local percentLabel = Instance.new("TextLabel")
percentLabel.Name = "Percent"
percentLabel.Size = UDim2.new(1, 0, 0, 36)
percentLabel.Position = UDim2.new(0, 0, 0, 30)
percentLabel.BackgroundTransparency = 1
percentLabel.Text = "5%"
percentLabel.TextColor3 = Color3.fromRGB(255, 255, 255)
percentLabel.TextSize = 32
percentLabel.Font = Enum.Font.GothamBold
percentLabel.Parent = container

-- Create status label
local statusLabel = Instance.new("TextLabel")
statusLabel.Name = "Status"
statusLabel.Size = UDim2.new(1, 0, 0, 16)
statusLabel.Position = UDim2.new(0, 0, 1, -22)
statusLabel.BackgroundTransparency = 1
statusLabel.Text = "Peaceful"
statusLabel.TextColor3 = Color3.fromRGB(100, 200, 100)
statusLabel.TextSize = 12
statusLabel.Font = Enum.Font.Gotham
statusLabel.Parent = container

-- Current PSI for display
local currentPsi = 0.05

--[[
    Get color for PSI level (green to red gradient).
]]
local function getPsiColor(psi: number): Color3
    if psi < 0.2 then
        return Color3.fromRGB(100, 200, 100) -- Green
    elseif psi < 0.4 then
        return Color3.fromRGB(200, 200, 100) -- Yellow
    elseif psi < 0.6 then
        return Color3.fromRGB(255, 180, 80)  -- Orange
    elseif psi < 0.8 then
        return Color3.fromRGB(255, 100, 80)  -- Red-orange
    else
        return Color3.fromRGB(255, 60, 60)   -- Red
    end
end

--[[
    Update the HUD display.
]]
local function updateDisplay(psi: number)
    currentPsi = psi

    -- Update percentage
    local percent = math.floor(psi * 100 + 0.5)
    percentLabel.Text = tostring(percent) .. "%"

    -- Update status text and color
    local status = SDT.getStressLevel(psi)
    local color = getPsiColor(psi)

    statusLabel.Text = status
    statusLabel.TextColor3 = color

    -- Also color the percentage based on stress
    percentLabel.TextColor3 = color
end

--[[
    Handle PSI updates from server.
]]
PsiUpdateEvent.OnClientEvent:Connect(function(psi: number)
    updateDisplay(psi)
end)

--[[
    Handle full state updates (if needed for additional display).
]]
StateUpdateEvent.OnClientEvent:Connect(function(state: {N: number, E: number, W: number, S: number, psi: number})
    updateDisplay(state.psi)
end)

-- Set initial display
updateDisplay(currentPsi)

print("HUD ready!")
print("  Displaying Political Stress percentage at top-center")
