--!strict
--[[
    HUD.client.luau

    Minimal HUD displaying Political Stress Index and current year.

    Features:
    - Political Stress percentage at top-center
    - Current year display (BCE/CE)
    - Clean, unobtrusive design
    - Updates from server state broadcasts
]]

local Players = game:GetService("Players")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

-- Wait for RemoteEvents
local RemoteEvents = ReplicatedStorage:WaitForChild("RemoteEvents")
local PsiUpdateEvent = RemoteEvents:WaitForChild("PsiUpdate")
local StateUpdateEvent = RemoteEvents:WaitForChild("StateUpdate")
local YearUpdateEvent = RemoteEvents:WaitForChild("YearUpdate", 10)

-- Wait for shared modules
local Shared = ReplicatedStorage:WaitForChild("Shared")
local SDT = require(Shared:WaitForChild("SDT"))

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
container.Size = UDim2.new(0, 300, 0, 100)
container.Position = UDim2.new(0.5, -150, 0, 20)
container.BackgroundColor3 = Color3.fromRGB(20, 20, 30)
container.BackgroundTransparency = 0.3
container.BorderSizePixel = 0
container.Parent = screenGui

-- Add corner rounding
local corner = Instance.new("UICorner")
corner.CornerRadius = UDim.new(0, 12)
corner.Parent = container

-- Create year label at top
local yearLabel = Instance.new("TextLabel")
yearLabel.Name = "Year"
yearLabel.Size = UDim2.new(1, 0, 0, 20)
yearLabel.Position = UDim2.new(0, 0, 0, 6)
yearLabel.BackgroundTransparency = 1
yearLabel.Text = "100 BCE"
yearLabel.TextColor3 = Color3.fromRGB(255, 220, 150)
yearLabel.TextSize = 16
yearLabel.Font = Enum.Font.GothamBold
yearLabel.Parent = container

-- Create title label
local titleLabel = Instance.new("TextLabel")
titleLabel.Name = "Title"
titleLabel.Size = UDim2.new(1, 0, 0, 18)
titleLabel.Position = UDim2.new(0, 0, 0, 28)
titleLabel.BackgroundTransparency = 1
titleLabel.Text = "POLITICAL STRESS"
titleLabel.TextColor3 = Color3.fromRGB(180, 180, 200)
titleLabel.TextSize = 12
titleLabel.Font = Enum.Font.GothamBold
titleLabel.Parent = container

-- Create percentage label
local percentLabel = Instance.new("TextLabel")
percentLabel.Name = "Percent"
percentLabel.Size = UDim2.new(1, 0, 0, 36)
percentLabel.Position = UDim2.new(0, 0, 0, 44)
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

-- Current state for display
local currentPsi = 0.05
local currentYear = -100

--[[
    Get color for PSI level (green to red gradient).
]]
local function getPsiColor(psi: number): Color3
    if psi < 0.2 then
        return Color3.fromRGB(100, 200, 100) -- Green
    elseif psi < 0.4 then
        return Color3.fromRGB(200, 200, 100) -- Yellow
    elseif psi < 0.6 then
        return Color3.fromRGB(255, 180, 80) -- Orange
    elseif psi < 0.8 then
        return Color3.fromRGB(255, 100, 80) -- Red-orange
    else
        return Color3.fromRGB(255, 60, 60) -- Red
    end
end

--[[
    Format year for display with BCE/CE suffix.
]]
local function formatYear(year: number): string
    if year < 0 then
        return string.format("%d BCE", -year)
    elseif year == 0 then
        return "1 BCE" -- No year 0 in historical calendar
    else
        return string.format("%d CE", year)
    end
end

--[[
    Update the PSI display.
]]
local function updatePsiDisplay(psi: number)
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
    Update the year display.
]]
local function updateYearDisplay(year: number)
    currentYear = year
    yearLabel.Text = formatYear(year)
end

--[[
    Handle PSI updates from server.
]]
PsiUpdateEvent.OnClientEvent:Connect(function(psi: number)
    updatePsiDisplay(psi)
end)

--[[
    Handle full state updates.
]]
StateUpdateEvent.OnClientEvent:Connect(function(state: { N: number, E: number, W: number, S: number, psi: number })
    updatePsiDisplay(state.psi)
end)

--[[
    Handle year updates from server.
]]
if YearUpdateEvent then
    YearUpdateEvent.OnClientEvent:Connect(function(year: number)
        updateYearDisplay(year)
    end)
    print("  Year display connected")
else
    warn("[HUD] YearUpdate event not found - year display disabled")
end

-- Set initial display
updatePsiDisplay(currentPsi)
updateYearDisplay(currentYear)

print("HUD ready!")
print("  Displaying Political Stress percentage")
print("  Displaying current year (BCE/CE)")
