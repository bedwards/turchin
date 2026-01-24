--!strict
--[[
    TimeSystem.server.luau

    Game time management system for the Rome simulation.

    Features:
    - 1 game year = ~10 real seconds
    - Tracks current simulation year (starting from 100 BCE)
    - Broadcasts year updates to clients via RemoteEvent
    - Integrates with SDT simulation timing

    The time system creates a sense of historical progression
    as players watch the Roman Republic evolve through its
    secular cycles of stability and crisis.
]]

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService = game:GetService("RunService")

-- Configuration
local SECONDS_PER_YEAR = 10 -- 1 game year = 10 real seconds
local STARTING_YEAR = -100 -- 100 BCE (negative = BCE)
local YEAR_BROADCAST_INTERVAL = 1.0 -- Broadcast year every second

-- Wait for RemoteEvents folder (created by GameLoop)
local RemoteEvents = ReplicatedStorage:WaitForChild("RemoteEvents", 30)
if not RemoteEvents then
    warn("[TimeSystem] RemoteEvents folder not found, creating it")
    RemoteEvents = Instance.new("Folder")
    RemoteEvents.Name = "RemoteEvents"
    RemoteEvents.Parent = ReplicatedStorage
end

-- Create RemoteEvent for year updates
local YearUpdateEvent = Instance.new("RemoteEvent")
YearUpdateEvent.Name = "YearUpdate"
YearUpdateEvent.Parent = RemoteEvents

-- Time state
local elapsedRealTime: number = 0
local currentYear: number = STARTING_YEAR

print("==============================================")
print("  TIME SYSTEM INITIALIZING")
print("==============================================")
print("")
print(string.format("Starting Year: %d %s", math.abs(STARTING_YEAR), if STARTING_YEAR < 0 then "BCE" else "CE"))
print(string.format("Time Scale: 1 game year = %d real seconds", SECONDS_PER_YEAR))
print("")

--[[
    Format year for display.
    Returns year with BCE/CE suffix.
]]
local function formatYear(year: number): string
    if year < 0 then
        return string.format("%d BCE", -year)
    elseif year == 0 then
        return "1 BCE" -- There is no year 0, jump from 1 BCE to 1 CE
    else
        return string.format("%d CE", year)
    end
end

--[[
    Calculate current year from elapsed time.
]]
local function calculateYear(): number
    local yearsPassed = math.floor(elapsedRealTime / SECONDS_PER_YEAR)
    local year = STARTING_YEAR + yearsPassed

    -- Handle year 0 (doesn't exist in historical calendar)
    if year >= 0 and STARTING_YEAR < 0 then
        year = year + 1
    end

    return year
end

--[[
    Broadcast current year to all clients.
]]
local function broadcastYear()
    YearUpdateEvent:FireAllClients(currentYear)
end

-- Track broadcast timing
local broadcastAccumulator: number = 0

--[[
    Main update loop - runs every frame.
]]
RunService.Heartbeat:Connect(function(dt: number)
    -- Accumulate elapsed time
    elapsedRealTime = elapsedRealTime + dt

    -- Calculate current year
    local newYear = calculateYear()

    -- Check for year change
    if newYear ~= currentYear then
        currentYear = newYear
        print(string.format("[TimeSystem] Year: %s", formatYear(currentYear)))

        -- Immediately broadcast year change
        broadcastYear()
        broadcastAccumulator = 0
    end

    -- Periodic broadcast for clients that join mid-game
    broadcastAccumulator = broadcastAccumulator + dt
    if broadcastAccumulator >= YEAR_BROADCAST_INTERVAL then
        broadcastYear()
        broadcastAccumulator = 0
    end
end)

-- Initial broadcast
task.defer(function()
    broadcastYear()
end)

print(string.format("[TimeSystem] Started at %s", formatYear(currentYear)))
print("Year updates will be broadcast to clients.")
