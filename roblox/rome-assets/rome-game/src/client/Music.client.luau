--!strict
--[[
    Music.client.luau

    Client-side music system with PSI-based crossfading.

    Features:
    - Preloads all 6 music tracks
    - Smooth crossfading based on PSI value
    - Uses TweenService for volume transitions
]]

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local SoundService = game:GetService("SoundService")
local TweenService = game:GetService("TweenService")

-- Wait for shared modules
local Shared = ReplicatedStorage:WaitForChild("Shared")
local MusicManager = require(Shared:WaitForChild("MusicManager"))

-- Wait for RemoteEvents
local RemoteEvents = ReplicatedStorage:WaitForChild("RemoteEvents")
local PsiUpdateEvent = RemoteEvents:WaitForChild("PsiUpdate")

print("==============================================")
print("  MUSIC SYSTEM INITIALIZING")
print("==============================================")
print("")

-- Sound storage
local sounds: {Sound} = {}
local currentTrackIndex = 0
local currentPsi = 0

-- Tween configuration
local TWEEN_INFO = TweenInfo.new(
    MusicManager.getCrossfadeDuration(),
    Enum.EasingStyle.Quad,
    Enum.EasingDirection.InOut
)

--[[
    Create a Sound object for a track.
]]
local function createSound(track: MusicManager.MusicTrack, index: number): Sound
    local sound = Instance.new("Sound")
    sound.Name = track.name
    sound.SoundId = track.id
    sound.Volume = 0
    sound.Looped = true
    sound.Parent = SoundService

    print(string.format("  [%d] Loading: %s", index, track.name))

    return sound
end

--[[
    Initialize and preload all music tracks.
]]
local function initializeTracks()
    print("Loading music tracks...")

    local tracks = MusicManager.getTracks()

    for i, track in ipairs(tracks) do
        local sound = createSound(track, i)
        table.insert(sounds, sound)
    end

    -- Start all sounds playing at volume 0 (for instant crossfade)
    for _, sound in ipairs(sounds) do
        sound:Play()
    end

    print(string.format("  Loaded %d tracks", #sounds))
end

--[[
    Tween a sound's volume to a target value.
]]
local function tweenVolume(sound: Sound, targetVolume: number)
    local tween = TweenService:Create(sound, TWEEN_INFO, {
        Volume = targetVolume
    })
    tween:Play()
end

--[[
    Update music volumes based on PSI value.
    Uses smooth crossfading calculation from MusicManager.
]]
local function updateMusicForPsi(psi: number)
    currentPsi = psi

    -- Get target volumes for all tracks
    local volumes = MusicManager.calculateCrossfadeVolumes(psi)

    -- Apply volume changes with tweening
    for i, sound in ipairs(sounds) do
        local targetVolume = volumes[i] or 0

        -- Only tween if significant change
        if math.abs(sound.Volume - targetVolume) > 0.01 then
            tweenVolume(sound, targetVolume)
        end
    end

    -- Track which track is primary
    local newTrackIndex = MusicManager.getTrackIndexForPsi(psi)
    if newTrackIndex ~= currentTrackIndex then
        local tracks = MusicManager.getTracks()
        local track = tracks[newTrackIndex]
        print(string.format("[Music] PSI %.2f -> Track: %s", psi, track.name))
        currentTrackIndex = newTrackIndex
    end
end

--[[
    Handle PSI updates from server.
]]
PsiUpdateEvent.OnClientEvent:Connect(function(psi: number)
    updateMusicForPsi(psi)
end)

-- Initialize
initializeTracks()

-- Set initial state (low PSI = first track)
updateMusicForPsi(0.05)

print("")
print("Music system running!")
print("  Tracks crossfade based on Political Stress Index")
print("  Current track: " .. MusicManager.getTrackName(currentPsi))
