--!strict
--[[
    SoundEffects.client.luau

    Sound effects system for immersive Roman atmosphere.

    Features:
    - Pressure plate step sounds
    - Ambient crowd murmur (volume scales with population)
    - Environmental ambience (birds, wind)
    - Sounds respond to game state

    Sound Design:
    - Peaceful: Birds, gentle wind, distant crowd
    - Tense: Louder crowd murmur, less birds
    - Crisis: Agitated crowd, strong wind
]]

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local SoundService = game:GetService("SoundService")
local TweenService = game:GetService("TweenService")
local Players = game:GetService("Players")

-- Wait for shared modules
local Shared = ReplicatedStorage:WaitForChild("Shared")
local AssetConfig = require(Shared:WaitForChild("AssetConfig"))

-- Wait for RemoteEvents
local RemoteEvents = ReplicatedStorage:WaitForChild("RemoteEvents")
local StateUpdateEvent = RemoteEvents:WaitForChild("StateUpdate")
local ControlInputEvent = RemoteEvents:WaitForChild("ControlInput")

print("==============================================")
print("  SOUND EFFECTS SYSTEM INITIALIZING")
print("==============================================")
print("")

-- Configuration
local TWEEN_DURATION = 2.0
-- AMBIENT_UPDATE_RATE could be used for periodic volume updates

-- Current state
local currentPsi: number = 0.05
local currentPopulation: number = 0.5

-- Tween info
local TWEEN_INFO = TweenInfo.new(TWEEN_DURATION, Enum.EasingStyle.Quad, Enum.EasingDirection.InOut)

-- Sound instances
local ambientSounds: { [string]: Sound } = {}
local controlStepSound: Sound?

-- Local player reference (for future positional audio)
local _localPlayer = Players.LocalPlayer

--[[
    Create an ambient sound with looping.
]]
local function createAmbientSound(name: string, soundId: string, defaultVolume: number): Sound
    local sound = Instance.new("Sound")
    sound.Name = name
    sound.SoundId = soundId
    sound.Volume = defaultVolume
    sound.Looped = true
    sound.RollOffMode = Enum.RollOffMode.Linear
    sound.RollOffMaxDistance = 500
    sound.RollOffMinDistance = 10

    -- Parent to SoundService for global ambient
    sound.Parent = SoundService

    return sound
end

--[[
    Create a one-shot sound effect.
]]
local function createOneShotSound(name: string, soundId: string, volume: number): Sound
    local sound = Instance.new("Sound")
    sound.Name = name
    sound.SoundId = soundId
    sound.Volume = volume
    sound.Looped = false
    sound.Parent = SoundService

    return sound
end

--[[
    Initialize all sound effects.
]]
local function initializeSounds()
    -- Ambient crowd murmur (volume scales with population and PSI)
    ambientSounds.crowd = createAmbientSound("CrowdMurmur", AssetConfig.Sounds.CrowdMurmur, 0.2)

    -- Birds (louder during peaceful times)
    ambientSounds.birds = createAmbientSound("Birds", AssetConfig.Sounds.Birds, 0.3)

    -- Wind (louder during crisis)
    ambientSounds.wind = createAmbientSound("Wind", AssetConfig.Sounds.Wind, 0.1)

    -- Control plate step sound (one-shot)
    controlStepSound = createOneShotSound("PlateStep", AssetConfig.Sounds.PlateStep, 0.5)

    print("[SoundEffects] Created ambient sounds:")
    print("  - Crowd murmur")
    print("  - Birds")
    print("  - Wind")
    print("  - Plate step")
end

--[[
    Play the pressure plate step sound.
]]
local function playPlateStepSound()
    if controlStepSound then
        -- Clone for overlapping sounds
        local soundClone = controlStepSound:Clone()
        soundClone.Parent = SoundService
        soundClone:Play()

        -- Clean up after playing
        soundClone.Ended:Connect(function()
            soundClone:Destroy()
        end)
    end
end

--[[
    Update ambient sound volumes based on game state.
]]
local function updateAmbientSounds(psi: number, population: number)
    -- Crowd volume: louder with more population and higher stress
    local crowdBase = 0.1 + population * 0.2 -- 0.1 to 0.3 based on population
    local crowdStress = psi * 0.3 -- Add up to 0.3 for stress
    local crowdVolume = math.clamp(crowdBase + crowdStress, 0.05, 0.5)

    -- Birds volume: quieter during crisis
    local birdsVolume = math.clamp(0.35 - psi * 0.3, 0.05, 0.35)

    -- Wind volume: louder during crisis
    local windVolume = math.clamp(0.05 + psi * 0.25, 0.05, 0.3)

    -- Apply volume changes with tweening
    if ambientSounds.crowd then
        local tween = TweenService:Create(ambientSounds.crowd, TWEEN_INFO, {
            Volume = crowdVolume,
        })
        tween:Play()
    end

    if ambientSounds.birds then
        local tween = TweenService:Create(ambientSounds.birds, TWEEN_INFO, {
            Volume = birdsVolume,
        })
        tween:Play()
    end

    if ambientSounds.wind then
        local tween = TweenService:Create(ambientSounds.wind, TWEEN_INFO, {
            Volume = windVolume,
        })
        tween:Play()
    end
end

--[[
    Update crowd pitch based on stress (higher pitch = more agitated).
]]
local function updateCrowdPitch(psi: number)
    if ambientSounds.crowd then
        -- Pitch: 0.9 (calm) to 1.2 (agitated)
        local pitch = 0.9 + psi * 0.3

        local tween = TweenService:Create(ambientSounds.crowd, TWEEN_INFO, {
            PlaybackSpeed = pitch,
        })
        tween:Play()
    end
end

--[[
    Start all ambient sounds.
]]
local function startAmbientSounds()
    for _, sound in pairs(ambientSounds) do
        if not sound.IsPlaying then
            sound:Play()
        end
    end

    print("[SoundEffects] Ambient sounds playing")
end

--[[
    Handle state updates from server.
]]
StateUpdateEvent.OnClientEvent:Connect(function(state: { N: number, E: number, W: number, S: number, psi: number })
    local oldPsi = currentPsi

    currentPsi = state.psi
    currentPopulation = state.N

    -- Update ambient volumes if PSI changed significantly
    if math.abs(oldPsi - currentPsi) > 0.03 then
        updateAmbientSounds(currentPsi, currentPopulation)
        updateCrowdPitch(currentPsi)
    end
end)

--[[
    Handle control input events (for step sounds).
]]
ControlInputEvent.OnClientEvent:Connect(function(_controlType: string, isActivating: boolean)
    if isActivating then
        playPlateStepSound()
    end
end)

-- Initialize
initializeSounds()

-- Wait a moment before starting sounds
task.wait(2)
startAmbientSounds()

-- Initial volume update
updateAmbientSounds(currentPsi, currentPopulation)

print("")
print("Sound effects system running!")
print("  Crowd murmur scales with population and stress")
print("  Bird sounds decrease during crisis")
print("  Wind increases during crisis")
print("  Plate step sounds on control activation")
print(string.format("  Initial state: PSI=%.2f, N=%.2f", currentPsi, currentPopulation))
