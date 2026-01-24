--!strict
--[[
    MusicManager.luau

    Manages PSI-based music crossfading for the cliodynamics simulation.

    Music transitions based on Political Stress Index (PSI):
    - Low PSI: Peaceful, ambient music
    - High PSI: Intense, dramatic music

    Uses smooth crossfading between tracks to avoid jarring transitions.
]]

local MusicManager = {}

-- Music track configuration
-- Each track has a maxPsi threshold - track plays when PSI is below this value
export type MusicTrack = {
    maxPsi: number,
    id: string,
    name: string,
}

-- PSI-based music tracks (from peaceful to crisis)
local MUSIC_TRACKS: { MusicTrack } = {
    { maxPsi = 0.08, id = "rbxassetid://123517555949581", name = "Dawn" },
    { maxPsi = 0.18, id = "rbxassetid://92805060343276", name = "Forum" },
    { maxPsi = 0.32, id = "rbxassetid://122383615563909", name = "Eternal" },
    { maxPsi = 0.52, id = "rbxassetid://139982351033320", name = "March" },
    { maxPsi = 0.76, id = "rbxassetid://94282741928731", name = "Colosseum" },
    { maxPsi = 1.00, id = "rbxassetid://83679996557602", name = "Caesar" },
}

-- Crossfade configuration
local CROSSFADE_DURATION = 2.0 -- seconds for crossfade
local MAX_VOLUME = 0.5 -- maximum track volume

--[[
    Get the list of music tracks.
    Used by the client to preload all tracks.
]]
function MusicManager.getTracks(): { MusicTrack }
    return MUSIC_TRACKS
end

--[[
    Get the track index for a given PSI value.
    Returns 1-6 (index into MUSIC_TRACKS).
]]
function MusicManager.getTrackIndexForPsi(psi: number): number
    for i, track in ipairs(MUSIC_TRACKS) do
        if psi <= track.maxPsi then
            return i
        end
    end
    return #MUSIC_TRACKS
end

--[[
    Get the track for a given PSI value.
]]
function MusicManager.getTrackForPsi(psi: number): MusicTrack
    local index = MusicManager.getTrackIndexForPsi(psi)
    return MUSIC_TRACKS[index]
end

--[[
    Get crossfade duration in seconds.
]]
function MusicManager.getCrossfadeDuration(): number
    return CROSSFADE_DURATION
end

--[[
    Get maximum volume for music tracks.
]]
function MusicManager.getMaxVolume(): number
    return MAX_VOLUME
end

--[[
    Calculate volume levels for smooth crossfading between tracks.
    Returns volumes for current and next track based on PSI position.

    When PSI is near a threshold, we crossfade between tracks:
    - Within 0.02 of threshold: begin fading out current
    - At threshold: fully transitioned to next track
]]
function MusicManager.calculateCrossfadeVolumes(psi: number): { [number]: number }
    local volumes: { [number]: number } = {}

    -- Initialize all volumes to 0
    for i = 1, #MUSIC_TRACKS do
        volumes[i] = 0
    end

    local currentIndex = MusicManager.getTrackIndexForPsi(psi)
    local currentTrack = MUSIC_TRACKS[currentIndex]
    local prevThreshold = if currentIndex > 1 then MUSIC_TRACKS[currentIndex - 1].maxPsi else 0
    local currentThreshold = currentTrack.maxPsi

    -- Calculate position within current track's range
    local rangeSize = currentThreshold - prevThreshold
    local positionInRange = (psi - prevThreshold) / rangeSize

    -- Crossfade zone is last 20% of range
    local crossfadeStart = 0.8

    if positionInRange < crossfadeStart then
        -- Fully in current track
        volumes[currentIndex] = MAX_VOLUME
    else
        -- In crossfade zone
        local crossfadeProgress = (positionInRange - crossfadeStart) / (1 - crossfadeStart)
        volumes[currentIndex] = MAX_VOLUME * (1 - crossfadeProgress)

        if currentIndex < #MUSIC_TRACKS then
            volumes[currentIndex + 1] = MAX_VOLUME * crossfadeProgress
        end
    end

    return volumes
end

--[[
    Get track name for display purposes.
]]
function MusicManager.getTrackName(psi: number): string
    local track = MusicManager.getTrackForPsi(psi)
    return track.name
end

--[[
    Get all track names as a list (for debug display).
]]
function MusicManager.getAllTrackNames(): { string }
    local names: { string } = {}
    for _, track in ipairs(MUSIC_TRACKS) do
        table.insert(names, track.name)
    end
    return names
end

return MusicManager
