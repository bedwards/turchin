--!strict
--[[
    AssetConfig.luau

    Configuration module containing all 3D model asset IDs for Roman buildings.
    These are mesh asset IDs from the Roblox library.

    Usage:
        local AssetConfig = require(path.to.AssetConfig)
        local meshId = AssetConfig.Meshes.Colosseum
]]

local AssetConfig = {}

-- Mesh Asset IDs
AssetConfig.Meshes = {
    Colosseum = "rbxassetid://126740068391615",
    Pantheon = "rbxassetid://99309794928903",
    RomanTemple = "rbxassetid://119800018358438",
    TriumphalArch = "rbxassetid://80950297053943",
    RomanAqueduct = "rbxassetid://120006200986942",
    DoricColumn = "rbxassetid://99822185467408",
    MarketStall = "rbxassetid://82241393018988",
}

-- Default materials for buildings
AssetConfig.Materials = {
    Stone = Enum.Material.Limestone,
    Marble = Enum.Material.Marble,
    Wood = Enum.Material.Wood,
    Bronze = Enum.Material.Metal,
}

-- Default colors
AssetConfig.Colors = {
    Marble = Color3.fromRGB(245, 240, 230),
    Limestone = Color3.fromRGB(220, 210, 190),
    Sandstone = Color3.fromRGB(210, 180, 140),
    Bronze = Color3.fromRGB(180, 140, 90),
    WornStone = Color3.fromRGB(180, 175, 165),
    Wood = Color3.fromRGB(140, 100, 60),
    RedTile = Color3.fromRGB(180, 80, 60),
}

-- Building placement configurations
AssetConfig.Placements = {
    Colosseum = {
        scale = Vector3.new(15, 15, 15),
        rotation = 0,
        yOffset = 0,
    },
    Pantheon = {
        scale = Vector3.new(12, 12, 12),
        rotation = 180,
        yOffset = 0,
    },
    RomanTemple = {
        scale = Vector3.new(8, 8, 8),
        rotation = 0,
        yOffset = 0,
    },
    TriumphalArch = {
        scale = Vector3.new(6, 6, 6),
        rotation = 90,
        yOffset = 0,
    },
    RomanAqueduct = {
        scale = Vector3.new(10, 10, 10),
        rotation = 0,
        yOffset = 0,
    },
    DoricColumn = {
        scale = Vector3.new(3, 3, 3),
        rotation = 0,
        yOffset = 0,
    },
    MarketStall = {
        scale = Vector3.new(4, 4, 4),
        rotation = 0,
        yOffset = 0,
    },
}

-- Sound Asset IDs (Roblox default sounds)
AssetConfig.Sounds = {
    -- UI/Interaction
    PlateStep = "rbxassetid://6042053626", -- UI click
    PlateActivate = "rbxassetid://421058925", -- Soft chime

    -- Ambient (replaced 403-erroring IDs with working public sounds)
    CrowdMurmur = "rbxassetid://926658585", -- People talking background
    Birds = "rbxassetid://4915214793", -- Ambient birds
    Wind = "rbxassetid://3645269782", -- Wind sound effect

    -- Effects
    Footstep = "rbxassetid://5892287796", -- Stone footstep
}

-- Tutorial hint configurations
AssetConfig.Tutorial = {
    HintColor = Color3.fromRGB(255, 255, 150),
    GlowColor = Color3.fromRGB(100, 255, 100),
    PathParticleColor = Color3.fromRGB(255, 220, 100),
    FadeDuration = 2.0,
}

return AssetConfig
