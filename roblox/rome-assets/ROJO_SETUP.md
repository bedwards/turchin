# Rojo Setup Guide for Rome Game

This document describes how to set up and use Rojo for the Rome cliodynamics game.

## Prerequisites

1. **Install Rojo CLI** (requires Rust):
   ```bash
   cargo install rojo
   ```

   Or use [Aftman](https://github.com/LPGhatguy/aftman):
   ```bash
   aftman add rojo-rbx/rojo
   ```

2. **Install Rojo Plugin** in Roblox Studio:
   - Download from [Rojo releases](https://github.com/rojo-rbx/rojo/releases)
   - Or install via Roblox Plugin Marketplace

## Project Structure

```
rome-game/
├── default.project.json    # Rojo project configuration
├── selene.toml            # Linter configuration (std = "roblox")
├── .gitignore             # Ignores .rbxl/.rbxlx files
└── src/
    ├── server/            # ServerScriptService scripts
    │   └── GenerateTerrain.server.luau
    └── shared/            # ReplicatedStorage modules
        ├── TerrainGenerator.luau
        └── TerrainUtils.luau
```

## Usage

### Start Development Server

From the `rome-game/` directory:

```bash
cd roblox/rome-assets/rome-game
rojo serve
```

This starts a live-sync server on port 34872.

### Connect in Roblox Studio

1. Open Roblox Studio
2. Open or create a new place
3. Click the Rojo plugin icon
4. Click "Connect" (it will auto-connect to localhost:34872)
5. Changes to `.luau` files will sync automatically

### Build Place File

To build a standalone `.rbxlx` file:

```bash
rojo build -o rome-game.rbxlx
```

This creates a place file you can open directly in Studio.

## File Naming Conventions

Rojo uses file suffixes to determine script types:

| Suffix | Type | Roblox Class |
|--------|------|--------------|
| `.server.luau` | Server script | Script |
| `.client.luau` | Client script | LocalScript |
| `.luau` | Module | ModuleScript |

## Configuration (default.project.json)

```json
{
  "name": "rome-game",
  "tree": {
    "$className": "DataModel",
    "ServerScriptService": {
      "$className": "ServerScriptService",
      "Server": {
        "$path": "src/server"
      }
    },
    "ReplicatedStorage": {
      "$className": "ReplicatedStorage",
      "Shared": {
        "$path": "src/shared"
      }
    }
  }
}
```

## Linting with Selene

The project uses [Selene](https://github.com/Kampfkarren/selene) for linting.

Install:
```bash
cargo install selene
```

Run:
```bash
selene src/
```

The `selene.toml` file configures it for Roblox's standard library.

## Module Imports

In server scripts, import shared modules like this:

```lua
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local Shared = ReplicatedStorage:WaitForChild("Shared")
local TerrainGenerator = require(Shared:WaitForChild("TerrainGenerator"))
```

In shared modules, import sibling modules like this:

```lua
local TerrainGenerator = require(script.Parent.TerrainGenerator)
```

## Terrain System Overview

### TerrainGenerator.luau

Procedural terrain generation with:
- Seeded Perlin noise for reproducibility
- Multiple noise octaves (large valleys, medium hills, small detail)
- Flat zones for building placement
- Tiber River with water containment

Key functions:
- `generate(config?)` - Generate terrain with optional config
- `getHeightAt(x, z)` - Get height at coordinates
- `getMaterialAt(x, z, height)` - Get material at coordinates
- `getFlatZones()` - Get list of building zones

### TerrainUtils.luau

Ground elevation API for placing objects:
- `getGroundY(x, z)` - Raycast to find ground elevation
- `placeOnGround(model, x, z)` - Place model at correct height
- `placePartOnGround(part, x, z)` - Place part at correct height
- `isOnSolidGround(x, z)` - Check if position is not water
- `getSlopeAngle(x, z)` - Get terrain slope in degrees

### GenerateTerrain.server.luau

Server script that runs on game start to generate terrain.
Uses seed 753 (the legendary founding year of Rome).

## Testing Terrain

1. Start Rojo: `rojo serve`
2. Connect Studio
3. Play test - terrain generates automatically
4. Use TerrainUtils to test elevation:

```lua
-- In Studio command bar
local TerrainUtils = require(game.ReplicatedStorage.Shared.TerrainUtils)
print(TerrainUtils.getGroundY(0, 0)) -- Height at Forum center
```
