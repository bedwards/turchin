# Mac Studio Session Kickoff

Use this prompt to start a fresh Claude Code session on the Mac Studio.

## Initial Prompt

```
I'm continuing work on the cliodynamics project - replicating Peter Turchin's research on mathematical modeling of historical dynamics.

## Quick Context

This project builds a complete pipeline from Seshat historical data to instability forecasting, with long-form essays documenting our work on GitHub Pages.

Read these files first:
1. CLAUDE.md - Full project context and worker instructions
2. MANAGER_ROLE.md - Your responsibilities as orchestrator
3. TURCHIN.md - Research background
4. WRITING_STYLE.md - Essay guidelines

## Current State

**Completed (8 issues merged):**
- #1 Project scaffolding
- #2 Seshat data download (Equinox-2020)
- #3 Data access layer (SeshatDB)
- #4 Core SDT equations
- #5 ODE solver and simulation
- #6 Parameter calibration framework
- #12 GitHub Pages + first essay
- #14 Seshat Databank essay

**Published:** https://bedwards.github.io/turchin/

**Open Work Issues (in dependency order):**
- #26 Seshat API client → #27 Polaris-2025 download → #7 Roman case study
- #8 US case study (depends on #6)
- #9 Visualization module
- #10 Instability forecasting (depends on #6, #8, #9)
- #30 PostgreSQL database
- #31 Monte Carlo simulations
- #32 Large ensemble forecasting
- #33 Roblox game

**Open Essay Issues:**
- #15 Essay: SDT Mathematics (needs #4, #5 - READY)
- #16 Essay: Calibration (needs #6 - READY)
- #17 Essay: Roman Empire (needs #7)
- #18 Essay: United States (needs #8)
- #19 Essay: Visualization (needs #9)
- #20 Essay: Forecasting (needs #10)

**Image Fix Issues:**
- #28 Fix global-map-seshat.png (Rome label in ocean)
- #29 Fix chart-timeline.png (compressed, unreadable)

## This Machine's Advantages

Mac Studio M2 Ultra:
- 24 CPU cores
- 76 GPU cores
- 192 GB RAM
- 3.74 TB free storage

Good for: Monte Carlo (#31), large ensembles (#32), heavy computation.

## Suggested Next Steps

1. **Fix images first** (#28, #29) - quick wins, improves site quality
2. **Polaris-2025 integration** (#26, #27) - get latest Seshat data
3. **Write Essay #15** (SDT Mathematics) - dependencies met
4. **Roman case study** (#7) - first real SDT application

## Important Notes

- Use `cliodynamics.viz.charts` and `cliodynamics.viz.images` for consistent visuals
- Visually verify ALL generated images before committing
- Essays must be 12,000+ words (verify with scripts/word_count.py)
- Prefer Polaris-2025 for new work, keep Equinox-2020 support
- Run `gh issue list` to see all open issues

Ready to continue. What would you like to work on?
```

## Environment Setup

After cloning, run:
```bash
# Clone repo
git clone git@github.com:bedwards/turchin.git
cd turchin

# Create conda environment
conda create -n turchin python=3.11 -y
conda activate turchin

# Install dependencies
pip install -e ".[dev]"

# Set up API key
echo "GEMINI_API_KEY=your_key" > .env

# Verify installation
python -c "from cliodynamics.models import SDTModel; print('OK')"
```

## File Locations

- Project root: `~/turchin/`
- Essays: `docs/essays/`
- Images: `docs/assets/images/`
- Code: `src/cliodynamics/`
- Tests: `tests/`

## GitHub CLI Authentication

If needed:
```bash
gh auth login
```
