# Project: Cliodynamics Replication

This project replicates Peter Turchin's cliodynamics research—mathematical modeling of historical dynamics using Structural-Demographic Theory (SDT).

## Goal

Build a complete pipeline from historical data to instability forecasting:
1. Ingest Seshat Global History Databank
2. Implement SDT differential equations
3. Calibrate models to historical data
4. Replicate published analyses (Roman Empire, U.S. *Ages of Discord*)
5. Generate instability forecasts

## Architecture

```
src/
  cliodynamics/
    __init__.py
    data/              # Seshat data access
      download.py      # Fetch datasets
      parser.py        # Parse raw CSV
      db.py            # Query interface (SeshatDB class)
    models/            # Mathematical models
      sdt.py           # SDT equations
      params.py        # Parameter dataclasses
    simulation/        # ODE solving
      simulator.py     # Run simulations
      events.py        # Event detection
    calibration/       # Parameter fitting
      calibrator.py    # Optimization wrapper
      uncertainty.py   # Confidence intervals
    viz/               # Visualization
      plots.py         # Time series, phase space
      cycles.py        # Secular cycle detection
    forecast/          # Prediction pipeline
      forecaster.py    # Probabilistic forecasts
tests/
data/                  # Downloaded datasets (.gitignore'd)
notebooks/             # Analysis notebooks
docs/                  # GitHub Pages: essays and visualizations
```

## Key References

- Turchin, P. (2016). *Ages of Discord*. Beresta Books.
- Turchin & Nefedov (2009). *Secular Cycles*. Princeton University Press.
- Seshat Equinox-2020: https://doi.org/10.5281/zenodo.6642229

## Worker Instructions

You are a one-shot worker implementing a specific GitHub issue. Your responsibilities:

1. **Read the issue carefully** - Understand scope and acceptance criteria
2. **Implement the feature** - Write clean, tested code
3. **Create a PR** - Push to feature branch and open PR with:
   - Summary of changes
   - How to test
   - Reference to issue: "Closes #N"
4. **Do not merge** - The orchestrator handles merges after review

### Conventions

- Python 3.11+
- Use type hints
- Docstrings for public functions (Google style)
- Tests in `tests/` mirroring `src/` structure
- Format with `ruff format`, lint with `ruff check`

### Dependencies

Check issue dependencies. If a dependency isn't merged yet, note it in your PR.

## Data Sources

- **Seshat Equinox-2020**: Primary historical dataset
  - Zenodo: https://doi.org/10.5281/zenodo.6642229
  - GitHub: https://github.com/seshatdb/Equinox_Data
- **U.S. Economic Data**: FRED, Historical Statistics of the United States
- **Elite Indicators**: ABA (lawyers), NSF (PhDs)

## SDT Model Overview

Core variables:
- `N` - Population
- `E` - Elite population
- `W` - Real wages / well-being
- `S` - State fiscal health
- `ψ` (psi) - Political Stress Index

Dynamics are coupled ODEs capturing feedback loops:
- Population growth limited by carrying capacity
- Elite overproduction from wealth accumulation
- Wage depression from labor oversupply
- State fiscal strain from elite competition
- Instability rises with elite surplus and popular immiseration

## docs/ Directory

**IMPORTANT**: `docs/` is reserved for GitHub Pages content:
- Essays explaining cliodynamics concepts
- Interactive visualizations
- Analysis writeups

Do NOT put package documentation, API docs, or other artifacts in `docs/`.
