# Project: Cliodynamics Replication

This project replicates Peter Turchin's cliodynamics research—mathematical modeling of historical dynamics using Structural-Demographic Theory (SDT).

## Contact

- **Author**: Brian Edwards
- **Email**: brian.mabry.edwards@gmail.com
- **Phone**: 512-584-6841
- **Location**: Waco, Texas, USA
- **Project Started**: January 22, 2026
- **Built With**: Claude Code 2.1.15 (Opus 4.5)

## Goal

Build a complete pipeline from historical data to instability forecasting:
1. Ingest Seshat Global History Databank
2. Implement SDT differential equations
3. Calibrate models to historical data
4. Replicate published analyses (Roman Empire, U.S. *Ages of Discord*)
5. Generate instability forecasts
6. Publish essays documenting our findings on GitHub Pages

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
      animations.py    # Animated visualizations
    forecast/          # Prediction pipeline
      forecaster.py    # Probabilistic forecasts
tests/
data/                  # Downloaded datasets (.gitignore'd)
notebooks/             # Analysis notebooks
docs/                  # GitHub Pages: essays and visualizations
  assets/
    images/            # Generated illustrations (Gemini)
    charts/            # Data visualizations (Altair)
    animations/        # Animated visualizations
  essays/              # Long-form essays
  index.html           # Home page listing essays
scripts/
  word_count.py        # Essay metadata generator
```

## Environment Variables

API keys are stored in `.env` (git-ignored):

```
GEMINI_API_KEY=your_key_here
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

## Visualization Stack

### Illustrations (Gemini)
Use Google's Gemini API for text-to-image generation to create conceptual illustrations for essays. The API key is in `.env`.

```python
import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
```

### Data Visualizations (Polars + Altair)
Use Polars for data manipulation and Altair for declarative chart generation:

```python
import polars as pl
import altair as alt

# Load and transform data with Polars
df = pl.read_csv("data/seshat.csv")
# Create interactive charts with Altair
chart = alt.Chart(df.to_pandas()).mark_line().encode(x="year", y="psi")
chart.save("docs/assets/charts/instability.html")
```

### Animated Visualizations
Use the project's `cliodynamics.viz.animations` module for animated time series, phase space trajectories, and secular cycle visualizations.

## docs/ Directory (GitHub Pages)

**IMPORTANT**: `docs/` is reserved for GitHub Pages content.

### Structure
- `docs/index.html` - Home page listing all essays with metadata
- `docs/essays/` - Individual essay HTML files
- `docs/assets/images/` - Gemini-generated illustrations
- `docs/assets/charts/` - Altair-generated interactive charts
- `docs/assets/animations/` - Animated visualizations

### Style
- Polished, professional design
- Lexend font (or similar clean sans-serif)
- Essays are long-form (1+ hour reading time)
- See `WRITING_STYLE.md` for content guidelines

### Essay Metadata
Every essay must include metadata generated by `scripts/word_count.py`:
- Word count (computed, never estimated)
- Reading time (computed at 200 words/minute)
- Tags for categorization
- Publication date

Do NOT put package documentation, API docs, or other artifacts in `docs/`.

## Essay Content Guidelines

Essays document our cliodynamics replication work and serve two purposes:

1. **Explain findings** - Outcomes, insights, visualizations from our analysis
2. **Document process** - How we built this (Claude Code, workers, GitHub issues, code reviews)

Primary source material is our own work in this project. Web research provides broader context but is secondary.

See `WRITING_STYLE.md` for detailed writing guidelines.
