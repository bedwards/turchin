# Cliodynamics

Mathematical modeling of historical dynamics using Peter Turchin's Structural-Demographic Theory (SDT).

## Overview

This project replicates and extends cliodynamics research, building a pipeline from historical data to instability forecasting:

1. Ingest data from the Seshat Global History Databank
2. Implement SDT differential equations
3. Calibrate models to historical data
4. Replicate published analyses (Roman Empire, U.S. *Ages of Discord*)
5. Generate instability forecasts
6. Document findings in long-form essays on our GitHub Pages site

## Essays

We publish in-depth essays documenting our findings and process at:

**[bedwards.github.io/turchin](https://bedwards.github.io/turchin)**

Each essay explores cliodynamics concepts from first principles, presents visualizations generated from our analysis, and reflects on the process of building this project with Claude Code.

## Installation

### Requirements

- Python 3.11 or higher
- Gemini API key (for illustrations, stored in `.env`)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/bedwards/turchin.git
   cd turchin
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

4. (Optional) Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

5. Create `.env` file with your API key:
   ```bash
   echo "GEMINI_API_KEY=your_key_here" > .env
   ```

## Usage

```python
from cliodynamics import __version__
print(f"Cliodynamics version: {__version__}")
```

## Development

### Running Tests

```bash
pytest
```

### Code Quality

Format code:
```bash
ruff format src/ tests/
```

Lint code:
```bash
ruff check src/ tests/
```

## Project Structure

```
src/
  cliodynamics/
    __init__.py
    data/              # Seshat data access
    models/            # Mathematical models (SDT equations)
    analysis/          # Analysis tools
    viz/               # Visualization tools
tests/                 # Test suite
data/                  # Local datasets (git-ignored)
docs/                  # GitHub Pages site
  index.html           # Essay listing
  essays/              # Long-form essays
  assets/              # Images, charts, animations
scripts/
  word_count.py        # Essay metadata generator
```

## Visualization Stack

- **Gemini**: Text-to-image for conceptual illustrations
- **Polars + Altair**: Data manipulation and interactive charts
- **cliodynamics.viz**: Project-specific visualizations and animations

## References

- Turchin, P. (2016). *Ages of Discord*. Beresta Books.
- Turchin & Nefedov (2009). *Secular Cycles*. Princeton University Press.
- Seshat Equinox-2020: https://doi.org/10.5281/zenodo.6642229

## Contact

- **Brian Edwards**
- Email: brian.mabry.edwards@gmail.com
- Phone: 512-584-6841
- Waco, Texas, USA

Built with Claude Code 2.1.15 (Opus 4.5), January 2026.

## License

MIT
