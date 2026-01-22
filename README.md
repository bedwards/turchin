# Cliodynamics

Mathematical modeling of historical dynamics using Peter Turchin's Structural-Demographic Theory (SDT).

## Overview

This project replicates and extends cliodynamics research, building a pipeline from historical data to instability forecasting:

1. Ingest data from the Seshat Global History Databank
2. Implement SDT differential equations
3. Calibrate models to historical data
4. Replicate published analyses (Roman Empire, U.S. *Ages of Discord*)
5. Generate instability forecasts

## Installation

### Requirements

- Python 3.11 or higher

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
tests/                 # Test suite
data/                  # Local datasets (git-ignored)
```

## References

- Turchin, P. (2016). *Ages of Discord*. Beresta Books.
- Turchin & Nefedov (2009). *Secular Cycles*. Princeton University Press.
- Seshat Equinox-2020: https://doi.org/10.5281/zenodo.6642229

## License

MIT
