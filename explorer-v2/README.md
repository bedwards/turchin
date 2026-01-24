# Cliodynamics Explorer v2

Interactive exploration of Structural-Demographic Theory (SDT) dynamics using
TypeScript and React.

## Overview

This is a native browser implementation of the SDT model, replacing the Pyodide-based
Explorer v1. Key advantages:

- **Fast**: Native JavaScript ODE solving (40-100x faster than Pyodide)
- **Instant Load**: No WASM bundle to download
- **Type Safe**: Full TypeScript implementation
- **Modern Stack**: React 19, Vite 6, Plotly.js

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Run tests
npm test

# Build for production
npm run build
```

## Architecture

```
src/
  models/
    sdt.ts          # SDT ODE equations + RK4 solver
    parameters.ts   # Type definitions and presets
  components/
    ParameterSliders.tsx
    TimeSeriesChart.tsx
    PhaseSpaceChart.tsx
    InstabilityChart.tsx
  App.tsx           # Main application
```

## SDT Model

The model implements the five coupled ODEs from Turchin's Structural-Demographic
Theory:

1. **Population (N)**: Logistic growth modified by real wages
2. **Elites (E)**: Growth from upward mobility, decline from attrition
3. **Wages (W)**: Labor supply/demand with elite extraction
4. **State (S)**: Revenue minus expenditure and elite burden
5. **Instability (psi)**: Political Stress Index accumulation

Integration uses fourth-order Runge-Kutta (RK4) for accuracy.

## Presets

- **Stable Society**: Balanced parameters, sustainable growth
- **American Crisis**: 2010-2020 polarization dynamics
- **Roman Collapse**: Late Western Empire dynamics
- **Medieval Expansion**: Early Capetian France dynamics

## Phase 2 Roadmap

- [ ] 3D phase space visualization (Three.js)
- [ ] Additional presets
- [ ] CSV export
- [ ] Cloudflare Pages deployment
- [ ] Web Workers for heavy simulations

## References

- Turchin, P. (2016). *Ages of Discord*. Beresta Books.
- Turchin, P. & Nefedov, S. (2009). *Secular Cycles*. Princeton University Press.
