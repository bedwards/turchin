/**
 * 2D Phase space visualization (3D planned for Phase 2).
 */

import Plot from 'react-plotly.js';
import { SimulationResult } from '../models/sdt';
import type { Data, Layout } from 'plotly.js';

interface PhaseSpaceChartProps {
  result: SimulationResult | null;
  xVar: 'N' | 'E' | 'W' | 'S' | 'psi';
  yVar: 'N' | 'E' | 'W' | 'S' | 'psi';
  darkMode: boolean;
}

const VAR_LABELS: Record<string, string> = {
  N: 'Population (N)',
  E: 'Elites (E)',
  W: 'Wages (W)',
  S: 'State Health (S)',
  psi: 'Instability (psi)',
};

export function PhaseSpaceChart({ result, xVar, yVar, darkMode }: PhaseSpaceChartProps) {
  if (!result) {
    return (
      <div className="chart-placeholder">
        <p>Run simulation to see phase space</p>
      </div>
    );
  }

  const x = result[xVar];
  const y = result[yVar];

  // Color by time (early = blue, late = red)
  const colors = result.time.map((t) => t);

  const trace: Data = {
    x,
    y,
    mode: 'lines+markers',
    type: 'scatter',
    marker: {
      size: 4,
      color: colors,
      colorscale: 'Viridis',
      showscale: true,
      colorbar: {
        title: { text: 'Time' },
        thickness: 15,
      },
    },
    line: {
      width: 1,
      color: 'rgba(100, 100, 100, 0.3)',
    },
    hovertemplate: xVar + ': %{x:.3f}<br>' + yVar + ': %{y:.3f}<br>Time: %{marker.color:.1f}<extra></extra>',
  };

  // Mark start and end points
  const startEnd: Data = {
    x: [x[0], x[x.length - 1]],
    y: [y[0], y[y.length - 1]],
    mode: 'markers',
    type: 'scatter',
    marker: {
      size: 12,
      color: ['#22c55e', '#ef4444'],
      symbol: ['circle', 'square'],
    },
    name: 'Start / End',
    hovertemplate: '%{text}<extra></extra>',
    text: ['Start', 'End'],
  };

  const bgColor = darkMode ? '#161b22' : '#ffffff';
  const textColor = darkMode ? '#e6edf3' : '#212529';
  const gridColor = darkMode ? '#30363d' : '#e5e7eb';

  const layout: Partial<Layout> = {
    title: {
      text: 'Phase Space: ' + xVar + ' vs ' + yVar,
      font: { size: 16, color: textColor },
    },
    xaxis: {
      title: { text: VAR_LABELS[xVar] },
      color: textColor,
      gridcolor: gridColor,
    },
    yaxis: {
      title: { text: VAR_LABELS[yVar] },
      color: textColor,
      gridcolor: gridColor,
    },
    paper_bgcolor: bgColor,
    plot_bgcolor: bgColor,
    font: { color: textColor },
    showlegend: false,
    margin: { t: 50, r: 80, b: 60, l: 60 },
    hovermode: 'closest',
  };

  return (
    <Plot
      data={[trace, startEnd]}
      layout={layout}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: '100%', height: '100%' }}
    />
  );
}
