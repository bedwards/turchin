/**
 * Time series visualization using Plotly.js
 */

import Plot from 'react-plotly.js';
import { SimulationResult } from '../models/sdt';
import type { Data, Layout } from 'plotly.js';

interface TimeSeriesChartProps {
  result: SimulationResult | null;
  darkMode: boolean;
}

export function TimeSeriesChart({ result, darkMode }: TimeSeriesChartProps) {
  if (!result) {
    return (
      <div className="chart-placeholder">
        <p>Run simulation to see results</p>
      </div>
    );
  }

  const colors = {
    N: '#3b82f6',    // Blue - Population
    E: '#ef4444',    // Red - Elites
    W: '#22c55e',    // Green - Wages
    S: '#a855f7',    // Purple - State
    psi: '#f97316',  // Orange - Instability
  };

  const traces: Data[] = [
    {
      x: result.time,
      y: result.N,
      name: 'N (Population)',
      type: 'scatter',
      mode: 'lines',
      line: { color: colors.N, width: 2 },
    },
    {
      x: result.time,
      y: result.E,
      name: 'E (Elites)',
      type: 'scatter',
      mode: 'lines',
      line: { color: colors.E, width: 2 },
    },
    {
      x: result.time,
      y: result.W,
      name: 'W (Wages)',
      type: 'scatter',
      mode: 'lines',
      line: { color: colors.W, width: 2 },
    },
    {
      x: result.time,
      y: result.S,
      name: 'S (State)',
      type: 'scatter',
      mode: 'lines',
      line: { color: colors.S, width: 2 },
    },
    {
      x: result.time,
      y: result.psi,
      name: 'psi (Instability)',
      type: 'scatter',
      mode: 'lines',
      line: { color: colors.psi, width: 2 },
    },
  ];

  const bgColor = darkMode ? '#161b22' : '#ffffff';
  const textColor = darkMode ? '#e6edf3' : '#212529';
  const gridColor = darkMode ? '#30363d' : '#e5e7eb';

  const layout: Partial<Layout> = {
    title: {
      text: 'SDT Dynamics Over Time',
      font: { size: 16, color: textColor },
    },
    xaxis: {
      title: { text: 'Time (years)' },
      color: textColor,
      gridcolor: gridColor,
    },
    yaxis: {
      title: { text: 'Value (normalized)' },
      color: textColor,
      gridcolor: gridColor,
    },
    paper_bgcolor: bgColor,
    plot_bgcolor: bgColor,
    font: { color: textColor },
    legend: {
      orientation: 'h',
      y: -0.2,
    },
    margin: { t: 50, r: 30, b: 80, l: 60 },
    hovermode: 'x unified',
  };

  return (
    <Plot
      data={traces}
      layout={layout}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: '100%', height: '100%' }}
    />
  );
}
