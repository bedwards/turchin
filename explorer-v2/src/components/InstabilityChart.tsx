/**
 * Political Stress Index (PSI) chart with crisis threshold.
 */

import Plot from 'react-plotly.js';
import { SimulationResult } from '../models/sdt';
import type { Data, Layout } from 'plotly.js';

interface InstabilityChartProps {
  result: SimulationResult | null;
  darkMode: boolean;
}

export function InstabilityChart({ result, darkMode }: InstabilityChartProps) {
  if (!result) {
    return (
      <div className="chart-placeholder">
        <p>Run simulation to see instability</p>
      </div>
    );
  }

  // PSI threshold for crisis (Turchin's ~0.6)
  const crisisThreshold = 0.6;

  // Find crisis periods
  const crisisStart: number[] = [];
  const crisisEnd: number[] = [];
  let inCrisis = false;

  for (let i = 0; i < result.psi.length; i++) {
    if (result.psi[i] >= crisisThreshold && !inCrisis) {
      crisisStart.push(result.time[i]);
      inCrisis = true;
    } else if (result.psi[i] < crisisThreshold && inCrisis) {
      crisisEnd.push(result.time[i]);
      inCrisis = false;
    }
  }
  if (inCrisis) {
    crisisEnd.push(result.time[result.time.length - 1]);
  }

  // Create shapes for crisis periods
  const crisisShapes = crisisStart.map((start, i) => ({
    type: 'rect' as const,
    xref: 'x' as const,
    yref: 'paper' as const,
    x0: start,
    x1: crisisEnd[i],
    y0: 0,
    y1: 1,
    fillcolor: 'rgba(239, 68, 68, 0.2)',
    line: { width: 0 },
  }));

  const traces: Data[] = [
    {
      x: result.time,
      y: result.psi,
      name: 'Political Stress Index',
      type: 'scatter',
      mode: 'lines',
      fill: 'tozeroy',
      line: { color: '#f97316', width: 2 },
      fillcolor: 'rgba(249, 115, 22, 0.2)',
    },
  ];

  const bgColor = darkMode ? '#161b22' : '#ffffff';
  const textColor = darkMode ? '#e6edf3' : '#212529';
  const gridColor = darkMode ? '#30363d' : '#e5e7eb';

  const layout: Partial<Layout> = {
    title: {
      text: 'Political Stress Index (PSI)',
      font: { size: 16, color: textColor },
    },
    xaxis: {
      title: { text: 'Time (years)' },
      color: textColor,
      gridcolor: gridColor,
    },
    yaxis: {
      title: { text: 'PSI' },
      color: textColor,
      gridcolor: gridColor,
      range: [0, Math.max(1, Math.max(...result.psi) * 1.1)],
    },
    paper_bgcolor: bgColor,
    plot_bgcolor: bgColor,
    font: { color: textColor },
    showlegend: false,
    margin: { t: 50, r: 30, b: 60, l: 60 },
    shapes: [
      // Crisis threshold line
      {
        type: 'line',
        xref: 'paper',
        x0: 0,
        x1: 1,
        yref: 'y',
        y0: crisisThreshold,
        y1: crisisThreshold,
        line: {
          color: '#ef4444',
          width: 2,
          dash: 'dash',
        },
      },
      ...crisisShapes,
    ],
    annotations: [
      {
        x: 1,
        y: crisisThreshold,
        xref: 'paper',
        yref: 'y',
        text: 'Crisis Threshold',
        showarrow: false,
        xanchor: 'right',
        yanchor: 'bottom',
        font: { color: '#ef4444', size: 10 },
      },
    ],
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
