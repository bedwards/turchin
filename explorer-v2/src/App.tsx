/**
 * Cliodynamics Explorer v2 - Main Application
 *
 * Interactive exploration of Structural-Demographic Theory dynamics.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { SDTModel, SimulationResult } from './models/sdt';
import { SDTParams, SDTState, DEFAULT_PARAMS, DEFAULT_STATE } from './models/parameters';
import { ParameterSliders } from './components/ParameterSliders';
import { TimeSeriesChart } from './components/TimeSeriesChart';
import { PhaseSpaceChart } from './components/PhaseSpaceChart';
import { InstabilityChart } from './components/InstabilityChart';
import { formatMetric } from './utils/formatNumber';
import './App.css';

function App() {
  const [params, setParams] = useState<SDTParams>(DEFAULT_PARAMS);
  const [initialState, setInitialState] = useState<SDTState>(DEFAULT_STATE);
  const [duration, setDuration] = useState(200);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [execTime, setExecTime] = useState<number | null>(null);
  const [autoRun, setAutoRun] = useState(true);

  // Track if parameters have changed since last simulation
  const [hasUnsimulatedChanges, setHasUnsimulatedChanges] = useState(false);
  const lastSimulatedConfig = useRef<string | null>(null);

  // Detect dark mode
  const [darkMode, setDarkMode] = useState(() =>
    window.matchMedia('(prefers-color-scheme: dark)').matches
  );

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (e: MediaQueryListEvent) => setDarkMode(e.matches);
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  // Create a config key to track what's been simulated
  const getConfigKey = useCallback(() => {
    return JSON.stringify({ params, initialState, duration });
  }, [params, initialState, duration]);

  // Run simulation
  const runSimulation = useCallback(() => {
    setIsRunning(true);
    const start = performance.now();
    const configKey = getConfigKey();

    // Run in next tick to allow UI update
    setTimeout(() => {
      const model = new SDTModel(params);
      const simResult = model.simulate(initialState, duration);
      const elapsed = performance.now() - start;

      setResult(simResult);
      setExecTime(elapsed);
      setIsRunning(false);
      lastSimulatedConfig.current = configKey;
      setHasUnsimulatedChanges(false);
    }, 0);
  }, [params, initialState, duration, getConfigKey]);

  // Track when parameters change but haven't been simulated yet
  useEffect(() => {
    const currentConfig = getConfigKey();
    if (lastSimulatedConfig.current !== currentConfig) {
      setHasUnsimulatedChanges(true);
    }
  }, [params, initialState, duration, getConfigKey]);

  // Auto-run on parameter change
  useEffect(() => {
    if (autoRun) {
      const timeoutId = setTimeout(runSimulation, 100);
      return () => clearTimeout(timeoutId);
    }
  }, [params, initialState, duration, autoRun, runSimulation]);

  // Run on initial load
  useEffect(() => {
    runSimulation();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const statusClass = isRunning ? 'loading' : 'ready';
  const statusText = isRunning
    ? 'Simulating...'
    : execTime
      ? 'Completed in ' + execTime.toFixed(1) + 'ms'
      : 'Ready';

  const finalPsi = result ? result.psi[result.psi.length - 1] : 0;
  const psiDanger = finalPsi > 0.6;

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <span className="logo">Cliodynamics Explorer</span>
            <span className="version-badge">v2 (TypeScript)</span>
          </div>
          <nav className="header-nav">
            <a href="../">Essays</a>
            <a href="https://github.com/bedwards/turchin" target="_blank" rel="noopener">GitHub</a>
          </nav>
        </div>
      </header>

      <div className="app-container">
        <aside className="sidebar">
          {/* Status bar */}
          <div className="status-bar">
            <span className={'status-dot ' + statusClass}></span>
            <span>{statusText}</span>
          </div>

          <ParameterSliders
            params={params}
            state={initialState}
            onParamsChange={setParams}
            onStateChange={setInitialState}
          />

          {/* Simulation Controls */}
          <section className="section">
            <h3 className="section-title">Simulation</h3>
            <div className="control-row">
              <label>Duration:</label>
              <input
                type="number"
                value={duration}
                onChange={(e) => setDuration(parseInt(e.target.value) || 200)}
                min={50}
                max={500}
              />
              <span className="unit">years</span>
            </div>
            <div className="control-row checkbox-row">
              <label>
                <input
                  type="checkbox"
                  checked={autoRun}
                  onChange={(e) => setAutoRun(e.target.checked)}
                />
                Auto-run on change
              </label>
            </div>
            <button
              className={'run-btn' + (autoRun && !hasUnsimulatedChanges ? ' auto-run-active' : '')}
              onClick={runSimulation}
              disabled={isRunning || (autoRun && !hasUnsimulatedChanges)}
              title={autoRun && !hasUnsimulatedChanges ? 'Auto-run is enabled - simulation runs automatically on parameter changes' : ''}
            >
              {isRunning ? 'Running...' : autoRun && !hasUnsimulatedChanges ? 'Auto-Running' : 'Run Simulation'}
            </button>
          </section>
        </aside>

        <main className="main-content">
          {/* Summary Cards */}
          {result && (
            <div className="info-grid">
              <div className="info-card">
                <div className="info-label">Final Population</div>
                <div className="info-value">{formatMetric(result.N[result.N.length - 1], 'population')}</div>
              </div>
              <div className="info-card">
                <div className="info-label">Final Elites</div>
                <div className="info-value">{formatMetric(result.E[result.E.length - 1], 'elites')}</div>
              </div>
              <div className="info-card">
                <div className="info-label">Final Wages</div>
                <div className="info-value">{formatMetric(result.W[result.W.length - 1], 'wages')}</div>
              </div>
              <div className="info-card">
                <div className="info-label">Final State</div>
                <div className="info-value">{formatMetric(result.S[result.S.length - 1], 'state')}</div>
              </div>
              <div className="info-card">
                <div className={'info-label' + (psiDanger ? ' danger' : '')}>
                  Final PSI
                </div>
                <div className={'info-value' + (psiDanger ? ' danger' : '')}>
                  {formatMetric(result.psi[result.psi.length - 1], 'psi')}
                </div>
              </div>
            </div>
          )}

          {/* Charts */}
          <div className="charts-grid">
            <div className="chart-card full-width">
              <h3 className="chart-title">Time Series Dynamics</h3>
              <p className="chart-subtitle">Evolution of all state variables over time</p>
              <div className="chart-container tall">
                <TimeSeriesChart result={result} darkMode={darkMode} />
              </div>
            </div>

            <div className="chart-card">
              <h3 className="chart-title">Phase Space: Population vs Wages</h3>
              <p className="chart-subtitle">Trajectory through state space</p>
              <div className="chart-container">
                <PhaseSpaceChart result={result} xVar="N" yVar="W" darkMode={darkMode} />
              </div>
            </div>

            <div className="chart-card">
              <h3 className="chart-title">Phase Space: Elites vs Instability</h3>
              <p className="chart-subtitle">Elite overproduction dynamics</p>
              <div className="chart-container">
                <PhaseSpaceChart result={result} xVar="E" yVar="psi" darkMode={darkMode} />
              </div>
            </div>

            <div className="chart-card full-width">
              <h3 className="chart-title">Political Stress Index</h3>
              <p className="chart-subtitle">Instability dynamics with crisis threshold (psi = 0.6)</p>
              <div className="chart-container">
                <InstabilityChart result={result} darkMode={darkMode} />
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
