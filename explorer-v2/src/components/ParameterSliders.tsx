/**
 * Parameter sliders for real-time SDT model adjustment.
 */

import { SDTParams, SDTState, PRESETS, applyPreset } from '../models/parameters';

interface ParameterSlidersProps {
  params: SDTParams;
  state: SDTState;
  onParamsChange: (params: SDTParams) => void;
  onStateChange: (state: SDTState) => void;
}

interface SliderConfig {
  key: keyof SDTParams;
  label: string;
  min: number;
  max: number;
  step: number;
  description: string;
}

const PARAM_SLIDERS: SliderConfig[] = [
  { key: 'r_max', label: 'r_max', min: 0.001, max: 0.05, step: 0.001, description: 'Population growth rate' },
  { key: 'alpha', label: 'alpha', min: 0.001, max: 0.02, step: 0.001, description: 'Elite upward mobility' },
  { key: 'delta_e', label: 'delta_e', min: 0.01, max: 0.05, step: 0.005, description: 'Elite attrition rate' },
  { key: 'gamma', label: 'gamma', min: 0.5, max: 4.0, step: 0.1, description: 'Labor supply effect' },
  { key: 'eta', label: 'eta', min: 0.5, max: 3.0, step: 0.1, description: 'Elite extraction effect' },
  { key: 'lambda_psi', label: 'lambda_psi', min: 0.01, max: 0.2, step: 0.01, description: 'Instability growth rate' },
  { key: 'psi_decay', label: 'psi_decay', min: 0.01, max: 0.1, step: 0.005, description: 'Instability decay rate' },
];

interface StateSliderConfig {
  key: keyof SDTState;
  label: string;
  min: number;
  max: number;
  step: number;
}

const STATE_SLIDERS: StateSliderConfig[] = [
  { key: 'N', label: 'N (Population)', min: 0.1, max: 1.2, step: 0.05 },
  { key: 'E', label: 'E (Elites)', min: 0.01, max: 0.3, step: 0.01 },
  { key: 'W', label: 'W (Wages)', min: 0.3, max: 1.5, step: 0.05 },
  { key: 'S', label: 'S (State)', min: 0.2, max: 1.5, step: 0.05 },
  { key: 'psi', label: 'psi (Instability)', min: 0, max: 1.0, step: 0.05 },
];

export function ParameterSliders({
  params,
  state,
  onParamsChange,
  onStateChange,
}: ParameterSlidersProps) {
  const handleParamChange = (key: keyof SDTParams, value: number) => {
    onParamsChange({ ...params, [key]: value });
  };

  const handleStateChange = (key: keyof SDTState, value: number) => {
    onStateChange({ ...state, [key]: value });
  };

  const handlePresetClick = (presetKey: string) => {
    const { params: newParams, state: newState } = applyPreset(presetKey);
    onParamsChange(newParams);
    onStateChange(newState);
  };

  return (
    <div className="parameter-panel">
      {/* Presets */}
      <section className="section">
        <h3 className="section-title">Presets</h3>
        <div className="preset-grid">
          {Object.entries(PRESETS).map(([key, preset]) => (
            <button
              key={key}
              className="preset-btn"
              onClick={() => handlePresetClick(key)}
              title={preset.description}
            >
              <span className="preset-name">{preset.name}</span>
              <span className="preset-desc">{preset.description}</span>
            </button>
          ))}
        </div>
      </section>

      {/* Initial Conditions */}
      <section className="section">
        <h3 className="section-title">Initial Conditions</h3>
        {STATE_SLIDERS.map(({ key, label, min, max, step }) => (
          <div key={key} className="param-group">
            <div className="param-header">
              <label className="param-label">{label}</label>
              <span className="param-value">{state[key].toFixed(2)}</span>
            </div>
            <input
              type="range"
              className="param-slider"
              min={min}
              max={max}
              step={step}
              value={state[key]}
              onChange={(e) => handleStateChange(key, parseFloat(e.target.value))}
            />
          </div>
        ))}
      </section>

      {/* Model Parameters */}
      <section className="section">
        <h3 className="section-title">Model Parameters</h3>
        {PARAM_SLIDERS.map(({ key, label, min, max, step, description }) => (
          <div key={key} className="param-group">
            <div className="param-header">
              <label className="param-label" title={description}>{label}</label>
              <span className="param-value">{params[key].toFixed(3)}</span>
            </div>
            <input
              type="range"
              className="param-slider"
              min={min}
              max={max}
              step={step}
              value={params[key]}
              onChange={(e) => handleParamChange(key, parseFloat(e.target.value))}
            />
            <div className="param-hint">{description}</div>
          </div>
        ))}
      </section>
    </div>
  );
}
