/**
 * Parameter types for Structural-Demographic Theory models.
 *
 * Based on Turchin (2016) Ages of Discord, Chapter 2
 * and Turchin & Nefedov (2009) Secular Cycles.
 */

/**
 * SDT model parameters - calibrated values for different societies.
 */
export interface SDTParams {
  // Population dynamics
  r_max: number;   // Maximum population growth rate
  K_0: number;     // Carrying capacity
  beta: number;    // Wage sensitivity of growth

  // Elite dynamics
  mu: number;      // Elite extraction rate
  alpha: number;   // Upward mobility rate
  delta_e: number; // Elite death/downward mobility

  // Wage dynamics
  gamma: number;   // Labor supply effect
  eta: number;     // Elite competition effect

  // State dynamics
  rho: number;     // Revenue coefficient
  sigma: number;   // Expenditure rate
  epsilon: number; // Elite burden on state

  // Instability dynamics
  lambda_psi: number; // Instability growth rate
  theta_w: number;    // Wage contribution to PSI
  theta_e: number;    // Elite contribution to PSI
  theta_s: number;    // State weakness contribution
  psi_decay: number;  // Natural decay rate

  // Reference values
  W_0: number; // Baseline wage
  E_0: number; // Baseline elite population
  S_0: number; // Baseline state health
}

/**
 * State variables at a point in time.
 */
export interface SDTState {
  N: number;   // Population (normalized)
  E: number;   // Elite population
  W: number;   // Real wages
  S: number;   // State fiscal health
  psi: number; // Political Stress Index
}

/**
 * Default parameters (baseline society).
 */
export const DEFAULT_PARAMS: SDTParams = {
  r_max: 0.02,
  K_0: 1.0,
  beta: 1.0,
  mu: 0.2,
  alpha: 0.005,
  delta_e: 0.02,
  gamma: 2.0,
  eta: 1.0,
  rho: 0.2,
  sigma: 0.1,
  epsilon: 0.05,
  lambda_psi: 0.05,
  theta_w: 1.0,
  theta_e: 1.0,
  theta_s: 1.0,
  psi_decay: 0.02,
  W_0: 1.0,
  E_0: 0.1,
  S_0: 1.0,
};

/**
 * Default initial state.
 */
export const DEFAULT_STATE: SDTState = {
  N: 0.5,   // Half carrying capacity
  E: 0.05,  // Small elite fraction
  W: 1.0,   // Baseline wages
  S: 1.0,   // Baseline state health
  psi: 0.0, // No initial instability
};

/**
 * Preset configurations for different historical scenarios.
 */
export interface Preset {
  name: string;
  description: string;
  params: Partial<SDTParams>;
  state: Partial<SDTState>;
}

export const PRESETS: Record<string, Preset> = {
  stable: {
    name: 'Stable Society',
    description: 'Balanced parameters, sustainable growth',
    params: {
      r_max: 0.015,
      alpha: 0.003,
      lambda_psi: 0.03,
    },
    state: {
      N: 0.4,
      E: 0.04,
    },
  },
  american_crisis: {
    name: 'American Crisis',
    description: '2010-2020 polarization dynamics',
    params: {
      r_max: 0.01,
      alpha: 0.008,
      eta: 1.5,
      lambda_psi: 0.08,
      theta_e: 1.5,
    },
    state: {
      N: 0.9,
      E: 0.15,
      W: 0.7,
      psi: 0.3,
    },
  },
  roman_collapse: {
    name: 'Roman Collapse',
    description: 'Late Western Empire dynamics',
    params: {
      r_max: 0.005,
      alpha: 0.01,
      epsilon: 0.1,
      sigma: 0.15,
      lambda_psi: 0.1,
    },
    state: {
      N: 0.95,
      E: 0.2,
      W: 0.5,
      S: 0.4,
      psi: 0.5,
    },
  },
  medieval_expansion: {
    name: 'Medieval Expansion',
    description: 'Early Capetian France dynamics',
    params: {
      r_max: 0.025,
      K_0: 1.2,
      alpha: 0.004,
    },
    state: {
      N: 0.3,
      E: 0.03,
      W: 1.2,
      S: 1.1,
    },
  },
};

/**
 * Apply preset to default parameters and state.
 */
export function applyPreset(presetKey: string): { params: SDTParams; state: SDTState } {
  const preset = PRESETS[presetKey];
  if (!preset) {
    return { params: { ...DEFAULT_PARAMS }, state: { ...DEFAULT_STATE } };
  }
  return {
    params: { ...DEFAULT_PARAMS, ...preset.params },
    state: { ...DEFAULT_STATE, ...preset.state },
  };
}
