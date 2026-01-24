/**
 * Structural-Demographic Theory ODE solver in TypeScript.
 *
 * This implements the core SDT model equations directly in JavaScript,
 * eliminating the need for Pyodide/WebAssembly.
 *
 * References:
 * - Turchin, P. (2016). Ages of Discord. Chapter 2.
 * - Turchin & Nefedov (2009). Secular Cycles. Mathematical Appendix.
 */

import { SDTParams, SDTState, DEFAULT_PARAMS, DEFAULT_STATE } from './parameters';

/**
 * Simulation result at each time step.
 */
export interface SimulationResult {
  time: number[];
  N: number[];
  E: number[];
  W: number[];
  S: number[];
  psi: number[];
}

/**
 * SDT Model class implementing the coupled ODE system.
 */
export class SDTModel {
  params: SDTParams;

  constructor(params: Partial<SDTParams> = {}) {
    this.params = { ...DEFAULT_PARAMS, ...params };
  }

  /**
   * Population dynamics: dN/dt
   * Logistic growth modified by real wages.
   */
  populationDynamics(N: number, W: number): number {
    const p = this.params;
    const rEffective = p.r_max * Math.pow(W / p.W_0, p.beta);
    return rEffective * N * (1 - N / p.K_0);
  }

  /**
   * Elite dynamics: dE/dt
   * Elite growth from upward mobility minus attrition.
   */
  eliteDynamics(E: number, N: number, W: number): number {
    const p = this.params;
    const wageRatio = W / p.W_0;
    const surplusFactor = Math.max(0, 1 - wageRatio + p.mu);
    const upwardMobility = p.alpha * surplusFactor * N;
    const eliteLoss = p.delta_e * E;
    return upwardMobility - eliteLoss;
  }

  /**
   * Wage dynamics: dW/dt
   * Supply-demand with elite extraction.
   */
  wageDynamics(W: number, N: number, E: number): number {
    const p = this.params;
    const laborEffect = p.gamma * (1 - N / p.K_0);
    const eliteRatio = E / p.E_0;
    const extractionEffect = p.eta * (eliteRatio - 1);
    return W * (laborEffect - extractionEffect);
  }

  /**
   * State dynamics: dS/dt
   * Revenue minus expenditures and elite burden.
   */
  stateDynamics(S: number, N: number, E: number, W: number): number {
    const p = this.params;
    const output = W * N;
    const revenue = p.rho * output;
    const baseExpenditure = p.sigma * S;
    const eliteBurden = p.epsilon * E;
    return revenue - baseExpenditure - eliteBurden;
  }

  /**
   * Instability dynamics: dpsi/dt
   * Political Stress Index accumulation.
   */
  instabilityDynamics(psi: number, E: number, W: number, S: number): number {
    const p = this.params;

    // Stress contributions (positive when conditions worsen)
    const wageStress = p.theta_w * Math.max(0, p.W_0 / Math.max(W, 1e-6) - 1);
    const eliteStress = p.theta_e * Math.max(0, E / p.E_0 - 1);
    const stateStress = p.theta_s * Math.max(0, p.S_0 / Math.max(S, 1e-6) - 1);

    const accumulation = p.lambda_psi * (wageStress + eliteStress + stateStress);
    const decay = p.psi_decay * psi;

    return accumulation - decay;
  }

  /**
   * Compute all derivatives for state vector [N, E, W, S, psi].
   */
  system(state: SDTState): SDTState {
    // Clamp to ensure numerical stability
    const N = Math.max(state.N, 1e-10);
    const E = Math.max(state.E, 1e-10);
    const W = Math.max(state.W, 1e-10);
    const S = Math.max(state.S, 1e-10);
    const psi = Math.max(state.psi, 0);

    return {
      N: this.populationDynamics(N, W),
      E: this.eliteDynamics(E, N, W),
      W: this.wageDynamics(W, N, E),
      S: this.stateDynamics(S, N, E, W),
      psi: this.instabilityDynamics(psi, E, W, S),
    };
  }

  /**
   * Fourth-order Runge-Kutta integration step.
   */
  rk4Step(state: SDTState, dt: number): SDTState {
    // k1 = f(y)
    const k1 = this.system(state);

    // k2 = f(y + dt/2 * k1)
    const state2: SDTState = {
      N: state.N + (dt / 2) * k1.N,
      E: state.E + (dt / 2) * k1.E,
      W: state.W + (dt / 2) * k1.W,
      S: state.S + (dt / 2) * k1.S,
      psi: state.psi + (dt / 2) * k1.psi,
    };
    const k2 = this.system(state2);

    // k3 = f(y + dt/2 * k2)
    const state3: SDTState = {
      N: state.N + (dt / 2) * k2.N,
      E: state.E + (dt / 2) * k2.E,
      W: state.W + (dt / 2) * k2.W,
      S: state.S + (dt / 2) * k2.S,
      psi: state.psi + (dt / 2) * k2.psi,
    };
    const k3 = this.system(state3);

    // k4 = f(y + dt * k3)
    const state4: SDTState = {
      N: state.N + dt * k3.N,
      E: state.E + dt * k3.E,
      W: state.W + dt * k3.W,
      S: state.S + dt * k3.S,
      psi: state.psi + dt * k3.psi,
    };
    const k4 = this.system(state4);

    // y_next = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return {
      N: state.N + (dt / 6) * (k1.N + 2 * k2.N + 2 * k3.N + k4.N),
      E: state.E + (dt / 6) * (k1.E + 2 * k2.E + 2 * k3.E + k4.E),
      W: state.W + (dt / 6) * (k1.W + 2 * k2.W + 2 * k3.W + k4.W),
      S: state.S + (dt / 6) * (k1.S + 2 * k2.S + 2 * k3.S + k4.S),
      psi: state.psi + (dt / 6) * (k1.psi + 2 * k2.psi + 2 * k3.psi + k4.psi),
    };
  }

  /**
   * Run simulation for specified duration.
   *
   * @param initialState - Starting state
   * @param duration - Total simulation time (years)
   * @param dt - Time step (years)
   * @returns Simulation results
   */
  simulate(
    initialState: Partial<SDTState> = {},
    duration: number = 200,
    dt: number = 0.5
  ): SimulationResult {
    const state: SDTState = { ...DEFAULT_STATE, ...initialState };
    const nSteps = Math.ceil(duration / dt);

    const result: SimulationResult = {
      time: new Array(nSteps + 1),
      N: new Array(nSteps + 1),
      E: new Array(nSteps + 1),
      W: new Array(nSteps + 1),
      S: new Array(nSteps + 1),
      psi: new Array(nSteps + 1),
    };

    // Record initial state
    let currentState = { ...state };
    result.time[0] = 0;
    result.N[0] = currentState.N;
    result.E[0] = currentState.E;
    result.W[0] = currentState.W;
    result.S[0] = currentState.S;
    result.psi[0] = currentState.psi;

    // Integrate
    for (let i = 1; i <= nSteps; i++) {
      currentState = this.rk4Step(currentState, dt);

      // Ensure non-negative values
      currentState.N = Math.max(currentState.N, 1e-10);
      currentState.E = Math.max(currentState.E, 1e-10);
      currentState.W = Math.max(currentState.W, 1e-10);
      currentState.S = Math.max(currentState.S, 1e-10);
      currentState.psi = Math.max(currentState.psi, 0);

      result.time[i] = i * dt;
      result.N[i] = currentState.N;
      result.E[i] = currentState.E;
      result.W[i] = currentState.W;
      result.S[i] = currentState.S;
      result.psi[i] = currentState.psi;
    }

    return result;
  }
}
