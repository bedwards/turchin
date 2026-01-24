/**
 * Tests for SDT ODE solver.
 */

import { describe, it, expect } from 'vitest';
import { SDTModel } from '../src/models/sdt';
import { DEFAULT_PARAMS, DEFAULT_STATE } from '../src/models/parameters';

describe('SDTModel', () => {
  describe('constructor', () => {
    it('uses default parameters when none provided', () => {
      const model = new SDTModel();
      expect(model.params).toEqual(DEFAULT_PARAMS);
    });

    it('merges custom parameters with defaults', () => {
      const model = new SDTModel({ r_max: 0.03 });
      expect(model.params.r_max).toBe(0.03);
      expect(model.params.alpha).toBe(DEFAULT_PARAMS.alpha);
    });
  });

  describe('populationDynamics', () => {
    it('returns positive growth when N < K and W > 0', () => {
      const model = new SDTModel();
      const dN = model.populationDynamics(0.5, 1.0);
      expect(dN).toBeGreaterThan(0);
    });

    it('returns zero growth at carrying capacity', () => {
      const model = new SDTModel();
      const dN = model.populationDynamics(1.0, 1.0);
      expect(dN).toBeCloseTo(0, 5);
    });

    it('growth depends on wages', () => {
      const model = new SDTModel();
      const dN_high = model.populationDynamics(0.5, 1.5);
      const dN_low = model.populationDynamics(0.5, 0.5);
      expect(dN_high).toBeGreaterThan(dN_low);
    });
  });

  describe('eliteDynamics', () => {
    it('elite growth increases with lower wages (more surplus)', () => {
      const model = new SDTModel();
      const dE_low_wage = model.eliteDynamics(0.1, 0.5, 0.5);
      const dE_high_wage = model.eliteDynamics(0.1, 0.5, 1.5);
      expect(dE_low_wage).toBeGreaterThan(dE_high_wage);
    });
  });

  describe('wageDynamics', () => {
    it('wages rise when labor is scarce', () => {
      const model = new SDTModel();
      const dW = model.wageDynamics(1.0, 0.3, 0.1);
      expect(dW).toBeGreaterThan(0);
    });

    it('wages fall with elite overproduction', () => {
      const model = new SDTModel();
      const dW = model.wageDynamics(1.0, 0.5, 0.3);
      expect(dW).toBeLessThan(0);
    });
  });

  describe('instabilityDynamics', () => {
    it('instability increases with low wages', () => {
      const model = new SDTModel();
      const dpsi = model.instabilityDynamics(0, 0.1, 0.5, 1.0);
      expect(dpsi).toBeGreaterThan(0);
    });

    it('instability decays without driving forces', () => {
      const model = new SDTModel();
      // High wages, normal elites, strong state
      const dpsi = model.instabilityDynamics(0.5, 0.05, 1.5, 1.5);
      expect(dpsi).toBeLessThan(0);
    });
  });

  describe('simulate', () => {
    it('returns correct array lengths', () => {
      const model = new SDTModel();
      const result = model.simulate(DEFAULT_STATE, 100, 1.0);
      expect(result.time.length).toBe(101);
      expect(result.N.length).toBe(101);
      expect(result.E.length).toBe(101);
      expect(result.W.length).toBe(101);
      expect(result.S.length).toBe(101);
      expect(result.psi.length).toBe(101);
    });

    it('maintains non-negative values', () => {
      const model = new SDTModel();
      const result = model.simulate(DEFAULT_STATE, 200);
      for (let i = 0; i < result.time.length; i++) {
        expect(result.N[i]).toBeGreaterThanOrEqual(0);
        expect(result.E[i]).toBeGreaterThanOrEqual(0);
        expect(result.W[i]).toBeGreaterThanOrEqual(0);
        expect(result.S[i]).toBeGreaterThanOrEqual(0);
        expect(result.psi[i]).toBeGreaterThanOrEqual(0);
      }
    });

    it('completes in reasonable time', () => {
      const model = new SDTModel();
      const start = performance.now();
      model.simulate(DEFAULT_STATE, 500);
      const elapsed = performance.now() - start;
      // Should complete in under 100ms
      expect(elapsed).toBeLessThan(100);
    });
  });

  describe('RK4 integration accuracy', () => {
    it('converges with smaller time steps', () => {
      const model = new SDTModel();
      const result_coarse = model.simulate(DEFAULT_STATE, 100, 1.0);
      const result_fine = model.simulate(DEFAULT_STATE, 100, 0.1);

      // Final values should be similar
      const N_diff = Math.abs(
        result_coarse.N[result_coarse.N.length - 1] -
        result_fine.N[result_fine.N.length - 1]
      );
      expect(N_diff).toBeLessThan(0.05);
    });
  });
});
