/**
 * Tests for number formatting utilities.
 */

import { describe, it, expect } from 'vitest';
import { formatNumber, formatMetric } from '../src/utils/formatNumber';

describe('formatNumber', () => {
  describe('small numbers', () => {
    it('formats very small numbers in scientific notation', () => {
      expect(formatNumber(0.0001234)).toBe('1.23e-4');
      expect(formatNumber(0.0000001)).toBe('1.00e-7');
    });

    it('formats numbers less than 1 with 3 decimal places', () => {
      expect(formatNumber(0.123)).toBe('0.123');
      expect(formatNumber(0.5)).toBe('0.500');
      expect(formatNumber(0.99)).toBe('0.990');
    });

    it('formats numbers less than 10 with 2 decimal places', () => {
      expect(formatNumber(1.234)).toBe('1.23');
      expect(formatNumber(5.0)).toBe('5.00');
      expect(formatNumber(9.999)).toBe('10.00');
    });

    it('formats numbers less than 100 with 1 decimal place', () => {
      expect(formatNumber(12.34)).toBe('12.3');
      expect(formatNumber(99.9)).toBe('99.9');
    });

    it('formats numbers less than 10000 with commas', () => {
      expect(formatNumber(1234)).toBe('1,234');
      expect(formatNumber(9999)).toBe('9,999');
    });
  });

  describe('large numbers', () => {
    it('formats thousands with K suffix', () => {
      expect(formatNumber(12345)).toBe('12.3K');
      expect(formatNumber(100000)).toBe('100K');
      expect(formatNumber(999999)).toBe('1000K');
    });

    it('formats millions with M suffix', () => {
      expect(formatNumber(1234567)).toBe('1.23M');
      expect(formatNumber(12345678)).toBe('12.3M');
      expect(formatNumber(123456789)).toBe('123M');
    });

    it('formats billions with B suffix', () => {
      expect(formatNumber(1234567890)).toBe('1.23B');
      expect(formatNumber(12345678901)).toBe('12.3B');
    });

    it('formats trillions with T suffix', () => {
      expect(formatNumber(1234567890123)).toBe('1.23T');
    });
  });

  describe('negative numbers', () => {
    it('handles negative small numbers', () => {
      expect(formatNumber(-0.123)).toBe('-0.123');
      expect(formatNumber(-5.5)).toBe('-5.50');
    });

    it('handles negative large numbers', () => {
      expect(formatNumber(-1234567)).toBe('-1.23M');
    });
  });

  describe('edge cases', () => {
    it('handles zero', () => {
      expect(formatNumber(0)).toBe('0.000');
    });

    it('handles Infinity', () => {
      expect(formatNumber(Infinity)).toBe('Infinity');
      expect(formatNumber(-Infinity)).toBe('-Infinity');
    });

    it('handles NaN', () => {
      expect(formatNumber(NaN)).toBe('NaN');
    });
  });
});

describe('formatMetric', () => {
  describe('PSI values', () => {
    it('always formats PSI with 3 decimal places', () => {
      expect(formatMetric(0.123456, 'psi')).toBe('0.123');
      expect(formatMetric(0.5, 'psi')).toBe('0.500');
      expect(formatMetric(1.0, 'psi')).toBe('1.000');
    });
  });

  describe('population/state values', () => {
    it('formats small values with 3 decimal places', () => {
      expect(formatMetric(0.5, 'population')).toBe('0.500');
      expect(formatMetric(2.5, 'elites')).toBe('2.500');
      expect(formatMetric(1.23, 'wages')).toBe('1.230');
      expect(formatMetric(0.8, 'state')).toBe('0.800');
    });

    it('uses compact formatting for large values', () => {
      expect(formatMetric(1234567, 'population')).toBe('1.23M');
    });
  });
});
