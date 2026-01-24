/**
 * Number formatting utilities for human-readable display.
 */

/**
 * Format a number with human-readable suffixes (K, M, B, T).
 *
 * Examples:
 *   formatNumber(1234) => "1,234"
 *   formatNumber(12345) => "12.3K"
 *   formatNumber(1234567) => "1.23M"
 *   formatNumber(1234567890) => "1.23B"
 *   formatNumber(0.123456) => "0.123"
 *
 * @param value - The number to format
 * @param precision - Number of significant digits after decimal (default: 3)
 * @returns Formatted string
 */
export function formatNumber(value: number, precision: number = 3): string {
  // Handle edge cases
  if (!isFinite(value)) {
    return value.toString();
  }

  const absValue = Math.abs(value);

  // For small numbers (< 10000), use locale formatting or fixed decimal
  if (absValue < 10000) {
    // For very small numbers, show more precision
    if (absValue < 0.001 && absValue > 0) {
      return value.toExponential(2);
    }
    // For numbers < 1, show up to 3 decimal places
    if (absValue < 1) {
      return value.toFixed(precision);
    }
    // For numbers < 10, show 2 decimal places
    if (absValue < 10) {
      return value.toFixed(2);
    }
    // For numbers < 100, show 1 decimal place
    if (absValue < 100) {
      return value.toFixed(1);
    }
    // For numbers < 10000, use locale formatting (adds commas)
    return value.toLocaleString('en-US', {
      maximumFractionDigits: 0,
    });
  }

  // Suffixes for large numbers
  const suffixes = [
    { threshold: 1e12, suffix: 'T' },
    { threshold: 1e9, suffix: 'B' },
    { threshold: 1e6, suffix: 'M' },
    { threshold: 1e3, suffix: 'K' },
  ];

  for (const { threshold, suffix } of suffixes) {
    if (absValue >= threshold) {
      const scaled = value / threshold;
      // Determine decimal places based on magnitude
      const decimals = absValue >= threshold * 100 ? 0 : absValue >= threshold * 10 ? 1 : 2;
      return scaled.toFixed(decimals) + suffix;
    }
  }

  return value.toString();
}

/**
 * Format a number for metric display with appropriate precision.
 * This is specifically tuned for the SDT model metrics.
 *
 * @param value - The number to format
 * @param metricType - Type of metric for context-aware formatting
 * @returns Formatted string
 */
export function formatMetric(
  value: number,
  metricType: 'population' | 'elites' | 'wages' | 'state' | 'psi' = 'population'
): string {
  // PSI is always between 0 and 1, show 3 decimal places
  if (metricType === 'psi') {
    return value.toFixed(3);
  }

  // For normalized values (typically 0-2 range), show 3 decimal places
  if (Math.abs(value) < 10) {
    return value.toFixed(3);
  }

  // For larger values, use compact formatting
  return formatNumber(value);
}
