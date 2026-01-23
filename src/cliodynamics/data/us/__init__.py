"""U.S. historical data module for Ages of Discord replication.

This module provides access to U.S. historical data needed to replicate
Turchin's Ages of Discord analysis, including:

- Real wages and economic indicators
- Elite overproduction proxies (lawyers, PhDs, wealth inequality)
- Political instability indicators
- Demographic data

Data sources:
- Historical Statistics of the United States (HSUS)
- Bureau of Labor Statistics (BLS)
- FRED (Federal Reserve Economic Data)
- American Bar Association (ABA)
- National Science Foundation (NSF)

Example:
    >>> from cliodynamics.data.us import USHistoricalData
    >>> data = USHistoricalData()
    >>> wages = data.get_real_wages()
    >>> elites = data.get_elite_indicators()
"""

from cliodynamics.data.us.loader import (
    USHistoricalData,
    get_elite_indicators,
    get_instability_indicators,
    get_real_wages,
    get_relative_wages,
    get_wealth_inequality,
)

__all__ = [
    "USHistoricalData",
    "get_real_wages",
    "get_relative_wages",
    "get_elite_indicators",
    "get_wealth_inequality",
    "get_instability_indicators",
]
