"""Case studies applying SDT models to historical data.

This module contains implementations of SDT analyses for specific
historical societies, following the methodology from Turchin's
*Secular Cycles* (2009) and *Ages of Discord* (2016).

Each case study module provides:
- Data loading from Seshat
- Model calibration
- Simulation and comparison with historical record
- Report generation
"""

from cliodynamics.case_studies.roman_empire import (
    RomanEmpireStudy,
    calibrate,
    generate_report,
    load_data,
    simulate,
)

__all__ = [
    "RomanEmpireStudy",
    "load_data",
    "calibrate",
    "simulate",
    "generate_report",
]
