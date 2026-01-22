"""
Standardized chart creation for cliodynamics essays.

This module provides wrapper functions that ensure consistent, readable
chart formatting across all essays. All charts should use these functions
to maintain quality standards.

IMPORTANT: After generating any chart, visually verify it looks correct
before committing. Charts must be readable and properly formatted.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd

logger = logging.getLogger(__name__)

# Standard chart dimensions
CHART_WIDTH = 800
CHART_HEIGHT_SMALL = 400
CHART_HEIGHT_MEDIUM = 600
CHART_HEIGHT_LARGE = 900
CHART_HEIGHT_XLARGE = 1200  # For charts with many categories

# Standard font sizes (minimum for readability)
FONT_SIZE_TITLE = 20
FONT_SIZE_AXIS_LABEL = 14
FONT_SIZE_AXIS_TITLE = 16
FONT_SIZE_LEGEND_LABEL = 12
FONT_SIZE_LEGEND_TITLE = 14

# Standard scale factor for PNG export (2x for crisp rendering)
PNG_SCALE_FACTOR = 2


def configure_chart(
    chart: alt.Chart,
    title: str,
    width: int = CHART_WIDTH,
    height: int = CHART_HEIGHT_MEDIUM,
) -> alt.Chart:
    """
    Apply standard formatting to an Altair chart.

    This ensures all charts have readable fonts, proper dimensions,
    and consistent styling.

    Args:
        chart: The Altair chart to configure
        title: Chart title
        width: Chart width in pixels
        height: Chart height in pixels

    Returns:
        Configured chart ready for export

    Example:
        >>> chart = alt.Chart(df).mark_bar().encode(x='year', y='value')
        >>> chart = configure_chart(chart, "My Chart Title", height=900)
        >>> save_chart(chart, "output.png")
    """
    return (
        chart
        .properties(
            width=width,
            height=height,
            title=alt.TitleParams(
                text=title,
                fontSize=FONT_SIZE_TITLE,
                anchor='start'
            )
        )
        .configure_axis(
            labelFontSize=FONT_SIZE_AXIS_LABEL,
            titleFontSize=FONT_SIZE_AXIS_TITLE,
            labelLimit=300,  # Allow longer labels
        )
        .configure_legend(
            labelFontSize=FONT_SIZE_LEGEND_LABEL,
            titleFontSize=FONT_SIZE_LEGEND_TITLE,
            labelLimit=200,
        )
        .configure_title(
            fontSize=FONT_SIZE_TITLE,
        )
    )


def save_chart(
    chart: alt.Chart,
    path: str | Path,
    scale_factor: int = PNG_SCALE_FACTOR,
) -> Path:
    """
    Save chart to file with proper resolution.

    Args:
        chart: Configured Altair chart
        path: Output file path (.png, .svg, or .html)
        scale_factor: Resolution multiplier for PNG (default 2x)

    Returns:
        Path to saved file

    Raises:
        ValueError: If file format not supported
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.png':
        chart.save(str(path), scale_factor=scale_factor)
    elif suffix == '.svg':
        chart.save(str(path))
    elif suffix == '.html':
        chart.save(str(path))
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use .png, .svg, or .html")

    logger.info(f"Chart saved to {path}")
    return path


def create_timeline_chart(
    df: pd.DataFrame,
    y_column: str,
    start_column: str,
    end_column: str,
    color_column: str | None = None,
    title: str = "Timeline",
) -> alt.Chart:
    """
    Create a horizontal bar timeline chart.

    Automatically sizes height based on number of categories.

    Args:
        df: DataFrame with timeline data
        y_column: Column for Y-axis categories
        start_column: Column for bar start (e.g., start year)
        end_column: Column for bar end (e.g., end year)
        color_column: Optional column for color encoding
        title: Chart title

    Returns:
        Configured chart ready for export
    """
    n_categories = df[y_column].nunique()

    # Scale height based on number of categories
    # Minimum 20px per category, plus padding
    height = max(CHART_HEIGHT_MEDIUM, n_categories * 25 + 100)

    encoding: dict[str, Any] = {
        'y': alt.Y(f'{y_column}:N', title=None, sort=None),
        'x': alt.X(f'{start_column}:Q', title='Year'),
        'x2': alt.X2(f'{end_column}:Q'),
    }

    if color_column:
        encoding['color'] = alt.Color(
            f'{color_column}:N',
            legend=alt.Legend(title=color_column.replace('_', ' ').title())
        )

    chart = alt.Chart(df).mark_bar().encode(**encoding)

    return configure_chart(chart, title, height=height)


def create_line_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: str | None = None,
    title: str = "Time Series",
) -> alt.Chart:
    """
    Create a line chart for time series data.

    Args:
        df: DataFrame with time series data
        x_column: Column for X-axis (typically time)
        y_column: Column for Y-axis values
        color_column: Optional column for multiple series
        title: Chart title

    Returns:
        Configured chart ready for export
    """
    encoding: dict[str, Any] = {
        'x': alt.X(f'{x_column}:Q', title=x_column.replace('_', ' ').title()),
        'y': alt.Y(f'{y_column}:Q', title=y_column.replace('_', ' ').title()),
    }

    if color_column:
        encoding['color'] = alt.Color(f'{color_column}:N')

    chart = alt.Chart(df).mark_line(strokeWidth=2).encode(**encoding)

    return configure_chart(chart, title)


def create_bar_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: str | None = None,
    title: str = "Bar Chart",
    horizontal: bool = False,
) -> alt.Chart:
    """
    Create a bar chart.

    Args:
        df: DataFrame with data
        x_column: Column for X-axis
        y_column: Column for Y-axis
        color_column: Optional column for color
        title: Chart title
        horizontal: If True, create horizontal bars

    Returns:
        Configured chart ready for export
    """
    if horizontal:
        n_categories = df[y_column].nunique()
        height = max(CHART_HEIGHT_SMALL, n_categories * 25 + 100)
        encoding = {
            'y': alt.Y(f'{y_column}:N', title=None, sort='-x'),
            'x': alt.X(f'{x_column}:Q', title=x_column.replace('_', ' ').title()),
        }
    else:
        height = CHART_HEIGHT_MEDIUM
        encoding = {
            'x': alt.X(f'{x_column}:N', title=None),
            'y': alt.Y(f'{y_column}:Q', title=y_column.replace('_', ' ').title()),
        }

    if color_column:
        encoding['color'] = alt.Color(f'{color_column}:N')

    chart = alt.Chart(df).mark_bar().encode(**encoding)

    return configure_chart(chart, title, height=height)


def create_heatmap(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: str,
    title: str = "Heatmap",
) -> alt.Chart:
    """
    Create a heatmap visualization.

    Args:
        df: DataFrame with data
        x_column: Column for X-axis categories
        y_column: Column for Y-axis categories
        color_column: Column for color intensity
        title: Chart title

    Returns:
        Configured chart ready for export
    """
    n_y_categories = df[y_column].nunique()
    height = max(CHART_HEIGHT_MEDIUM, n_y_categories * 20 + 100)

    chart = alt.Chart(df).mark_rect().encode(
        x=alt.X(f'{x_column}:N', title=None),
        y=alt.Y(f'{y_column}:N', title=None),
        color=alt.Color(f'{color_column}:Q', title=color_column.replace('_', ' ').title()),
    )

    return configure_chart(chart, title, height=height)


# Verification reminder
VERIFICATION_REMINDER = """
╔══════════════════════════════════════════════════════════════════╗
║  IMPORTANT: Visually verify all charts before committing!        ║
║                                                                  ║
║  Check that:                                                     ║
║  1. All text is readable (not too small)                        ║
║  2. Chart is not squished or stretched                          ║
║  3. Data is displayed correctly                                 ║
║  4. Legend is visible and readable                              ║
║  5. Axis labels are not cut off                                 ║
╚══════════════════════════════════════════════════════════════════╝
"""


def verify_chart_exists(path: str | Path) -> bool:
    """Check if a chart file exists and has reasonable size."""
    path = Path(path)
    if not path.exists():
        logger.error(f"Chart not found: {path}")
        return False

    size = path.stat().st_size
    if size < 1000:  # Less than 1KB is suspicious
        logger.warning(f"Chart file suspiciously small ({size} bytes): {path}")
        return False

    logger.info(f"Chart exists: {path} ({size:,} bytes)")
    print(VERIFICATION_REMINDER)
    return True
