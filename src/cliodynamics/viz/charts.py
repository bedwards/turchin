"""
Standardized chart creation for cliodynamics essays.

This module provides wrapper functions that ensure consistent, readable
chart formatting across all essays. All charts should use these functions
to maintain quality standards.

IMPORTANT: After generating any chart, visually verify it looks correct
before committing. Charts must be readable and properly formatted.

Known Failure Modes:
--------------------
1. Runaway Chart Height: Charts can export with extreme dimensions (e.g., 53,000px)
   due to unknown Altair/Vega interactions. This module validates dimensions
   and warns/fails when charts exceed reasonable bounds.

2. Missing/Corrupt Export: File created but empty or corrupt. The verify_chart()
   function checks file size and dimensions.

3. Text Cutoff: Labels truncated at chart edges. Use labelLimit parameter.

4. Unreadable Fonts: Default Altair fonts are too small. This module enforces
   minimum font sizes for readability.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from shutil import move
from typing import Any

import altair as alt
import pandas as pd

try:
    import imagesize
    HAS_IMAGESIZE = True
except ImportError:
    HAS_IMAGESIZE = False

logger = logging.getLogger(__name__)

# Standard chart dimensions
CHART_WIDTH = 800
CHART_HEIGHT_SMALL = 400
CHART_HEIGHT_MEDIUM = 600
CHART_HEIGHT_LARGE = 900
CHART_HEIGHT_XLARGE = 1200  # For charts with many categories

# Maximum allowed dimensions (safeguard against runaway charts)
MAX_CHART_WIDTH = 3000
MAX_CHART_HEIGHT = 3000

# Standard font sizes (minimum for readability)
FONT_SIZE_TITLE = 20
FONT_SIZE_AXIS_LABEL = 14
FONT_SIZE_AXIS_TITLE = 16
FONT_SIZE_LEGEND_LABEL = 12
FONT_SIZE_LEGEND_TITLE = 14

# Standard scale factor for PNG export (2x for crisp rendering)
PNG_SCALE_FACTOR = 2


class ChartDimensionError(Exception):
    """Raised when a chart exceeds maximum allowed dimensions."""
    pass


class ChartValidationResult:
    """Result of chart dimension validation."""

    def __init__(
        self,
        width: int,
        height: int,
        file_size: int,
        is_valid: bool,
        message: str,
    ):
        self.width = width
        self.height = height
        self.file_size = file_size
        self.is_valid = is_valid
        self.message = message

    def __repr__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"ChartValidationResult({status}: {self.width}x{self.height}px, "
            f"{self.file_size:,} bytes - {self.message})"
        )


def validate_chart_dimensions(
    path: str | Path,
    max_width: int = MAX_CHART_WIDTH,
    max_height: int = MAX_CHART_HEIGHT,
) -> ChartValidationResult:
    """
    Validate that a saved chart has reasonable dimensions.

    Uses the lightweight imagesize library to check PNG dimensions without
    loading the full image into memory.

    Args:
        path: Path to the saved chart image
        max_width: Maximum allowed width in pixels
        max_height: Maximum allowed height in pixels

    Returns:
        ChartValidationResult with dimensions, validity, and message

    Raises:
        FileNotFoundError: If chart file does not exist

    Example:
        >>> result = validate_chart_dimensions("chart.png")
        >>> if not result.is_valid:
        ...     print(f"Warning: {result.message}")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Chart file not found: {path}")

    file_size = path.stat().st_size

    # Check file size first
    if file_size < 1000:
        return ChartValidationResult(
            width=0,
            height=0,
            file_size=file_size,
            is_valid=False,
            message=f"File too small ({file_size} bytes) - likely corrupt or empty",
        )

    if file_size > 50_000_000:  # 50MB
        return ChartValidationResult(
            width=0,
            height=0,
            file_size=file_size,
            is_valid=False,
            message=f"File too large ({file_size:,} bytes) - likely corrupted",
        )

    # Get dimensions
    if not HAS_IMAGESIZE:
        logger.warning(
            "imagesize library not installed - skipping dimension validation. "
            "Install with: pip install imagesize"
        )
        return ChartValidationResult(
            width=0,
            height=0,
            file_size=file_size,
            is_valid=True,
            message="Dimension validation skipped (imagesize not installed)",
        )

    try:
        width, height = imagesize.get(str(path))
    except Exception as e:
        return ChartValidationResult(
            width=0,
            height=0,
            file_size=file_size,
            is_valid=False,
            message=f"Failed to read image dimensions: {e}",
        )

    # Check if dimensions are valid (imagesize returns -1, -1 for unknown formats)
    if width < 0 or height < 0:
        return ChartValidationResult(
            width=width,
            height=height,
            file_size=file_size,
            is_valid=False,
            message="Could not determine image dimensions",
        )

    # Check against maximums
    issues = []
    if width > max_width:
        issues.append(f"width {width}px exceeds max {max_width}px")
    if height > max_height:
        issues.append(f"height {height}px exceeds max {max_height}px")

    if issues:
        return ChartValidationResult(
            width=width,
            height=height,
            file_size=file_size,
            is_valid=False,
            message=f"Chart too large: {', '.join(issues)}",
        )

    return ChartValidationResult(
        width=width,
        height=height,
        file_size=file_size,
        is_valid=True,
        message=f"Dimensions OK: {width}x{height}px",
    )


def configure_chart(
    chart: alt.Chart,
    title: str,
    width: int = CHART_WIDTH,
    height: int = CHART_HEIGHT_MEDIUM,
    max_height: int = MAX_CHART_HEIGHT,
) -> alt.Chart:
    """
    Apply standard formatting to an Altair chart.

    This ensures all charts have readable fonts, proper dimensions,
    and consistent styling. Height is clamped to max_height to prevent
    runaway chart dimensions.

    Args:
        chart: The Altair chart to configure
        title: Chart title
        width: Chart width in pixels
        height: Chart height in pixels
        max_height: Maximum allowed height (default 3000px)

    Returns:
        Configured chart ready for export

    Example:
        >>> chart = alt.Chart(df).mark_bar().encode(x='year', y='value')
        >>> chart = configure_chart(chart, "My Chart Title", height=900)
        >>> save_chart(chart, "output.png")
    """
    # Clamp height to maximum
    if height > max_height:
        logger.warning(
            f"Requested height {height}px exceeds max {max_height}px, clamping"
        )
        height = max_height

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
    max_width: int = MAX_CHART_WIDTH,
    max_height: int = MAX_CHART_HEIGHT,
    validate: bool = True,
    debug: bool = False,
) -> Path:
    """
    Save chart to file with proper resolution and dimension validation.

    For PNG files, the chart is first saved to a temporary file, validated,
    and only moved to the final path if dimensions are within bounds.

    Args:
        chart: Configured Altair chart
        path: Output file path (.png, .svg, or .html)
        scale_factor: Resolution multiplier for PNG (default 2x)
        max_width: Maximum allowed width in pixels (for PNG validation)
        max_height: Maximum allowed height in pixels (for PNG validation)
        validate: Whether to validate dimensions (default True, PNG only)
        debug: If True, print chart spec and save HTML alongside PNG

    Returns:
        Path to saved file

    Raises:
        ValueError: If file format not supported
        ChartDimensionError: If PNG dimensions exceed maximums and validate=True
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if debug:
        _print_chart_debug_info(chart)

    if suffix == '.png':
        if validate:
            # Save to temp file first for validation
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = Path(tmp.name)

            try:
                chart.save(str(tmp_path), scale_factor=scale_factor)
                result = validate_chart_dimensions(tmp_path, max_width, max_height)

                if not result.is_valid:
                    tmp_path.unlink()  # Clean up temp file
                    raise ChartDimensionError(
                        f"Chart dimension validation failed: {result.message}. "
                        f"File: {path}"
                    )

                # Validation passed, move to final location
                path.parent.mkdir(parents=True, exist_ok=True)
                move(str(tmp_path), str(path))
                logger.info(
                    f"Chart saved to {path} ({result.width}x{result.height}px, "
                    f"{result.file_size:,} bytes)"
                )
            except ChartDimensionError:
                raise
            except Exception as e:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise RuntimeError(f"Failed to save chart: {e}") from e
        else:
            # Save directly without validation
            chart.save(str(path), scale_factor=scale_factor)
            logger.info(f"Chart saved to {path} (validation skipped)")

        # Optionally save HTML for debugging
        if debug:
            html_path = path.with_suffix('.html')
            chart.save(str(html_path))
            logger.info(f"Debug HTML saved to {html_path}")

    elif suffix == '.svg':
        chart.save(str(path))
        logger.info(f"Chart saved to {path}")
    elif suffix == '.html':
        chart.save(str(path))
        logger.info(f"Chart saved to {path}")
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use .png, .svg, or .html")

    return path


def _print_chart_debug_info(chart: alt.Chart) -> None:
    """Print debug information about a chart."""
    try:
        spec = chart.to_dict()
        print("=" * 60)
        print("CHART DEBUG INFO")
        print("=" * 60)

        # Print dimensions if available
        if 'width' in spec:
            print(f"Width: {spec['width']}")
        if 'height' in spec:
            print(f"Height: {spec['height']}")

        # Print data summary if available
        if 'data' in spec and 'values' in spec['data']:
            data = spec['data']['values']
            if isinstance(data, list):
                print(f"Data rows: {len(data)}")
                if data:
                    print(f"Data columns: {list(data[0].keys())}")

        # Print encoding summary
        if 'encoding' in spec:
            print(f"Encodings: {list(spec['encoding'].keys())}")

        print("=" * 60)
    except Exception as e:
        print(f"Could not extract debug info: {e}")


def year_axis(
    field: str = "year",
    title: str = "Year",
    **kwargs,
) -> alt.X:
    """
    Create a properly formatted year axis (no commas in year labels).

    Years should display as "2025" not "2,025". This helper ensures
    correct formatting by using format='d' (integer format).

    Args:
        field: Column name for year data (default "year")
        title: Axis title
        **kwargs: Additional arguments passed to alt.X

    Returns:
        Configured alt.X encoding for year axis

    Example:
        >>> chart = alt.Chart(df).mark_line().encode(
        ...     x=year_axis("year", "Year (CE)"),
        ...     y="value:Q"
        ... )
    """
    return alt.X(
        f"{field}:Q",
        title=title,
        axis=alt.Axis(format="d"),  # Integer format, no commas
        **kwargs,
    )


def create_timeline_chart(
    df: pd.DataFrame,
    y_column: str,
    start_column: str,
    end_column: str,
    color_column: str | None = None,
    title: str = "Timeline",
    max_height: int = MAX_CHART_HEIGHT,
) -> alt.Chart:
    """
    Create a horizontal bar timeline chart.

    Automatically sizes height based on number of categories, with a
    maximum height safeguard.

    Args:
        df: DataFrame with timeline data
        y_column: Column for Y-axis categories
        start_column: Column for bar start (e.g., start year)
        end_column: Column for bar end (e.g., end year)
        color_column: Optional column for color encoding
        title: Chart title
        max_height: Maximum chart height (default 3000px)

    Returns:
        Configured chart ready for export
    """
    n_categories = df[y_column].nunique()

    # Scale height based on number of categories
    # Minimum 20px per category, plus padding
    height = max(CHART_HEIGHT_MEDIUM, n_categories * 25 + 100)

    # Apply max height safeguard
    if height > max_height:
        logger.warning(
            f"Timeline chart would be {height}px tall for {n_categories} categories. "
            f"Clamping to {max_height}px. Consider filtering data."
        )
        height = max_height

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

    return configure_chart(chart, title, height=height, max_height=max_height)


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
    max_height: int = MAX_CHART_HEIGHT,
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
        max_height: Maximum chart height for horizontal bars

    Returns:
        Configured chart ready for export
    """
    if horizontal:
        n_categories = df[y_column].nunique()
        height = max(CHART_HEIGHT_SMALL, n_categories * 25 + 100)
        if height > max_height:
            logger.warning(
                f"Bar chart would be {height}px tall for {n_categories} categories. "
                f"Clamping to {max_height}px."
            )
            height = max_height
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

    return configure_chart(chart, title, height=height, max_height=max_height)


def create_heatmap(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: str,
    title: str = "Heatmap",
    max_height: int = MAX_CHART_HEIGHT,
) -> alt.Chart:
    """
    Create a heatmap visualization.

    Args:
        df: DataFrame with data
        x_column: Column for X-axis categories
        y_column: Column for Y-axis categories
        color_column: Column for color intensity
        title: Chart title
        max_height: Maximum chart height

    Returns:
        Configured chart ready for export
    """
    n_y_categories = df[y_column].nunique()
    height = max(CHART_HEIGHT_MEDIUM, n_y_categories * 20 + 100)

    if height > max_height:
        logger.warning(
            f"Heatmap would be {height}px tall. Clamping to {max_height}px."
        )
        height = max_height

    color_title = color_column.replace('_', ' ').title()
    chart = alt.Chart(df).mark_rect().encode(
        x=alt.X(f'{x_column}:N', title=None),
        y=alt.Y(f'{y_column}:N', title=None),
        color=alt.Color(f'{color_column}:Q', title=color_title),
    )

    return configure_chart(chart, title, height=height, max_height=max_height)


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
    """
    Check if a chart file exists and has reasonable size.

    Deprecated: Use verify_chart() for comprehensive validation.
    """
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


def verify_chart(
    path: str | Path,
    max_width: int = MAX_CHART_WIDTH,
    max_height: int = MAX_CHART_HEIGHT,
    print_info: bool = True,
) -> ChartValidationResult:
    """
    Comprehensive chart verification with dimension reporting.

    This is the recommended way to verify charts after generation.
    It checks file existence, size, and dimensions.

    Args:
        path: Path to the chart file
        max_width: Maximum allowed width in pixels
        max_height: Maximum allowed height in pixels
        print_info: Whether to print verification info to console

    Returns:
        ChartValidationResult with full details

    Example:
        >>> save_chart(chart, "output.png")
        >>> result = verify_chart("output.png")
        >>> if not result.is_valid:
        ...     print(f"Chart verification failed: {result.message}")
    """
    path = Path(path)

    if not path.exists():
        result = ChartValidationResult(
            width=0,
            height=0,
            file_size=0,
            is_valid=False,
            message=f"File not found: {path}",
        )
        if print_info:
            print(f"[FAIL] {result.message}")
        return result

    result = validate_chart_dimensions(path, max_width, max_height)

    if print_info:
        status = "[OK]" if result.is_valid else "[FAIL]"
        print(f"{status} {path.name}: {result.message}")
        if result.is_valid:
            print(VERIFICATION_REMINDER)

    return result
