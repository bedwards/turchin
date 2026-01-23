"""
Download Seshat Polaris-2025 dataset from GitHub.

The Polaris-2025 dataset is the latest release from the Seshat Global History
Databank, providing updated and expanded coverage compared to Equinox-2020.

Dataset Repository: https://github.com/Seshat-Global-History-Databank/build_polaris_dataset
License: CC-BY-NC-SA

Key Differences from Equinox-2020:
    - Polaris-2025 is built from the live Seshat API, providing more recent data
    - Includes polity_threads.csv for polity temporal continuity information
    - Column naming may differ from Equinox-2020 (lowercase vs mixed case)
    - Expanded variable coverage

Usage:
    python -m cliodynamics.data.download_polaris

This will download the dataset to the data/polaris2025/ directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# GitHub raw URLs for Polaris-2025 dataset files
POLARIS_EXCEL_URL = (
    "https://raw.githubusercontent.com/Seshat-Global-History-Databank/"
    "build_polaris_dataset/main/Polaris2025.xlsx"
)
POLARIS_THREADS_URL = (
    "https://raw.githubusercontent.com/Seshat-Global-History-Databank/"
    "build_polaris_dataset/main/polity_threads.csv"
)

# Expected filenames
EXPECTED_EXCEL_FILE = "Polaris2025.xlsx"
EXPECTED_THREADS_FILE = "polity_threads.csv"

# Default download directory (relative to project root)
DEFAULT_DATA_DIR = Path("data")


def download_file(url: str, dest_path: Path, timeout: int = 300) -> None:
    """
    Download a file from URL to destination path.

    Args:
        url: URL to download from
        dest_path: Local path to save the file
        timeout: Request timeout in seconds

    Raises:
        requests.HTTPError: If download fails
    """
    logger.info(f"Downloading: {url}")
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    # Get total size for progress reporting
    total_size = int(response.headers.get("content-length", 0))
    if total_size > 0:
        logger.info(f"File size: {total_size / 1024 / 1024:.2f} MB")

    # Write to file
    downloaded = 0
    with dest_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                logger.debug(f"Downloaded {pct:.1f}%")

    logger.info(f"Saved to: {dest_path}")


def download_polaris(
    data_dir: Path | str | None = None,
    force: bool = False,
    include_threads: bool = True,
) -> Path:
    """
    Download the Seshat Polaris-2025 dataset from GitHub.

    Downloads the main Polaris2025.xlsx file and optionally the polity_threads.csv
    file containing polity temporal continuity information.

    Args:
        data_dir: Directory to store the downloaded data. Defaults to ./data/
        force: If True, re-download even if files already exist.
        include_threads: If True, also download polity_threads.csv (default: True)

    Returns:
        Path to the data directory containing the downloaded files.

    Raises:
        requests.HTTPError: If download fails.

    Example:
        >>> from cliodynamics.data.download_polaris import download_polaris
        >>> data_path = download_polaris()
        >>> print(data_path)
        data/polaris2025
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    else:
        data_dir = Path(data_dir)

    # Create output directory
    polaris_dir = data_dir / "polaris2025"
    polaris_dir.mkdir(parents=True, exist_ok=True)

    # Download Excel file
    excel_path = polaris_dir / EXPECTED_EXCEL_FILE
    if excel_path.exists() and not force:
        logger.info(f"Excel file already exists: {excel_path}")
    else:
        download_file(POLARIS_EXCEL_URL, excel_path)

    # Download polity threads file
    if include_threads:
        threads_path = polaris_dir / EXPECTED_THREADS_FILE
        if threads_path.exists() and not force:
            logger.info(f"Threads file already exists: {threads_path}")
        else:
            download_file(POLARIS_THREADS_URL, threads_path)

    logger.info(f"Download complete. Data stored in: {polaris_dir}")
    return polaris_dir


def main() -> None:
    """Main entry point for command-line usage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Find project root (go up from this file)
    # This module is at: src/cliodynamics/data/download_polaris.py
    # Project root has: pyproject.toml, data/
    module_path = Path(__file__).resolve()
    src_path = module_path.parent.parent.parent  # up from data/ to src/
    project_root = src_path.parent  # up from src/ to project root

    data_dir = project_root / "data"
    logger.info(f"Data directory: {data_dir}")

    try:
        output_path = download_polaris(data_dir)
        print(f"\nSeshat Polaris-2025 dataset downloaded to: {output_path}")
        print("\nAvailable files:")
        for f in sorted(output_path.iterdir()):
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name} ({size_mb:.2f} MB)")
    except requests.HTTPError as e:
        logger.error(f"Download failed: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        logger.error(f"Error: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
