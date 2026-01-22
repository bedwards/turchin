"""
Download Seshat Equinox-2020 dataset from Zenodo.

The Seshat Global History Databank (Equinox-2020 release) contains ~100 variables
for 373 societies spanning 9600 BCE to 1900 CE.

Dataset DOI: https://doi.org/10.5281/zenodo.6642229
License: CC-BY-NC-SA

Usage:
    python -m cliodynamics.data.download

This will download and extract the dataset to the data/ directory.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Zenodo record ID for Seshat Equinox-2020
ZENODO_RECORD_ID = "6642230"

# API endpoint for Zenodo
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Expected filename in the archive
EXPECTED_EXCEL_FILE = "Equinox_on_GitHub_June9_2022.xlsx"

# Default download directory (relative to project root)
DEFAULT_DATA_DIR = Path("data")


def get_zenodo_download_url() -> str:
    """
    Fetch the download URL for the Seshat dataset from Zenodo API.

    Returns:
        The direct download URL for the zip archive.

    Raises:
        requests.HTTPError: If the API request fails.
        ValueError: If no downloadable files are found in the record.
    """
    logger.info("Fetching Zenodo record metadata...")
    response = requests.get(ZENODO_API_URL, timeout=30)
    response.raise_for_status()

    record = response.json()
    files = record.get("files", [])

    if not files:
        raise ValueError(f"No files found in Zenodo record {ZENODO_RECORD_ID}")

    # Look for the zip file
    for file_info in files:
        filename = file_info.get("key", "")
        if filename.endswith(".zip"):
            download_url = file_info.get("links", {}).get("self")
            if download_url:
                logger.info(f"Found download URL for: {filename}")
                return download_url

    # Fallback to first file if no zip found
    first_file = files[0]
    download_url = first_file.get("links", {}).get("self")
    if download_url:
        return download_url

    raise ValueError("Could not find download URL in Zenodo record")


def download_and_extract(
    data_dir: Path | str | None = None,
    force: bool = False,
) -> Path:
    """
    Download and extract the Seshat Equinox-2020 dataset.

    Args:
        data_dir: Directory to store the downloaded data. Defaults to ./data/
        force: If True, re-download even if files already exist.

    Returns:
        Path to the extracted data directory.

    Raises:
        requests.HTTPError: If download fails.
        zipfile.BadZipFile: If the downloaded archive is corrupted.

    Example:
        >>> from cliodynamics.data.download import download_and_extract
        >>> data_path = download_and_extract()
        >>> print(data_path)
        data/seshat
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    else:
        data_dir = Path(data_dir)

    # Create output directory
    seshat_dir = data_dir / "seshat"

    # Check if already downloaded
    expected_file = seshat_dir / EXPECTED_EXCEL_FILE
    if expected_file.exists() and not force:
        logger.info(f"Dataset already exists at {expected_file}")
        return seshat_dir

    # Get download URL
    download_url = get_zenodo_download_url()

    # Download the file
    logger.info("Downloading Seshat dataset from Zenodo...")
    logger.info(f"URL: {download_url}")

    response = requests.get(download_url, stream=True, timeout=300)
    response.raise_for_status()

    # Get total size for progress reporting
    total_size = int(response.headers.get("content-length", 0))
    logger.info(f"Total size: {total_size / 1024 / 1024:.1f} MB")

    # Read content into memory
    content = io.BytesIO()
    downloaded = 0
    for chunk in response.iter_content(chunk_size=8192):
        content.write(chunk)
        downloaded += len(chunk)
        if total_size > 0:
            pct = downloaded / total_size * 100
            logger.debug(f"Downloaded {pct:.1f}%")

    content.seek(0)

    # Create directories
    seshat_dir.mkdir(parents=True, exist_ok=True)

    # Extract the archive
    logger.info(f"Extracting to {seshat_dir}...")
    with zipfile.ZipFile(content, "r") as zf:
        # List contents
        names = zf.namelist()
        logger.info(f"Archive contains {len(names)} files")

        # Extract all files
        for name in names:
            # Skip directories and hidden files
            if name.endswith("/") or name.startswith("."):
                continue

            # Get just the filename (strip any directory prefix)
            filename = Path(name).name

            # Skip non-data files
            if not (
                filename.endswith(".xlsx")
                or filename.endswith(".csv")
                or filename.endswith(".txt")
            ):
                continue

            # Extract to seshat_dir with flat structure
            logger.info(f"Extracting: {filename}")
            data = zf.read(name)
            (seshat_dir / filename).write_bytes(data)

    logger.info(f"Download complete. Data stored in: {seshat_dir}")
    return seshat_dir


def main() -> None:
    """Main entry point for command-line usage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Find project root (go up from this file)
    # This module is at: src/cliodynamics/data/download.py
    # Project root has: pyproject.toml, data/
    module_path = Path(__file__).resolve()
    src_path = module_path.parent.parent.parent  # up from data/ to src/
    project_root = src_path.parent  # up from src/ to project root

    data_dir = project_root / "data"
    logger.info(f"Data directory: {data_dir}")

    try:
        output_path = download_and_extract(data_dir)
        print(f"\nSeshat Equinox-2020 dataset downloaded to: {output_path}")
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
