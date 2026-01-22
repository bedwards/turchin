"""Basic tests for package installation and imports."""

import pytest


def test_import_cliodynamics():
    """Test that the main package can be imported."""
    import cliodynamics
    assert hasattr(cliodynamics, "__version__")


def test_version():
    """Test that version is defined."""
    from cliodynamics import __version__
    assert __version__ == "0.1.0"


def test_import_submodules():
    """Test that submodules can be imported."""
    from cliodynamics import data
    from cliodynamics import models
    from cliodynamics import analysis

    # Just verify they imported without error
    assert data is not None
    assert models is not None
    assert analysis is not None
