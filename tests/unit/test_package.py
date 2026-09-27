from importlib.metadata import version

import aura


def test_package_is_importable() -> None:
    """Verify that the source package is available to Python."""
    assert aura.__file__ is not None


def test_package_is_installed() -> None:
    """Verify that the AURA package is installed in the active environment."""
    assert version("aura-banking") == "0.1.0"
