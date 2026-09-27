from importlib.metadata import version


def test_package_is_installed() -> None:
    """Verify that the AURA package is installed in the active environment."""
    assert version("aura-banking") == "0.1.0"
