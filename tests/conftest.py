"""
Pytest configuration and fixtures.
"""

import pytest


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests by default unless explicitly requested."""
    if config.getoption("-m"):
        # If markers are specified, respect them
        return

    skip_slow = pytest.mark.skip(reason="Slow test - use -m slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
