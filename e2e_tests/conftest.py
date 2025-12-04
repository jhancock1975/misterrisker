"""
Pytest configuration for e2e tests.

These tests are separate from the unit tests in tests/ directory.
They require the server to be running and use Playwright for browser automation.
"""

import pytest


def pytest_configure(config):
    """Configure pytest for e2e tests."""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test requiring browser"
    )


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser context for all tests."""
    return {
        **browser_context_args,
        "viewport": {"width": 1280, "height": 800},
    }
