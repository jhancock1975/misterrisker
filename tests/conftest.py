"""
Pytest configuration and shared fixtures.

This module provides logging utilities and common fixtures for all tests.
"""

import logging
import pytest
from typing import Any, Callable
from functools import wraps
import json


# =============================================================================
# Logging Configuration
# =============================================================================

# Create a test logger
test_logger = logging.getLogger("test")
test_logger.setLevel(logging.DEBUG)


def safe_repr(obj, max_len=500):
    """Safely represent an object for logging, truncating if needed."""
    try:
        if hasattr(obj, '__dict__'):
            s = f"{type(obj).__name__}({obj.__dict__})"
        else:
            s = repr(obj)
        if len(s) > max_len:
            s = s[:max_len] + "..."
        return s
    except Exception:
        return f"<{type(obj).__name__}>"


@pytest.fixture(autouse=True)
def setup_test_logging(request, caplog):
    """Automatically log test start, end, and results for every test.
    
    This fixture runs automatically for every test and logs:
    - Test name and module
    - Test docstring (what's being tested)
    - Test parameters/fixtures used
    - Test result (PASSED/FAILED)
    """
    caplog.set_level(logging.DEBUG)
    
    # Get test info
    test_name = request.node.name
    test_module = request.node.module.__name__
    test_class = request.node.cls.__name__ if request.node.cls else "NoClass"
    test_docstring = request.node.obj.__doc__ or "No description"
    
    # Get fixture names used by this test
    fixture_names = request.fixturenames
    
    # Log test start
    test_logger.info("=" * 70)
    test_logger.info(f"TEST: {test_name}")
    test_logger.info(f"Module: {test_module}")
    test_logger.info(f"Class: {test_class}")
    test_logger.info(f"Description: {test_docstring.strip()}")
    test_logger.info(f"Fixtures: {', '.join(fixture_names)}")
    test_logger.info("=" * 70)
    
    yield
    
    # Log test end - check if test passed or failed
    if hasattr(request.node, 'rep_call'):
        if request.node.rep_call.passed:
            outcome = "✅ PASSED"
        elif request.node.rep_call.failed:
            outcome = "❌ FAILED"
        else:
            outcome = "⏭️ SKIPPED"
    else:
        outcome = "COMPLETED"
    
    test_logger.info(f"TEST RESULT: {test_name} - {outcome}")
    test_logger.info("=" * 70 + "\n")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test result for logging."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)
    
    # Log assertion failures with details
    if rep.when == "call" and rep.failed:
        test_logger.error(f"Test failed: {item.name}")
        if hasattr(rep, 'longrepr'):
            test_logger.error(f"Failure details: {rep.longrepr}")


@pytest.fixture
def log(caplog):
    """Fixture to provide a logger for tests to log custom messages.
    
    Usage:
        def test_something(log):
            log.info("Testing with input: %s", input_data)
            result = do_something(input_data)
            log.info("Result: %s", result)
    """
    caplog.set_level(logging.DEBUG)
    return test_logger


# =============================================================================
# Auto-logging for Mock Calls
# =============================================================================

@pytest.fixture(autouse=True)
def log_mock_calls(request, caplog):
    """Log all mock calls made during the test."""
    caplog.set_level(logging.DEBUG)
    yield
    
    # After test, log any mock calls that were made
    for fixture_name in request.fixturenames:
        try:
            fixture = request.getfixturevalue(fixture_name)
            if hasattr(fixture, 'call_args_list') and fixture.call_args_list:
                test_logger.debug(f"Mock '{fixture_name}' calls:")
                for i, call in enumerate(fixture.call_args_list[:10]):  # Limit to 10 calls
                    test_logger.debug(f"  Call {i+1}: {call}")
        except Exception:
            pass  # Ignore fixtures that can't be accessed


# =============================================================================
# Common Test Utilities
# =============================================================================

def log_test_data(logger: logging.Logger, **kwargs):
    """Log test data in a structured format.
    
    Args:
        logger: The logger to use
        **kwargs: Key-value pairs to log
    """
    for key, value in kwargs.items():
        logger.info(f"  {key}: {safe_repr(value)}")


@pytest.fixture
def log_data(log):
    """Fixture to log test data in a structured way.
    
    Usage:
        def test_something(log_data):
            log_data(input="test", expected="result")
    """
    def _log_data(**kwargs):
        log_test_data(log, **kwargs)
    return _log_data


# =============================================================================
# Assertion Helpers with Logging
# =============================================================================

@pytest.fixture
def assert_with_log(log):
    """Fixture providing assertion helpers that log their checks.
    
    Usage:
        def test_something(assert_with_log):
            result = do_something()
            assert_with_log.equals(result, expected, "result should match expected")
    """
    class AssertWithLog:
        def equals(self, actual, expected, message=""):
            log.info(f"Asserting equals: {safe_repr(actual)} == {safe_repr(expected)}")
            if message:
                log.info(f"  Context: {message}")
            assert actual == expected, f"{message}: {actual} != {expected}"
            log.info("  ✓ Assertion passed")
        
        def true(self, condition, message=""):
            log.info(f"Asserting true: {condition}")
            if message:
                log.info(f"  Context: {message}")
            assert condition, message
            log.info("  ✓ Assertion passed")
        
        def false(self, condition, message=""):
            log.info(f"Asserting false: {condition}")
            if message:
                log.info(f"  Context: {message}")
            assert not condition, message
            log.info("  ✓ Assertion passed")
        
        def contains(self, container, item, message=""):
            log.info(f"Asserting contains: {safe_repr(item)} in {safe_repr(container)}")
            if message:
                log.info(f"  Context: {message}")
            assert item in container, f"{message}: {item} not in {container}"
            log.info("  ✓ Assertion passed")
        
        def not_none(self, value, message=""):
            log.info(f"Asserting not None: {safe_repr(value)}")
            if message:
                log.info(f"  Context: {message}")
            assert value is not None, message
            log.info("  ✓ Assertion passed")
    
    return AssertWithLog()

