#!/usr/bin/env python3

# Ruff Settings
# ruff: noqa: D105

import os
import signal
import sys
import time
import unittest
from functools import wraps


class Test:
    """Decorator for test methods with timeout and performance tracking.

    Parameters
    ----------
    timeout : int, optional
        Maximum allowed time in seconds for the test to execute (default is 5).
    """

    def __init__(self, timeout=5):
        self.timeout = timeout

    def __call__(self, func):
        """Wrap the test function with timeout and elapsed time measurement.

        Parameters
        ----------
        func : callable
            The test method to wrap.

        Returns
        -------
        callable
            The wrapped test method.
        """
        func = TimeoutFunc(self.timeout)(func)
        func.__timeout__ = self.timeout

        @wraps(func)
        def wrapper(*args, **kwargs):
            args[0].starttime = time.perf_counter()
            result = func(*args, **kwargs)
            endtime = time.perf_counter()
            args[0].elapsed = endtime - args[0].starttime
            return result

        return wrapper


class TimeoutFunc:
    """Decorator to enforce a timeout on a function call.

    Parameters
    ----------
    max_seconds : int
        Maximum allowed time in seconds for the function to execute.

    Methods
    -------
    __call__(func)
        Wraps the function to enforce the timeout.
    """

    def __init__(self, max_seconds):
        self.max_seconds = max_seconds

    def __call__(self, func):
        """Wrap the function to enforce the timeout.

        Parameters
        ----------
        func : callable
            The function to wrap.

        Returns
        -------
        callable
            The wrapped function.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Windows signal library does not support SIGALRM
            if os.name != 'nt':

                def handle_timeout(signum, frame):
                    args[0].fail(f'Test case timed out. Max time: {self.max_seconds} seconds')

                signal.signal(signal.SIGALRM, handle_timeout)
                signal.alarm(self.max_seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
            else:
                result = func(*args, **kwargs)
            return result

        return wrapper


def blockPrint():
    """Disable printing to stdout."""
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    """Restore printing to stdout."""
    sys.stdout = sys.__stdout__


class HiddenPrints:
    """Context manager to suppress stdout temporarily."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class CustomTestCase(unittest.TestCase):
    """Custom TestCase with additional functionality.

    Attributes
    ----------
    starttime : float
        The time when the test started.
    elapsed : float
        The time elapsed during the test execution.
    """

    def id(self):
        """Return the identifier of the test case."""
        return self.shortDescription().split(':')[0]

    @property
    def timeout(self):
        """Return the timeout for the test case."""
        return getattr(getattr(self, self._testMethodName), '__timeout__', None)


if __name__ == '__main__':
    unittest.main()
