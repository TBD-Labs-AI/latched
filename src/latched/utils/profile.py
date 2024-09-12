# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from contextlib import contextmanager


@contextmanager
def profile(name: str):
    """profile a section of code

    Args:
        name (str): name of the section to profile

    usage:
    >>> with profile("my_function"):
    >>>     my_function()

    >>> my_function took 0.0123 seconds
    """
    start = time.time()
    print(f"\033[94mProfiling {name}...\033[0m", end="", flush=True)
    yield
    end = time.time()
    duration = end - start
    print(f"\r\033[97m[\033[0m\033[92m{name}\033[0m\033[97m]\033[0m took \033[93m{duration:.4f}\033[0m seconds")


# Wrapper for profile a function as a decorator
def profile_decorator(func):
    def wrapper(*args, **kwargs):
        with profile(func.__name__):
            return func(*args, **kwargs)

    return wrapper
