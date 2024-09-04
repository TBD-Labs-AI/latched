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
    yield
    end = time.time()
    print(f"{name} took {end - start:.4f} seconds")
