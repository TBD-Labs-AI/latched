# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from latched.model_wrappers.base import BaseModelWrapper


class BaseModelOptimizer:
    """
    Base class of the Optimizer.
    """

    @abstractmethod
    def run(cls, model_wrapper: BaseModelWrapper) -> BaseModelWrapper:
        raise NotImplementedError
