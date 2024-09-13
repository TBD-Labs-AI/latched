# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import TYPE_CHECKING

from latched.model_optimizers.base import BaseModelOptimizer
from latched.model_wrappers.huggingface import HuggingFaceModelWrapper
from latched.model_optimizers.torchao import HFQuantOptimizer

if TYPE_CHECKING:
    from latched.model_wrappers.base import BaseModelWrapper


class AutoOptimizer(BaseModelOptimizer):
    """AutoOptimizer to use the optimization techniques with a easiest way."""

    @classmethod
    def run(cls, model_wrapper: BaseModelWrapper) -> BaseModelWrapper:
        if isinstance(model_wrapper, HuggingFaceModelWrapper):
            return HFQuantOptimizer.run(model_wrapper)
        else:
            raise NotImplementedError(f"AutoOptimizer for {model_wrapper.library_name} is not implemented")
