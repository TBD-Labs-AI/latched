# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from latched.model_optimizers.base import BaseModelOptimizer
import torch.nn as nn


class OmniQuantOptimizer(BaseModelOptimizer):
    """
    OmniQuantOptimizer to use the OmniQuant optimization techniques.

    This optimizer implements the OmniQuant method for quantizing large language models,
    as described in the OmniQuant repository.
    """

    @classmethod
    def run(cls, model: nn.Module) -> nn.Module:
        pass
