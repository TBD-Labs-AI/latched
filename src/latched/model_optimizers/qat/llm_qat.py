# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from latched.model_optimizers.base import BaseModelOptimizer
import torch.nn as nn


class LLMQATOptimizer(BaseModelOptimizer):
    """
    LLMQATOptimizer to use the LLM QAT optimization techniques.

    This optimizer implements the LLM QAT method for quantizing large language models,
    as described in the LLM QAT repository.
    """

    @classmethod
    def run(cls, model: nn.Module) -> nn.Module:
        pass
