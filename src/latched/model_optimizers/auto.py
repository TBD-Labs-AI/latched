# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from latched.model_optimizers.base import BaseModelOptimizer

import torch.nn as nn


class AutoOptimizer(BaseModelOptimizer):
    """AutoOptimizer to use the optimization techniques with a easiest way."""

    @classmethod
    def run(cls, model: nn.Module) -> nn.Module:
        pass
