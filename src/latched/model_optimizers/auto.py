# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING
from latched.model_optimizers.base import BaseModelOptimizer

if TYPE_CHECKING:
    import torch.nn as nn


class AutoOptimizer(BaseModelOptimizer):
    """
    AutoOptimizer to use the optimization techniques with a easiest way.
    """

    def run(model: nn.Module):
        pass