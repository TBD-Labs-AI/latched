# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn    

class BaseModelOptimizer:
    """
    Base class of the Optimizer.
    """

    @abstractmethod
    def run(model: nn.Module):
        raise NotImplementedError