# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod

import torch.nn as nn


class BaseModelOptimizer:
    """
    Base class of the Optimizer.
    """

    @abstractmethod
    def run(cls, model: nn.Module) -> nn.Module:
        raise NotImplementedError
