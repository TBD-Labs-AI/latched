# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod

import torch.nn as nn


class BaseModelExporter:
    """
    Base class of the Exporter.
    """

    @abstractmethod
    def run(cls, model: nn.Module) -> nn.Module:
        raise NotImplementedError
