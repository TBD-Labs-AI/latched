# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from latched.configs.device import DeviceConfig


class BaseModelWrapper:
    """
    Base class for the model wrapper. Model wrapper converts any models from different libraries to a unified interface(nn.Module).

    Args:
        model (nn.Module): The model wrapper for handling the model from any library
    """

    def __init__(
        self,
        model: nn.Module,
        precision: str = "fp32",
        device_config: DeviceConfig | None = None,
    ):
        self.model = model
        self.precision = precision
        self.device_config = device_config
