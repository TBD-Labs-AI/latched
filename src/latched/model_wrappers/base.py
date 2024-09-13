# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

from latched.configs.device import DeviceConfig
from latched.model_wrappers import SUPPORTED_LIBRARIES


class BaseModelWrapper:
    """
    Base class for the model wrapper. Model wrapper converts any models from different libraries to a unified interface(nn.Module).

    Args:
        model (nn.Module): The model wrapper for handling the model from any library
        precision (str): The precision of the model. Defaults to "fp32".
        device_config (DeviceConfig): The device configuration. Defaults to None.
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

        self.library_name = self._get_library_name(model)
        if self.library_name not in SUPPORTED_LIBRARIES:
            raise ValueError(f"Unsupported library: {self.library_name}")

    def _get_library_name(self, model: nn.Module) -> str:
        """Get the library name from the model."""
        return model.__module__.split(".")[0]
