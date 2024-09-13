# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from typing import TYPE_CHECKING
import torch.nn as nn

from latched.model_wrappers.base import BaseModelWrapper
from latched.configs.device import DeviceConfig

if TYPE_CHECKING:
    from latched.model_wrappers.huggingface import HuggingFaceTokenizer


class AutoModelWrapper(BaseModelWrapper):
    """
    AutoModelWrapper is a wrapper around the model from any library.

    Args:
        model (nn.Module): The model from any library
        precision (str, optional): The precision of the model. Defaults to "fp32".
        device_config (DeviceConfig, optional): The device configuration. Defaults to None.
        tokenizer (AutoTokenizer, optional): The tokenizer from any library
    """

    def __init__(
        self,
        model: nn.Module,
        precision: str = "fp32",
        device_config: DeviceConfig | None = None,
        tokenizer: HuggingFaceTokenizer | None = None,
    ):
        super().__init__(model, precision, device_config)
        self.tokenizer = tokenizer

    def create(self) -> BaseModelWrapper:
        if self.library_name == "transformers":
            from latched.model_wrappers.huggingface import HuggingFaceModelWrapper

            return HuggingFaceModelWrapper(self.model, self.precision, self.device_config, self.tokenizer)
        else:
            raise NotImplementedError(f"AutoModelWrapper for {self.library_name} is not implemented")
