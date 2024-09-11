# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TypeAlias

import torch.nn as nn
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.models.phi3 import Phi3Model

from latched.model_wrappers.base import BaseModelWrapper
from latched.configs.device import DeviceConfig

HuggingFaceTokenizer: TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast


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

        # TODO: Remove this after designing the AutoModel for all the libraries
        if self._is_hf_phi3_model(model):
            self.model = model.model.model
        else:
            self.model = model

    def _is_hf_phi3_model(self, model: nn.Module) -> bool:
        return isinstance(model, Phi3Model)
