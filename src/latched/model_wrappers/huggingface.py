# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TypeAlias

import torch.nn as nn

from transformers.models.phi3 import Phi3Model
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from latched.configs.device import DeviceConfig
from latched.model_wrappers.base import BaseModelWrapper

HuggingFaceTokenizer: TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast


class HuggingFaceModelWrapper(BaseModelWrapper):
    """
    HuggingFace model wrapper.

    Args:
        model (nn.Module): The model from huggingface
        precision (str, optional): The precision of the model. Defaults to "fp32".
        device_config (DeviceConfig, optional): The device configuration. Defaults to None.
        tokenizer (AutoTokenizer, optional): The tokenizer from huggingface
    """

    def __init__(
        self,
        model: nn.Module,
        precision: str = "fp32",
        device_config: DeviceConfig | None = None,
        tokenizer: HuggingFaceTokenizer | None = None,
    ):
        super().__init__(model, precision, device_config)

        if self._is_phi3_model(model) and tokenizer is not None:
            self.model = model.model.model
            self.tokenizer = tokenizer
        else:
            self.model = model

    def _is_phi3_model(self, model: nn.Module) -> bool:
        """Check if the model is a Phi3 model."""
        return isinstance(model, Phi3Model)
