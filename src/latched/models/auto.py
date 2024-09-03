# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoTokenizer
import torch.nn as nn
from latched.models.base import BaseModelWrapper
from transformers.models.phi3 import Phi3Model


class AutoModelWrapper(BaseModelWrapper):
    """
    AutoModelWrapper is a wrapper around the model from any library.

    Args:
        model (BaseModelWrapper): The model wrapper from any library
        tokenizer (AutoTokenizer): The tokenizer from any library
    """

    def __init__(self, model: nn.Module, tokenizer: AutoTokenizer):
        super().__init__(model)
        self.tokenizer = tokenizer

        # TODO: Remove this after designing the AutoModel for all the libraries
        if self._is_hf_phi3_model(model):
            self.model = model.model.model
        else:
            self.model = model

    def _is_hf_phi3_model(self, model: nn.Module) -> bool:
        return isinstance(model, Phi3Model)
