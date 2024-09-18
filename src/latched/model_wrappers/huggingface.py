# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TypeAlias

import torch
from torch import nn
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from latched.configs.device import DeviceConfig
from latched.model_wrappers.base import BaseModelWrapper

HuggingFaceTokenizer: TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast
TracedModule: TypeAlias = torch.jit._trace.TopLevelTracedModule  # type: ignore


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
        if self._is_phi3_model(model):
            self.model = model.model
            self.original_model = model
        elif self._is_llama_model(model):
            self.model = model.model.model
            self.original_model = model
        else:
            self.model = model
            self.original_model = model

        if tokenizer is not None:
            self.tokenizer = tokenizer

    def _is_phi3_model(self, model: nn.Module) -> bool:
        """Check if the model is a Phi3 model."""
        from transformers.models.phi3 import Phi3Model

        return isinstance(model.model, Phi3Model)

    def _is_llama_model(self, model: nn.Module) -> bool:
        """Check if the model is a Llama model."""
        from transformers.models.llama import LlamaModel

        return isinstance(model, LlamaModel)

    @property
    def eos_token_id(self) -> int | None:
        """Get the EOS token ID."""
        if self.tokenizer is None:
            return None

        return self.tokenizer.eos_token_id


class LLMScriptModelWrapper(nn.Module):
    def __init__(self, model: TracedModule, eos_token_id: int):
        super().__init__()
        self.model = model
        self.eos = torch.tensor([[eos_token_id]], dtype=torch.long)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        sentence = tokens

        for _ in range(100):
            predictions, _ = self.model(sentence)
            token = torch.argmax(predictions[:, -1, :], dim=-1).unsqueeze(1)
            sentence = torch.cat([sentence, token], dim=1)

            if token == self.eos:
                break

        return sentence
