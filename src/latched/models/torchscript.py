# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TypeAlias

import torch
from torch import nn

TracedModule: TypeAlias = torch.jit._trace.TopLevelTracedModule


class ScriptModel(nn.Module):
    def __init__(self, model: TracedModule, **kwargs):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return NotImplementedError


class LLMScriptModel(ScriptModel):
    def __init__(self, model: TracedModule, **kwargs):
        super().__init__()
        self.model = model
        self.eos_token_id = kwargs.get("eos_token_id")
        if not self.eos_token_id:
            raise ValueError("LLMScriptModel needs eos_token_id")
        self.eos = torch.tensor([[self.eos_token_id]], dtype=torch.long)
        self.max_token_length = 100

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        sentence = tokens

        for _ in range(self.max_token_length):
            predictions, _ = self.model(sentence)
            token = torch.argmax(predictions[:, -1, :], dim=-1).unsqueeze(1)
            sentence = torch.cat([sentence, token], dim=1)

            if token == self.eos:
                break

        return sentence
