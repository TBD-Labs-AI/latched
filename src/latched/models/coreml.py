# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn


class CoreMLLanguageModel(nn.Module):
    """
    Model for the CoreML Language Model

    Args:
        eos_token_id (int): the id of the EOS token
    """

    # TODO: Update below code especially forward method
    def __init__(self, model: nn.Module, eos_token_id: int):
        super().__init__()
        self.model = model
        self.default_token = torch.tensor([[0]], dtype=torch.int64)
        self.eos = torch.tensor([[eos_token_id]], dtype=torch.int64)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(tokens)
