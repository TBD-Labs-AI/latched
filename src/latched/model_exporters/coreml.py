# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, TypeAlias

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

from latched.model_exporters.base import BaseModelExporter

CoreMLReturn: TypeAlias = Any


class CoreMLExporter(BaseModelExporter):
    """
    Export the model to CoreML.
    """

    @classmethod
    def model_wrapper(cls, model: nn.Module, eos_token_id: int) -> nn.Module:
        # TODO: Update below code especially forward method
        class CoreMLLanguageModel(nn.Module):
            def __init__(self, model: nn.Module, eos_token_id: int):
                super().__init__()
                self.model = model
                self.default_token = torch.tensor([[0]], dtype=torch.int64)
                self.eos = torch.tensor([[eos_token_id]], dtype=torch.int64)

            def forward(self, tokens: torch.Tensor) -> torch.Tensor:
                return self.model(tokens)

        return CoreMLLanguageModel(model, eos_token_id)

    @classmethod
    def run2(cls, model: nn.Module, eos_token_id: int | None) -> Any:
        if eos_token_id is None:
            raise ValueError("EOS token ID is required for CoreML export")
        wrapped_model = cls.model_wrapper(model, eos_token_id)
        wrapped_model.eval()
        sample_input = torch.randint(1000, (1, 100))
        traced_model = torch.jit.trace(wrapped_model, sample_input)
        scripted_model = torch.jit.script(traced_model)
        mlmodel = ct.convert(
            scripted_model,
            convert_to="mlprogram",
            inputs=[ct.TensorType(shape=sample_input.shape, dtype=np.int64)],
            source="pytorch",
        )
        return mlmodel
