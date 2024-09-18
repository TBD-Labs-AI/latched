# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import Any, TypeAlias, TYPE_CHECKING

import coremltools as ct
import numpy as np
import torch

from latched.model_exporters.base import BaseModelExporter
from latched.models.coreml import CoreMLLanguageModel

if TYPE_CHECKING:
    from latched.model_wrappers.base import BaseModelWrapper

CoreMLReturn: TypeAlias = Any

if TYPE_CHECKING:
    from latched.model_wrappers.base import BaseModelWrapper


class CoreMLExporter(BaseModelExporter):
    """
    Export the model to CoreML.
    """

    @classmethod
    def run(cls, model_wrapper: BaseModelWrapper, **kwargs) -> Any:
        eos_token_id = kwargs.get("eos_token_id")
        if eos_token_id is None:
            raise ValueError("EOS token ID is required for CoreML export")

        model = CoreMLLanguageModel(model_wrapper.model, eos_token_id)

        sample_input = torch.randint(1000, (1, 100))
        traced_model = torch.jit.trace(model, sample_input)
        scripted_model = torch.jit.script(traced_model)
        mlmodel = ct.convert(
            scripted_model,
            convert_to="mlprogram",
            inputs=[ct.TensorType(shape=sample_input.shape, dtype=np.int64)],
            source="pytorch",
        )
        return mlmodel
