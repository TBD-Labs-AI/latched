# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from torch.ao.quantization import quantize_dynamic

from latched.model_optimizers.base import BaseModelOptimizer

if TYPE_CHECKING:
    from latched.model_wrappers.base import BaseModelWrapper


class HFQuantOptimizer(BaseModelOptimizer):
    """
    HFQuantOptimizer to use the HuggingFace quantization optimization techniques.
    """

    @classmethod
    def run(cls, model_wrapper: BaseModelWrapper) -> BaseModelWrapper:
        torch.backends.quantized.engine = "qnnpack"
        quantize_dynamic(
            model_wrapper.model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        return model_wrapper
