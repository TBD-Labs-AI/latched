# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, TypeAlias

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

from latched.model_exporters.base import BaseModelExporter
from latched.model_wrappers.huggingface import HuggingFaceModelWrapper
from latched.utils import profile

CoreMLReturn: TypeAlias = Any

if TYPE_CHECKING:
    from latched.model_wrappers.base import BaseModelWrapper


class CoreMLExporter(BaseModelExporter):
    """
    Export the model to CoreML.
    """

    class ForwardModel(nn.Module):
        def __init__(self, model: nn.Module):
            super().__init__()
            self.model = model

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.model(input_ids)[0]

    @classmethod
    def run(cls, model: "BaseModelWrapper", output_name: str = "coreml_model.mlpackage") -> None:
        if not output_name.endswith(".mlpackage") and not output_name.endswith(".mlmodel"):
            raise ValueError("Output name must end with .mlpackage or .mlmodel")

        if isinstance(model, HuggingFaceModelWrapper):
            # print_model_dtypes(model.model)
            target_model = cls.ForwardModel(model.model)
            dummy_input = torch.randint(1000, (1, 512), dtype=torch.int32)
            target_model.eval()
            target_model.to("cuda")
            with profile("torch.jit.trace"):
                with torch.no_grad():
                    traced_model = torch.jit.trace(target_model, dummy_input.to("cuda"))
            target_model.to("cpu")
            target_model.eval()

            ct_shape = ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=512)))
            with profile("coremltools.convert"):
                mlmodel = ct.convert(
                    traced_model,
                    convert_to="mlprogram",
                    inputs=[ct.TensorType(shape=ct_shape, dtype=np.int32, name="input_ids")],
                    source="pytorch",
                    outputs=[ct.TensorType(name="logits", dtype=np.float32)],
                    compute_precision=ct.precision.FLOAT32,
                )
            mlmodel.save(output_name)  # type: ignore
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")


def print_model_dtypes(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")
