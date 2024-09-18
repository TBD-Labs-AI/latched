# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import Any, TypeAlias, TYPE_CHECKING

import coremltools as ct
import numpy as np
import torch

from latched.model_exporters.base import BaseModelExporter
from latched.model_wrappers.huggingface import HuggingFaceModelWrapper
from latched.models.coreml import CoreMLForwardModel
from latched.utils import profile

if TYPE_CHECKING:
    from latched.model_wrappers.base import BaseModelWrapper

CoreMLReturn: TypeAlias = Any


class CoreMLExporter(BaseModelExporter):
    """
    Export the model to CoreML.
    """

    @classmethod
    def run(cls, model_wrapper: BaseModelWrapper, **kwargs) -> Any:
        output_name = kwargs.get("output_name", "coreml_model.mlpackage")
        if not output_name.endswith(".mlpackage") and not output_name.endswith(".mlmodel"):
            raise ValueError("Output name must end with .mlpackage or .mlmodel")

        if isinstance(model_wrapper, HuggingFaceModelWrapper):
            # print_model_dtypes(model.model)
            target_model = CoreMLForwardModel(model_wrapper.model)

            device = cls().device
            dummy_input = torch.randint(1000, (1, 512), dtype=torch.int32, device=device)

            target_model.eval()
            with profile("torch.jit.trace"):
                with torch.no_grad():
                    traced_model = torch.jit.trace(target_model, dummy_input)

            ct_shape = ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=2048)))
            with profile("coremltools.convert"):
                mlmodel = ct.convert(
                    traced_model,
                    convert_to="mlprogram",
                    inputs=[ct.TensorType(shape=ct_shape, dtype=np.int32, name="input_ids")],
                    source="pytorch",
                    outputs=[ct.TensorType(name="logits", dtype=np.float32)],
                    compute_precision=ct.precision.FLOAT32,
                    minimum_deployment_target=ct.target.iOS18,
                )
            mlmodel.save(output_name)  # type: ignore
        else:
            raise ValueError(f"Unsupported model wrapper type: {type(model_wrapper)}")
