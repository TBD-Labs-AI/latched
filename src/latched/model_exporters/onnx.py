# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import TYPE_CHECKING

from latched.model_exporters.base import BaseModelExporter
from latched.model_wrappers.huggingface import HuggingFaceModelWrapper

if TYPE_CHECKING:
    from latched.model_wrappers.base import BaseModelWrapper


class ONNXExporter(BaseModelExporter):
    """
    ONNXExporter is a class for exporting the model to ONNX format.
    """

    @classmethod
    def run(cls, model_wrapper: BaseModelWrapper, **kwargs) -> None:
        if isinstance(model_wrapper, HuggingFaceModelWrapper):
            from optimum.exporters.onnx import onnx_export_from_model

            output_name = kwargs.get("output_name", "onnx_model")
            onnx_export_from_model(model_wrapper.original_model, output_name)

            print(f"\nModel successfully exported to {output_name}")
        else:
            raise NotImplementedError(f"Unsupported model wrapper: {type(model_wrapper)}")
