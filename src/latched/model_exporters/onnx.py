# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from latched.model_exporters.base import BaseModelExporter
from latched.model_wrappers.huggingface import HuggingFaceModelWrapper

if TYPE_CHECKING:
    from latched.model_wrappers.base import BaseModelWrapper


class ONNXExporter(BaseModelExporter):
    """
    ONNXExporter is a class for exporting the model to ONNX format.
    """

    @classmethod
    def run(cls, model_wrapper: BaseModelWrapper, output_name: str = "model.onnx") -> None:
        if isinstance(model_wrapper, HuggingFaceModelWrapper):
            dummy_input = model_wrapper.tokenizer("Hello, world!", return_tensors="pt")
            model_wrapper.model.eval()

            torch.onnx.export(
                model_wrapper.model,
                (dummy_input["input_ids"],),
                output_name,
                export_params=False,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input_ids"],
                output_names=["output"],
                dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}},
            )
            print(f"Model exported to {output_name}")
        else:
            raise NotImplementedError(f"Unsupported model wrapper: {type(model_wrapper)}")
