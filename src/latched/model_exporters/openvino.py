# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING
from latched.model_exporters.base import BaseModelExporter
from latched.model_wrappers.huggingface import HuggingFaceModelWrapper

if TYPE_CHECKING:
    from latched.model_wrappers.base import BaseModelWrapper


class OpenVINOExporter(BaseModelExporter):
    """
    Export the model to OpenVINO.
    """

    @classmethod
    def run(cls, model_wrapper: BaseModelWrapper, output_name: str = "openvino_model") -> None:
        if isinstance(model_wrapper, HuggingFaceModelWrapper):
            from optimum.exporters.openvino import export_from_model

            export_from_model(model_wrapper.original_model, output_name)

            print(f"Model exported to {output_name}")
        else:
            raise NotImplementedError(f"Unsupported model wrapper: {type(model_wrapper)}")
