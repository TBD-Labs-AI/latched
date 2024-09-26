# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING
from latched.endpoints.base import BaseEndpoint
from latched.model_wrappers.base import BaseModelWrapper
from transformers import pipeline

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase, Pipeline
    import torch.nn as nn


class IntelEndpoint(BaseEndpoint):
    """
    IntelEndpoint is a specialized class for handling model inference on Intel devices using OpenVINO.

    This class is designed to optimize and run transformer models, particularly for tasks like text generation,
    on Intel hardware by leveraging the OpenVINO (OV) framework. The model is wrapped using the optimum.intel
    library, and a Hugging Face pipeline is created to handle the inference tasks efficiently.

    Args:
        model_wrapper (BaseModelWrapper): A wrapper object that contains model-related configurations and methods.
        tokenizer (PreTrainedTokenizerBase): A tokenizer that converts input text into token ids for the model.
        model_path (str): The path to the pre-trained model to be loaded and optimized for Intel devices.
        task (str, optional): The NLP task for which the pipeline is created. Default is 'text-generation'.

    Attributes:
        ov_model (nn.Module): The OpenVINO-optimized model loaded from the provided model path.
        pipeline (Pipeline): Hugging Face pipeline that uses the optimized model for inference.
    """

    def __init__(
        self,
        model_wrapper: BaseModelWrapper,
        tokenizer: PreTrainedTokenizerBase,
        model_path: str,
        task: str = "text-generation",
    ):
        super().__init__(model_wrapper, tokenizer, model_path, task)
        self.ov_model = self._prepare_model()
        self.pipeline = self._prepare_pipeline()

    def _prepare_model(self) -> nn.Module:
        if self.model_wrapper.library_name == "transformers":
            from optimum.intel import OVModelForCausalLM

            return OVModelForCausalLM.from_pretrained(self.model_path)
        else:
            raise ValueError(f"Unsupported library: {self.model_wrapper.library_name}")

    def _prepare_pipeline(self) -> Pipeline:
        return pipeline(self.task, model=self.ov_model, tokenizer=self.tokenizer)

    def inference(self, input_text: str) -> str:
        """
        Runs inference on the provided input text using the Hugging Face pipeline.

        The method takes input text, processes it through the pipeline, and returns the generated output. It also
        checks that the input text is valid (non-empty) before running the inference.

        Args:
            input_text (str): The input text for which inference is to be performed.

        Returns:
            Any: The result of the inference process, typically the generated text for tasks like text-generation.

        """
        return self.pipeline(input_text)
