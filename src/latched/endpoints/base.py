# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from latched.model_wrappers.base import BaseModelWrapper
    from transformers import PreTrainedTokenizerBase


class BaseEndpoint:
    """
    Base class of the Endpoint.

    Args:
        model_wrapper (BaseModelWrapper): A wrapper object that contains model-related configurations and methods.
        tokenizer (PreTrainedTokenizerBase): A tokenizer that converts input text into token ids for the model.
        model_path (str): The path to the pre-trained model to be loaded and optimized.
        task (str, optional): The NLP task for which the pipeline is created. Default is 'text-generation'.
    """

    def __init__(
        self,
        model_wrapper: BaseModelWrapper,
        tokenizer: PreTrainedTokenizerBase,
        model_path: str,
        task: str = "text-generation",
    ):
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.task = task
