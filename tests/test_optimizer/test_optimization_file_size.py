# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from latched.model_optimizers.torchao import HFQuantOptimizer
from latched.model_wrappers.auto import AutoModelWrapper
from latched.utils.models import get_size_of_model

model_path = "microsoft/Phi-3.5-mini-instruct"


@pytest.fixture
def model():
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelWrapper(model=model, tokenizer=tokenizer)
    model.model.eval()
    return model


def test_file_size_after_quantization(model):
    quantized_model = HFQuantOptimizer.run(model)
    assert quantized_model is not None
    assert quantized_model.model.state_dict() is not None

    assert get_size_of_model(model.model) > get_size_of_model(quantized_model.model)
