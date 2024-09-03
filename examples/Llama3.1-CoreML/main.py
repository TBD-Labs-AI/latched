# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from latched.models.auto import AutoModel

# Load the huggingface tokenizer and model
model_path = "nvidia/Llama-3.1-Minitron-4B-Width-Base"
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = "cpu"
dtype = torch.bfloat16
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype)

latched_model = AutoModel(model=model)
