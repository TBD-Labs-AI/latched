# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from latched.configs import DeviceConfig
from latched.models.auto import AutoModel
from latched.model_optimizers.auto import AutoOptimizer
from transformers import AutoTokenizer, LlamaForCausalLM

# Load the huggingface tokenizer and model
model_path = "nvidia/Llama-3.1-Minitron-4B-Width-Base"
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = 'cpu'
dtype = torch.bfloat16
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device)

latched_model = AutoModel(model=model)