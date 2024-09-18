# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoModelForCausalLM, AutoTokenizer

from latched.model_exporters.coreml import CoreMLExporter

# from latched.model_optimizers.auto import AutoOptimizer
from latched.model_wrappers.auto import AutoModelWrapper

# Load the huggingface tokenizer and model
model_path = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torchscript=True)

# Wrap the model in Latched's AutoModel
latched_model_wrapper = AutoModelWrapper(model=model, tokenizer=tokenizer).create()

# Optimize the model
# optimized_model_wrapper = AutoOptimizer.run(latched_model_wrapper)

# Export the model to CoreML
CoreMLExporter.run(latched_model_wrapper)
