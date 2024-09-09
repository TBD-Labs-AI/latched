# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoTokenizer, AutoModelForCausalLM

from latched.model_wrappers.auto import AutoModelWrapper
from latched.model_optimizers.hf_qint8 import HFQuantOptimizer
from latched.model_exporters.coreml import CoreMLExporter

# Load the huggingface tokenizer and model
model_path = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")

# Wrap the model in Latched's AutoModel
latched_model_wrapper = AutoModelWrapper(model=model, tokenizer=tokenizer)

# Optimize the model
optimized_model = HFQuantOptimizer.run(latched_model_wrapper.model)

# Export the model to CoreML
exported_model = CoreMLExporter.run(optimized_model)
