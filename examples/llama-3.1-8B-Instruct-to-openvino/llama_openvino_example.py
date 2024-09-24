# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoTokenizer, AutoModelForCausalLM

from latched.model_wrappers.auto import AutoModelWrapper
from latched.model_exporters.auto import AutoExporter
from latched.endpoints.intel import IntelEndpoint
from latched.configs.device import DeviceConfig

# Setup the configuraiton
device_config = DeviceConfig(type="intel_i5_13900k")

# Step 1. Load the huggingface tokenizer and model
model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")

# Step 2. Wrap the model in Latched's AutoModel
latched_model_wrapper = AutoModelWrapper(model=model, tokenizer=tokenizer, device_config=device_config).create()

# Step 3. Export the model to OpenVINO
AutoExporter.run(latched_model_wrapper, output_path="ov_model")

# Step 4. Inference with device SDK
# This step should be executed at Intel devices.
intel_endpoint = IntelEndpoint(latched_model_wrapper, tokenizer, "ov_model")
results = intel_endpoint.inference("What is the love?")
print(results)
