import coremltools as ct
import torch
from transformers import AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct", torchscript=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

# model.eval()

# model.to("cuda")

msg = """<|system|>
You are a helpful assistant.<|end|>
<|user|>
How to explain Internet for a medieval knight?<|end|>
<|assistant|>
"""

# dummy_input = torch.randint(1000, (1, 100))

# traced_model = torch.jit.trace(model, dummy_input.to("cuda"))

# inputs = tokenizer(msg, return_tensors="pt").to("cuda")

# print(inputs["input_ids"].shape)  # type: ignore
# print(inputs["input_ids"])

# eos_token_id = tokenizer.eos_token_id

# eos_token = torch.tensor([[eos_token_id]], dtype=torch.long, device="cuda")

# input_ids = inputs["input_ids"]
# for _ in range(64):
#     predictions, _ = model(input_ids)

#     token = torch.argmax(predictions[:, -1, :], dim=-1).unsqueeze(1)
#     input_ids = torch.cat([input_ids, token], dim=1)  # type: ignore

#     if token == eos_token:
#         break


# print(input_ids.shape)  # type: ignore

# print(tokenizer.decode(input_ids.squeeze(0).tolist()))  # type: ignore

inputs = tokenizer.encode(msg, return_tensors="pt")
print(inputs)
mlmodel = ct.models.MLModel("Phi3.5-mini-scripted-int64.mlpackage")
coreml_inputs = {"input_ids": inputs.to(torch.int32).numpy()}

prediction = mlmodel.predict(coreml_inputs)

print(prediction)
