import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latched.utils import profile, profile_decorator

model_path = "microsoft/Phi-3.5-mini-instruct"


@profile_decorator
def prepare_model():
    model = AutoModelForCausalLM.from_pretrained(model_path, torchscript=True)
    model.eval()
    return model


@profile_decorator
def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


@profile_decorator
def jit_trace_model(model, input_tokens):
    traced_model = torch.jit.trace(model, input_tokens)
    return traced_model


class LLMModelWrapper(torch.nn.Module):
    def __init__(self, model, eos_token_id):
        super().__init__()
        self.model = model
        self.default_token = torch.tensor([[0]], dtype=torch.long)
        self.eos = torch.tensor([[eos_token_id]], dtype=torch.long)

    def forward(self, tokens):
        sentence = tokens

        for i in range(64):  # Set a maximum length to avoid infinite loops
            predictions, _ = self.model(sentence)
            token = torch.argmax(predictions[:, -1, :], dim=-1).unsqueeze(1)
            sentence = torch.cat([sentence, token], dim=1)

            if token == self.eos:
                break

        return sentence

    def forward_with_tokenizer(self, tokens, tokenizer):
        sentence = tokens

        for i in range(64):  # Set a maximum length to avoid infinite loops
            predictions, _ = self.model(sentence)
            token = torch.argmax(predictions[:, -1, :], dim=-1).unsqueeze(1)
            print(token)
            sentence = torch.cat([sentence, token], dim=1)
            print(tokenizer.decode(sentence.squeeze(0).tolist()))

            if token.item() == self.eos.item():
                break

        return sentence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", action="store_true")
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    import os

    args = parse_args()
    tokenizer = prepare_tokenizer()
    text_input = tokenizer.encode("Hello, how are you?", return_tensors="pt")
    sample_input = torch.randint(1000, (1, 10))

    if args.trace:
        print("Tracing model...")
        model = prepare_model()
        model.eval()
        with profile("jit_trace_model"):
            traced_model = jit_trace_model(model, sample_input)
        torch.jit.save(traced_model, f"{model_path.replace('/', '_')}_traced_model.pt")
    else:
        print("Loading traced model...")
        if os.path.exists(f"{model_path.replace('/', '_')}_traced_model.pt"):
            traced_model = torch.jit.load(f"{model_path.replace('/', '_')}_traced_model.pt")
        else:
            raise ValueError("Traced model not found")

    if args.script:
        print("Scripting model...")
        wrapped_model = LLMModelWrapper(traced_model, tokenizer.eos_token_id)
        with profile("torch.jit.script"):
            scripted_model = torch.jit.script(wrapped_model)
        scripted_model.save(f"{model_path.replace('/', '_')}_scripted_model.pt")
    else:
        print("Loading scripted model...")
        if os.path.exists(f"{model_path.replace('/', '_')}_scripted_model.pt"):
            scripted_model = torch.jit.load(f"{model_path.replace('/', '_')}_scripted_model.pt")
        else:
            raise ValueError("Scripted model not found")

    import coremltools as ct
    import numpy as np

    scripted_model = scripted_model.eval()
    scripted_model = scripted_model.to("cpu")
    with profile("coremltools.convert"):
        mlmodel = ct.convert(
            scripted_model,
            convert_to="mlprogram",
            inputs=[ct.TensorType(shape=sample_input.shape, dtype=np.int32)],
            source="pytorch",
        )

    mlmodel.save("Phi3.5-mini-scripted.mlpackage")  # type: ignore
