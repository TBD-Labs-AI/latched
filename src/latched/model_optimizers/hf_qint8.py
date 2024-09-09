import torch.nn as nn
from optimum.quanto import freeze, qint8, quantize

from latched.model_optimizers.base import BaseModelOptimizer


class HFQuantOptimizer(BaseModelOptimizer):
    """
    HFQuantOptimizer to use the HuggingFace quantization optimization techniques.
    """

    @classmethod
    def run(cls, model: nn.Module) -> nn.Module:
        quantize(model, weights=qint8, activations=None)
        freeze(model)
        return model
