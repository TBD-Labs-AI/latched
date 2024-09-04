from optimum.quanto import quantize, qint8, freeze

from latched.model_optimizers.base import BaseModelOptimizer
import torch.nn as nn


class HFQuantOptimizer(BaseModelOptimizer):
    """
    HFQuantOptimizer to use the HuggingFace quantization optimization techniques.
    """

    @classmethod
    def run(cls, model: nn.Module) -> nn.Module:
        quantize(model, weights=qint8, activation=None)
        freeze(model)
        return model
