# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch.nn as nn


class BaseModelWrapper(nn.Module):
    """
    Base class for the model wrapper. Model wrapper converts any models from different libraries to a unified interface(nn.Module).

    Args:
        model (nn.Module): The model wrapper for handling the model from any library
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
