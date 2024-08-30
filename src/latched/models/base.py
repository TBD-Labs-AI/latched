# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import torch.nn as nn

class BaseModel(nn.Module):
    """
    Base class for the model.

    Args:
        model (nn.Module): The model from any library
    """
    def __init__(self, model: nn.Module):
        self.model = model