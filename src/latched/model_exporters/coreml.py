# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from latched.model_exporters.base import BaseModelExporter


class CoreMLExporter(BaseModelExporter):
    """
    Export the model to CoreML.
    """

    @classmethod
    def run(cls, model: nn.Module) -> nn.Module:
        pass
