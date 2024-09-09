# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from latched.model_exporters.base import BaseModelExporter


class OpenVINOExporter(BaseModelExporter):
    """
    Export the model to OpenVINO.
    """

    @classmethod
    def run(cls, model: nn.Module) -> nn.Module:
        pass
