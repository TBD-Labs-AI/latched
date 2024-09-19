# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from latched.model_wrappers.base import BaseModelWrapper


class BaseModelExporter:
    """
    Base class of the Exporter.
    """

    @abstractmethod
    def run(cls, model_wrapper: BaseModelWrapper, **kwargs) -> None:
        raise NotImplementedError

    @property
    def device(self):
        # TODO: consider the accelerator
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
