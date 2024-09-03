# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod

from latched.models.base import BaseModel


class BaseModelExporter:
    """
    Base class of the Exporter.
    """

    @abstractmethod
    def run(model: BaseModel) -> BaseModel:
        raise NotImplementedError
