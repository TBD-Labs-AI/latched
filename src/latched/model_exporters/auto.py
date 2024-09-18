# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING
from latched.model_exporters.base import BaseModelExporter

if TYPE_CHECKING:
    from latched.model_wrappers.base import BaseModelWrapper


class AutoExporter(BaseModelExporter):
    """
    AutoExporter is a model exporter that automatically selects the best exporter for the given model.
    """

    @classmethod
    def run(cls, model_wrapper: BaseModelWrapper, **kwargs) -> None:
        raise NotImplementedError
