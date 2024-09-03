# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from latched.model_optimizers.base import BaseModelOptimizer

from latched.models.base import BaseModel

class SmoothQuantOptimizer(BaseModelOptimizer):
    """
    SmoothQuantOptimizer to use the SmoothQuant optimization techniques.

    This optimizer implements the SmoothQuant method for quantizing large language models,
    as described in the MIT-Han-Lab repository.
    """

    @classmethod
    def run(model: BaseModel) -> BaseModel:
        pass