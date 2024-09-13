# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from latched.model_wrappers.base import ModelWrapper

class EndpointSDK:
    def __init__(
            self, 
            model_wrapper: ModelWrapper,

        ):
        self.model_wrapper = model_wrapper
