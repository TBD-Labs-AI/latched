# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from enum import Enum


class Device(Enum):
    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value
