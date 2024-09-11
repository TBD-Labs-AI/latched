# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

AVAILABLE_DEVICES = {
    "apple": [
        "iphone_15_pro",
    ],
    "samsung": [
        "galaxy_s_24",
    ],
    "nvidia": ["jetson_orin_nano_8gb"],
    "rebellion": ["atom"],
    "intel": ["gaudi2"],
}


@dataclass
class DeviceConfig:
    """Configuration for the device.

    Attributes:
        type (str): the type of device
        limit_mem_gb (int):
    """

    type: str
    limit_mem_gb: int = 2
