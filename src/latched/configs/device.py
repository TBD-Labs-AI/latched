# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from latched.configs.apple import Apple
from latched.configs.base import Device

AVAILABLE_DEVICES = {
    "apple": [
        "iphone_15_pro",
    ],
    "samsung": [
        "galaxy_s_24",
    ],
    "nvidia": ["jetson_orin_nano_8gb"],
    "rebellion": ["atom"],
    "intel": ["gaudi3"],
}


@dataclass
class DeviceConfig:
    """Configuration for the device.

    Attributes:
        type (str): the type of device
        limit_mem_gb (int):
    """

    type: str
    dtype: Device
    limit_mem_gb: int = 2


if __name__ == "__main__":
    devices = [Apple.IPHONE_15_PRO, Apple.IPHONE_15_PRO_MAX]
    target_device = Apple.IPHONE_15_PRO

    print(devices)
    print(target_device.processor)
    print(target_device == Apple.IPHONE_15_PRO)
