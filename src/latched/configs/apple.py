# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import TypeAlias

from .base import Device

iphones: TypeAlias = str


class Apple(Device):
    """Apple devices."""

    IPHONE_15_PRO = "iphone_15_pro"
    IPHONE_15_PRO_MAX = "iphone_15_pro_max"
    IPHONE_15 = "iphone_15"
    IPHONE_15_PLUS = "iphone_15_plus"
    IPHONE_14_PRO_MAX = "iphone_14_pro_max"
    IPHONE_14_PRO = "iphone_14_pro"
    IPHONE_14 = "iphone_14"
    IPHONE_13_PRO = "iphone_13_pro"
    IPHONE_13_PRO_MAX = "iphone_13_pro_max"
    IPHONE_13 = "iphone_13"

    @property
    def processor(self):
        """Get the processor for the device."""

        # TODO: Check if this is correct
        match self:
            case self.IPHONE_15_PRO:
                return "A17 Bionic"
            case self.IPHONE_15_PRO_MAX:
                return "A17 Bionic"
            case self.IPHONE_15:
                return "A16 Bionic"
            case self.IPHONE_15_PLUS:
                return "A16 Bionic"
            case self.IPHONE_14_PRO_MAX:
                return "A16 Bionic"
            case self.IPHONE_14_PRO:
                return "A15 Bionic"
            case self.IPHONE_14:
                return "A15 Bionic"
            case self.IPHONE_13_PRO:
                return "A14 Bionic"
