# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn as nn


def get_model_size(model: nn.Module, unit: str = "gb") -> float:
    """Get the size of the model in MB or GB.

    Args:
        model (nn.Module): The model to get the size of.
        unit (str): The unit to return the size in. Defaults to "gb".

    Returns:
        float: The size of the model in MB or GB.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    size_all_gb = size_all_mb / 1024
    if unit == "mb":
        return size_all_mb
    elif unit == "gb":
        return size_all_gb
    else:
        raise ValueError(f"Invalid unit: {unit}. Please specify 'mb' or 'gb'.")


def print_size_of_model(model: nn.Module, label: str = ""):
    """Print the size of the model in MB or GB.

    Args:
        model (nn.Module): The model to get the size of.
        label (str): The label of the model. Defaults to "".
    """
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model: ", label, " \t", "Size (KB):", size / 1e3)
    os.remove("temp.p")
    return size
