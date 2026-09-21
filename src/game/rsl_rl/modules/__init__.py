# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Building blocks for neural models."""

from .gated_mha import GatedMHA

__all__ = [
    "GatedMHA",
]
