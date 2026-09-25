"""Recurrent Convolutional Layer (RCL) building block.

Additive RCL formulation from R2U-Net (Alom et al., 2018):
    h_0 = Nonlin(Norm(Dropout(Conv_feed(x))))
    h_t = Nonlin(Norm(Dropout(Conv_feed(x) + Conv_recurrent(h_{t-1}))))

Both Conv_feed and Conv_recurrent share weights across all time steps.
"""

from typing import List, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list


class RecurrentConvDropoutNormReLU(nn.Module):
    """Recurrent convolutional layer that unrolls over `time_steps` iterations.

    Drop-in replacement for ConvDropoutNormReLU with shared-weight temporal
    recurrence. When time_steps=1, degenerates to a standard conv block.
    """

    def __init__(
        self,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]],
        time_steps: int = 2,
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        nonlin_first: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.time_steps = max(1, min(time_steps, 4))

        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        # Build the post-conv op sequence once (match ConvDropoutNormReLU pattern).
        # Create feed-forward conv first, then wrap with norm/dropout/nonlin.
        conv = conv_op(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding=[(i - 1) // 2 for i in kernel_size],
            dilation=1,
            bias=conv_bias,
        )
        ops: List[nn.Module] = [conv]

        if dropout_op is not None:
            self.dropout = dropout_op(**(dropout_op_kwargs or {}))
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **(norm_op_kwargs or {}))
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**(nonlin_kwargs or {}))
            ops.append(self.nonlin)

        if nonlin_first and norm_op is not None and nonlin is not None:
            ops[-1], ops[-2] = ops[-2], ops[-1]

        self.all_modules = nn.Sequential(*ops)

        # Recurrent convolution — shared across time steps
        self.conv_recurrent = conv_op(
            output_channels,
            output_channels,
            kernel_size,
            1,  # stride=1 for recurrent path
            padding=[(i - 1) // 2 for i in kernel_size],
            dilation=1,
            bias=conv_bias,
        )

    @property
    def conv_feed(self) -> nn.Module:
        """The feed-forward convolution (first module in all_modules)."""
        return self.all_modules[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial projection: Conv -> [Dropout] -> Norm -> Nonlin
        h = self.all_modules(x)

        if self.time_steps == 1:
            return h

        # Recurrent steps: add recurrent path, then re-apply norm/nonlin ops
        # (skip the initial conv — ops[1:] are dropout, norm, nonlin)
        ops = self.all_modules[1:] if len(self.all_modules) > 1 else []
        for _ in range(1, self.time_steps):
            recurrent = self.conv_recurrent(h)
            # Add feed-forward contribution + recurrent at the feature level
            f_raw = self.conv_feed(x)
            combined = f_raw + recurrent
            h = combined
            for op in ops:
                h = op(h)

        return h

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        output_size = [i // j for i, j in zip(input_size, self.stride)]
        return np.prod([self.output_channels, *output_size], dtype=np.int64)
