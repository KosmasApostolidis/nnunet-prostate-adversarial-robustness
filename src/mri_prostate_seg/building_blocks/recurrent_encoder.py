"""Recurrent Encoder — replaces StackedConvBlocks depth with RCL time steps.

Each encoder stage applies one RecurrentConvDropoutNormReLU with
T = n_conv_per_stage[s] (from plans), instead of stacking N sequential
ConvDropoutNormReLU blocks. The skip-connection contract (List[Tensor]
with one entry per stage) is identical to PlainConvEncoder, so UNetDecoder
can be reused unchanged.
"""

from typing import List, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import (
    get_matching_pool_op,
    maybe_convert_scalar_to_list,
)

from mri_prostate_seg.building_blocks.recurrent_conv import (
    RecurrentConvDropoutNormReLU,
)


class RecurrentEncoder(nn.Module):
    """Encoder where each stage uses a recurrent conv block.

    Interface is compatible with PlainConvEncoder — stores conv_op, norm_op,
    etc. attributes that UNetDecoder reads. Returns List[Tensor] for skip
    connections when return_skips=True.
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        nonlin_first: bool = False,
        pool: str = "conv",
    ):
        super().__init__()

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        assert len(kernel_sizes) == n_stages
        assert len(n_conv_per_stage) == n_stages
        assert len(features_per_stage) == n_stages
        assert len(strides) == n_stages

        stages = []
        for s in range(n_stages):
            stage_modules = []

            if pool == "max" or pool == "avg":
                if (
                    (isinstance(strides[s], int) and strides[s] != 1)
                    or (
                        isinstance(strides[s], (tuple, list))
                        and any(i != 1 for i in strides[s])
                    )
                ):
                    stage_modules.append(
                        get_matching_pool_op(conv_op, pool_type=pool)(
                            kernel_size=strides[s], stride=strides[s]
                        )
                    )
                conv_stride = 1
            elif pool == "conv":
                conv_stride = strides[s]
            else:
                raise RuntimeError(f"Unknown pool type: {pool}")

            # Use T = n_conv_per_stage[s] from plans as the number of recurrent steps
            stage_modules.append(
                RecurrentConvDropoutNormReLU(
                    conv_op=conv_op,
                    input_channels=input_channels,
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    stride=conv_stride,
                    time_steps=n_conv_per_stage[s],
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    nonlin_first=nonlin_first,
                )
            )
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # Stored for decoder compatibility
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, "compute_conv_feature_map_size"):
                        output += self.stages[s][-1].compute_conv_feature_map_size(
                            input_size
                        )
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [
                i // j for i, j in zip(input_size, self.strides[s])
            ]
        return output
