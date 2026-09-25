"""nnU-Net trainers with recurrent convolutional encoder.

Provides:
- nnUNetTrainerRecurrentEncoder: standard nnU-Net training with recurrent encoder
- nnUNetTrainerRobustLossRecurrentEncoder: TRADES adversarial training + recurrent encoder
"""

from typing import List, Tuple, Union

import pydoc
import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from training.nnUNetTrainerRobustLoss import nnUNetTrainerRobustLoss

from mri_prostate_seg.architectures.recurrent_unet import RecurrentEncoderUNet


class nnUNetTrainerRecurrentEncoder(nnUNetTrainer):
    """Standard nnU-Net trainer that uses RecurrentEncoderUNet architecture.

    torch.compile is disabled because the recurrent control flow (shared-weight
    conv called at each time step with different inputs) triggers inductor
    backend bugs in some configurations (e.g. T=2 with 3D Conv3d on small patches).
    """

    def _do_i_compile(self):
        return False

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        # Resolve string class references (conv_op, norm_op, dropout_op, nonlin)
        # that get_network_from_plans normally handles.
        resolved_kwargs = dict(arch_init_kwargs)
        for key in arch_init_kwargs_req_import:
            if key in resolved_kwargs and resolved_kwargs[key] is not None:
                resolved_kwargs[key] = pydoc.locate(resolved_kwargs[key])

        network = RecurrentEncoderUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            **resolved_kwargs,
        )
        network.apply(network.initialize)
        return network


class nnUNetTrainerRobustLossRecurrentEncoder(
    nnUNetTrainerRecurrentEncoder,
    nnUNetTrainerRobustLoss,
):
    """TRADES adversarial training with recurrent convolutional encoder.

    MRO: nnUNetTrainerRecurrentEncoder → nnUNetTrainerRobustLoss → nnUNetTrainer
    build_network_architecture() comes from nnUNetTrainerRecurrentEncoder.
    train_step() adversarial logic comes from nnUNetTrainerRobustLoss.
    """
