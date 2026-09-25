"""
nnU-Net v2 trainer with TRADES-style robust loss (adversarial KL + gradient difference penalty).
"""
from __future__ import annotations

from typing import Dict, List, Tuple
from time import time

import numpy as np
import torch
from torch import autocast
from torch import distributed as dist

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.loss.robust_loss_utils import (adversarial_kl_voxel, boundary_band_mask, detached_probs_from_logits, fgsm_trades, gradient_difference_penalty, mi_fgsm_trades, pgd_trades, region_target_for_robust, safe_weighted_mean, target_to_float_mask)
from nnunetv2.utilities.helpers import dummy_context

# Default phase lengths for the four-stage robust schedule
# (clean warm-up → FGSM → MI-FGSM → PGD). Total epochs = sum of these four.
# Defaults match the values used for the published checkpoints; override per run via
# the `robust_loss` block in plans.json or dataset.json.
DEFAULT_WARM_ADV_EPOCHS = 1000
DEFAULT_FGSM_EPOCHS = 1000
DEFAULT_MI_FGSM_EPOCHS = 1000
DEFAULT_PGD_EPOCHS = 1000

# Perturbation budget bounds (L_inf, fraction of input range).
# The schedule currently applies `eps_max` as a constant budget after warm-up;
# `eps_min` is the validated lower bound (reserved for future warmup interpolation).
# A run with `eps_min > eps_max` is invalid and rejected at construction time.
DEFAULT_EPS_MIN = 0.01
DEFAULT_EPS_MAX = 0.05

DEFAULT_LAMBDA_ADV_BASE = 1.0
DEFAULT_LAMBDA_GP_BASE = 1.0
DEFAULT_PGD_STEPS = 3
DEFAULT_STEP_SCALE = 1.0
DEFAULT_MI_STEPS = 10
DEFAULT_MOMENTUM = 0.9
DEFAULT_PGD_RESTARTS = 1
DEFAULT_MAX_ADV_TO_CLEAN_RATIO = 3.0
DEFAULT_MAX_GP_TO_CLEAN_RATIO = 3.0
DEFAULT_RATIO_EMA_MOMENTUM = 0.95

# Numerical guards used across loss composition.
_RATIO_FLOOR = 1e-8
_LOG_EPS = 1e-12
_PROB_CLAMP = 1e-4  # avoids log(0) in the per-region KL term
_GRAD_CLIP_MAX_NORM = 12.0
_SIGMOID_DECISION_THRESHOLD = 0.5


class nnUNetTrainerRobustLoss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        robust_cfg = self._get_robust_config()
        self.warm_adv_epochs        = DEFAULT_WARM_ADV_EPOCHS
        self.fgsm_epochs            = DEFAULT_FGSM_EPOCHS
        self.mi_fgsm_epochs         = DEFAULT_MI_FGSM_EPOCHS
        self.pgd_epochs             = DEFAULT_PGD_EPOCHS
        self.eps_min                = DEFAULT_EPS_MIN
        self.eps_max                = DEFAULT_EPS_MAX
        self.lambda_adv_base        = DEFAULT_LAMBDA_ADV_BASE
        self.lambda_gp_base         = DEFAULT_LAMBDA_GP_BASE
        self.pgd_steps              = DEFAULT_PGD_STEPS
        self.step_scale             = DEFAULT_STEP_SCALE
        self.mi_steps               = DEFAULT_MI_STEPS
        self.momentum               = DEFAULT_MOMENTUM
        self.pgd_restarts           = DEFAULT_PGD_RESTARTS
        self.max_adv_to_clean_ratio = DEFAULT_MAX_ADV_TO_CLEAN_RATIO
        self.max_gp_to_clean_ratio  = DEFAULT_MAX_GP_TO_CLEAN_RATIO
        self.ratio_ema_momentum     = DEFAULT_RATIO_EMA_MOMENTUM

        # Allow lightweight per-experiment overrides via plans/dataset json.
        self.warm_adv_epochs        = int(robust_cfg.get("warm_adv_epochs", self.warm_adv_epochs))
        self.fgsm_epochs            = int(robust_cfg.get("fgsm_epochs", self.fgsm_epochs))
        self.mi_fgsm_epochs         = int(robust_cfg.get("mi_fgsm_epochs", self.mi_fgsm_epochs))
        self.pgd_epochs             = int(robust_cfg.get("pgd_epochs", self.pgd_epochs))
        self.eps_min                = float(robust_cfg.get("eps_min", self.eps_min))
        self.eps_max                = float(robust_cfg.get("eps_max", self.eps_max))
        self.lambda_adv_base        = float(robust_cfg.get("lambda_adv_base", self.lambda_adv_base))
        self.lambda_gp_base         = float(robust_cfg.get("lambda_gp_base", self.lambda_gp_base))
        self.pgd_steps              = int(robust_cfg.get("pgd_steps", self.pgd_steps))
        self.step_scale             = float(robust_cfg.get("step_scale", self.step_scale))
        self.mi_steps               = int(robust_cfg.get("mi_steps", self.mi_steps))
        self.momentum               = float(robust_cfg.get("momentum", self.momentum))
        self.pgd_restarts           = int(robust_cfg.get("pgd_restarts", self.pgd_restarts))
        self.max_adv_to_clean_ratio = float(robust_cfg.get("max_adv_to_clean_ratio", self.max_adv_to_clean_ratio))
        self.max_gp_to_clean_ratio  = float(robust_cfg.get("max_gp_to_clean_ratio", self.max_gp_to_clean_ratio))
        self.ratio_ema_momentum     = float(robust_cfg.get("ratio_ema_momentum", self.ratio_ema_momentum))

        if self.eps_min > self.eps_max:
            raise ValueError(
                f"Invalid perturbation schedule: eps_min ({self.eps_min}) > eps_max ({self.eps_max}). "
                "eps_min must be a non-negative lower bound for the L_inf budget; eps_max is the upper bound."
            )
        if self.eps_min < 0.0:
            raise ValueError(f"eps_min must be non-negative, got {self.eps_min}.")

        self.num_epochs = (
            int(self.warm_adv_epochs)
            + int(self.fgsm_epochs)
            + int(self.mi_fgsm_epochs)
            + int(self.pgd_epochs)
        )

        self._eps_curr              = float(self.eps_min)
        self._lambda_adv_epoch      = float(0.0)
        self._lambda_gp_epoch       = float(0.0)
        self._use_fgsm              = bool(False)
        self._use_mi_fgsm           = bool(False)
        self._phase_name            = str("CLEAN")
        self._robust_log_r_adv      = float(0.0)
        self._robust_log_r_gp       = float(0.0)
        self._robust_log_train_dice = float(0.0)
        self._adv_ratio_ema         = float(1.0)
        self._gp_ratio_ema          = float(1.0)

        # Custom on_epoch_end line replaces default nnUNet train_loss/val_loss/Pseudo dice/Epoch time/Yayy prints
        self.log_standard_epoch_metrics = False

    def _get_robust_config(self) -> Dict:
        # nnUNetTrainer keeps the plans dict on plans_manager, not self.plans
        plans_dict = self.plans_manager.plans if hasattr(self, "plans_manager") else {}
        plans_cfg = plans_dict.get("robust_loss", {}) if isinstance(plans_dict, dict) else {}
        ds_cfg = self.dataset_json.get("robust_loss", {}) if isinstance(self.dataset_json, dict) else {}
        out: Dict = {}
        if isinstance(plans_cfg, dict):
            out.update(plans_cfg)
        if isinstance(ds_cfg, dict):
            out.update(ds_cfg)
        return out

    def _do_i_compile(self):
        """
        Robust training uses input gradients with create_graph=True (gradient-difference penalty and TRADES-style
        objectives). torch.compile's AOT backward uses donated buffers and rejects second-order autograd.
        """
        if super()._do_i_compile():
            self.print_to_log_file(
                "INFO: torch.compile disabled for nnUNetTrainerRobustLoss: input-gradients with create_graph=True "
                "(gradient-difference penalty) are incompatible with compiled donated-buffer backward."
            )
        return False

    def _build_loss(self):
        return super()._build_loss()

    def _phase_progress(self, current: int, start: int, length: int) -> float:
        if length <= 0:
            return 1.0
        return min(1.0, max(0.0, float(current - start) / float(length)))

    def _safe_ratio(self, num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
        return num / torch.clamp(den, min=_RATIO_FLOOR)

    def _foreground_tp_fp_fn(self, output: torch.Tensor, target: torch.Tensor):
        """Hard tp/fp/fn for foreground Dice (same logic as validation_step)."""
        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > _SIGMOID_DECISION_THRESHOLD).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target = target.clone()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
        tp_hard = tp.detach().float().cpu()
        fp_hard = fp.detach().float().cpu()
        fn_hard = fn.detach().float().cpu()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
        return tp_hard, fp_hard, fn_hard

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

        epoch_1based = self.current_epoch + 1
        warm = max(int(self.warm_adv_epochs), 0)
        f_e = max(int(self.fgsm_epochs), 0)
        mi_e = max(int(self.mi_fgsm_epochs), 0)
        pgd_e = max(int(self.pgd_epochs), 0)
        pgd_e_safe = max(pgd_e, 1)

        fgsm_end = warm + f_e
        mi_end = fgsm_end + mi_e
        pgd_end = mi_end + pgd_e

        eps_curr = self.eps_max

        if epoch_1based <= warm:
            warm_adv_factor = 0.0
            warm_gp_factor = 0.0
            use_fgsm = False
            use_mi_fgsm = False
            phase_name = "CLEAN"
        elif f_e > 0 and epoch_1based <= fgsm_end:
            fgsm_progress = self._phase_progress(epoch_1based, warm, f_e)
            warm_adv_factor = 1.0
            warm_gp_factor = 0.8 + 0.2 * fgsm_progress
            use_fgsm = True
            use_mi_fgsm = False
            phase_name = "FGSM"
        elif mi_e > 0 and epoch_1based <= mi_end:
            mi_progress = self._phase_progress(epoch_1based, fgsm_end, mi_e)
            warm_adv_factor = 1.0
            warm_gp_factor = 0.8 + 0.2 * mi_progress
            use_fgsm = False
            use_mi_fgsm = True
            phase_name = "MI-FGSM"
        else:
            pgd_progress = min(
                1.0, max(0.0, (epoch_1based - mi_end) / float(pgd_e_safe))
            )
            warm_adv_factor = 0.3 + 0.4 * pgd_progress
            warm_gp_factor = 0.1 + 0.4 * pgd_progress
            use_fgsm = False
            use_mi_fgsm = False
            phase_name = f"PGD-{max(self.pgd_steps, 1)}" if pgd_e > 0 else "ADV-EPS"

        self._eps_curr = float(eps_curr)
        self._lambda_adv_epoch = float(self.lambda_adv_base * warm_adv_factor)
        self._lambda_gp_epoch = float(self.lambda_gp_base * warm_gp_factor)
        self._use_fgsm = use_fgsm
        self._use_mi_fgsm = use_mi_fgsm
        self._phase_name = phase_name

    def on_train_epoch_end(self, train_outputs: List[dict]):
        # Build aggregates even when this rank received zero batches: in DDP all ranks
        # must participate in the all_reduce calls below or the world will deadlock.
        if len(train_outputs) == 0:
            self.print_to_log_file("WARNING: on_train_epoch_end received empty train_outputs on this rank; "
                                   "contributing zeros to DDP aggregation.")
            zero = torch.zeros((), device=self.device, dtype=torch.float32)
            loss_sum = zero.clone()
            l_clean_sum = zero.clone()
            adv_sum = zero.clone()
            gp_sum = zero.clone()
            steps = zero.clone()
            num_classes = self.label_manager.num_segmentation_heads
            tp_dim = num_classes if self.label_manager.has_regions else max(num_classes - 1, 1)
            tp = torch.zeros(tp_dim, device=self.device, dtype=torch.float32)
            fp = torch.zeros(tp_dim, device=self.device, dtype=torch.float32)
            fn = torch.zeros(tp_dim, device=self.device, dtype=torch.float32)
        else:
            loss_sum = torch.stack([o['loss'] for o in train_outputs]).sum().to(self.device)
            l_clean_sum = torch.stack([o['l_clean'] for o in train_outputs]).sum().to(self.device)
            adv_sum = torch.stack([o['adv'] for o in train_outputs]).sum().to(self.device)
            gp_sum = torch.stack([o['gp'] for o in train_outputs]).sum().to(self.device)
            steps = torch.tensor(float(len(train_outputs)), device=self.device, dtype=torch.float32)

            tp = torch.stack([o['tp_hard'] for o in train_outputs]).sum(0).to(self.device)
            fp = torch.stack([o['fp_hard'] for o in train_outputs]).sum(0).to(self.device)
            fn = torch.stack([o['fn_hard'] for o in train_outputs]).sum(0).to(self.device)

        if self.is_ddp:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(l_clean_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(adv_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(gp_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(steps, op=dist.ReduceOp.SUM)
            dist.all_reduce(tp, op=dist.ReduceOp.SUM)
            dist.all_reduce(fp, op=dist.ReduceOp.SUM)
            dist.all_reduce(fn, op=dist.ReduceOp.SUM)

        loss_here = float((loss_sum / torch.clamp(steps, min=1.0)).item())
        mean_l_clean = float((l_clean_sum / torch.clamp(steps, min=1.0)).item())
        mean_adv = float((adv_sum / torch.clamp(steps, min=1.0)).item())
        mean_gp = float((gp_sum / torch.clamp(steps, min=1.0)).item())

        self.logger.log('train_losses', loss_here, self.current_epoch)

        global_dc_per_class = (2 * tp) / torch.clamp(2 * tp + fp + fn, min=_RATIO_FLOOR)
        train_mean_fg_dice = float(torch.nanmean(global_dc_per_class).item())

        la = self._lambda_adv_epoch
        lg = self._lambda_gp_epoch
        self._robust_log_r_adv = float((la * mean_adv) / (mean_l_clean + _LOG_EPS))
        self._robust_log_r_gp = float((lg * mean_gp) / (mean_l_clean + _LOG_EPS))
        self._robust_log_train_dice = train_mean_fg_dice

    def on_epoch_end(self):
        try:
            val_dice = float(self.logger.get_value('mean_fg_dice', step=-1))
        except (IndexError, KeyError, TypeError, ValueError):
            val_dice = float("nan")
        epoch_time_s = float(
            np.round(time() - self.logger.get_value('epoch_start_timestamps', step=-1), decimals=2)
        )
        self.print_to_log_file(
            f"Epoch {self.current_epoch} [{self._phase_name}] | eps={self._eps_curr:.4f} | "
            f"r_adv: {self._robust_log_r_adv:.3f} r_gp: {self._robust_log_r_gp:.3f} | "
            f"Train Dice: {self._robust_log_train_dice:.4f} | Val Dice: {val_dice:.4f} | "
            f"Epoch time: {epoch_time_s} s"
        )
        super().on_epoch_end()

    def _forward_logits_robust(self, x_in: torch.Tensor) -> torch.Tensor:
        """Force fp32 for the adversarial forward.

        AMP softmax/sigmoid + create_graph=True (gradient-difference penalty) and
        TRADES KL on perturbed inputs overflow under fp16, silently producing NaN
        gradients that bypass GradScaler.
        """
        ctx = autocast(self.device.type, enabled=False) if self.device.type == 'cuda' else dummy_context()
        with ctx:
            out = self.network(x_in)
        lo = out[0] if isinstance(out, (list, tuple)) else out
        return lo.float()

    def _build_adv_example(
        self,
        x_f: torch.Tensor,
        eps: float,
        p_clean: torch.Tensor,
        binary: bool,
        multilabel_sigmoid: bool = False,
    ) -> torch.Tensor:
        fwd = self._forward_logits_robust
        if self._use_fgsm:
            return fgsm_trades(fwd, x_f, eps, p_clean, binary, multilabel_sigmoid=multilabel_sigmoid)
        if self._use_mi_fgsm:
            return mi_fgsm_trades(
                fwd,
                x_f,
                eps,
                p_clean,
                binary,
                mi_steps=self.mi_steps,
                momentum=self.momentum,
                multilabel_sigmoid=multilabel_sigmoid,
            )
        step = (self.step_scale * eps) / float(max(self.pgd_steps, 1))
        return pgd_trades(
            fwd,
            x_f,
            eps,
            step,
            max(self.pgd_steps, 1),
            p_clean,
            binary,
            restarts=max(self.pgd_restarts, 1),
            multilabel_sigmoid=multilabel_sigmoid,
        )

    def _compute_adv_and_gp_terms(
        self,
        data_float: torch.Tensor,
        logits_clean: torch.Tensor,
        target: torch.Tensor | List[torch.Tensor],
        la: float,
        lg: float) -> Tuple[torch.Tensor, torch.Tensor]:

        num_classes = logits_clean.shape[1]
        # `binary` here means "single-channel sigmoid head" (routes through
        # probs_from_logits_binary). Standard nnUNet softmax outputs >=2 channels
        # even for 2-class problems, so this is only true for 1-channel heads.
        binary = num_classes == 1
        t0 = target[0] if isinstance(target, list) else target
        ignore = self.label_manager.ignore_label if self.label_manager.has_ignore_label else None
        if self.label_manager.has_regions:
            y_for_gp = region_target_for_robust(t0, self.label_manager.has_ignore_label)
            y_band = y_for_gp.mean(dim=1, keepdim=True)
        else:
            y_for_gp = target_to_float_mask(t0, num_classes, binary, ignore)
            # nnUNet non-region targets are [B,1,*spatial]; slice the channel dim
            # explicitly for both 2D ([B,1,H,W]) and 3D ([B,1,D,H,W]) configs.
            y_band = t0[:, 0:1].float() if t0.ndim >= 4 else t0.float()
        x_f = data_float

        was_training = self.network.training
        self.network.eval()
        try:
            if self.label_manager.has_regions:
                p_clean_for_attack = torch.sigmoid(logits_clean).detach()
                x_adv = self._build_adv_example(
                    x_f, self._eps_curr, p_clean_for_attack, binary=False, multilabel_sigmoid=True
                )
            else:
                p_clean = detached_probs_from_logits(logits_clean, binary)
                x_adv = self._build_adv_example(x_f, self._eps_curr, p_clean, binary)
        finally:
            self.network.train(was_training)

        if self.label_manager.has_regions:
            p_clean = torch.sigmoid(logits_clean).detach()
            logits_adv = self._forward_logits_robust(x_adv)
            p_adv = torch.sigmoid(logits_adv)
            p_c = torch.clamp(p_clean, _PROB_CLAMP, 1.0 - _PROB_CLAMP)
            p_a = torch.clamp(p_adv, _PROB_CLAMP, 1.0 - _PROB_CLAMP)
            kl_regions = p_c * torch.log(p_c / p_a) + (1.0 - p_c) * torch.log((1.0 - p_c) / (1.0 - p_a))
            kl_vox = kl_regions.sum(dim=1, keepdim=True)
            band = boundary_band_mask(y_band)
            adv_t = safe_weighted_mean(kl_vox, band) if la > 0 else logits_clean.new_zeros(())
            gp_t = gradient_difference_penalty(
                self._forward_logits_robust,
                x_f,
                x_adv,
                y_for_gp,
                band_only=True,
                binary=False,
                multilabel_sigmoid=True,
            ) if lg > 0 else logits_clean.new_zeros(())
            return adv_t, gp_t

        p_clean = detached_probs_from_logits(logits_clean, binary)

        if la > 0:
            logits_adv = self._forward_logits_robust(x_adv)
            kl_vox = adversarial_kl_voxel(p_clean, logits_adv, binary)
            mask = boundary_band_mask(y_band)
            adv_t = safe_weighted_mean(kl_vox, mask)
        else:
            adv_t = logits_clean.new_zeros(())

        if lg > 0:
            gp_t = gradient_difference_penalty(
                self._forward_logits_robust,
                x_f,
                x_adv,
                y_for_gp,
                band_only=True,
                binary=binary,
            )
        else:
            gp_t = logits_clean.new_zeros(())

        return adv_t, gp_t

    def _compose_total_loss(
        self,
        l_clean: torch.Tensor,
        adv_t: torch.Tensor,
        gp_t: torch.Tensor,
        la: float,
        lg: float) -> torch.Tensor:

        clean = l_clean.float().detach()
        clean_den = torch.clamp(clean, min=_RATIO_FLOOR)
        adv_ratio_now = float(self._safe_ratio(abs(la) * adv_t.float().detach(), clean_den).item()) if la > 0 else 1.0
        gp_ratio_now = float(self._safe_ratio(abs(lg) * gp_t.float().detach(), clean_den).item()) if lg > 0 else 1.0

        # In DDP, average per-batch ratios across ranks before updating the EMA so all
        # ranks apply the same lambda clamp and compose identical effective losses.
        if self.is_ddp:
            ratio_t = torch.tensor([adv_ratio_now, gp_ratio_now], device=self.device, dtype=torch.float32)
            dist.all_reduce(ratio_t, op=dist.ReduceOp.SUM)
            world_size = float(dist.get_world_size())
            adv_ratio_now = float(ratio_t[0].item()) / world_size
            gp_ratio_now = float(ratio_t[1].item()) / world_size

        m = float(np.clip(self.ratio_ema_momentum, 0.0, 0.9999))
        self._adv_ratio_ema = m * self._adv_ratio_ema + (1.0 - m) * adv_ratio_now
        self._gp_ratio_ema = m * self._gp_ratio_ema + (1.0 - m) * gp_ratio_now

        la_eff = float(la)
        lg_eff = float(lg)
        if la > 0 and self._adv_ratio_ema > self.max_adv_to_clean_ratio:
            la_eff *= self.max_adv_to_clean_ratio / max(self._adv_ratio_ema, _RATIO_FLOOR)
        if lg > 0 and self._gp_ratio_ema > self.max_gp_to_clean_ratio:
            lg_eff *= self.max_gp_to_clean_ratio / max(self._gp_ratio_ema, _RATIO_FLOOR)

        return l_clean.float() + la_eff * adv_t.float() + lg_eff * gp_t.float()

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            l_clean = self.loss(output, target)

        out_dice = output[0] if isinstance(output, (list, tuple)) else output
        t_dice = target[0] if isinstance(target, list) else target
        tp_hard, fp_hard, fn_hard = self._foreground_tp_fp_fn(out_dice, t_dice)

        logits_clean = out_dice.float().detach()
        data_float = data.float()

        la = self._lambda_adv_epoch
        lg = self._lambda_gp_epoch

        adv_t = logits_clean.new_zeros(())
        gp_t = logits_clean.new_zeros(())

        if la <= 0 and lg <= 0:
            l = l_clean
        else:
            adv_t, gp_t = self._compute_adv_and_gp_terms(data_float, logits_clean, target, la, lg)
            l = self._compose_total_loss(l_clean, adv_t, gp_t, la, lg)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), _GRAD_CLIP_MAX_NORM)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), _GRAD_CLIP_MAX_NORM)
            self.optimizer.step()

        lc = l_clean.detach().float().mean()
        return {
            'loss'      : l.detach().float().cpu(),
            'l_clean'   : lc.detach().float().cpu(),
            'adv'       : adv_t.detach().float().cpu(),
            'gp'        : gp_t.detach().float().cpu(),
            'tp_hard'   : tp_hard,
            'fp_hard'   : fp_hard,
            'fn_hard'   : fn_hard,
        }