# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional

import torch
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import AlgoConfig, agg_loss, register_policy_loss

__all__ = ["compute_policy_loss_future_kl_kalman", "backward_scan", "forward_window_var"]

def backward_scan(z: torch.Tensor, gamma: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    EXP_LIMIT = 60.0

    bsz, seqlen = z.shape
    device = z.device
    z32 = z.float()
    log_gamma = torch.log(gamma.float().clamp_min(1e-6))

    max_abs_log = float(log_gamma.abs().max())
    if max_abs_log > 0:
        chunk_size = max(1, min(chunk_size, int(EXP_LIMIT / max_abs_log)))

    out = torch.zeros_like(z32)
    carry = torch.zeros(bsz, device=device, dtype=torch.float32)
    zero_col = torch.zeros(bsz, 1, device=device, dtype=torch.float32)

    for start in reversed(range(0, seqlen, chunk_size)):
        end = min(start + chunk_size, seqlen)
        lg = log_gamma[:, start:end]

        cum = torch.cat([zero_col, torch.cumsum(lg, dim=1)], dim=1)

        weighted = torch.exp(cum[:, :-1]) * z32[:, start:end]
        tail = torch.exp(cum[:, -1]) * carry

        suffix = torch.flip(torch.cumsum(torch.flip(weighted, [1]), dim=1), [1])
        chunk_out = torch.exp(-cum[:, :-1]) * (suffix + tail.unsqueeze(1))

        out[:, start:end] = chunk_out
        carry = chunk_out[:, 0]

    return out

def forward_window_var(
    signal: torch.Tensor,
    valid: torch.Tensor,
    window: int,
    fallback: torch.Tensor,
) -> torch.Tensor:
    bsz, seqlen = signal.shape
    device = signal.device
    masked = (signal * valid).float()
    zero_col = torch.zeros(bsz, 1, device=device, dtype=torch.float32)

    c1 = torch.cat([zero_col, torch.cumsum(masked, dim=1)], dim=1)
    c2 = torch.cat([zero_col, torch.cumsum(masked * masked, dim=1)], dim=1)
    cn = torch.cat([zero_col, torch.cumsum(valid.float(), dim=1)], dim=1)

    pos = torch.arange(seqlen, device=device)
    lo = torch.clamp(pos + 1, max=seqlen)
    hi = torch.clamp(pos + 1 + window, max=seqlen)

    s1 = c1[:, hi] - c1[:, lo]
    s2 = c2[:, hi] - c2[:, lo]
    n = cn[:, hi] - cn[:, lo]

    n_safe = n.clamp_min(1.0)
    mean = s1 / n_safe
    var = (s2 / n_safe - mean * mean).clamp_min(0.0)
    return torch.where(n >= 2.0, var, fallback)

@register_policy_loss("future_kl_kalman")
def compute_policy_loss_future_kl_kalman(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
):
    assert config is not None
    assert not isinstance(config, AlgoConfig)

    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)
    assert clip_ratio_c > 1.0, f"clip_ratio_c must exceed 1.0, got {clip_ratio_c}"

    pl = config.policy_loss

    decay_rate = pl.get("decay_rate", 32.0)
    gamma_target = float(pl.get("gamma_target", 2.0 ** (-1.0 / decay_rate)))
    var_window = int(pl.get("var_window", 64))
    gamma_min = float(pl.get("gamma_min", 0.5))
    gamma_max = float(pl.get("gamma_max", 0.9995))
    chunk_size = int(pl.get("chunk_size", 64))
    fkl_clip_ratio = float(pl.get("future_kl_clip_ratio", 0.2))
    clip_high_only = bool(pl.get("future_kl_clip_high_only", False))
    safety_thresh = float(pl.get("safety_thresh", 3.0))
    eps = 1e-8

    device = log_prob.device
    resp_mask_f = response_mask.to(log_prob.dtype)
    resp_mask_b = response_mask.bool()

    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    filter_threshold = math.log(clip_ratio_c)
    participation = (negative_approx_kl <= filter_threshold)
    kl_response_premask = negative_approx_kl * resp_mask_f
    kl_response = kl_response_premask * participation.to(log_prob.dtype)
    contributing = (response_mask * participation).to(log_prob.dtype)

    n_contrib = contributing.sum(dim=1, keepdim=True).clamp_min(1.0)
    row_mean = (kl_response.float() * contributing).sum(dim=1, keepdim=True) / n_contrib
    row_var = (((kl_response.float() - row_mean) ** 2) * contributing).sum(dim=1, keepdim=True) / n_contrib
    row_var = row_var.clamp_min(0.0).expand(-1, log_prob.shape[1])

    r_obs = forward_window_var(kl_response, contributing, var_window, row_var) + eps

    r_valid = r_obs[resp_mask_b]
    if r_valid.numel() > 0:
        r_median = torch.median(r_valid)
    else:
        r_median = torch.tensor(eps, device=device, dtype=torch.float32)

    if float(r_median) <= eps:

        gamma = torch.full_like(r_obs, gamma_target)
        q_process = torch.tensor(0.0, device=device, dtype=torch.float32)
    else:
        q_process = gamma_target * r_median / (1.0 - gamma_target)
        gamma = (q_process / (q_process + r_obs)).clamp(gamma_min, gamma_max)

    gamma = torch.where(resp_mask_b, gamma, torch.full_like(gamma, gamma_target))

    future_kl = backward_scan(kl_response, gamma, chunk_size=chunk_size).to(log_prob.dtype)
    future_kl = torch.clamp(future_kl, min=-20.0, max=20.0)

    raw_influence = torch.exp(future_kl)
    if fkl_clip_ratio != 0.0:
        upper_bound = 1.0 + fkl_clip_ratio
        lower_bound = 1.0 if clip_high_only else 1.0 - fkl_clip_ratio
    else:
        upper_bound, lower_bound = 10.0, 0.0
    influence_weights = torch.clamp(raw_influence, min=lower_bound, max=upper_bound).detach()

    mask_neg_high_is = (advantages < 0) & (ratio > safety_thresh)
    influence_weights = torch.where(
        mask_neg_high_is, torch.clamp(influence_weights, min=0.8, max=1.0), influence_weights
    )

    weighted_advantages = advantages * influence_weights

    pg_losses1 = -weighted_advantages * ratio
    pg_losses2 = -weighted_advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -weighted_advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    lower_clip_mask = (advantages < 0) & (clip_pg_losses1 > pg_losses3) & resp_mask_b
    seq_valid = (lower_clip_mask.sum(dim=1) <= 1).unsqueeze(1)
    final_mask = (resp_mask_b & seq_valid).to(log_prob.dtype)

    pg_losses = torch.where(weighted_advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=final_mask, loss_agg_mode=loss_agg_mode)

    gamma_valid = gamma[resp_mask_b]
    iw_valid = influence_weights[resp_mask_b]
    raw_valid = raw_influence[resp_mask_b]

    def _stat(fn, tensor, default=0.0):
        return float(fn(tensor)) if tensor.numel() > 0 else default

    clip_frac_upper = verl_F.masked_mean((influence_weights >= upper_bound - 1e-7).float(), response_mask)
    clip_frac_lower = verl_F.masked_mean((influence_weights <= lower_bound + 1e-7).float(), response_mask)

    metrics = {

        "actor/fipok_gamma_mean": _stat(torch.mean, gamma_valid),
        "actor/fipok_gamma_std": _stat(torch.std, gamma_valid),
        "actor/fipok_gamma_median": _stat(torch.median, gamma_valid),
        "actor/fipok_gamma_min": _stat(torch.min, gamma_valid),
        "actor/fipok_gamma_max": _stat(torch.max, gamma_valid),
        "actor/fipok_gamma_target": gamma_target,

        "actor/fipok_eff_horizon_mean": _stat(lambda x: torch.mean(1.0 / (1.0 - x).clamp_min(1e-6)), gamma_valid),

        "actor/fipok_R_median": float(r_median),
        "actor/fipok_Q": float(q_process),

        "actor/influence_weights_mean": _stat(torch.mean, iw_valid),
        "actor/influence_weights_min": _stat(torch.min, iw_valid),
        "actor/influence_weights_max": _stat(torch.max, iw_valid),
        "actor/influence_weights_mean_raw": _stat(torch.mean, raw_valid),
        "actor/raw_influence_weights_min": _stat(torch.min, raw_valid),
        "actor/raw_influence_weights_max": _stat(torch.max, raw_valid),
        "actor/IW_upper_clip_ratio": float(clip_frac_upper),
        "actor/IW_lower_clip_ratio": float(clip_frac_lower),
        "actor/IW_overall_clip_ratio": float(clip_frac_upper + clip_frac_lower),
    }

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, metrics
