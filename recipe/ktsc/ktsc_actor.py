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

from __future__ import annotations

import torch

from verl import DataProto
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor

KTSC_MEASURE_KEYS = ["responses", "response_mask", "input_ids", "attention_mask", "position_ids", "old_log_probs"]

def k3_kl(new_log_prob: torch.Tensor, old_log_prob: torch.Tensor) -> torch.Tensor:
    rho = (new_log_prob - old_log_prob).clamp(min=-20.0, max=20.0)
    return torch.exp(rho) - 1.0 - rho

def ktsc_measure_indices(n: int, stride: int) -> torch.Tensor:
    stride = max(int(stride), 1)
    return torch.arange(0, n, stride)

class KtscDataParallelPPOActor(DataParallelPPOActor):

    def update_policy(self, data: DataProto):
        meta = data.meta_info
        mult = float(meta.get("ktsc_lr_mult", 1.0))
        stride = int(meta.get("ktsc_measure_stride", 0))
        temperature = meta["temperature"]

        saved = [g["lr"] for g in self.actor_optimizer.param_groups]
        if mult != 1.0:
            for g in self.actor_optimizer.param_groups:
                g["lr"] = g["lr"] * mult
        lr_effective = float(self.actor_optimizer.param_groups[0]["lr"])

        try:
            metrics = super().update_policy(data)
        finally:
            for g, lr in zip(self.actor_optimizer.param_groups, saved):
                g["lr"] = lr

        metrics["actor/ktsc_lr_mult_applied"] = [mult]
        metrics["actor/ktsc_lr_effective"] = [lr_effective]

        if stride > 0:
            r = self._ktsc_measure(data, stride=stride, temperature=temperature)
            metrics["actor/ktsc_kl_k3_post"] = [r["k3"]]
            metrics["actor/ktsc_kl_k3c_post"] = [r["k3c"]]
            metrics["actor/ktsc_kl_k1_post"] = [r["k1"]]
            metrics["actor/ktsc_rho_max"] = [r["rho_max"]]
            metrics["actor/ktsc_rho_p999"] = [r["rho_p999"]]
            metrics["actor/ktsc_measure_tokens"] = [float(r["n"])]
        return metrics

    @torch.no_grad()
    def _ktsc_measure(self, data: DataProto, stride: int, temperature) -> dict:
        has_mm = "multi_modal_inputs" in data.non_tensor_batch.keys()
        keys = [k for k in KTSC_MEASURE_KEYS if k in data.batch.keys()]
        sub = data.select(batch_keys=keys, non_tensor_batch_keys=["multi_modal_inputs"] if has_mm else [])
        idx = ktsc_measure_indices(len(sub), stride).to(data.batch["responses"].device)
        sub = sub.select_idxs(idx)

        self.actor_module.eval()
        try:
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = prepare_dynamic_batch(sub, max_token_len=max_token_len)
            else:
                micro_batches = sub.split(self.config.ppo_micro_batch_size_per_gpu)

            dev = data.batch["responses"].device
            k3_sum = torch.zeros((), dtype=torch.float64, device=dev)
            k3c_sum = torch.zeros_like(k3_sum)
            k1_sum = torch.zeros_like(k3_sum)
            n_tok = torch.zeros_like(k3_sum)
            rho_max = torch.full((), -1e9, dtype=torch.float32, device=dev)
            rho_all = []
            for mb in micro_batches:
                model_inputs = {**mb.batch, **mb.non_tensor_batch}
                _, new_lp = self._forward_micro_batch(model_inputs, temperature=temperature, calculate_entropy=False)
                old_lp = model_inputs["old_log_probs"]
                mask_b = model_inputs["response_mask"].bool()
                mask = mask_b.to(new_lp.dtype)
                rho = (new_lp - old_lp).clamp(min=-20.0, max=20.0)

                k3 = torch.exp(rho) - 1.0 - rho
                rho_c = rho.clamp(min=-5.0, max=5.0)
                k3c = torch.exp(rho_c) - 1.0 - rho_c
                k3_sum += (k3 * mask).sum().double()
                k3c_sum += (k3c * mask).sum().double()
                k1_sum += (-rho * mask).sum().double()
                n_tok += mask.sum().double()
                if mask_b.any():
                    rho_max = torch.maximum(rho_max, rho[mask_b].max().float())
                    rho_all.append(rho[mask_b].float().flatten())
        finally:
            self.actor_module.train()

        n = max(float(n_tok.item()), 1.0)
        rho_p999 = float(torch.quantile(torch.cat(rho_all), 0.999).item()) if rho_all else 0.0
        return {
            "k3": float(k3_sum.item()) / n,
            "k3c": float(k3c_sum.item()) / n,
            "k1": float(k1_sum.item()) / n,
            "n": int(n),
            "rho_max": float(rho_max.item()),
            "rho_p999": rho_p999,
        }
