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

from collections import defaultdict

import torch

from .pso_v5_ray_trainer import RayPSOV5Trainer

class RayPSOV6Trainer(RayPSOV5Trainer):

    def _apply_pso_weights(self, batch):
        advantages = batch.batch["advantages"]
        old_log_probs = batch.batch["old_log_probs"]
        response_mask = batch.batch["response_mask"]
        token_rewards = batch.batch["token_level_rewards"]
        uids = batch.non_tensor_batch["uid"]

        scores = token_rewards.sum(dim=-1)
        pso_amplify = self.config.algorithm.get("pso_amplify", 2.0)
        pso_top_ratio = self.config.algorithm.get("pso_top_ratio", 0.2)

        uid2correct_lps = defaultdict(list)
        uid2correct_idxs = defaultdict(list)
        uid2wrong_lps = defaultdict(list)
        uid2wrong_idxs = defaultdict(list)
        uid2all_idxs = defaultdict(list)

        bsz = scores.shape[0]
        for i in range(bsz):
            uid = uids[i]
            uid2all_idxs[uid].append(i)
            if scores[i] > 0:
                uid2correct_lps[uid].append(old_log_probs[i])
                uid2correct_idxs[uid].append(i)
            else:
                uid2wrong_lps[uid].append(old_log_probs[i])
                uid2wrong_idxs[uid].append(i)

        new_advantages = advantages.clone()

        n_total_prompts = len(uid2all_idxs)
        n_contrast = 0
        correct_fracs = []
        all_pivotality_vals = []
        all_weight_vals = []
        all_weight_stds = []
        all_frac_amplified = []
        all_piv_scales = []
        all_pos_fracs = []

        for uid in uid2all_idxs:
            correct_lps = uid2correct_lps.get(uid, [])
            correct_idxs = uid2correct_idxs.get(uid, [])
            wrong_lps = uid2wrong_lps.get(uid, [])
            wrong_idxs = uid2wrong_idxs.get(uid, [])
            all_idxs = uid2all_idxs[uid]

            correct_fracs.append(len(correct_lps) / max(len(all_idxs), 1))

            if not correct_lps or not wrong_lps:
                continue

            n_contrast += 1

            correct_masks = torch.stack([response_mask[i] for i in correct_idxs])
            wrong_masks = torch.stack([response_mask[i] for i in wrong_idxs])

            correct_lps_t = torch.stack(correct_lps) * correct_masks
            wrong_lps_t = torch.stack(wrong_lps) * wrong_masks

            correct_count = correct_masks.sum(0).clamp(min=1)
            wrong_count = wrong_masks.sum(0).clamp(min=1)

            mean_correct = correct_lps_t.sum(0) / correct_count
            mean_wrong = wrong_lps_t.sum(0) / wrong_count

            contrastive = mean_correct - mean_wrong
            pivotality = contrastive.clamp(min=0)

            union_mask = ((correct_masks.sum(0) + wrong_masks.sum(0)) > 0)
            pivotality = pivotality * union_mask.float()

            n_valid = int(union_mask.sum().item())
            n_pivotal = max(1, int(n_valid * pso_top_ratio))

            if n_valid > 0:
                n_pos = int(((pivotality > 0) & union_mask).sum().item())
                all_pos_fracs.append(n_pos / n_valid)

            if n_valid > 0 and pivotality[union_mask].max() > 0:
                threshold = torch.topk(pivotality, n_pivotal).values[-1]

                is_pivotal = (pivotality >= threshold) & union_mask & (pivotality > 0)

                piv_vals = pivotality[is_pivotal]
                n_piv = piv_vals.numel()
                if n_piv > 0:
                    order = torch.argsort(piv_vals)
                    ranks = torch.empty(n_piv, device=piv_vals.device, dtype=piv_vals.dtype)
                    ranks[order] = torch.arange(1, n_piv + 1, device=piv_vals.device, dtype=piv_vals.dtype)
                    piv_scale = torch.zeros_like(pivotality)
                    piv_scale[is_pivotal] = ranks / float(n_piv)
                    all_piv_scales.append((ranks / float(n_piv)).mean().item())
                else:
                    piv_scale = torch.zeros_like(pivotality)

                amplify = torch.tensor(pso_amplify, device=pivotality.device, dtype=pivotality.dtype)
                pso_weight = torch.ones_like(pivotality) + (amplify - 1.0) * piv_scale
            else:
                pso_weight = torch.ones_like(pivotality)

            if union_mask.any():
                valid_weights = pso_weight[union_mask]
                all_pivotality_vals.append(pivotality[union_mask].mean().item())
                all_weight_vals.append(valid_weights.mean().item())
                all_weight_stds.append(valid_weights.std().item())
                all_frac_amplified.append((valid_weights > 1.0).float().mean().item())

            for i in all_idxs:
                traj_mask = response_mask[i].bool()
                if not traj_mask.any():
                    continue
                traj_weight = torch.where(traj_mask, pso_weight, torch.ones_like(pso_weight))
                mean_w = traj_weight[traj_mask].mean().clamp(min=1e-8)
                traj_weight = traj_weight / mean_w
                new_advantages[i] = advantages[i] * traj_weight

        batch.batch["advantages"] = new_advantages

        pso_metrics = {
            "pso/contrast_frac": n_contrast / max(n_total_prompts, 1),
            "pso/correct_frac": sum(correct_fracs) / max(len(correct_fracs), 1),
        }
        if all_pivotality_vals:
            pso_metrics["pso/mean_pivotality"] = sum(all_pivotality_vals) / len(all_pivotality_vals)
        if all_weight_vals:
            pso_metrics["pso/mean_weight"] = sum(all_weight_vals) / len(all_weight_vals)
        if all_weight_stds:
            pso_metrics["pso/weight_std"] = sum(all_weight_stds) / len(all_weight_stds)
        if all_frac_amplified:
            pso_metrics["pso/frac_amplified"] = sum(all_frac_amplified) / len(all_frac_amplified)
        if all_piv_scales:
            pso_metrics["pso/mean_piv_scale"] = sum(all_piv_scales) / len(all_piv_scales)
        if all_pos_fracs:

            pso_metrics["pso/signed_pos_frac"] = sum(all_pos_fracs) / len(all_pos_fracs)

        return batch, pso_metrics
