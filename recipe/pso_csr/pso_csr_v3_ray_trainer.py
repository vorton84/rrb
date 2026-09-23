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

from .pso_csr_ray_trainer import RayPSOCSRTrainer

class RayPSOCSRv3Trainer(RayPSOCSRTrainer):

    def _apply_csr_rewards(self, batch):
        old_log_probs = batch.batch["old_log_probs"]
        token_rewards = batch.batch["token_level_rewards"]
        response_mask = batch.batch["response_mask"]
        uids = batch.non_tensor_batch["uid"]

        scores = token_rewards.sum(dim=-1)
        csr_alpha = self.config.algorithm.get("csr_alpha", 0.05)
        pso_top_ratio = self.config.algorithm.get("pso_top_ratio", 0.2)

        uid2correct_lps = defaultdict(list)
        uid2wrong_lps = defaultdict(list)
        uid2all_idxs = defaultdict(list)

        bsz = scores.shape[0]
        for i in range(bsz):
            uid = uids[i]
            uid2all_idxs[uid].append(i)
            if scores[i] > 0:
                uid2correct_lps[uid].append(old_log_probs[i])
            else:
                uid2wrong_lps[uid].append(old_log_probs[i])

        new_rewards = token_rewards.clone()

        n_total_prompts = len(uid2all_idxs)
        n_contrast = 0
        correct_fracs = []
        all_bonus_sums = []
        all_outcome_abs = []
        all_contrastive_stds = []

        for uid in uid2all_idxs:
            correct_lps = uid2correct_lps.get(uid, [])
            wrong_lps = uid2wrong_lps.get(uid, [])
            all_idxs = uid2all_idxs[uid]

            correct_fracs.append(len(correct_lps) / max(len(all_idxs), 1))

            if not correct_lps or not wrong_lps:
                continue

            n_contrast += 1
            mean_correct = torch.stack(correct_lps).mean(0)
            mean_wrong = torch.stack(wrong_lps).mean(0)
            contrastive = mean_correct - mean_wrong

            mask = response_mask[all_idxs[0]]
            pivotality = contrastive.abs() * mask

            n_valid = int(mask.sum().item())
            n_pivotal = max(1, int(n_valid * pso_top_ratio))
            if pivotality.max() > 0:
                threshold = torch.topk(pivotality, n_pivotal).values[-1]
                pivotal_mask = (pivotality >= threshold).float()
            else:
                pivotal_mask = torch.zeros_like(pivotality)

            first_mask_bool = mask.bool()
            valid_contrastive = contrastive[first_mask_bool]
            std = valid_contrastive.std() + 1e-8
            all_contrastive_stds.append(std.item())
            contrastive_norm = (contrastive / std) * mask.float()
            contrastive_at_pivotal = contrastive_norm * pivotal_mask

            for i in all_idxs:
                sign = 1.0 if scores[i] > 0 else -1.0
                csr_step = csr_alpha * sign * contrastive_at_pivotal * response_mask[i].float()
                new_rewards[i] = token_rewards[i] + csr_step
                all_bonus_sums.append(csr_step.abs().sum().item())
                all_outcome_abs.append(abs(scores[i].item()))

        batch.batch["token_level_rewards"] = new_rewards

        csr_metrics = {
            "csr/contrast_frac": n_contrast / max(n_total_prompts, 1),
            "csr/correct_frac": sum(correct_fracs) / max(len(correct_fracs), 1),
        }
        if all_bonus_sums:
            mean_bonus = sum(all_bonus_sums) / len(all_bonus_sums)
            mean_outcome = sum(all_outcome_abs) / max(len(all_outcome_abs), 1)
            csr_metrics["csr/mean_bonus_sum"] = mean_bonus
            csr_metrics["csr/bonus_to_outcome_ratio"] = mean_bonus / (mean_outcome + 1e-8)
        if all_contrastive_stds:
            csr_metrics["csr/mean_contrastive_std"] = sum(all_contrastive_stds) / len(all_contrastive_stds)

        return batch, csr_metrics
