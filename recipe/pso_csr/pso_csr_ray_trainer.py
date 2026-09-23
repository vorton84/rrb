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

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip

class RayPSOCSRTrainer(RayPPOTrainer):

    def _apply_csr_rewards(self, batch):
        old_log_probs = batch.batch["old_log_probs"]
        token_rewards = batch.batch["token_level_rewards"]
        response_mask = batch.batch["response_mask"]
        uids = batch.non_tensor_batch["uid"]

        scores = token_rewards.sum(dim=-1)
        csr_alpha = self.config.algorithm.get("csr_alpha", 0.3)

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

            first_mask = response_mask[all_idxs[0]].bool()
            valid_contrastive = contrastive[first_mask]
            std = valid_contrastive.std() + 1e-8
            all_contrastive_stds.append(std.item())
            contrastive_norm = (contrastive / std) * response_mask[all_idxs[0]].float()

            for i in all_idxs:
                sign = 1.0 if scores[i] > 0 else -1.0
                csr_step = csr_alpha * sign * contrastive_norm * response_mask[i].float()
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

        new_advantages = advantages.clone()

        n_total_prompts = len(uid2all_idxs)
        n_contrast = 0
        correct_fracs = []
        all_pivotality_vals = []
        all_weight_vals = []

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
            pivotality = (mean_correct - mean_wrong).abs()

            mask = response_mask[all_idxs[0]]
            pivotality = pivotality * mask

            n_valid = int(mask.sum().item())
            n_pivotal = max(1, int(n_valid * pso_top_ratio))

            if pivotality.max() > 0:
                threshold = torch.topk(pivotality, n_pivotal).values[-1]
                pso_weight = torch.where(
                    pivotality >= threshold,
                    torch.tensor(pso_amplify, device=pivotality.device, dtype=pivotality.dtype),
                    torch.ones(1, device=pivotality.device, dtype=pivotality.dtype),
                )
            else:
                pso_weight = torch.ones_like(pivotality)

            valid_mask = mask.bool()
            if valid_mask.any():
                all_pivotality_vals.append(pivotality[valid_mask].mean().item())
                all_weight_vals.append(pso_weight[valid_mask].mean().item())

            for i in all_idxs:
                new_advantages[i] = advantages[i] * pso_weight

        batch.batch["advantages"] = new_advantages

        pso_metrics = {
            "pso/contrast_frac": n_contrast / max(n_total_prompts, 1),
            "pso/correct_frac": sum(correct_fracs) / max(len(correct_fracs), 1),
        }
        if all_pivotality_vals:
            pso_metrics["pso/mean_pivotality"] = sum(all_pivotality_vals) / len(all_pivotality_vals)
        if all_weight_vals:
            pso_metrics["pso/mean_weight"] = sum(all_weight_vals) / len(all_weight_vals)

        return batch, pso_metrics

    def fit(self):
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.trainer.profile_steps
            if self.config.trainer.profile_steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            new_batch.batch["reward_baselines"] = reward_baseline_tensor
                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                self.gen_steps += 1
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    batch, csr_metrics = self._apply_csr_rewards(batch)
                    metrics.update(csr_metrics)

                    with marked_timer("adv", timing_raw, "brown"):
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    batch, pso_metrics = self._apply_pso_weights(batch)
                    metrics.update(pso_metrics)

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    global_log_buffer = None
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)

                        if "global_log_buffer" in actor_output.non_tensor_batch:
                            global_log_buffer = actor_output.non_tensor_batch.pop("global_log_buffer")

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                        if global_log_buffer is not None:
                            import json
                            import os
                            save_path = os.path.join(
                                rollout_data_dir,
                                f"negative_approx_kl_global_step_{self.global_steps}.jsonl",
                            )
                            with open(save_path, "w") as f:
                                for item in global_log_buffer:
                                    neg_kl = item["negative_approx_kl"]
                                    resps = item["responses"]
                                    if isinstance(neg_kl, torch.Tensor):
                                        neg_kl = neg_kl.tolist()
                                    if isinstance(resps, torch.Tensor):
                                        resps = resps.tolist()
                                    if (
                                        isinstance(neg_kl, list)
                                        and isinstance(resps, list)
                                        and len(neg_kl) == len(resps)
                                        and not isinstance(neg_kl[0], list)
                                    ):
                                        for nk, r in zip(neg_kl, resps):
                                            json.dump({"negative_approx_kl": nk, "response": r}, f)
                                            f.write("\n")
                                    else:
                                        json.dump({"negative_approx_kl": neg_kl, "responses": resps}, f)
                                        f.write("\n")

                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.trainer.profile_steps
                        if self.config.trainer.profile_steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
