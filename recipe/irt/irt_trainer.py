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

import json
import os
from typing import Any

from omegaconf import OmegaConf

from .irt_selector import IrtDataLoader, IrtRewardObserver
from .irt_tracker import BetaTracker

IRT_STATE_FILE = "irt_state.json"

_DEFAULTS = {
    "enable": True,
    "pool_factor": 4,
    "epsilon": 0.25,
    "halflife_steps": 150.0,
    "prior_alpha": 1.0,
    "prior_beta": 1.0,
    "seed": 1234,
    "log_every": 10,
}

class IrtTrainerMixin:
    def fit(self):
        self._irt_install()
        return super().fit()

    def _irt_cfg(self) -> dict:
        raw = OmegaConf.select(self.config, "trainer.irt", default=None)
        d = OmegaConf.to_container(raw, resolve=True) if raw is not None else {}
        cfg = dict(_DEFAULTS)
        cfg.update({k: v for k, v in (d or {}).items() if k in _DEFAULTS})
        return cfg

    def _irt_install(self) -> None:
        cfg = self._irt_cfg()
        self.irt_cfg = cfg
        if not cfg["enable"]:
            print("[irt] disabled by config; dataloader and reward_fn untouched")
            self.irt_tracker = None
            return

        pad_id = self.tokenizer.pad_token_id
        assert pad_id is not None, "IRT hashing needs tokenizer.pad_token_id"
        group_size = int(self.config.actor_rollout_ref.rollout.n)
        gen_bsz = int(self.config.data.get("gen_batch_size", self.config.data.train_batch_size))
        step_fn = lambda: int(getattr(self, "global_steps", 0))

        self.irt_tracker = BetaTracker(
            halflife_steps=cfg["halflife_steps"],
            prior_alpha=cfg["prior_alpha"],
            prior_beta=cfg["prior_beta"],
            group_size=group_size,
        )
        self.train_dataloader = IrtDataLoader(
            inner=self.train_dataloader,
            tracker=self.irt_tracker,
            pad_token_id=pad_id,
            batch_size=gen_bsz,
            pool_factor=cfg["pool_factor"],
            epsilon=cfg["epsilon"],
            seed=cfg["seed"],
            step_fn=step_fn,
            log_every=cfg["log_every"],
        )
        self.reward_fn = IrtRewardObserver(
            inner=self.reward_fn,
            tracker=self.irt_tracker,
            pad_token_id=pad_id,
            step_fn=step_fn,
            log_every=cfg["log_every"],
        )
        print(
            f"[irt] installed: gen_bsz={gen_bsz} G={group_size} pool={cfg['pool_factor']}x "
            f"epsilon={cfg['epsilon']} halflife={cfg['halflife_steps']} "
            f"prior=Beta({cfg['prior_alpha']},{cfg['prior_beta']}) pad_id={pad_id}"
        )

    def _irt_state_path(self) -> str:
        return os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}", IRT_STATE_FILE)

    def _save_checkpoint(self):
        super()._save_checkpoint()
        if getattr(self, "irt_tracker", None) is None:
            return
        path = self._irt_state_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.irt_tracker.state_dict(), f)
        print(f"[irt] saved tracker ({len(self.irt_tracker.state)} prompts) to {path}")

    def _load_checkpoint(self):
        super()._load_checkpoint()
        if getattr(self, "irt_tracker", None) is None:
            return
        if int(getattr(self, "global_steps", 0)) <= 0:
            return
        path = self._irt_state_path()
        if os.path.isfile(path):
            with open(path) as f:
                self.irt_tracker.load_state_dict(json.load(f))
            print(f"[irt] restored tracker ({len(self.irt_tracker.state)} prompts) from {path}")
        else:
            print(f"[irt] WARNING: resuming at step {self.global_steps} without {path}; tracker starts empty")
