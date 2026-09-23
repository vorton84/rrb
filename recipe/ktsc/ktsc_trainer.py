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
from typing import Any, Optional

import numpy as np
from omegaconf import OmegaConf

from .ktsc_controller import KtscConfig, KtscController

KTSC_STATE_FILE = "ktsc_state.json"

class KtscTrainerMixin:

    def init_workers(self):
        super().init_workers()
        self._ktsc_install()

    def _ktsc_build(self) -> None:
        raw = OmegaConf.select(self.config, "trainer.ktsc", default=None)
        cfg_dict = OmegaConf.to_container(raw, resolve=True) if raw is not None else {}
        if not isinstance(cfg_dict, dict):
            cfg_dict = {}
        self.ktsc_cfg = KtscConfig.from_dict(cfg_dict)
        self.ktsc = KtscController(self.ktsc_cfg)

    def _ktsc_install(self) -> None:
        self._ktsc_build()
        if not self.ktsc_cfg.enable:
            print("[ktsc] disabled by config (trainer.ktsc.enable=false); update_actor left untouched")
            return
        wg = self.actor_rollout_wg
        assert hasattr(wg, "update_actor"), "actor worker group has no update_actor to wrap"
        self._ktsc_orig_update_actor = wg.update_actor
        wg.update_actor = self._ktsc_update_actor
        print(
            f"[ktsc] installed: delta_target={self.ktsc_cfg.delta_target:.3e} kl_key={self.ktsc_cfg.kl_key} "
            f"gain={self.ktsc_cfg.gain} ema_beta={self.ktsc_cfg.ema_beta} "
            f"warmup_steps={self.ktsc_cfg.warmup_steps} adv_aware={self.ktsc_cfg.adv_aware} "
            f"measure_stride={self.ktsc_cfg.measure_stride}"
        )

    @staticmethod
    def _ktsc_adv_max(batch) -> Optional[float]:
        try:
            adv = batch.batch["advantages"]
        except Exception:
            return None
        if adv is None or adv.numel() == 0:
            return None
        mask = batch.batch.get("response_mask", None) if hasattr(batch.batch, "get") else None
        if mask is not None:
            adv = adv * mask.to(adv.dtype)
        return float(adv.abs().max().item())

    def _ktsc_update_actor(self, batch):
        cfg, ctl = self.ktsc_cfg, self.ktsc
        mult_applied = ctl.lr_mult
        adv_max = self._ktsc_adv_max(batch)

        batch.meta_info["ktsc_lr_mult"] = float(mult_applied)
        batch.meta_info["ktsc_measure_stride"] = int(cfg.measure_stride)

        out = self._ktsc_orig_update_actor(batch)

        m = out.meta_info.setdefault("metrics", {})
        vals = m.get(cfg.kl_key)
        if vals is None or len(vals) == 0:
            print(f"[ktsc] WARNING: metric {cfg.kl_key!r} missing from actor output; holding lr_mult")
        else:
            kl = float(np.mean(vals))
            ctl.update(kl, global_step=int(self.global_steps), adv_max=adv_max)

        for k, v in ctl.metrics(mult_applied=mult_applied, adv_max=adv_max).items():
            m[k] = [v]
        return out

    def _ktsc_state_path(self) -> str:
        return os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}", KTSC_STATE_FILE)

    def _save_checkpoint(self):
        super()._save_checkpoint()
        if not getattr(self, "ktsc", None) or not self.ktsc_cfg.enable:
            return
        path = self._ktsc_state_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.ktsc.state_dict(), f, indent=2)
        print(f"[ktsc] saved controller state to {path} (lr_mult={self.ktsc.lr_mult:.4f})")

    def _load_checkpoint(self):
        super()._load_checkpoint()
        if not getattr(self, "ktsc", None) or not self.ktsc_cfg.enable:
            return
        if int(getattr(self, "global_steps", 0)) <= 0:
            return
        path = self._ktsc_state_path()
        if os.path.isfile(path):
            with open(path) as f:
                self.ktsc.load_state_dict(json.load(f))
            print(f"[ktsc] restored controller state from {path} (lr_mult={self.ktsc.lr_mult:.4f})")
        else:
            print(
                f"[ktsc] WARNING: resuming at step {self.global_steps} but {path} not found; "
                "controller starts fresh at lr_mult=1.0 and re-seeds its EMA"
            )
