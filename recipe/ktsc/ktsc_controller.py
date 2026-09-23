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

import dataclasses
import statistics
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class KtscConfig:
    enable: bool = True

    delta_target: float = 1.124e-4

    kl_key: str = "actor/ppo_kl"

    stat: str = "median"
    window: int = 5
    ema_beta: float = 0.7

    gain: float = 0.25

    step_clamp_lo: float = 0.7
    step_clamp_hi: float = 1.0 / 0.7

    mult_min: float = 0.2
    mult_max: float = 4.0

    warmup_steps: int = 20

    seed_steps: int = 5

    adv_aware: bool = False
    adv_ref: float = 5.48

    kl_floor_frac: float = 0.01

    measure_stride: int = 4

    @classmethod
    def from_dict(cls, d: Optional[dict[str, Any]]) -> "KtscConfig":
        d = d or {}
        names = {f.name for f in dataclasses.fields(cls)}
        cfg = cls(**{k: v for k, v in d.items() if k in names})
        assert cfg.stat in ("median", "ema"), f"stat must be 'median' or 'ema', got {cfg.stat!r}"
        assert cfg.window >= 1
        return cfg

    def behaviour_key(self) -> dict[str, Any]:
        return {"stat": self.stat, "window": self.window, "kl_key": self.kl_key, "delta_target": self.delta_target}

class KtscController:

    def __init__(self, cfg: KtscConfig):
        self.cfg = cfg
        self.lr_mult: float = 1.0
        self.kl_hist: list[float] = []
        self.kl_ema: Optional[float] = None
        self.n_updates: int = 0
        self.n_held: int = 0
        self.n_seen: int = 0
        self.last: dict[str, float] = {}

    def delta_effective(self, adv_max: Optional[float]) -> float:
        d = float(self.cfg.delta_target)
        if self.cfg.adv_aware and adv_max is not None and adv_max > self.cfg.adv_ref > 0:
            d *= self.cfg.adv_ref / float(adv_max)
        return d

    def _statistic(self) -> float:
        if self.cfg.stat == "median":
            return float(statistics.median(self.kl_hist))
        return float(self.kl_ema)

    def _ingest(self, kl: float) -> None:
        self.n_seen += 1
        self.kl_hist.append(kl)
        if len(self.kl_hist) > self.cfg.window:
            del self.kl_hist[: len(self.kl_hist) - self.cfg.window]
        if self.kl_ema is None or self.cfg.ema_beta <= 0.0:
            self.kl_ema = kl
        else:
            self.kl_ema = self.cfg.ema_beta * self.kl_ema + (1.0 - self.cfg.ema_beta) * kl

    def _record(self, kl_raw: float, delta_eff: float, ratio: float, held: bool) -> None:
        self.last = {
            "kl_raw": kl_raw,
            "kl_stat": float("nan") if not self.kl_hist else self._statistic(),
            "kl_ema": float("nan") if self.kl_ema is None else self.kl_ema,
            "delta_eff": delta_eff,
            "ratio": ratio,
            "held": 1.0 if held else 0.0,
            "lr_mult_next": self.lr_mult,
        }

    def update(self, kl_measured: float, global_step: int, adv_max: Optional[float] = None) -> float:
        cfg = self.cfg
        kl_raw = float(kl_measured)
        kl = max(kl_raw, 0.0)
        delta_eff = self.delta_effective(adv_max)

        if global_step <= cfg.warmup_steps:
            self.n_held += 1
            self._record(kl_raw, delta_eff, 1.0, held=True)
            return self.lr_mult

        self._ingest(kl)
        if self.n_seen <= cfg.seed_steps:
            self.n_held += 1
            self._record(kl_raw, delta_eff, 1.0, held=True)
            return self.lr_mult

        stat_used = max(self._statistic(), delta_eff * cfg.kl_floor_frac)
        raw_ratio = (delta_eff / stat_used) ** (0.5 * cfg.gain)
        ratio = min(max(raw_ratio, cfg.step_clamp_lo), cfg.step_clamp_hi)
        self.lr_mult = min(max(self.lr_mult * ratio, cfg.mult_min), cfg.mult_max)
        self.n_updates += 1
        self._record(kl_raw, delta_eff, ratio, held=False)
        return self.lr_mult

    def metrics(self, mult_applied: float, adv_max: Optional[float] = None) -> dict[str, float]:
        out = {
            "actor/ktsc_lr_mult": float(mult_applied),
            "actor/ktsc_lr_mult_next": float(self.lr_mult),
            "actor/ktsc_delta_target": float(self.cfg.delta_target),
        }
        if adv_max is not None:
            out["actor/ktsc_adv_max"] = float(adv_max)
        for k in ("kl_raw", "kl_stat", "kl_ema", "delta_eff", "ratio", "held"):
            if k in self.last:
                out[f"actor/ktsc_{k}"] = float(self.last[k])
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "lr_mult": self.lr_mult,
            "kl_hist": list(self.kl_hist),
            "kl_ema": self.kl_ema,
            "n_updates": self.n_updates,
            "n_held": self.n_held,
            "n_seen": self.n_seen,
            "cfg": dataclasses.asdict(self.cfg),
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.lr_mult = float(sd["lr_mult"])
        self.n_updates = int(sd.get("n_updates", 0))
        self.n_held = int(sd.get("n_held", 0))
        saved_cfg = sd.get("cfg") or {}
        saved_key = {k: saved_cfg.get(k) for k in self.cfg.behaviour_key()}
        if saved_key != self.cfg.behaviour_key():

            print(f"[ktsc] controller config changed since checkpoint ({saved_key} -> {self.cfg.behaviour_key()}); "
                  "keeping lr_mult, resetting KL history and re-seeding")
            self.kl_hist, self.kl_ema, self.n_seen = [], None, 0
            return
        self.kl_hist = [float(x) for x in sd.get("kl_hist", [])][-self.cfg.window:]
        self.kl_ema = None if sd.get("kl_ema") is None else float(sd["kl_ema"])
        self.n_seen = int(sd.get("n_seen", len(self.kl_hist)))
