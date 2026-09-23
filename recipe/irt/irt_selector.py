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

import random
from typing import Any, Callable, Optional

import numpy as np
import torch

from .irt_tracker import BetaTracker, prompt_hash

class _Row:
    __slots__ = ("data", "hash", "score")

    def __init__(self, data: dict, h: str):
        self.data = data
        self.hash = h
        self.score = 0.0

def _split_rows(batch_dict: dict) -> list[dict]:
    n = None
    for v in batch_dict.values():
        n = len(v)
        break
    return [{k: v[i] for k, v in batch_dict.items()} for i in range(n)]

def _collate(rows: list[dict], template: dict) -> dict:
    out = {}
    for k in template:
        vals = [r[k] for r in rows]
        if isinstance(template[k], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            arr = np.empty(len(vals), dtype=object)
            for i, v in enumerate(vals):
                arr[i] = v
            out[k] = arr
    return out

class IrtDataLoader:
    def __init__(
        self,
        inner,
        tracker: BetaTracker,
        pad_token_id: int,
        batch_size: int,
        pool_factor: int = 4,
        epsilon: float = 0.25,
        seed: int = 1234,
        step_fn: Optional[Callable[[], int]] = None,
        log_every: int = 10,
    ):
        self.inner = inner
        self.tracker = tracker
        self.pad_token_id = int(pad_token_id)
        self.batch_size = int(batch_size)
        self.pool_target = int(pool_factor) * self.batch_size
        self.pool_max = 2 * self.pool_target
        self.epsilon = float(epsilon)
        self.rng = random.Random(seed)
        self.step_fn = step_fn or (lambda: 0)
        self.log_every = int(log_every)
        self.pool: list[_Row] = []
        self.template: Optional[dict] = None
        self.n_yielded = 0

    def __getattr__(self, name):

        return getattr(self.inner, name)

    def __len__(self):
        return len(self.inner)

    def state_dict(self):
        return self.inner.state_dict()

    def load_state_dict(self, sd):

        return self.inner.load_state_dict(sd)

    def _ingest(self, batch_dict: dict) -> None:
        if self.template is None:
            self.template = {k: v for k, v in batch_dict.items()}
        for data in _split_rows(batch_dict):
            ids = data["input_ids"]
            ids = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
            self.pool.append(_Row(data, prompt_hash(ids, self.pad_token_id)))

    def _select(self) -> list[_Row]:
        step = int(self.step_fn())
        for r in self.pool:
            r.score = self.tracker.score(r.hash, step)

        n_explore = int(round(self.epsilon * self.batch_size))
        n_top = self.batch_size - n_explore

        order = sorted(range(len(self.pool)), key=lambda i: self.pool[i].score, reverse=True)
        picked, seen = [], set()
        for i in order:
            if len(picked) >= n_top:
                break
            h = self.pool[i].hash
            if h in seen:
                continue
            seen.add(h)
            picked.append(i)
        remaining = [i for i in range(len(self.pool)) if i not in set(picked)]
        self.rng.shuffle(remaining)
        for i in remaining:
            if len(picked) >= self.batch_size:
                break
            if self.pool[i].hash in seen and len(remaining) > 2 * self.batch_size:
                continue
            seen.add(self.pool[i].hash)
            picked.append(i)

        if len(picked) < self.batch_size:
            leftover = [i for i in range(len(self.pool)) if i not in set(picked)]
            picked.extend(leftover[: self.batch_size - len(picked)])

        picked_set = set(picked)
        rows = [self.pool[i] for i in picked]
        self.pool = [r for i, r in enumerate(self.pool) if i not in picked_set]

        if len(self.pool) > self.pool_max:
            self.pool.sort(key=lambda r: r.score, reverse=True)
            del self.pool[self.pool_target:]
        return rows

    def __iter__(self):
        it = iter(self.inner)
        exhausted = False
        while True:
            while not exhausted and len(self.pool) < self.pool_target:
                try:
                    self._ingest(next(it))
                except StopIteration:
                    exhausted = True
            if len(self.pool) < self.batch_size:
                return
            rows = self._select()
            self.n_yielded += 1
            if self.log_every and self.n_yielded % self.log_every == 1:
                sc = sorted(r.score for r in rows)
                pool_sc = [r.score for r in self.pool] or [float("nan")]
                print(
                    f"[irt] round {self.n_yielded} gstep {self.step_fn()}: "
                    f"selected score min/med/max = {sc[0]:.3f}/{sc[len(sc) // 2]:.3f}/{sc[-1]:.3f}  "
                    f"pool {len(self.pool)} (mean {sum(pool_sc) / len(pool_sc):.3f})  "
                    f"tracked {len(self.tracker.state)}"
                )
            yield _collate([r.data for r in rows], self.template)

class IrtRewardObserver:
    def __init__(self, inner, tracker: BetaTracker, pad_token_id: int, step_fn: Callable[[], int], log_every: int = 10):
        self.inner = inner
        self.tracker = tracker
        self.pad_token_id = int(pad_token_id)
        self.step_fn = step_fn
        self.log_every = int(log_every)
        self.n_calls = 0

    def __call__(self, batch, *args, **kwargs) -> Any:
        out = self.inner(batch, *args, **kwargs)
        try:
            self._record(batch, out, bool(kwargs.get("return_dict", False)))
        except Exception as e:
            print(f"[irt] WARNING: observer failed ({type(e).__name__}: {e}); skipping this round")
        return out

    def _record(self, batch, out, is_dict: bool) -> None:
        nt = getattr(batch, "non_tensor_batch", None)
        bt = getattr(batch, "batch", None)
        if nt is None or bt is None or "uid" not in nt.keys() or "prompts" not in bt.keys():
            return

        uids = nt["uid"]
        if is_dict and isinstance(out, dict) and "reward_extra_info" in out and "acc" in out["reward_extra_info"]:
            correct = [float(a) > 0 for a in out["reward_extra_info"]["acc"]]
        else:
            rt = out["reward_tensor"] if is_dict and isinstance(out, dict) else out
            correct = (rt.sum(dim=-1) > 0).tolist()

        prompts = bt["prompts"]
        step = int(self.step_fn())
        groups: dict[str, list] = {}
        first_row: dict[str, int] = {}
        for i, u in enumerate(uids):
            groups.setdefault(u, []).append(bool(correct[i]))
            first_row.setdefault(u, i)

        mixed = 0
        for u, ks in groups.items():
            h = prompt_hash(prompts[first_row[u]].tolist(), self.pad_token_id)
            k = sum(ks)
            self.tracker.observe(h, k, len(ks), step)
            if 0 < k < len(ks):
                mixed += 1

        self.n_calls += 1
        if self.log_every and self.n_calls % self.log_every == 1:
            print(
                f"[irt] obs round {self.n_calls} gstep {step}: groups {len(groups)}  "
                f"mixed {mixed} ({mixed / max(len(groups), 1):.1%})  "
                f"tracked {len(self.tracker.state)}  total obs {self.tracker.n_observations}"
            )
