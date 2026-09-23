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

import math
from typing import Any, Optional

def mixed_prob(alpha: float, beta: float, G: int) -> float:
    pa = pb = 1.0
    ab = alpha + beta
    for i in range(G):
        pa *= (alpha + i) / (ab + i)
        pb *= (beta + i) / (ab + i)
    return max(0.0, 1.0 - pa - pb)

class BetaTracker:
    def __init__(
        self,
        halflife_steps: float = 150.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        group_size: int = 32,
    ):
        assert halflife_steps > 0 and prior_alpha > 0 and prior_beta > 0
        self.halflife = float(halflife_steps)
        self.a0 = float(prior_alpha)
        self.b0 = float(prior_beta)
        self.G = int(group_size)

        self.state: dict[str, list] = {}
        self.n_observations = 0

    def _decayed(self, h: str, step: int) -> tuple[float, float]:
        rec = self.state.get(h)
        if rec is None:
            return self.a0, self.b0
        a, b, last = rec
        dt = max(0, int(step) - int(last))
        if dt > 0:
            lam = 2.0 ** (-dt / self.halflife)
            a = self.a0 + lam * (a - self.a0)
            b = self.b0 + lam * (b - self.b0)
        return a, b

    def observe(self, h: str, k: int, G: int, step: int) -> None:
        a, b = self._decayed(h, step)
        k = min(max(int(k), 0), int(G))
        self.state[h] = [a + k, b + (int(G) - k), int(step)]
        self.n_observations += 1

    def score(self, h: str, step: int, G: Optional[int] = None) -> float:
        a, b = self._decayed(h, step)
        return mixed_prob(a, b, self.G if G is None else int(G))

    def mean(self, h: str, step: int) -> float:
        a, b = self._decayed(h, step)
        return a / (a + b)

    def state_dict(self) -> dict[str, Any]:
        return {
            "halflife": self.halflife,
            "a0": self.a0,
            "b0": self.b0,
            "G": self.G,
            "n_observations": self.n_observations,
            "state": {h: list(v) for h, v in self.state.items()},
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.n_observations = int(sd.get("n_observations", 0))
        self.state = {h: [float(v[0]), float(v[1]), int(v[2])] for h, v in sd.get("state", {}).items()}

def prompt_hash(token_ids, pad_token_id: int) -> str:
    import hashlib

    ids = list(token_ids)
    lo, hi = 0, len(ids)
    while lo < hi and ids[lo] == pad_token_id:
        lo += 1
    while hi > lo and ids[hi - 1] == pad_token_id:
        hi -= 1
    return hashlib.md5(",".join(map(str, ids[lo:hi])).encode()).hexdigest()
