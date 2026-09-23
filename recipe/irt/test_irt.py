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
import math
import random
import sys

from recipe.irt.irt_tracker import BetaTracker, mixed_prob, prompt_hash

RESULTS: list[tuple[str, bool]] = []

def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

def part_a():
    print("\nPart A -- discounted-Beta tracker")
    G = 32

    check("A1 prior score", abs(mixed_prob(1, 1, G) - (1 - 2 / 33)) < 1e-12, f"{mixed_prob(1, 1, G):.6f}")

    e_fail = 1.0
    for i in range(G):
        e_fail *= (33 + i) / (34 + i)
    check("A1 one all-wrong visit", abs(mixed_prob(1, 33, G) - (1 - e_fail - mixed_prob_ep(1, 33, G))) < 1e-12,
          f"score={mixed_prob(1, 33, G):.4f} (expect ~0.492)")

    rng = random.Random(0)
    for a, b in [(1.0, 1.0), (2.5, 40.0), (17.0, 3.0)]:
        n = 200_000
        acc = 0.0
        for _ in range(n):
            p = rng.betavariate(a, b)
            acc += 1.0 - p**G - (1 - p) ** G
        mc = acc / n
        cf = mixed_prob(a, b, G)
        check(f"A2 MC agreement Beta({a},{b})", abs(mc - cf) < 5e-3, f"closed {cf:.4f} vs MC {mc:.4f}")

    tr = BetaTracker(halflife_steps=150, group_size=G)
    tr.observe("h", 8, 32, step=10)
    check("A3 posterior mean after 8/32", abs(tr.mean("h", 10) - 9 / 34) < 1e-12, f"{tr.mean('h', 10):.4f}")
    tr.observe("h2", 99, 32, step=10)
    check("A3 k clamped to G", abs(tr.mean("h2", 10) - 33 / 34) < 1e-12)

    tr = BetaTracker(halflife_steps=100, group_size=G)
    tr.observe("h", 0, 32, step=0)
    a, b = tr._decayed("h", 100)
    check("A4 half-life halves evidence", abs(a - 1.0) < 1e-12 and abs(b - (1 + 0.5 * 32)) < 1e-12, f"beta={b:.2f}")

    s0, s1, s2 = tr.score("h", 0), tr.score("h", 100), tr.score("h", 1000)
    check("A4 hopeless prompt revives toward prior", s0 < s1 < s2 and abs(s2 - mixed_prob(1, 1, G)) < 0.01,
          f"{s0:.3f} -> {s1:.3f} -> {s2:.3f} (prior {mixed_prob(1, 1, G):.3f})")

    tr = BetaTracker(halflife_steps=1e9, group_size=G)
    scores = []
    for t in range(3):
        scores.append(tr.score("h", t))
        tr.observe("h", 0, 32, step=t)
    check("A5 monotone demotion under all-wrong", scores[0] > scores[1] > scores[2],
          " -> ".join(f"{s:.3f}" for s in scores))

    tr = BetaTracker(halflife_steps=150, group_size=G)
    for i in range(50):
        tr.observe(f"h{i}", i % 33, 32, step=i)
    sd = json.loads(json.dumps(tr.state_dict()))
    tr2 = BetaTracker(halflife_steps=150, group_size=G)
    tr2.load_state_dict(sd)
    same = all(abs(tr.score(f"h{i}", 60) - tr2.score(f"h{i}", 60)) < 1e-12 for i in range(50))
    check("A6 state round trip", same and tr2.n_observations == 50)

    pad = 151643
    base = [5, 6, 7, 8]
    h1 = prompt_hash([pad, pad] + base, pad)
    h2 = prompt_hash(base + [pad, pad, pad], pad)
    h3 = prompt_hash([pad] + base + [pad], pad)
    h4 = prompt_hash([5, 6, 7, 9], pad)
    check("A7 hash pad-invariant", h1 == h2 == h3)
    check("A7 hash content-sensitive", h4 != h1)
    check("A7 interior pad preserved", prompt_hash([5, pad, 6], pad) != prompt_hash([5, 6], pad))

def mixed_prob_ep(a, b, G):
    v = 1.0
    for i in range(G):
        v *= (a + i) / (a + b + i)
    return v

def part_b():
    print("\nPart B -- IrtDataLoader")
    try:
        import numpy as np
        import torch

        from recipe.irt.irt_selector import IrtDataLoader, _collate, _split_rows
    except Exception as e:
        print(f"  [SKIP] torch not importable here ({e.__class__.__name__}); run Part B on the pod")
        return

    PAD = 0

    def make_batch(ids_rows):
        n = len(ids_rows)
        width = max(len(r) for r in ids_rows)
        ii = torch.full((n, width), PAD, dtype=torch.long)
        for i, r in enumerate(ids_rows):
            ii[i, width - len(r):] = torch.tensor(r)
        raw = np.empty(n, dtype=object)
        for i, r in enumerate(ids_rows):
            raw[i] = list(r)
        return {"input_ids": ii, "attention_mask": (ii != PAD).long(), "raw_prompt_ids": raw}

    bd = make_batch([[1, 2], [3, 4, 5], [6]])
    rows = _split_rows(bd)
    rb = _collate(rows, bd)
    check("B1 split/collate round trip",
          torch.equal(rb["input_ids"], bd["input_ids"]) and list(rb["raw_prompt_ids"]) == list(bd["raw_prompt_ids"]))

    class FakeInner:

        def __init__(self, batches):
            self.batches = batches
            self.state_calls = 0

        def __iter__(self):
            return iter(self.batches)

        def __len__(self):
            return len(self.batches)

        def state_dict(self):
            self.state_calls += 1
            return {"pos": 7}

        def load_state_dict(self, sd):
            self.loaded = sd

    class FakeTracker:

        def __init__(self):
            self.state = {}

        def score(self, h, step, G=None):
            return self.scores.get(h, 0.5)

    prompts = [[100 + i] for i in range(40)]
    batches = [make_batch(prompts[i:i + 8]) for i in range(0, 40, 8)]
    from recipe.irt.irt_tracker import prompt_hash as ph

    tr = FakeTracker()
    tr.scores = {ph(p, PAD): p[0] / 1000 for p in prompts}

    tr.scores[ph([138], PAD)] = 0.99
    tr.scores[ph([139], PAD)] = 0.98

    dl = IrtDataLoader(FakeInner(batches), tr, PAD, batch_size=4, pool_factor=10, epsilon=0.0, seed=0, log_every=0)
    out = list(iter(dl))
    check("B2 yields full batches of batch_size", all(len(b["input_ids"]) == 4 for b in out), f"{len(out)} batches")
    first_ids = sorted(int(x) for x in out[0]["input_ids"].max(dim=1).values)
    check("B2 top scorers selected first (whole stream pooled)",
          first_ids == [136, 137, 138, 139], f"first batch ids {first_ids}")

    dl_small = IrtDataLoader(FakeInner(batches), tr, PAD, batch_size=4, pool_factor=2, epsilon=0.0, seed=0, log_every=0)
    ids_small = sorted(int(x) for x in next(iter(dl_small))["input_ids"].max(dim=1).values)
    check("B2 small pool = top-k of the 8 arrived rows", ids_small == [104, 105, 106, 107], f"{ids_small}")

    dup_prompts = [[7], [8]] * 20
    dup_batches = [make_batch(dup_prompts[i:i + 8]) for i in range(0, 40, 8)]
    tr2 = FakeTracker()
    tr2.scores = {ph([7], PAD): 0.9, ph([8], PAD): 0.8}
    dl2 = IrtDataLoader(FakeInner(dup_batches), tr2, PAD, batch_size=2, pool_factor=2, epsilon=0.0, seed=0, log_every=0)
    b0 = next(iter(dl2))
    ids0 = sorted(int(x) for x in b0["input_ids"].max(dim=1).values)
    check("B3 dedup within a batch", ids0 == [7, 8], f"{ids0}")

    only7 = [make_batch([[7]] * 8)]
    dl2b = IrtDataLoader(FakeInner(only7), tr2, PAD, batch_size=2, pool_factor=1, epsilon=0.0, seed=0, log_every=0)
    ids7 = sorted(int(x) for x in next(iter(dl2b))["input_ids"].max(dim=1).values)
    check("B3 fallback fills with duplicates when nothing else exists", ids7 == [7, 7], f"{ids7}")

    tr3 = FakeTracker()
    tr3.scores = {ph(p, PAD): p[0] / 1000 for p in prompts}
    dl3 = IrtDataLoader(FakeInner([make_batch(prompts)]), tr3, PAD, batch_size=4, pool_factor=1, epsilon=1.0, seed=3, log_every=0)
    b0 = next(iter(dl3))
    ids0 = sorted(int(x) for x in b0["input_ids"].max(dim=1).values)
    check("B4 epsilon=1 explores (not the top-4)", ids0 != [136, 137, 138, 139], f"{ids0}")

    prompts42 = [[200 + i] for i in range(42)]
    batches42 = [make_batch(prompts42[i:i + 7]) for i in range(0, 42, 7)]
    inner = FakeInner(batches42)
    dl4 = IrtDataLoader(inner, tr, PAD, batch_size=4, pool_factor=2, epsilon=0.0, seed=0, log_every=0)
    e1 = list(iter(dl4))
    pool_after_e1 = len(dl4.pool)
    e2_first = next(iter(dl4), None)
    check("B5 pool persists across epochs", len(e1) == 10 and pool_after_e1 == 2 and e2_first is not None,
          f"{len(e1)} batches, pool {pool_after_e1} after epoch 1")
    check("B5 state_dict forwarded", dl4.state_dict() == {"pos": 7} and inner.state_calls == 1)
    dl4.load_state_dict({"pos": 3})
    check("B5 load_state_dict forwarded", inner.loaded == {"pos": 3})

    many = [[500 + i] for i in range(400)]
    mb = [make_batch(many[i:i + 40]) for i in range(0, 400, 40)]
    tr4 = FakeTracker()
    tr4.scores = {}
    dl5 = IrtDataLoader(FakeInner(mb), tr4, PAD, batch_size=4, pool_factor=2, epsilon=0.0, seed=0, log_every=0)
    for _ in iter(dl5):
        check_pool = len(dl5.pool) <= dl5.pool_max
        if not check_pool:
            break
    check("B6 pool bounded by pool_max", len(dl5.pool) <= dl5.pool_max, f"{len(dl5.pool)} <= {dl5.pool_max}")

def part_c():
    print("\nPart C -- IrtRewardObserver")
    try:
        import numpy as np
        import torch

        from recipe.irt.irt_selector import IrtRewardObserver
    except Exception as e:
        print(f"  [SKIP] torch not importable here ({e.__class__.__name__}); run Part C on the pod")
        return

    PAD = 0

    class FakeBatch:
        def __init__(self, uids, prompt_rows, has_prompts=True):
            n = len(uids)
            width = max(len(r) for r in prompt_rows)
            pm = torch.full((n, width), PAD, dtype=torch.long)
            for i, r in enumerate(prompt_rows):
                pm[i, width - len(r):] = torch.tensor(r)
            self.non_tensor_batch = {"uid": np.array(uids, dtype=object)}
            self.batch = {"prompts": pm} if has_prompts else {}

    tr = BetaTracker(halflife_steps=1e9, group_size=4)

    uids = ["u1"] * 4 + ["u2"] * 4
    rows = [[11]] * 4 + [[22]] * 4
    rt = torch.tensor([1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]).unsqueeze(1)

    def inner_fn(batch, return_dict=False):
        return {"reward_tensor": rt, "reward_extra_info": {"acc": [1, 1, 0, 0, 0, 0, 0, 0]}} if return_dict else rt

    obs = IrtRewardObserver(inner_fn, tr, PAD, step_fn=lambda: 5, log_every=0)
    out = obs(FakeBatch(uids, rows), return_dict=True)
    check("C1 inner result passed through", isinstance(out, dict) and "reward_tensor" in out)
    h11 = prompt_hash([11], PAD)
    h22 = prompt_hash([22], PAD)
    check("C1 acc path recorded 2/4 and 0/4",
          abs(tr.mean(h11, 5) - 3 / 6) < 1e-12 and abs(tr.mean(h22, 5) - 1 / 6) < 1e-12,
          f"means {tr.mean(h11, 5):.3f}, {tr.mean(h22, 5):.3f}")

    tr2 = BetaTracker(halflife_steps=1e9, group_size=4)
    obs2 = IrtRewardObserver(inner_fn, tr2, PAD, step_fn=lambda: 5, log_every=0)
    obs2(FakeBatch(uids, rows))
    check("C2 fallback path records", abs(tr2.mean(h11, 5) - 3 / 6) < 1e-12 and tr2.n_observations == 2)

    tr3 = BetaTracker(halflife_steps=1e9, group_size=4)
    obs3 = IrtRewardObserver(inner_fn, tr3, PAD, step_fn=lambda: 5, log_every=0)

    class NoUid:
        non_tensor_batch = {}
        batch = {}

    obs3(NoUid(), return_dict=True)
    check("C3 no-uid batch skipped", tr3.n_observations == 0)

    def bad_inner(batch, return_dict=False):
        return {"reward_tensor": "not a tensor"} if return_dict else rt

    obs4 = IrtRewardObserver(bad_inner, tr3, PAD, step_fn=lambda: 5, log_every=0)
    try:
        obs4(FakeBatch(uids, rows), return_dict=True)
        check("C4 observer failure contained", True)
    except Exception as e:
        check("C4 observer failure contained", False, str(e))

def main():
    part_a()
    part_b()
    part_c()
    passed = sum(1 for _, ok in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n{'=' * 60}\n{passed}/{total} passed")
    if passed != total:
        print("FAILED -- do not launch training")
        return 1
    print("All checks passed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
